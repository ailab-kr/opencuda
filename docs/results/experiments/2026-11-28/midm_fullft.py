# Is full fine-tuning of Mi:dm 2.0 Mini possible and sound on one Intel Arc Pro B70 (PyTorch XPU)?
#   mode mixed: fp32 weights and gradients, bf16 compute (autocast), 8-bit AdamW (bitsandbytes)
#   mode bf16:  bf16 weights, gradients and AdamW states (small updates may round away)
# Every parameter is trained. Reports per step: loss, time; then peak memory, tokens/s, the share
# of weights a step actually changed, NaN/inf checks, and a generation after training.
import sys, time, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name, mode = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
seq = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
ckpt = (sys.argv[5] if len(sys.argv) > 5 else "on") == "on"
lr = 1e-5
dev = "xpu"
tok = AutoTokenizer.from_pretrained(name)
dtype = torch.float32 if mode == "mixed" else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").to(dev)
if ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
nparam = sum(p.numel() for p in model.parameters())
print(f"{name}: {nparam/1e9:.2f}B parameters, all trainable; mode {mode}, seq {seq}, checkpointing {ckpt}", flush=True)
params = [p for p in model.parameters() if p.requires_grad]
if mode == "mixed":
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
else:
    try:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
        print("optimizer: fused AdamW", flush=True)
    except Exception as e:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=True)
        print(f"optimizer: foreach AdamW (fused not available: {e})", flush=True)

# code-like text: this file and the transformers modeling code, packed into one stream
import transformers.models.llama.modeling_llama as ml
text = (open(__file__).read() + open(ml.__file__).read()) * 8
ids_all = tok(text, return_tensors="pt")["input_ids"][0]
print(f"training stream: {ids_all.numel()} tokens", flush=True)

probe = [p for p in params if p.dim() == 2][:8]   # weight matrices to watch
model.train()
torch.xpu.reset_peak_memory_stats()
t0, ntok, losses, bad = None, 0, [], 0
for step in range(steps):
    off = (step * 1013) % (ids_all.numel() - seq - 1)
    ids = ids_all[off:off + seq].unsqueeze(0).to(dev)
    if step == 2:
        torch.xpu.synchronize(); t0 = time.time(); ntok = 0
    ts = time.time()
    with torch.autocast("xpu", dtype=torch.bfloat16, enabled=(mode == "mixed")):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    before = [p.detach().clone() for p in probe] if step == steps - 1 else None
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.xpu.synchronize()
    lv = loss.item()
    bad += int(not math.isfinite(lv) or not math.isfinite(float(gn)))
    losses.append(lv)
    if t0 is not None: ntok += seq
    print(f"step {step:2d} loss {lv:.4f} grad_norm {float(gn):.3f} {time.time()-ts:.2f}s peak {torch.xpu.max_memory_allocated()/2**30:.1f} GiB", flush=True)
changed = sum(int((b != p.detach()).sum()) for b, p in zip(before, probe)) / sum(p.numel() for p in probe)
torch.xpu.synchronize()
dt = time.time() - t0
nonfinite = sum(int((~torch.isfinite(p)).sum()) for p in params)
print(f"RESULT {mode} seq {seq} ckpt {ckpt}: {ntok/dt:.0f} tokens/s, peak {torch.xpu.max_memory_allocated()/2**30:.1f} GiB, "
      f"loss {losses[0]:.3f} -> {losses[-1]:.3f} (min {min(losses):.3f}), non-finite steps {bad}, non-finite weights {nonfinite}, "
      f"weights changed by the last step {100*changed:.1f}%", flush=True)
model.eval()
msgs = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}]
p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
with torch.no_grad(), torch.autocast("xpu", dtype=torch.bfloat16, enabled=(mode == "mixed")):
    g = model.generate(**p, max_new_tokens=80, do_sample=False)
print("GEN:", tok.decode(g[0][p["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " | ")[:300], flush=True)

# LoRA fine-tuning of Mi:dm 2.0 on an Intel GPU with PyTorch XPU (no CUDA, no OpenCuda).
# Small synthetic Korean instruction data; measures loss, throughput, peak memory.
import sys, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

name = sys.argv[1] if len(sys.argv) > 1 else "K-intelligence/Midm-2.0-Mini-Instruct"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
seq, bs = 512, 4
dev = "xpu"
assert torch.xpu.is_available()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev)
if "Base" in name:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

random.seed(0)
facts = [("OpenCuda", "CUDA 프로그램을 고치지 않고 다른 칩에서 실행하는 오픈소스 도구"),
         ("Arc Pro B70", "Intel의 32GB 워크스테이션 GPU"),
         ("SPIR-V", "Khronos가 정의한 GPU 중간 표현"),
         ("HeCBench", "이종 컴퓨팅 벤치마크 프로그램 모음")]
def sample():
    k, v = random.choice(facts)
    msgs = [{"role": "user", "content": f"{k}이(가) 무엇인지 한 문장으로 설명해 주세요."},
            {"role": "assistant", "content": f"{k}은(는) {v}입니다."}]
    return tok.apply_chat_template(msgs, tokenize=False)
def batch():
    texts = [" ".join(sample() for _ in range(12)) for _ in range(bs)]
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=seq, padding="max_length")
    ids = enc["input_ids"].to(dev); att = enc["attention_mask"].to(dev)
    labels = ids.masked_fill(att == 0, -100)
    return ids, att, labels

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
torch.xpu.reset_peak_memory_stats()
losses, t0, ntok = [], None, 0
for step in range(steps):
    ids, att, labels = batch()
    if step == 2:  # after warm-up (kernel compilation)
        torch.xpu.synchronize(); t0 = time.time(); ntok = 0
    out = model(input_ids=ids, attention_mask=att, labels=labels)
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    losses.append(out.loss.item())
    if t0 is not None: ntok += int(att.sum())
    if step % 5 == 0 or step == steps - 1:
        print(f"step {step:3d} loss {out.loss.item():.4f}", flush=True)
torch.xpu.synchronize()
dt = time.time() - t0
print(f"loss first {losses[0]:.3f} -> last {losses[-1]:.3f}")
print(f"throughput {ntok / dt:.0f} tokens/s ({(steps - 2) / dt:.2f} steps/s, batch {bs} x {seq})")
print(f"peak memory {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
model.eval()
msgs = [{"role": "user", "content": "OpenCuda이(가) 무엇인지 한 문장으로 설명해 주세요."}]
p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
with torch.no_grad():
    g = model.generate(**p, max_new_tokens=40, do_sample=False)
print("answer:", tok.decode(g[0][p["input_ids"].shape[1]:], skip_special_tokens=True).strip())

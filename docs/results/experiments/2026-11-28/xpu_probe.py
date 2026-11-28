import sys, time, torch
what = sys.argv[1]
torch.empty(1, device="xpu"); torch.xpu.synchronize()
if what == "compute":
    t = time.time(); x = torch.ones(64 << 20, device="xpu"); y = float((x * 2).sum()); torch.xpu.synchronize()
    print(f"compute on device: sum {y:.0f} (expect {2*(64<<20)}) in {time.time()-t:.2f}s", flush=True)
else:
    for mb in [1, 16, 256, 1024]:
        x = torch.ones(mb << 18)
        t = time.time(); y = x.to("xpu"); torch.xpu.synchronize(); dt = time.time() - t
        ok = bool((y.cpu() == 1).all())
        print(f"H2D {mb:5d} MiB: {dt*1e3:8.1f} ms {mb/1024/dt:7.2f} GiB/s ok={ok}", flush=True)

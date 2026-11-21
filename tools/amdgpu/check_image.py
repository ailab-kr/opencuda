#!/usr/bin/env python3
"""Run the checks of tests/backend/test_metal_cu2msl.c (probe_kernel.cu) or
tests/metal/test_metal_features.c (cu2msl_features.cu) on a real GPU, the
same checks the C test drivers make, through
  - the HIP runtime (tools/amdgpu/hiprt.py) for an amdgcn code object made
    by tools/amdgpu/ocu_amd_cc.py (the AMD backend's image), or
  - OpenCL (tools/amdgpu/oclpy.py) for an OCUOCL1 image made by
    compiler/opencl/cu2ocl.py.
The form is chosen from the file: ELF -> HIP, OCUOCL1 -> OpenCL.

    check_image.py probe    <image of compiler/probe/probe_kernel.cu>
    check_image.py features <image of tests/metal/cu2msl_features.cu>

Exit 0 when every check passes, 1 otherwise, 77 when no GPU takes the image.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oclpy  # noqa: E402

fails = checks = 0


class HipKernel:
    def __init__(self, f, params):
        self.f, self.params = f, params


class HipModule:
    """The calls the checks make, over the HIP runtime (backends/amd/README.md
    section 2: module_load = hipModuleLoadData, module_get_kernel =
    hipModuleGetFunction, module_get_global = hipModuleGetGlobal)."""

    def __init__(self, dev, image: bytes):
        import elfsyms
        self.dev, self.h = dev, dev.h
        self.m = self.h.load(image)
        self.form = "amdgcn code object"
        self.globals = {n: s for n, t, s in elfsyms.symbols(image) if t == 1 and not n.endswith(".kd")}
        self.params = elfsyms.kernel_params(image)   # what amd_meta.c reads for kernel_arg_sizes

    def kernel(self, name):
        return HipKernel(self.h.function(self.m, name), self.params.get(name))

    def global_ptr(self, name):
        return self.h.global_(self.m, name)


class HipDevice:
    def __init__(self):
        import hiprt
        self.h = hiprt.Hip()
        self.name, self.arch, self.version = self.h.name, "amdgcn", f"HIP runtime {self.h.runtime_version}"

    def load(self, image):
        return HipModule(self, image)

    def alloc(self, n):
        return self.h.malloc(n)

    def free(self, p):
        self.h.free(p)

    def copy_in(self, p, data):
        self.h.copy_in(p, data)

    def copy_out(self, p, n):
        return self.h.copy_out(p, n)

    def launch(self, k, grid, block, args, shared=0):
        self.h.launch(k.f, grid, block, args, shared)


def check(ok: bool, what: str):
    global fails, checks
    checks += 1
    if not ok:
        fails += 1
        print(f"FAIL {what}")


def report(what: str, bad: int):
    check(bad == 0, what)
    print(f"{what:<32} {'WRONG' if bad else 'ok'}")


def f32(xs):
    return struct.pack(f"<{len(xs)}f", *xs)


def i32(xs):
    return struct.pack(f"<{len(xs)}i", *xs)


def ptr(p):
    return struct.pack("<Q", p)


def as_f32(b):
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def as_i32(b):
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def r32(x):
    """x rounded to float32 (the host test computes in float)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def dev_from(d, data: bytes) -> int:
    p = d.alloc(len(data))
    d.copy_in(p, data)
    return p


# ---- probe_kernel.cu: saxpy, block_sum -------------------------------------------------
def probe(d, m):
    saxpy = m.kernel("_Z5saxpyifPKfPf")
    bsum = m.kernel("_Z9block_sumPKfPf")
    if saxpy.params is not None:
        check(len(saxpy.params) == 4, "saxpy has 4 params")
    count = 100000
    hx = [(i % 777) * 0.25 for i in range(count)]
    dx = dev_from(d, f32(hx))
    dy = dev_from(d, f32([2.0] * count))
    d.launch(saxpy, ((count + 127) // 128,), (128,), [i32([count]), f32([1.5]), ptr(dx), ptr(dy)])
    hy = as_f32(d.copy_out(dy, count * 4))
    bad = sum(hy[i] != r32(r32(1.5 * r32(hx[i])) + 2.0) for i in range(count))
    report("translated saxpy", bad)
    blocks, threads = 32, 256
    dout = d.alloc(blocks * 4)
    d.launch(bsum, (blocks,), (threads,), [ptr(dx), ptr(dout)])
    hout = as_f32(d.copy_out(dout, blocks * 4))
    bad = 0
    for b in range(blocks):
        want = sum(hx[b * threads + t] for t in range(threads))
        bad += hout[b] != want
    report("translated block_sum", bad)
    for p in (dx, dy, dout):
        d.free(p)


# ---- cu2msl_features.cu ------------------------------------------------------------------
def features(d, m, image):
    # 1. address spaces
    T, B = 64, 8
    N = T * B
    inp = [r32((i % 37) * 0.25) for i in range(N)]
    din, dout = dev_from(d, f32(inp)), dev_from(d, f32(inp))
    d.launch(m.kernel("_Z6spacesPKfPf"), (B,), (T,), [ptr(din), ptr(dout)])
    out = as_f32(d.copy_out(dout, N * 4))
    bad = 0
    for blk in range(B):
        for t in range(T):
            i = blk * T + t
            r = 0.0
            if t + 2 < T:
                s = r32(r32(inp[i] + inp[i + 1]) + inp[i + 2])
                r = r32(r32(s + s) + 6.0)
            r = r32(r + 0.5)
            want = r32(r + r32(inp[i] + 1.0))
            bad += out[i] != want
    report("address-space overloads", bad)
    d.free(din), d.free(dout)

    # 2. templates
    N = 1024
    f = [r32((i % 50) * 0.25) for i in range(N)]
    k = [i % 300 - 20 for i in range(N)]
    df, dk = dev_from(d, f32(f)), dev_from(d, i32(k))
    d.launch(m.kernel("_Z7scale_nIfLi1000ELb1EEvPT_S0_"), (4,), (256,), [ptr(df), f32([1.5])])
    d.launch(m.kernel("_Z7scale_nIiLi500ELb0EEvPT_S0_"), (4,), (256,), [ptr(dk), i32([3])])
    fo, ko = as_f32(d.copy_out(df, N * 4)), as_i32(d.copy_out(dk, N * 4))
    bad = 0
    for i in range(N):
        wf = r32(r32(f[i] * 1.5) * 2.0) if i < 1000 else f[i]
        wk = k[i] * 3 if i < 500 else k[i]
        bad += (fo[i] != wf) + (ko[i] != wk)
    report("kernel templates", bad)
    d.launch(m.kernel("_Z9clamp_allPiPfi"), (4,), (256,), [ptr(dk), ptr(df), i32([N])])
    kc, fc = as_i32(d.copy_out(dk, N * 4)), as_f32(d.copy_out(df, N * 4))
    bad = sum((kc[i] != min(ko[i], 100)) + (fc[i] != (0.5 if fo[i] > 0.5 else fo[i])) for i in range(N))
    report("device function templates", bad)
    d.free(df), d.free(dk)

    # 3. module variables
    gw, bw = m.global_ptr("c_weights")
    go, bo = m.global_ptr("c_offset")
    gc, bc = m.global_ptr("d_counter")
    gt, bt = m.global_ptr("d_table")
    check("no_such_variable" not in m.globals, "unknown variable not found")
    check((bw, bo, bc, bt) == (16, 4, 4, 32), "variable sizes")
    w = as_f32(d.copy_out(gw, 16))
    off = as_i32(d.copy_out(go, 4))[0]
    cnt = struct.unpack("<I", d.copy_out(gc, 4))[0]
    check(w == [0.5, -1.0, 2.0, 0.0], "c_weights initial value")
    check(off == 7, "c_offset initial value")
    check(cnt == 0, "d_counter zero")
    table = [i * 0.5 for i in range(8)]
    d.copy_in(gt, f32(table))
    d.copy_in(go, i32([-2]))
    N = 1000
    inp = [((i % 9) - 4) * 0.25 for i in range(N)]
    positive = sum(x > 0 for x in inp)
    din, dout = dev_from(d, f32(inp)), dev_from(d, f32(inp))
    d.launch(m.kernel("_Z11use_globalsPKfPfi"), ((N + 127) // 128,), (128,), [ptr(din), ptr(dout), i32([N])])
    out = as_f32(d.copy_out(dout, N * 4))
    cnt = struct.unpack("<I", d.copy_out(gc, 4))[0]
    bad = sum(out[i] != r32(r32(r32(inp[i] * w[i % 4]) + -2.0) + table[i % 8]) for i in range(N))
    bad += cnt != positive
    report("__constant__/__device__ variables", bad)
    d.free(din), d.free(dout)

    # 4. struct arguments: Params {u64 src, dst; char mode; short tag; float scale; float shift[2]} (32 B), Pair {float; uchar} (8 B)
    N = 300
    src = [float(i % 11) for i in range(N)]
    dsrc, ddst = dev_from(d, f32(src)), dev_from(d, f32(src))
    kk = m.kernel("_Z11with_struct6Paramsi4Pair")
    if kk.params is not None:
        check(kk.params == [32, 4, 8], f"with_struct params {kk.params}")
    for mode in (1, 2):
        params = struct.pack("<QQbxhfff", dsrc, ddst, mode, 3, 2.0, 100.0, 0.25)
        pair = struct.pack("<fB3x", 1.5, 200)
        d.launch(kk, ((N + 63) // 64,), (64,), [params, i32([N]), pair])
        dst = as_f32(d.copy_out(ddst, N * 4))
        bad = 0
        for i in range(N):
            v = src[i]
            v = v + 2.0 * 3 + 0.25 if mode == 1 else v * 2.0
            bad += dst[i] != v + 1.5 + 200.0 + 3 + 1
        report("struct arguments (add)" if mode == 1 else "struct arguments (mul)", bad)
    d.free(dsrc), d.free(ddst)

    # 5. dynamic shared memory
    T, B = 128, 6
    N = T * B
    inp = [float(i) for i in range(N)]
    din, dout = dev_from(d, f32(inp)), dev_from(d, f32(inp))
    d.launch(m.kernel("_Z13reverse_blockPKfPf"), (B,), (T,), [ptr(din), ptr(dout)], shared=T * 4)
    out = as_f32(d.copy_out(dout, N * 4))
    bad = sum(out[b * T + t] != inp[b * T + T - 1 - t] for b in range(B) for t in range(T))
    report("dynamic shared memory", bad)
    d.free(din), d.free(dout)

    # a second module from the same image has its own variables
    m2 = d.load(image)
    g1, _ = m.global_ptr("c_offset")
    g2, _ = m2.global_ptr("c_offset")
    check(g1 != g2, "second module has its own variables")
    check(as_i32(d.copy_out(g2, 4))[0] == 7, "second module's c_offset is 7")


def main(argv):
    if len(argv) != 2 or argv[0] not in ("probe", "features"):
        print(__doc__)
        return 2
    image = Path(argv[1]).read_bytes()
    try:
        d = HipDevice() if image[:4] == b"\x7fELF" else oclpy.Device()
    except Exception as e:
        print(f"SKIP ({e})")
        return 77
    try:
        m = d.load(image)
    except Exception as e:
        msg = str(e)
        print(("SKIP " if "no form" in msg else "FAIL ") + msg)
        return 77 if "no form" in msg else 1
    print(f"device: {d.name} ({d.arch}, {d.version}); form: {m.form}")
    (probe if argv[0] == "probe" else lambda d, m: features(d, m, image))(d, m)
    print(f"check_image {argv[0]}: {'FAIL' if fails else 'ok'} ({checks} checks, {fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

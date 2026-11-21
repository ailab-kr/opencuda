#!/usr/bin/env python3
"""Compile tests/amd/amd_lang.cu for the AMD GPU in this machine
(tools/amdgpu/ocu_amd_cc.py, the driver's amd_comgr) and run every kernel
through the HIP runtime, comparing each thread's result with the meaning
the CUDA programming guide and math API document. Exit 0 all pass, 1 a
check failed, 77 no AMD GPU / driver.

    check_amd_lang.py [--arch gfx1103]
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "amdgpu"))

fails = checks = 0


def check(ok, what, detail=""):
    global fails, checks
    checks += 1
    if not ok:
        fails += 1
    print(f"{what:<44} {'ok' if ok else 'WRONG ' + detail}")


def u32(x):
    return x & 0xffffffff


def s32(x):
    x &= 0xffffffff
    return x - (1 << 32) if x & 0x80000000 else x


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pack(fmt, xs):
    return struct.pack(f"<{len(xs)}{fmt}", *xs)


def unpack(fmt, b):
    return list(struct.unpack(f"<{len(b) // struct.calcsize(fmt)}{fmt}", b))


def P(x):
    return struct.pack("<Q", x)


def I(x):
    return struct.pack("<i", x)


class Run:
    def __init__(self, h, m):
        self.h, self.m, self.allocs = h, m, []

    def buf(self, data: bytes) -> int:
        p = self.h.malloc(max(len(data), 4))
        self.h.copy_in(p, data)
        self.allocs.append(p)
        return p

    def zeros(self, n: int) -> int:
        return self.buf(b"\0" * n)

    def get(self, p, n):
        return self.h.copy_out(p, n)

    def launch(self, name, grid, block, args, shared=0):
        self.h.launch(self.h.function(self.m, name), grid, block, args, shared)


# ---- references (CUDA semantics) ------------------------------------------------------
def shfl_ref(kind, lane, arg, w):
    base = lane & ~(w - 1)
    if kind == 0:
        return base + (arg & (w - 1))
    if kind == 1:
        return lane if lane - base < arg else lane - arg
    if kind == 2:
        return lane if lane - base + arg >= w else lane + arg
    t = lane ^ arg
    return lane if t >= base + w else t


def cvt_i32(r):
    if r != r:
        return 0
    if r >= 2147483647:
        return 2147483647
    if r <= -2147483648:
        return -2147483648
    return int(r)


def cvt_u32(r):
    if r != r or r <= 0:
        return 0
    if r >= 4294967295:
        return 4294967295
    return int(r)


def rne(x):
    if x != x or math.isinf(x):
        return x
    return float(round(x))          # Python rounds half to even


def ulp_diff(a, b):
    if math.isnan(a) and math.isnan(b):
        return 0
    if math.isinf(b) or math.isinf(a):
        return 0 if a == b else 1e9
    ia = struct.unpack("<i", struct.pack("<f", a))[0]
    ib = struct.unpack("<i", struct.pack("<f", f32(b)))[0]
    if ia < 0:
        ia = -2147483648 - ia
    if ib < 0:
        ib = -2147483648 - ib
    return abs(ia - ib)


# ---- the checks ------------------------------------------------------------------------
def t_builtins(r):
    grid, block = (3, 2, 2), (8, 4, 2)
    nthreads = 3 * 2 * 2 * 8 * 4 * 2
    out = r.zeros(12 * 4 * nthreads)
    r.launch("_Z8builtinsPj", grid, block, [P(out)])
    got = unpack("I", r.get(out, 12 * 4 * nthreads))
    bad = 0
    for bz in range(2):
        for by in range(2):
            for bx in range(3):
                for tz in range(2):
                    for ty in range(4):
                        for tx in range(8):
                            b = bx + 3 * (by + 2 * bz)
                            t = tx + 8 * (ty + 4 * tz)
                            rec = got[12 * (b * 64 + t):12 * (b * 64 + t) + 12]
                            bad += rec != [tx, ty, tz, bx, by, bz, 8, 4, 2, 3, 2, 2]
    check(bad == 0, "threadIdx/blockIdx/blockDim/gridDim (3-D)", f"{bad} threads")


def t_shuffles(r):
    out = r.zeros(4 * 128)
    cases = [(0, 5, 32), (0, 3, 8), (0, 29, 16), (1, 1, 32), (1, 3, 8), (2, 2, 32), (2, 5, 16),
             (3, 1, 32), (3, 16, 32), (3, 4, 8), (3, 7, 8)]
    names = {0: "__shfl_sync", 1: "__shfl_up_sync", 2: "__shfl_down_sync", 3: "__shfl_xor_sync"}
    for kind, arg, w in cases:
        r.launch("_Z8shufflesPiiii", (2,), (64,), [P(out), I(kind), I(arg), I(w)])
        got = unpack("i", r.get(out, 4 * 128))
        bad = 0
        for blk in range(2):
            for tid in range(64):
                warp, lane = tid // 32, tid % 32
                src = shfl_ref(kind, lane, arg, w)
                bad += got[blk * 64 + tid] != 1000 * blk + warp * 32 + src
        check(bad == 0, f"{names[kind]}(arg {arg}, width {w})", f"{bad} lanes")
    o64, of = r.zeros(8 * 64), r.zeros(4 * 64)
    r.launch("_Z13shuffles_widePxPf", (1,), (64,), [P(o64), P(of)])
    g64, gf = unpack("q", r.get(o64, 8 * 64)), unpack("f", r.get(of, 4 * 64))
    bad = 0
    for tid in range(64):
        w0 = (tid // 32) * 32
        s = w0 + ((tid % 32) ^ 5)
        bad += g64[tid] != ((s + 1) << 40) | (7 * s)
        sd = tid if (tid % 32) + 3 >= 32 else tid + 3
        bad += gf[tid] != sd * 0.5
    check(bad == 0, "64-bit and float shuffles", f"{bad}")


def t_votes(r):
    out = r.zeros(6 * 4 * 64)
    r.launch("_Z5votesPj", (1,), (64,), [P(out)])
    got = unpack("I", r.get(out, 6 * 4 * 64))
    want_ballot = sum(1 << l for l in range(0, 32, 3))
    bad = 0
    for tid in range(64):
        lane = tid % 32
        o = got[6 * tid:6 * tid + 6]
        bad += o != [want_ballot, 1, 1, 0, 0xff << (8 * (lane // 8)), 0xffffffff]
    check(bad == 0, "__ballot/__any/__all/__match_any/__activemask", f"{bad} lanes; e.g. {got[:6]}")


def t_atomics(r):
    N = 8 * 128
    ci = r.buf(pack("i", [0, 0]))
    cu = r.zeros(4)
    cull = r.zeros(8)
    cf = r.zeros(4)
    cd = r.zeros(8)
    inc = r.zeros(4)
    dec = r.zeros(4)
    mx = r.buf(I(-2147483648))
    mn = r.buf(I(2147483647))
    bits = r.buf(pack("I", [0, 0, 0xffffffff]))
    cas = r.zeros(4)
    r.launch("_Z7atomicsPiPjPyPfPdS0_S0_S_S_S0_S_", (8,), (128,),
             [P(ci), P(cu), P(cull), P(cf), P(cd), P(inc), P(dec), P(mx), P(mn), P(bits), P(cas)])
    g = {
        "atomicAdd/Sub int": unpack("i", r.get(ci, 8)) == [N, -2 * N],
        "atomicAdd unsigned": unpack("I", r.get(cu, 4))[0] == u32(N * (N - 1) // 2),
        "atomicAdd unsigned long long": unpack("Q", r.get(cull, 8))[0] == N << 33,
        "atomicAdd float": unpack("f", r.get(cf, 4))[0] == 0.5 * N,
        "atomicAdd double": unpack("d", r.get(cd, 8))[0] == 0.25 * N,
        "atomicInc (wrap at 9)": unpack("I", r.get(inc, 4))[0] == N % 10,
        "atomicDec (wrap at 9)": unpack("I", r.get(dec, 4))[0] == (-N) % 10,
        "atomicMax / atomicMin": (unpack("i", r.get(mx, 4))[0], unpack("i", r.get(mn, 4))[0])
        == (max(t * 7 % 1013 for t in range(N)), min(-(t * 5 % 977) for t in range(N))),
        "atomicOr / Xor / And": unpack("I", r.get(bits, 12)) == [0xffffffff, 0b01111, 0xffffff80],
        "atomicCAS loop": unpack("i", r.get(cas, 4))[0] == 3 * N,
    }
    for k, v in g.items():
        check(v, k)


def t_block_votes(r):
    out = r.zeros(6 * 4 * 3)
    r.launch("_Z11block_votesPi", (3,), (256,), [P(out)])
    got = unpack("i", r.get(out, 6 * 4 * 3))
    want = [64, 1, 0, 1, 0, 255 * 256 // 2] * 3
    check(got == want, "__syncthreads_count/and/or, shared atomicAdd", f"{got[:6]}")


def t_intrinsics(r):
    a = [0, 1, -1, 2147483647, -2147483648, 0x12345678, -0x0abcdef1, 0x00ffffff, 0x01000001, 77, -5, 1 << 20]
    b = [3, -7, 0x7fff, 2, 65536, -0x3456, 0x2468ace, -1, 0x00800000, -91, 13, 1 << 11]
    f = [float("nan"), float("inf"), float("-inf"), 2.5, -2.5, 3.5, 1e10, -1e10, -0.4, 0.5, 1.75, -3.7]
    d = [float("nan"), 1e300, -1e300, 2.5, -2.5, 3.5, 4294967296.0, -0.5, 1.5, 0.49999999999999994, -7.5, 123456.5]
    n = len(a)
    da, db = r.buf(pack("i", a)), r.buf(pack("i", b))
    df, dd = r.buf(pack("f", f)), r.buf(pack("d", d))
    out = r.zeros(24 * 4 * n)
    r.launch("_Z10intrinsicsPKiS0_PKfPKdPii", (1,), (32,), [P(da), P(db), P(df), P(dd), P(out), I(n)])
    got = unpack("i", r.get(out, 24 * 4 * n))
    names = ["__popc", "__clz", "__ffs", "__brev", "__mul24", "__umul24", "__mulhi", "__umulhi", "__byte_perm",
             "__float2int_rn", "__float2int_rz", "__float2int_rd", "__float2int_ru", "__float2uint_rn",
             "__double2int_rn", "__saturatef", "__popcll", "__clzll", "__mul64hi lo", "__mul64hi hi", "__sad",
             "min", "max", "fminf"]
    bad = {k: 0 for k in names}
    for i in range(n):
        ua, ub = u32(a[i]), u32(b[i])
        sx = lambda v: ((v & 0xffffff) ^ 0x800000) - 0x800000
        sel = (i * 0x1357) & 0x7777
        x8 = (ub << 32) | ua
        bp = 0
        for k in range(4):
            bp |= ((x8 >> (8 * ((sel >> (4 * k)) & 7))) & 0xff) << (8 * k)
        mh = ((a[i] * 65537) * (b[i] * 8191)) >> 64
        sat = 0.0 if f[i] != f[i] else min(max(f[i], 0.0), 1.0)
        ref = [bin(ua).count("1"), 32 - ua.bit_length(), (ua & -ua).bit_length(),
               s32(int(f"{ua:032b}"[::-1], 2)), s32(sx(a[i]) * sx(b[i])), s32((ua & 0xffffff) * (ub & 0xffffff)),
               s32((a[i] * b[i]) >> 32), s32((ua * ub) >> 32), s32(bp),
               cvt_i32(rne(f[i])), cvt_i32(math.trunc(f[i]) if math.isfinite(f[i]) else f[i]),
               cvt_i32(math.floor(f[i]) if math.isfinite(f[i]) else f[i]),
               cvt_i32(math.ceil(f[i]) if math.isfinite(f[i]) else f[i]),
               s32(cvt_u32(rne(f[i]))), cvt_i32(rne(d[i])),
               struct.unpack("<i", struct.pack("<f", sat))[0],
               bin((ua << 32) | ub).count("1"), 64 - u32(a[i] if a[i] >= 0 else 0).bit_length() if a[i] >= 0 else 0,
               s32(mh & 0xffffffff), s32(mh >> 32), s32(abs(a[i] - b[i]) + 5),
               min(a[i], b[i]), max(a[i], b[i]),
               struct.unpack("<i", struct.pack("<f", 1.5 if f[i] != f[i] else min(f[i], 1.5)))[0]]
        for k in range(24):
            if got[24 * i + k] != ref[k]:
                if bad[names[k]] == 0:
                    print(f"  {names[k]}(input {i}): got {got[24 * i + k]}, want {ref[k]}")
                bad[names[k]] += 1
    wrong = [k for k, v in bad.items() if v]
    check(not wrong, f"integer/conversion intrinsics ({len(names)} kinds)", ", ".join(wrong))


def t_math(r):
    xs = [f32(-6.0 + 12.0 * k / 509.0) for k in range(510)] + [0.0, -0.0, 1e-30, 88.5, -87.0, 0.6931472]
    n = len(xs)
    dx, out = r.buf(pack("f", xs)), r.zeros(12 * 4 * n)
    r.launch("_Z5mathfPKfPfi", ((n + 127) // 128,), (128,), [P(dx), P(out), I(n)])
    got = unpack("f", r.get(out, 12 * 4 * n))
    # arguments rounded to float as the kernel computes them, then the exact function
    fns = [("expf", lambda v: math.exp(v)), ("logf", lambda v: math.log(f32(abs(v) + f32(1e-3)))),
           ("sinf", math.sin), ("cosf", math.cos), ("tanhf", math.tanh), ("sqrtf", lambda v: math.sqrt(abs(v))),
           ("powf", lambda v: math.pow(f32(abs(v) + 0.5), 1.75)), ("atan2f", lambda v: math.atan2(v, 0.75)),
           ("erff", math.erf), ("exp2f", lambda v: 2.0 ** v), ("log2f", lambda v: math.log2(f32(abs(v) + f32(1e-3)))),
           ("fmodf", lambda v: math.fmod(f32(v * 13.0), 3.25))]
    worst = {}
    for k, (name, fn) in enumerate(fns):
        m = 0
        for i, v in enumerate(xs):
            try:
                want = fn(v)
            except (OverflowError, ValueError):
                continue
            if abs(want) > 3.4e38:
                continue
            m = max(m, ulp_diff(got[12 * i + k], want))
        worst[name] = m
    limit = {"sqrtf": 0, "fmodf": 0}
    over = {k: v for k, v in worst.items() if v > limit.get(k, 2)}
    check(not over, "device math (12 functions, <= 2 ulp; sqrtf, fmodf exact)",
          ", ".join(f"{k} {v} ulp" for k, v in over.items()))
    print("    max ulp: " + ", ".join(f"{k} {v}" for k, v in worst.items()))


def t_reduce_and_globals(r):
    nb, bs = 16, 256
    xs = [float((i * 37) % 101 - 50) for i in range(nb * bs)]
    din, dout = r.buf(pack("f", xs)), r.zeros(4 * nb * bs // 32)
    r.launch("_Z11warp_reducePKfPf", (nb,), (bs,), [P(din), P(dout)])
    got = unpack("f", r.get(dout, 4 * nb * bs // 32))
    want = [sum(xs[32 * w:32 * w + 32]) for w in range(nb * bs // 32)]
    check(got == want, "warp reduction with __shfl_down_sync")
    ints = [(i * 7919) % 1000 for i in range(4 * 64 * 4)]
    di = r.buf(pack("i", ints))
    g, size = r.h.global_(r.m, "g_total")
    r.h.copy_in(g, b"\0" * 8)
    r.launch("_Z8sum_intoIiLi4EEvPKT_", (4,), (64,), [P(di)])
    total = unpack("Q", r.h.copy_out(g, 8))[0]
    check(size == 8 and total == sum(ints), "template kernel + __device__ variable atomics", f"{total} vs {sum(ints)}")


def main(argv):
    import hiprt
    from comgr import ComgrError
    from ocu_amd_cc import compile_cu, default_arch
    arch = argv[argv.index("--arch") + 1] if "--arch" in argv else default_arch()
    try:
        h = hiprt.Hip()
    except Exception as e:
        print(f"SKIP ({e})")
        return 77
    try:
        co = compile_cu(HERE / "amd_lang.cu", arch)
    except ComgrError as e:
        print(f"FAIL: amd_lang.cu does not compile for {arch}:\n{e}")
        return 1
    print(f"device: {h.name} ({arch}); code object {len(co)} bytes")
    m = h.load(co)
    r = Run(h, m)
    for t in (t_builtins, t_shuffles, t_votes, t_atomics, t_block_votes, t_intrinsics, t_math, t_reduce_and_globals):
        t(r)
    for p in r.allocs:
        h.free(p)
    h.unload(m)
    print(f"check_amd_lang: {'FAIL' if fails else 'ok'} ({checks} checks, {fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

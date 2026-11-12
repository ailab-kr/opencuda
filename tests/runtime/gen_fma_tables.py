#!/usr/bin/env python3
"""Generates tests/runtime/fma_exact_table.h (expected fma/fmaf results) and
tests/runtime/powf_table.h (expected powf results).

Everything is computed with exact rational arithmetic (fractions.Fraction)
and decimal at 80 digits, from the definitions only:

  IEEE 754-2019 5.4.1 fusedMultiplyAdd(x, y, z): x*y + z computed as if with
    unbounded range and precision, rounded once to the destination format;
  4.3 rounding-direction attributes (to nearest ties to even, toward zero,
    toward +inf, toward -inf), 7.4 overflow (to nearest: infinity; toward
    zero: the largest finite value; directed toward an infinity: that
    infinity for the matching sign, else the largest finite value),
  6.3 the sign of zero (an exact zero sum of operands of opposite sign is +0,
    or -0 when rounding toward negative; x*y and z zeros of one sign keep
    it; a nonzero exact result that rounds to zero keeps its sign).

The rounding below shares no code with the implementation under test.
Run:  python3 gen_fma_tables.py     (rewrites the two headers; deterministic)
"""
import random
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as Fr
from pathlib import Path

FMT = {
    "f": dict(p=24, emin=-126, emax=127, bits=32, ebits=8),
    "d": dict(p=53, emin=-1022, emax=1023, bits=64, ebits=11),
}
MODES = ("rn", "rz", "ru", "rd")


def fl(v):
    import struct
    return struct.unpack("<I", struct.pack("<f", v))[0]


def dl(v):
    import struct
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def pw(k):
    return Fr(2) ** k


def unpack(bits, f):
    """(sign, magnitude as Fraction) of a finite number."""
    mb = f["p"] - 1
    s = bits >> (f["bits"] - 1)
    e = (bits >> mb) & ((1 << f["ebits"]) - 1)
    m = bits & ((1 << mb) - 1)
    if e == (1 << f["ebits"]) - 1:
        raise ValueError("not finite")
    if e == 0:
        return s, Fr(m) * pw(f["emin"] - mb)
    return s, Fr(m | (1 << mb)) * pw(e - f["emax"] - mb)


def pack(s, n, q, f):
    """n * 2^q (exactly representable, n < 2^p after normalising) as bits."""
    p, mb = f["p"], f["p"] - 1
    sbit = s << (f["bits"] - 1)
    if n == 0:
        return sbit
    while n.bit_length() > p:
        assert n & 1 == 0
        n >>= 1
        q += 1
    if n.bit_length() == p:
        e = q + mb
        assert f["emin"] <= e <= f["emax"], (e, f)
        return sbit | ((e + f["emax"]) << mb) | (n & ((1 << mb) - 1))
    assert q == f["emin"] - mb, (n, q)
    return sbit | n


def inf_bits(s, f):
    return (s << (f["bits"] - 1)) | (((1 << f["ebits"]) - 1) << (f["p"] - 1))


def max_bits(s, f):
    mb = f["p"] - 1
    return (s << (f["bits"] - 1)) | (((1 << f["ebits"]) - 2) << mb) | ((1 << mb) - 1)


def round_to(v, s, mode, f):
    """Round the positive Fraction v (sign s) to the format in `mode`."""
    p, mb, emin, emax = f["p"], f["p"] - 1, f["emin"], f["emax"]
    e = v.numerator.bit_length() - v.denominator.bit_length()
    while pw(e) > v:
        e -= 1
    while pw(e + 1) <= v:
        e += 1
    q = max(e, emin) - mb
    scaled = v / pw(q)
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    up = (
        (mode == "rn" and (rem > Fr(1, 2) or (rem == Fr(1, 2) and n & 1)))
        or (mode == "ru" and s == 0 and rem > 0)
        or (mode == "rd" and s == 1 and rem > 0)
    )
    n += 1 if up else 0
    while n.bit_length() > p:
        n >>= 1
        q += 1
    if n.bit_length() == p and q + mb > emax:  # overflow
        to_inf = mode == "rn" or (mode == "ru" and s == 0) or (mode == "rd" and s == 1)
        return inf_bits(s, f) if to_inf else max_bits(s, f)
    return pack(s, n, q, f)


def fma_ref(a, b, c, mode, f):
    """Bits of fma(a, b, c) in `mode` for finite operands (bit patterns)."""
    sa, va = unpack(a, f)
    sb, vb = unpack(b, f)
    sc, vc = unpack(c, f)
    sp = sa ^ sb
    prod = va * vb
    exact = (-prod if sp else prod) + (-vc if sc else vc)
    if exact == 0:
        if (va == 0 or vb == 0) and vc == 0:
            s = sp if sp == sc else (1 if mode == "rd" else 0)
        else:
            s = 1 if mode == "rd" else 0
        return s << (f["bits"] - 1)
    s = 1 if exact < 0 else 0
    return round_to(abs(exact), s, mode, f)


def make(f, rnd, s, e, mode="rand"):
    """A number of sign s with exponent e (2^e <= |x| < 2^(e+1), or a subnormal
    when e < emin).  mode: 'rand' random significand, 'one' exactly 2^e,
    'max' all ones."""
    p, mb, emin = f["p"], f["p"] - 1, f["emin"]
    if e >= emin:
        low = {"rand": rnd.getrandbits(mb), "one": 0, "max": (1 << mb) - 1}[mode]
        return pack(s, (1 << mb) | low, e - mb, f)
    q = emin - mb
    top = e - q
    assert top >= 0
    low = {"rand": rnd.getrandbits(top) if top else 0, "one": 0, "max": (1 << top) - 1}[mode]
    return pack(s, (1 << top) | low, q, f)


def rows_for(f, rnd, exps, n_random, n_cancel, near_pairs=None, bset=None):
    rows = set()
    p, mb, emin, emax = f["p"], f["p"] - 1, f["emin"], f["emax"]

    def add(a, b, c):
        rows.add((a, b, c))

    def num(s, e, mode="rand"):
        return make(f, rnd, s, e, mode)

    # signed zeros, units, the smallest and the largest numbers
    smallest = pack(0, 1, emin - mb, f)
    biggest = max_bits(0, f)
    base = [0, 1 << (f["bits"] - 1)]
    for s in (0, 1):
        base += [num(s, 0, "one"), smallest | (s << (f["bits"] - 1)), biggest | (s << (f["bits"] - 1))]
    for a in base:
        for b in base:
            for c in base:
                add(a, b, c)
    # random exponents, all signs
    for _ in range(n_random):
        add(num(rnd.getrandbits(1), rnd.choice(exps), rnd.choice(("rand", "rand", "one", "max"))),
            num(rnd.getrandbits(1), rnd.choice(exps), rnd.choice(("rand", "rand", "one", "max"))),
            num(rnd.getrandbits(1), rnd.choice(exps), rnd.choice(("rand", "rand", "one", "max"))))
    # fast-path boundaries: every exponent from a small set, all sign mixes
    if bset:
        for ea in bset:
            for eb in bset:
                for ec in bset:
                    for sg in ((0, 0, 1), (1, 1, 1)):
                        add(num(sg[0], ea), num(sg[1], eb), num(sg[2], ec))
        for ea in bset:
            for eb in bset:
                add(num(0, ea, "max"), num(0, eb, "max"), num(1, ea + eb, "one") if emin <= ea + eb <= f["emax"] else 0)
    # products near 2^-2*(boundary), addend around the product's last bits
    if near_pairs:
        for ea in near_pairs[0]:
            for eb in near_pairs[1]:
                for dc in near_pairs[2]:
                    ec = ea + eb + dc
                    if ec < emin - mb:
                        continue
                    add(num(0, ea), num(rnd.getrandbits(1), eb), num(rnd.getrandbits(1), ec))
    # cancellation: c = -RN(a*b) +- k ulp
    for _ in range(n_cancel):
        ea, eb = rnd.choice(exps), rnd.choice(exps)
        if not (emin <= ea + eb <= emax - 1):
            continue
        a, b = num(0, ea), num(0, eb)
        sa, va = unpack(a, f)
        sb, vb = unpack(b, f)
        r = fma_ref(a, b, 0, "rn", f)
        if r == inf_bits(0, f):
            continue
        for k in (0, 1, -1, 2):
            c = (r ^ (1 << (f["bits"] - 1))) + k
            if 0 <= c < (1 << f["bits"]) and (c >> mb) & ((1 << f["ebits"]) - 1) != (1 << f["ebits"]) - 1:
                add(a, b, c)
                add(b, a, c)
    # ties: a*b is exactly half (or just over half) an ulp of c
    for _ in range(max(n_cancel // 2, 100)):
        ec = rnd.choice([e for e in exps if emin + mb <= e <= emax - 1] or [0])
        sc = rnd.getrandbits(1)
        c = num(sc, ec)
        i = rnd.randint(-8, 8)
        j = (ec - mb - 1) - i
        if not (emin - mb <= i <= emax and emin - mb <= j <= emax):
            continue
        for bm in ("one", "max"):
            for sg in (0, 1):
                a = num(0, i, "one") if i >= emin else None
                b = num(sg, j, "one") if j >= emin else None
                if a is None or b is None:
                    continue
                add(a, b, c)
                if bm == "max":  # just over half an ulp
                    add(a, num(sg, j, "max"), c)
    # subnormal and half-way cases at the bottom of the range
    for ea in range(emin - mb, emin + 3):
        for eb in (-1, -2, -mb - 1, -mb, -mb + 1):
            for kc in (1, 2, 3):
                for sa in (0, 1):
                    for sc in (0, 1):
                        add(num(sa, ea, "one"), num(0, eb, "one"), pack(sc, kc, emin - mb, f))
    return sorted(rows)



def ties_by_construction(f, rnd, exps, count):
    """a*b is an odd integer of p+1 bits times a power of two: exactly half way
    between two floats.  c is zero, or a tiny number of either sign far below
    it, or one just below: the sum is a tie, or just above or below it
    (the case where the rounding needs the bits that the alignment dropped)."""
    p, mb, emin, emax = f["p"], f["p"] - 1, f["emin"], f["emax"]
    ha, hb = (p + 1) // 2, (p + 1) - (p + 1) // 2
    rows = set()
    tries = 0
    while len(rows) < count and tries < count * 50:
        tries += 1
        ma = rnd.getrandbits(ha) | 1 | (1 << (ha - 1))
        mb_ = rnd.getrandbits(hb) | 1 | (1 << (hb - 1))
        P = ma * mb_
        if P.bit_length() != p + 1:
            continue
        # a = ma * 2^i, b = mb_ * 2^j with the pieces exactly representable
        ep = rnd.choice(exps)  # exponent of the product, 2^ep <= P*2^(i+j) < 2^(ep+1)
        ij = ep - P.bit_length() + 1
        i = ij // 2
        j = ij - i
        if not (emin - mb <= i and emin - mb <= j):
            continue
        sa, sb = rnd.getrandbits(1), rnd.getrandbits(1)
        # ma has ha bits: write it as a normal or subnormal number by scaling
        def num_of(m, e2, s):
            n = m
            q = e2
            if q < emin - mb:
                return None
            # shift m up to p bits when the format allows (exact: low bits zero)
            bl = m.bit_length()
            if bl < p and q - (p - bl) >= emin - mb:
                n, q = m << (p - bl), q - (p - bl)
            try:
                return pack(s, n, q, f)
            except AssertionError:
                return None
        a = num_of(ma, i, sa)
        b = num_of(mb_, j, sb)
        if a is None or b is None:
            continue
        sp = sa ^ sb
        top = ep  # P's exponent
        for dc in (None, -(p + 3), -(p + 40), -(2 * p + 20), -(2 * p + 70), -(p + 1)):
            for sc in (0, 1):
                if dc is None:
                    c = 0
                else:
                    ec = top + dc
                    if ec < emin - mb:
                        continue
                    c = make(f, rnd, sc, ec, "one")
                rows.add((a, b, c))
    return sorted(rows)



def sticky_rows(f, rnd, count):
    """Double only.  a*b = A*B*2^k with A, B 53-bit odd and A*B = T + 1 where T
    ends in the pattern of a tie (low 53 bits of the 106-bit product are
    10...01), and c = -w with w just below one unit of the product's last bit:
    the exact sum is a hair above a tie.  A result computed from the sum with
    the bits of c below the alignment dropped, and no sticky bit, reads a tie
    and rounds to even instead of up."""
    assert f["p"] == 53
    rows = set()
    mod = 1 << 53
    while len(rows) < count:
        A = rnd.getrandbits(53) | 1 | (1 << 52)
        B = (((1 << 52) + 1) * pow(A, -1, mod)) % mod
        if B < (1 << 52) or (A * B).bit_length() != 106:
            continue
        ea = rnd.choice([-1000, -700, -500, -401, -400, -399, -300, -100, -53, -1, 0, 1, 53, 100, 300, 399, 400, 401, 500, 700, 900])
        eb = rnd.choice([-600, -401, -400, -399, -200, -53, -1, 0, 1, 53, 200, 399, 400, 401, 600])
        ia, ib = ea - 52, eb - 52
        if not (-1074 <= ia and -1074 <= ib and ia + ib - 53 >= -1074 and ea + eb < 1022):
            continue
        for jp in (1, 2, 5, 1 << 20, 1 << 32):
            for s in (0, 1):
                a = pack(s, A, ia, f) if ia >= f["emin"] - 52 else None
                b = pack(0, B, ib, f) if ib >= f["emin"] - 52 else None
                if a is None or b is None:
                    continue
                c = pack(1 - s, (1 << 53) - jp, ia + ib - 53, f)
                rows.add((a, b, c))
                rows.add((b, a, c))
    return sorted(rows)



def cancel_rows(f, rnd, pairs, reps=1):
    """c = -RN(a*b) +- k ulp for the given exponent pairs (every sign of a, b)."""
    mb, emin, emax = f["p"] - 1, f["emin"], f["emax"]
    rows = set()
    for ea, eb in pairs:
        if not (emin <= ea + eb <= emax - 1):
            continue
        for rep in range(reps):
            a, b = make(f, rnd, 0, ea), make(f, rnd, 0, eb)
            r = fma_ref(a, b, 0, "rn", f)
            if r == inf_bits(0, f):
                continue
            for k in (0, 1, -1, 2):
                c = (r ^ (1 << (f["bits"] - 1))) + k
                if 0 <= c < (1 << f["bits"]) and (c >> mb) & ((1 << f["ebits"]) - 1) != (1 << f["ebits"]) - 1:
                    for sa in (0, 1):
                        rows.add((a | (sa << (f["bits"] - 1)), b, c ^ (sa << (f["bits"] - 1))))
                    rows.add((b, a, c))
    return sorted(rows)


def hexs(v, f):
    return ("0x%08x" if f["bits"] == 32 else "0x%016x") % v + ("u" if f["bits"] == 32 else "ull")


def write_fma(path):
    out = []
    out.append("// Generated by gen_fma_tables.py (exact rational arithmetic, IEEE 754-2019\n"
               "// 5.4.1 fusedMultiplyAdd, 4.3, 6.3, 7.4).  Do not edit.\n"
               "#pragma once\n#include <stdint.h>\n")
    out.append("struct FmaRowF { uint32_t a, b, c, rn, rz, ru, rd; };\n"
               "struct FmaRowD { uint64_t a, b, c, rn, rz, ru, rd; };\n")
    rnd = random.Random(20261113)
    # float
    f = FMT["f"]
    exps_f = [-149, -140, -127, -126, -125, -100, -75, -50, -24, -12, -1, 0, 1, 12, 24, 50, 75, 100, 126, 127]
    rows = rows_for(f, rnd, exps_f, 1200, 250, near_pairs=([-63, -62, -75, -64], [-63, -62, -75, -64], [-26, -25, -24, -1, 0, 1, 2]),
                    bset=[-149, -127, -126, -64, -63, 0, 63, 64, 127])
    rows = sorted(set(rows) | set(ties_by_construction(f, rnd, [e for e in exps_f if -100 <= e <= 100], 400)))
    Lf = [-100, -75, -64, -63, -24, 0, 24, 63, 64]
    rows = sorted(set(rows) | set(cancel_rows(f, rnd, [(x, y) for x in Lf for y in Lf])))
    # the three documented samples must be present
    H = lambda t: fl(float.fromhex(t))
    docs = [
        (H("0x1.02fbdap+0"), H("0x1.fffffep-1"), H("-0x1.02fbd8p+0"), H("0x1.fa084cp-25")),
        (H("0x1p-126"), H("0x1p-24"), H("0x1p-149"), H("0x1p-148")),
    ]
    for a, b, c, want in docs:
        rows = sorted(set(rows) | {(a, b, c)})
        assert fma_ref(a, b, c, "rn", f) == want, (hex(a), hex(fma_ref(a, b, c, "rn", f)), hex(want))
    out.append("static const FmaRowF fma_rows_f[] = {\n")
    for a, b, c in rows:
        r = [fma_ref(a, b, c, m, f) for m in MODES]
        out.append("{%s,%s,%s,%s,%s,%s,%s},\n" % tuple(hexs(v, f) for v in (a, b, c, *r)))
    out.append("};\nstatic const int fma_rows_f_n = %d;\n" % len(rows))
    nf = len(rows)
    # double
    f = FMT["d"]
    exps_d = [-1074, -1060, -1023, -1022, -1021, -900, -520, -500, -480, -450, -420, -801, -800, -799, -536, -402, -401, -400, -399, -398, -54, -53, -1, 0, 1,
              53, 54, 398, 399, 400, 401, 402, 511, 512, 799, 800, 1000, 1022, 1023]
    rows = rows_for(f, rnd, exps_d, 1800, 250,
                    near_pairs=(list(range(-405, -394)), list(range(-405, -394)), [-54, -53, -52, -1, 0, 1, 2]),
                    bset=[-401, -400, -399, -1, 0, 399, 400, 401])
    H = lambda t: dl(float.fromhex(t))
    rows = sorted(set(rows) | set(ties_by_construction(f, rnd, [e for e in exps_d if -1000 <= e <= 1000], 700)))
    rows = sorted(set(rows) | set(rows_for(f, rnd, exps_d, 0, 0, near_pairs=([-520, -500, -480, -450, -420], [-520, -500, -480, -450, -420], [-54, -53, -1, 0, 1, 2]))))
    L = [-700, -500, -450, -420, -401, -400, -399, -200, 0, 400, 401]
    rows = sorted(set(rows) | set(cancel_rows(f, rnd, [(x, y) for x in L for y in L], 4)))
    rows = sorted(set(rows) | set(sticky_rows(f, rnd, 150)))
    docs = [(H("0x1p-1022"), H("0x1p-53"), H("0x1p-1074"), H("0x0.0000000000002p-1022"))]
    for a, b, c, want in docs:
        rows = sorted(set(rows) | {(a, b, c)})
        assert fma_ref(a, b, c, "rn", f) == want
    out.append("static const FmaRowD fma_rows_d[] = {\n")
    for a, b, c in rows:
        r = [fma_ref(a, b, c, m, f) for m in MODES]
        out.append("{%s,%s,%s,%s,%s,%s,%s},\n" % tuple(hexs(v, f) for v in (a, b, c, *r)))
    out.append("};\nstatic const int fma_rows_d_n = %d;\n" % len(rows))
    Path(path).write_text("".join(out), encoding="utf-8", newline="\n")
    print("fma rows: float %d, double %d" % (nf, len(rows)))


# ---- powf -------------------------------------------------------------------------

def powf_ref(x, y):
    """Correctly rounded float of x**y (x, y floats given as bits-exact Fractions)."""
    getcontext().prec = 90
    f = FMT["f"]
    sx, vx = unpack(x, f)
    sy, vy = unpack(y, f)
    X = Decimal(vx.numerator) / Decimal(vx.denominator)
    Y = Decimal(vy.numerator) / Decimal(vy.denominator)
    if sy:
        Y = -Y
    neg = False
    if sx:
        if vy.denominator != 1:
            return None
        neg = int(vy) % 2 == 1
    if X == 0:
        return None
    r = (Y * X.ln()).exp()
    if r > Decimal(2) ** 129 or r < Decimal(2) ** -152:
        return None
    # to a Fraction (exactly, the Decimal has 90 digits)
    fr = Fr(r)
    return round_to(fr, 1 if neg else 0, "rn", f)


def write_powf(path):
    rnd = random.Random(7)
    f = FMT["f"]
    out = ["// Generated by gen_fma_tables.py: correctly rounded float x**y from 90-digit\n"
           "// decimal arithmetic.  Do not edit.\n#pragma once\n#include <stdint.h>\n"
           "struct PowRow { uint32_t x, y, want; };\nstatic const PowRow pow_rows[] = {\n"]
    n = 0
    seen = set()

    def emit(x, y):
        nonlocal n
        if (x, y) in seen:
            return
        w = powf_ref(x, y)
        if w is None:
            return
        seen.add((x, y))
        out.append("{0x%08xu,0x%08xu,0x%08xu},\n" % (x, y, w))
        n += 1

    import math
    import struct
    for _ in range(1500):
        e = rnd.randint(-100, 100)
        x = make(f, rnd, 0, e)
        xv = struct.unpack("<f", struct.pack("<I", x))[0]
        l = math.log2(xv) or 1.0
        y = fl(rnd.uniform(-120.0, 120.0) / l)
        emit(x, y)
    for _ in range(300):  # integer y, negative x
        x = make(f, rnd, 1, rnd.randint(-10, 10))
        emit(x, fl(float(rnd.randint(-30, 30))))
    for _ in range(300):  # x near 1, large y
        d = rnd.randint(-2000, 2000)
        xv = 1.0 + d * 2.0 ** -23
        emit(fl(xv), fl(rnd.uniform(-1, 1) * 2.0 ** rnd.randint(8, 20)))
    for xv, yv in ((2.0, 10.0), (2.0, -10.0), (10.0, 3.0), (3.0, 0.5), (0.5, 100.0), (1.5, 2.5), (7.0, -1.0)):
        emit(fl(xv), fl(yv))
    out.append("};\nstatic const int pow_rows_n = %d;\n" % n)
    Path(path).write_text("".join(out), encoding="utf-8", newline="\n")
    print("powf rows:", n)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    write_fma(here / "fma_exact_table.h")
    write_powf(here / "powf_table.h")

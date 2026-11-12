#!/usr/bin/env python3
"""Generate runtime/include/ocu_device_mathf_tables.h: the constants, tables
and polynomial coefficients of the float-only device math functions.

Everything is derived here from first principles, with Python's exact
integer and decimal arithmetic only:

  * pi from Machin's formula (pi/4 = 4 atan(1/5) - atan(1/239)), as an
    integer to 400 bits; the bits of 2/pi for the large-argument sin/cos
    reduction (Payne-Hanek) come from it by exact integer division.
  * constants (ln 2, pi/2, ...) split into sums of floats, each part the
    float nearest to what the previous parts leave.
  * the log table: c_i = float(1 / (1 + i/64)), and -ln(c_i) of that exact
    float value, to 50 digits, as two floats.
  * polynomial coefficients: minimax fits by the Remez exchange algorithm
    on a dense grid, the linear systems solved in exact rationals. The
    coefficients are rounded to float one at a time, lowest degree first,
    and the remaining ones refitted after each rounding, so the rounded
    polynomial stays close to the best one. Function values come from
    Taylor series (or Decimal's exp/ln/sqrt) at 50 digits.

Rerun to check the committed header:

    python3 tools/gen_mathf_tables.py > runtime/include/ocu_device_mathf_tables.h

With --report it prints the fit errors instead.
"""
import math
import struct
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 60
D = Decimal

# ---- pi and friends -------------------------------------------------------------


def atan_inv_fixed(n: int, bits: int) -> int:
    """atan(1/n) * 2^bits, truncated series in integers."""
    one = 1 << (bits + 16)
    term = one // n
    total = 0
    k = 0
    n2 = n * n
    while term:
        total += term // (2 * k + 1) if k % 2 == 0 else -(term // (2 * k + 1))
        term //= n2
        k += 1
    return total >> 16


PI_BITS = 400
PI_FIXED = 4 * (4 * atan_inv_fixed(5, PI_BITS) - atan_inv_fixed(239, PI_BITS))  # pi * 2^400
PI_FRAC = Fraction(PI_FIXED, 1 << PI_BITS)  # within 2^-390 of pi
PI = D(PI_FIXED) / D(2) ** PI_BITS


def f32(x) -> float:
    """Nearest float (ties to even) of a float/Fraction/Decimal value."""
    if isinstance(x, Fraction):
        x = float(x)  # correctly rounded to double first
        # double rounding is harmless for the values used here (checked below)
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_exact(q: Fraction) -> float:
    """Nearest float of an exact rational, without double rounding."""
    if q == 0:
        return 0.0
    s = -1 if q < 0 else 1
    q = abs(q)
    e = math.floor(math.log2(q))
    while Fraction(2) ** e > q:
        e -= 1
    while Fraction(2) ** (e + 1) <= q:
        e += 1
    e = max(e, -126)
    scale = Fraction(2) ** (23 - e)
    n = q * scale
    fl = math.floor(n)
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    return s * float(Fraction(fl) / scale)


def split(q: Fraction, parts: int, first_bits: int = 24):
    """q as a sum of floats: the first with first_bits significant bits."""
    out = []
    rest = q
    for i in range(parts):
        if i == 0 and first_bits < 24:
            e = math.floor(math.log2(abs(rest)))
            scale = Fraction(2) ** (first_bits - 1 - e)
            v = float(Fraction(round(rest * scale)) / scale)
        else:
            v = f32_exact(rest)
        out.append(v)
        rest -= Fraction(v)
    return out


def ln_frac(q: Fraction, digits: int = 60) -> Fraction:
    getcontext().prec = digits
    v = D(q.numerator) / D(q.denominator)
    r = v.ln()
    getcontext().prec = 60
    return Fraction(r)


# ---- Decimal function values ----------------------------------------------------------


def d_sin(x: Decimal) -> Decimal:
    s, t, k = D(0), x, 1
    x2 = x * x
    while abs(t) > D(10) ** -70:
        s += t
        t = -t * x2 / ((k + 1) * (k + 2))
        k += 2
    return s


def d_cos(x: Decimal) -> Decimal:
    s, t, k = D(0), D(1), 0
    x2 = x * x
    while abs(t) > D(10) ** -70:
        s += t
        t = -t * x2 / ((k + 1) * (k + 2))
        k += 2
    return s


def d_atan(t: Decimal) -> Decimal:
    if t < 0:
        return -d_atan(-t)
    if t > 1:
        return PI / 2 - d_atan(1 / t)
    mult = 1
    for _ in range(3):  # atan t = 2 atan(t / (1 + sqrt(1 + t^2)))
        t = t / (1 + (1 + t * t).sqrt())
        mult *= 2
    s, term, k = D(0), t, 0
    t2 = t * t
    while abs(term) > D(10) ** -70:
        s += term / (2 * k + 1) if k % 2 == 0 else -term / (2 * k + 1)
        term *= t2
        k += 1
    return mult * s


def d_erf(x: Decimal) -> Decimal:
    """erf(x) = 2/sqrt(pi) e^{-x^2} sum 2^n x^(2n+1) / (1 3 5 ... (2n+1)), x >= 0."""
    s, term, n = D(0), x, 0
    x2 = x * x
    while term > s * D(10) ** -(getcontext().prec + 2):
        s += term
        n += 1
        term = term * 2 * x2 / (2 * n + 1)
    return 2 / PI.sqrt() * (-x2).exp() * s


def d_erfc(x: Decimal) -> Decimal:
    old = getcontext().prec
    getcontext().prec = 130  # erfc(10) ~ 2e-45: 1 - erf needs the extra digits
    r = 1 - d_erf(x)
    getcontext().prec = old
    return +r


# ---- Remez exchange on a grid ----------------------------------------------------


def solve(a, b):
    """Gaussian elimination in exact rationals."""
    n = len(b)
    m = [[Fraction(v) for v in row] + [Fraction(bv)] for row, bv in zip(a, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def peval(c, t):
    s = 0.0
    for v in reversed(c):
        s = s * t + v
    return s


def remez(ts, gs, ws, deg, fixed, iters=60):
    """Minimise max_i ws[i] |p(ts[i]) - gs[i]| over the coefficients of
    degree 0..deg not in `fixed` (a dict degree -> value)."""
    free = [k for k in range(deg + 1) if k not in fixed]
    n = len(free)
    N = len(ts)
    ref = [round(i * (N - 1) / n) for i in range(n + 1)]
    best = None
    for _ in range(iters):
        rows, rhs = [], []
        for j, i in enumerate(ref):
            t = Fraction(ts[i])
            rows.append([t ** k for k in free] + [Fraction((-1) ** j) / Fraction(ws[i])])
            rhs.append(Fraction(gs[i]) - sum(Fraction(v) * t ** k for k, v in fixed.items()))
        sol = solve(rows, rhs)
        coef = [0.0] * (deg + 1)
        for k, v in fixed.items():
            coef[k] = v
        for k, v in zip(free, sol):
            coef[k] = float(v)
        lev = abs(float(sol[-1]))
        err = [ws[i] * (peval(coef, ts[i]) - gs[i]) for i in range(N)]
        emax = max(abs(e) for e in err)
        if best is None or emax < best[1]:
            best = (coef, emax)
        if lev > 0 and emax <= lev * (1 + 1e-6):
            break
        # new reference: the largest error in each run of one sign
        ext = []
        i = 0
        while i < N:
            j = i
            s = err[i] >= 0
            k = i
            while j < N and (err[j] >= 0) == s:
                if abs(err[j]) > abs(err[k]):
                    k = j
                j += 1
            ext.append(k)
            i = j
        while len(ext) > n + 1:
            # drop the smaller end, or the adjacent pair with the smallest errors
            if len(ext) - (n + 1) == 1 or len(ext) == n + 2:
                if abs(err[ext[0]]) < abs(err[ext[-1]]):
                    ext.pop(0)
                else:
                    ext.pop()
                continue
            pi = min(range(len(ext) - 1),
                     key=lambda p: max(abs(err[ext[p]]), abs(err[ext[p + 1]])))
            del ext[pi:pi + 2]
        if len(ext) < n + 1:
            break
        if ext == ref:
            break
        ref = ext
    return best


def fit(fn, a, b, deg, weight="rel", npts=1500, g0=None, split0=False):
    """Float coefficients c[0..deg] of p(t) ~ fn(t) on [a, b].

    Coefficient j is rounded after c[0..j-1] are: the rest of the polynomial
    is refitted to (fn - those terms) / t^j with weight |t|^j, an ordinary
    minimax problem (a Haar system even when 0 is in [a, b])."""
    ts = sorted({(a + b) / 2 + (b - a) / 2 * math.cos(math.pi * i / (npts - 1)) for i in range(npts)})
    gs = []
    for t in ts:
        if t == 0 and g0 is not None:
            gs.append(g0)
        else:
            gs.append(float(fn(D(t))))
    ws = [1 / abs(g) for g in gs] if weight == "rel" else [1.0] * len(gs)
    coef = []
    fit.unrounded = remez(ts, gs, ws, deg, {})[1]
    for j in range(deg + 1):
        pts = [i for i in range(len(ts)) if j == 0 or ts[i] != 0]
        tj = [ts[i] for i in pts]
        hj = [(gs[i] - peval(coef, ts[i])) / ts[i] ** j for i in pts]
        wj = [ws[i] * abs(ts[i]) ** j for i in pts]
        c, _ = remez(tj, hj, wj, deg - j, {})
        if j == 0 and split0:  # constant term as hi + lo
            hi = f32(c[0])
            fit.lo = f32(c[0] - hi)
            coef.append(hi + fit.lo)
        else:
            coef.append(f32(c[0]))
    emax = max(ws[i] * abs(peval(coef, ts[i]) - gs[i]) for i in range(len(ts)))
    return coef, emax


# ---- the fits -----------------------------------------------------------------------------

LN2 = Fraction(D(2).ln())


def g_exp(r):  # (e^r - 1 - r - r^2/2) / r^3
    return (r.exp() - 1 - r - r * r / 2) / (r * r * r)


def g_log(r):  # (log1p(r) - r + r^2/2) / r^3
    return ((1 + r).ln() - r + r * r / 2) / (r * r * r)


def g_sin(z):  # (sin r - r) / r^3, z = r^2
    r = z.sqrt()
    return (d_sin(r) - r) / (z * r)


def g_cos(z):  # (cos r - 1 + z/2) / z^2
    r = z.sqrt()
    return (d_cos(r) - 1 + z / 2) / (z * z)


def g_atan(z):  # (atan s - s) / s^3, z = s^2
    s = z.sqrt()
    return (d_atan(s) - s) / (z * s)


def g_asin(z):
    s = z.sqrt()
    return (d_atan(s / (1 - z).sqrt()) - s) / (z * s)


def g_sinh(z):
    x = z.sqrt()
    return ((x.exp() - (-x).exp()) / 2 - x) / (z * x)


def g_tanh(z):
    x = z.sqrt()
    e = (2 * x).exp()
    return ((e - 1) / (e + 1) - x) / (z * x)


def g_erf(z):  # erf(x)/x - 1
    x = z.sqrt()
    return d_erf(x) / x - 1


def g_erfc(u):  # log(erfc(x) / t) + x^2 with t = 1 / (1 + x/2), u = t - 1/2
    t = u + D("0.5")
    x = 2 / t - 2
    old = getcontext().prec
    getcontext().prec = 130
    v = (d_erfc(x) / t).ln() + x * x
    getcontext().prec = old
    return +v


def g_rcbrt(m):  # m^(-1/3)
    return (-(m.ln()) / 3).exp()


SIN_R = 0.7955  # |r| after the Cody-Waite reduction, with margin
FITS = [
    # name, function, interval, degree, weight, value at 0
    ("EXP_Q", g_exp, -0.3467, 0.3467, 4, "rel", 1 / 6),
    ("LOG_Q", g_log, -0.0079, 0.0079, 3, "rel", 1 / 3),
    ("SIN_S", g_sin, 0.0, SIN_R * SIN_R, 3, "rel", -1 / 6),
    ("COS_C", g_cos, 0.0, SIN_R * SIN_R, 3, "rel", 1 / 24),
    ("ATAN_A", g_atan, 0.0, 1.0, 9, "rel", -1 / 3),
    ("ASIN_P", g_asin, 0.0, 0.25, 5, "rel", 1 / 6),
    ("SINH_P", g_sinh, 0.0, 1.0, 3, "rel", 1 / 6),
    ("TANH_P", g_tanh, 0.0, 0.3025, 5, "rel", -1 / 3),
    ("ERF_G", g_erf, 0.0, 1.0, 6, "abs", float(2 / PI.sqrt() - 1)),
    ("ERFC_P", g_erfc, -0.3345, 0.3005, 9, "abs", None),  # constant term split
    ("RCBRT_P", g_rcbrt, 1.0, 8.0, 4, "rel", None),
]


def hexf(v: float) -> str:
    if v == 0:
        return "0.0f"
    return float(v).hex().replace("0000000p", "p").replace("000000p", "p") + "f"


def main():
    report = "--report" in sys.argv
    out = []
    w = out.append
    w("/* Generated by tools/gen_mathf_tables.py; do not edit. */")
    w("#ifndef OCU_DEVICE_MATHF_TABLES_H")
    w("#define OCU_DEVICE_MATHF_TABLES_H")
    w("#include <cstdint>")
    w("namespace ocu_dm {")
    w("namespace mf {")
    w("")
    w("/* Constants as sums of floats (hi + mid + lo). */")
    consts = [
        ("LN2_E", LN2, 2, 24),          # exp reduction: k * hi exact for |k| < 2^8 enough
        ("LN2_L", LN2, 3, 16),          # log: e * hi exact for |e| < 2^8
        ("LOG2E", 1 / LN2, 2, 24),
        ("LN10", Fraction(D(10).ln()), 2, 24),
        ("LOG10E", 1 / Fraction(D(10).ln()), 2, 24),
        ("PIO2", PI_FRAC / 2, 3, 24),
        ("PI", PI_FRAC, 2, 24),
        ("TWO_OVER_PI", 2 / PI_FRAC, 1, 24),
    ]
    for name, q, parts, fb in consts:
        vals = split(q, parts, fb)
        suff = ["_H", "_M", "_L"] if parts == 3 else ["_H", "_L"] if parts == 2 else [""]
        w("constexpr float " + ", ".join(f"{name}{s} = {hexf(v)}" for s, v in zip(suff, vals)) + ";")
    w("")
    # 2/pi bits: word j, bit b (0 = most significant) has the value 2^-(32 j + b - 31);
    # word 0 holds the integer part (zero).
    nw = 10
    v = (2 << (32 * nw - 32 + PI_BITS)) // PI_FIXED  # floor(2/pi * 2^(32 nw - 32))
    words = [(v >> (32 * (nw - 1 - j))) & 0xFFFFFFFF for j in range(nw)]
    w("/* Bits of 2/pi, most significant first; word 0 is the integer part (0). */")
    w(f"constexpr std::uint32_t TWO_OVER_PI_BITS[{nw}] = {{")
    w("    " + ", ".join(f"0x{x:08x}u" for x in words) + "};")
    w("")
    # log table
    w("/* log table: for m near 1 + i/64 (halved to near (1 + i/64)/2 when i >= 27),")
    w("   c ~ 1/m and -ln(c) = nlh + nll. */")
    w("struct log_entry { float c, nlh, nll, pad; };")
    w("alignas(16) OCU_DMF_TABLE log_entry LOG_TABLE[65] = {")
    for i in range(65):
        q = 1 / (1 + Fraction(i, 64))
        if i >= 27:
            q *= 2
        c = f32_exact(q)
        nl = -ln_frac(Fraction(c), 70)
        hi = f32_exact(nl)
        lo = f32_exact(nl - Fraction(hi))
        w(f"    {{{hexf(c)}, {hexf(hi)}, {hexf(lo)}, 0.0f}},")
    w("};")
    w("")
    w("/* Polynomial coefficients, constant term first (see the header comment")
    w("   of each function for what is approximated). */")
    for name, fn, a, b, deg, weight, g0 in FITS:
        split0 = name == "ERFC_P"
        coef, emax = fit(fn, a, b, deg, weight, g0=g0, split0=split0)
        if report:
            print(f"{name:8s} deg {deg:2d} on [{a}, {b}]: max {weight} error {emax:.3e} "
                  f"(2^{math.log2(emax):.1f}); unrounded 2^{math.log2(fit.unrounded):.1f}", file=sys.stderr)
        w(f"/* max {weight} error {emax:.2e} on [{a}, {b}] */")
        if split0:
            hi = f32(coef[0])
            w(f"constexpr float {name}0_H = {hexf(hi)}, {name}0_L = {hexf(fit.lo)};")
            w(f"constexpr float {name}[{deg}] = {{  /* degree 1 and up */")
            w("    " + ", ".join(hexf(c) for c in coef[1:]) + "};")
        else:
            w(f"constexpr float {name}[{deg + 1}] = {{")
            w("    " + ", ".join(hexf(c) for c in coef) + "};")
    w("")
    w("} // namespace mf")
    w("} // namespace ocu_dm")
    w("#endif")
    if not report:
        print("\n".join(out))


if __name__ == "__main__":
    main()

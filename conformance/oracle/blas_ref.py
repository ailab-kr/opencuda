"""Mathematical reference answers for the BLAS operations in the phase-0 set.

Semantics follow the public BLAS standard (the netlib reference BLAS
definitions): strided vectors with positive or negative increments,
column-major matrices with a leading dimension, 1-based index results.
Nothing here depends on any vendor implementation.

Precision: every result is computed exactly (rational arithmetic, or
math.fsum where that is exact) and rounded once to the requested float
type. These are the "mathematically determined answers" of
conformance/README.md, rule 2.

Standard library only.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Sequence

F32 = "f32"
F64 = "f64"


def round_to(value: float | Fraction, kind: str) -> float:
    """Round once to float32 or float64 (NaN and infinities pass through)."""
    if isinstance(value, Fraction):
        value = _fraction_to_float(value)
    if kind == F64:
        return float(value)
    if math.isnan(value) or math.isinf(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _fraction_to_float(value: Fraction) -> float:
    try:
        return float(value)  # correctly rounded by CPython
    except OverflowError:
        return math.copysign(math.inf, value)


def representable(values: Sequence[float], kind: str) -> list[float]:
    """Round inputs to the element type, as a test would store them."""
    return [round_to(v, kind) for v in values]


# ---- strided access ------------------------------------------------------

def vector_indices(n: int, inc: int) -> list[int]:
    """Element positions of a BLAS vector with increment `inc`.

    For inc < 0 the vector starts at the far end: element i lives at
    (n - 1 - i) * |inc|, as in the reference BLAS.
    """
    if n <= 0:
        return []
    if inc == 0:
        raise ValueError("increment must not be zero")
    if inc > 0:
        return [i * inc for i in range(n)]
    return [(n - 1 - i) * (-inc) for i in range(n)]


def gather(buf: Sequence[float], n: int, inc: int) -> list[float]:
    return [buf[k] for k in vector_indices(n, inc)]


def scatter(buf: list[float], n: int, inc: int, values: Sequence[float]) -> list[float]:
    out = list(buf)
    for k, v in zip(vector_indices(n, inc), values):
        out[k] = v
    return out


def _has_special(values: Sequence[float]) -> bool:
    return any(math.isnan(v) or math.isinf(v) for v in values)


def _exact(v: float) -> Fraction:
    return Fraction(v)


def exact_dot(xs: Sequence[float], ys: Sequence[float]) -> Fraction | float:
    """sum(x*y) exactly; float (NaN/inf) when a special value is involved."""
    if _has_special(xs) or _has_special(ys):
        return math.fsum(x * y for x, y in zip(xs, ys)) if xs else 0.0
    return sum((_exact(x) * _exact(y) for x, y in zip(xs, ys)), Fraction(0))


# ---- level 1 -------------------------------------------------------------

def axpy(n: int, alpha: float, x: Sequence[float], incx: int,
         y: Sequence[float], incy: int, kind: str) -> list[float]:
    """y := alpha*x + y. n <= 0 leaves y unchanged."""
    if n <= 0:
        return list(y)
    xs, ys = gather(x, n, incx), gather(y, n, incy)
    out = []
    for xv, yv in zip(xs, ys):
        if _has_special((alpha, xv, yv)):
            out.append(round_to(alpha * xv + yv, kind))
        else:
            out.append(round_to(_exact(alpha) * _exact(xv) + _exact(yv), kind))
    return scatter(list(y), n, incy, out)


def dot(n: int, x: Sequence[float], incx: int, y: Sequence[float], incy: int,
        kind: str) -> float:
    if n <= 0:
        return 0.0
    return round_to(exact_dot(gather(x, n, incx), gather(y, n, incy)), kind)


def scal(n: int, alpha: float, x: Sequence[float], incx: int, kind: str) -> list[float]:
    if n <= 0 or incx <= 0:
        return list(x)  # reference BLAS: no-op for incx <= 0
    xs = gather(x, n, incx)
    out = [round_to(alpha * v if _has_special((alpha, v))
                    else _exact(alpha) * _exact(v), kind) for v in xs]
    return scatter(list(x), n, incx, out)


def nrm2(n: int, x: Sequence[float], incx: int, kind: str) -> float:
    """sqrt(sum x^2) without intermediate overflow or underflow."""
    if n <= 0 or incx <= 0:
        return 0.0
    xs = gather(x, n, incx)
    if any(math.isnan(v) for v in xs):
        return math.nan
    if any(math.isinf(v) for v in xs):
        return math.inf
    total = sum((_exact(v) ** 2 for v in xs), Fraction(0))
    if total == 0:
        return 0.0
    getcontext().prec = 60
    root = (Decimal(total.numerator) / Decimal(total.denominator)).sqrt()
    return round_to(float(root), kind)


def asum(n: int, x: Sequence[float], incx: int, kind: str) -> float:
    if n <= 0 or incx <= 0:
        return 0.0
    xs = gather(x, n, incx)
    if _has_special(xs):
        return round_to(math.fsum(abs(v) for v in xs), kind)
    return round_to(sum((abs(_exact(v)) for v in xs), Fraction(0)), kind)


def iamax(n: int, x: Sequence[float], incx: int) -> int:
    """1-based index of the first element with the largest |x|; 0 if none."""
    if n <= 0 or incx <= 0:
        return 0
    xs = gather(x, n, incx)
    best, best_i = -1.0, 0
    for i, v in enumerate(xs):
        a = abs(v)
        if math.isnan(a):
            continue  # implementations differ on NaN; tests avoid it
        if a > best:
            best, best_i = a, i
    return best_i + 1


def copy(n: int, x: Sequence[float], incx: int, y: Sequence[float],
         incy: int) -> list[float]:
    if n <= 0:
        return list(y)
    return scatter(list(y), n, incy, gather(x, n, incx))


def swap(n: int, x: Sequence[float], incx: int, y: Sequence[float],
         incy: int) -> tuple[list[float], list[float]]:
    if n <= 0:
        return list(x), list(y)
    xs, ys = gather(x, n, incx), gather(y, n, incy)
    return scatter(list(x), n, incx, ys), scatter(list(y), n, incy, xs)


# ---- matrices (column-major) ---------------------------------------------

def mat_get(a: Sequence[float], lda: int, i: int, j: int) -> float:
    return a[i + j * lda]


def _op(a: Sequence[float], lda: int, trans: str, i: int, j: int) -> float:
    return mat_get(a, lda, j, i) if trans in ("T", "C") else mat_get(a, lda, i, j)


def gemv(trans: str, m: int, n: int, alpha: float, a: Sequence[float], lda: int,
         x: Sequence[float], incx: int, beta: float, y: Sequence[float],
         incy: int, kind: str) -> list[float]:
    """y := alpha*op(A)*x + beta*y, A is m x n. beta == 0 never reads y."""
    if m <= 0 or n <= 0:
        return list(y)
    rows, cols = (n, m) if trans in ("T", "C") else (m, n)
    xs = gather(x, cols, incx)
    ys = gather(y, rows, incy)
    out = []
    for i in range(rows):
        arow = [_op(a, lda, trans, i, j) for j in range(cols)]
        s = exact_dot(arow, xs)
        out.append(_combine(alpha, s, beta, ys[i], kind))
    return scatter(list(y), rows, incy, out)


def _combine(alpha: float, s: Fraction | float, beta: float, c: float,
             kind: str) -> float:
    """alpha*s + beta*c with the BLAS rule that beta == 0 ignores c."""
    if isinstance(s, float) or _has_special((alpha,)) or \
            (beta != 0 and _has_special((beta, c))):
        sv = float(s) if isinstance(s, Fraction) else s
        val = alpha * sv + (beta * c if beta != 0 else 0.0)
        return round_to(val, kind)
    val = _exact(alpha) * s
    if beta != 0:
        val += _exact(beta) * _exact(c)
    return round_to(val, kind)


def ger(m: int, n: int, alpha: float, x: Sequence[float], incx: int,
        y: Sequence[float], incy: int, a: Sequence[float], lda: int,
        kind: str) -> list[float]:
    """A := alpha*x*y^T + A."""
    out = list(a)
    if m <= 0 or n <= 0:
        return out
    xs, ys = gather(x, m, incx), gather(y, n, incy)
    for j in range(n):
        for i in range(m):
            k = i + j * lda
            if _has_special((alpha, xs[i], ys[j], a[k])):
                out[k] = round_to(alpha * xs[i] * ys[j] + a[k], kind)
            else:
                out[k] = round_to(_exact(alpha) * _exact(xs[i]) * _exact(ys[j])
                                  + _exact(a[k]), kind)
    return out


def gemm(transa: str, transb: str, m: int, n: int, k: int, alpha: float,
         a: Sequence[float], lda: int, b: Sequence[float], ldb: int,
         beta: float, c: Sequence[float], ldc: int, kind: str) -> list[float]:
    """C := alpha*op(A)*op(B) + beta*C, C is m x n. beta == 0 never reads C."""
    out = list(c)
    if m <= 0 or n <= 0:
        return out
    for j in range(n):
        bcol = [_op(b, ldb, transb, p, j) for p in range(k)]
        for i in range(m):
            arow = [_op(a, lda, transa, i, p) for p in range(k)]
            s = exact_dot(arow, bcol) if k > 0 else Fraction(0)
            out[i + j * ldc] = _combine(alpha, s, beta, c[i + j * ldc], kind)
    return out


def syrk(uplo: str, trans: str, n: int, k: int, alpha: float,
         a: Sequence[float], lda: int, beta: float, c: Sequence[float],
         ldc: int, kind: str) -> list[float]:
    """C := alpha*op(A)*op(A)^T + beta*C on the `uplo` triangle only.

    op(A) is n x k (trans 'N') or A^T with A k x n (trans 'T')."""
    out = list(c)
    if n <= 0:
        return out
    for j in range(n):
        rows = range(0, j + 1) if uplo == "U" else range(j, n)
        for i in rows:
            if trans == "N":
                ai = [mat_get(a, lda, i, p) for p in range(k)]
                aj = [mat_get(a, lda, j, p) for p in range(k)]
            else:
                ai = [mat_get(a, lda, p, i) for p in range(k)]
                aj = [mat_get(a, lda, p, j) for p in range(k)]
            s = exact_dot(ai, aj) if k > 0 else Fraction(0)
            out[i + j * ldc] = _combine(alpha, s, beta, c[i + j * ldc], kind)
    return out


def _tri(a: Sequence[float], lda: int, n: int, uplo: str, trans: str,
         diag: str) -> list[list[Fraction]]:
    """op(A) as an exact dense n x n matrix, zeros outside the triangle."""
    full = [[Fraction(0)] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            inside = i <= j if uplo == "U" else i >= j
            if not inside:
                continue
            v = Fraction(1) if (i == j and diag == "U") else _exact(mat_get(a, lda, i, j))
            full[i][j] = v
    if trans in ("T", "C"):
        full = [[full[j][i] for j in range(n)] for i in range(n)]
    return full


def _dec(v: Fraction) -> Decimal:
    return Decimal(v.numerator) / Decimal(v.denominator)


def _solve_tri(t: list[list[Fraction]], rhs: list[Fraction]) -> list[Decimal]:
    """Solve t*x = rhs for triangular t (either orientation).

    Exact rationals grow without bound in a triangular solve, so this works
    in 100-digit decimal arithmetic: about 1e-90 relative error, far below
    any float64 rounding."""
    getcontext().prec = 100
    n = len(rhs)
    lower = all(t[i][j] == 0 for i in range(n) for j in range(i + 1, n))
    td = [[_dec(v) for v in row] for row in t]
    x = [Decimal(0)] * n
    order = range(n) if lower else range(n - 1, -1, -1)
    for i in order:
        s = _dec(rhs[i])
        cols = range(0, i) if lower else range(i + 1, n)
        for j in cols:
            s -= td[i][j] * x[j]
        if td[i][i] == 0:
            raise ZeroDivisionError("singular triangular matrix")
        x[i] = s / td[i][i]
    return x


def trsv(uplo: str, trans: str, diag: str, n: int, a: Sequence[float], lda: int,
         x: Sequence[float], incx: int, kind: str) -> list[float]:
    """x := inv(op(A)) * x for triangular A. Inputs must be finite."""
    if n <= 0:
        return list(x)
    t = _tri(a, lda, n, uplo, trans, diag)
    sol = _solve_tri(t, [_exact(v) for v in gather(x, n, incx)])
    return scatter(list(x), n, incx, [round_to(float(v), kind) for v in sol])


def trsm(side: str, uplo: str, trans: str, diag: str, m: int, n: int,
         alpha: float, a: Sequence[float], lda: int, b: Sequence[float],
         ldb: int, kind: str) -> list[float]:
    """B := alpha*inv(op(A))*B (side 'L', A m x m) or
    B := alpha*B*inv(op(A)) (side 'R', A n x n). Inputs must be finite."""
    out = list(b)
    if m <= 0 or n <= 0:
        return out
    al = _exact(alpha)
    if side == "L":
        t = _tri(a, lda, m, uplo, trans, diag)
        for j in range(n):
            col = [al * _exact(mat_get(b, ldb, i, j)) for i in range(m)]
            sol = _solve_tri(t, col)
            for i in range(m):
                out[i + j * ldb] = round_to(float(sol[i]), kind)
    else:
        # X*op(A) = alpha*B  <=>  op(A)^T * X^T = alpha*B^T
        t = _tri(a, lda, n, uplo, trans, diag)
        tt = [[t[j][i] for j in range(n)] for i in range(n)]
        for i in range(m):
            row = [al * _exact(mat_get(b, ldb, i, j)) for j in range(n)]
            sol = _solve_tri(tt, row)
            for j in range(n):
                out[i + j * ldb] = round_to(float(sol[j]), kind)
    return out

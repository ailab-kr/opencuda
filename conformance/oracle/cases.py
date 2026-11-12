"""Deterministic test cases for the phase-0 BLAS set, with reference answers.

Each case is a plain dict that can be written to JSON:

    {"id": "L4.blas.saxpy.n17_incx-2_incy1", "card": "blas-saxpy",
     "fn": "axpy", "kind": "f32", "args": {...},
     "expect": {"y": [...]}, "scale": {"y": [...]}, "terms": {"y": k}}

`expect` holds the mathematically exact answer rounded once. `scale[i]`
is the sum of the absolute values of the terms that make output i, and
`terms` how many there are; together they bound the rounding error a
correct implementation may show (see conformance/harness/ocu_conf/blas_check.py).

Inputs cover the rules in conformance/README.md: sizes 0, 1, small primes
and near powers of two; positive and negative increments; zero alpha and
beta; NaN in an output buffer that beta == 0 must ignore.
"""

from __future__ import annotations

import math
import random
from typing import Iterator

from . import blas_ref as R

F32, F64 = R.F32, R.F64

SIZES_1D = (0, 1, 2, 7, 16, 17, 31, 64, 129)
INCS = (1, 2, -1, -3)
MAT_SIZES = ((1, 1, 1), (3, 5, 2), (8, 8, 8), (17, 9, 13), (32, 1, 31))


def _vals(rng: random.Random, count: int, kind: str, lo=-4.0, hi=4.0) -> list[float]:
    return R.representable([rng.uniform(lo, hi) for _ in range(count)], kind)


def _buf_len(n: int, inc: int) -> int:
    return 1 + (n - 1) * abs(inc) if n > 0 else 1


def _abs_scale_vec(n, alpha, x, incx, y, incy):
    xs, ys = R.gather(x, n, incx), R.gather(y, n, incy)
    return [abs(alpha * a) + abs(b) for a, b in zip(xs, ys)]


def _case(fid: str, card: str, fn: str, kind: str, args: dict, expect: dict,
          scale: dict | None = None, terms: dict | None = None) -> dict:
    return {"id": fid, "card": card, "fn": fn, "kind": kind, "args": args,
            "expect": expect, "scale": scale or {}, "terms": terms or {}}


def level1_cases(seed: int = 1) -> Iterator[dict]:
    rng = random.Random(seed)
    for kind, p in ((F32, "s"), (F64, "d")):
        for n in SIZES_1D:
            for incx, incy in ((1, 1), (2, -1), (-3, 2)):
                x = _vals(rng, _buf_len(n, incx), kind)
                y = _vals(rng, _buf_len(n, incy), kind)
                tag = f"n{n}_incx{incx}_incy{incy}"
                for alpha in (R.round_to(1.5, kind), 0.0):
                    yo = R.axpy(n, alpha, x, incx, y, incy, kind)
                    sc = R.scatter([0.0] * len(y), n, incy,
                                   _abs_scale_vec(n, alpha, x, incx, y, incy))
                    yield _case(f"L4.blas.{p}axpy.{tag}_a{alpha:g}", f"blas-{p}axpy",
                                "axpy", kind,
                                {"n": n, "alpha": alpha, "x": x, "incx": incx,
                                 "y": y, "incy": incy},
                                {"y": yo}, {"y": sc}, {"y": 2})
                d = R.dot(n, x, incx, y, incy, kind)
                xs, ys = R.gather(x, n, incx), R.gather(y, n, incy)
                yield _case(f"L4.blas.{p}dot.{tag}", f"blas-{p}dot", "dot", kind,
                            {"n": n, "x": x, "incx": incx, "y": y, "incy": incy},
                            {"result": d},
                            {"result": math.fsum(abs(a * b) for a, b in zip(xs, ys))},
                            {"result": max(n, 1)})
                if p == "s":
                    yield _case(f"L4.blas.scopy.{tag}", "blas-scopy", "copy", kind,
                                {"n": n, "x": x, "incx": incx, "y": y, "incy": incy},
                                {"y": R.copy(n, x, incx, y, incy)})
                    xo, yo2 = R.swap(n, x, incx, y, incy)
                    yield _case(f"L4.blas.sswap.{tag}", "blas-sswap", "swap", kind,
                                {"n": n, "x": x, "incx": incx, "y": y, "incy": incy},
                                {"x": xo, "y": yo2})
            for incx in (1, 3):
                x = _vals(rng, _buf_len(n, incx), kind)
                tag = f"n{n}_incx{incx}"
                alpha = R.round_to(-0.75, kind)
                yield _case(f"L4.blas.{p}scal.{tag}", f"blas-{p}scal", "scal", kind,
                            {"n": n, "alpha": alpha, "x": x, "incx": incx},
                            {"x": R.scal(n, alpha, x, incx, kind)},
                            {"x": [abs(alpha * v) for v in x]}, {"x": 1})
                xs = R.gather(x, n, incx)
                yield _case(f"L4.blas.{p}nrm2.{tag}", f"blas-{p}nrm2", "nrm2", kind,
                            {"n": n, "x": x, "incx": incx},
                            {"result": R.nrm2(n, x, incx, kind)},
                            {"result": R.nrm2(n, x, incx, F64)},
                            {"result": max(n, 1)})
                if p == "s":
                    yield _case(f"L4.blas.sasum.{tag}", "blas-sasum", "asum", kind,
                                {"n": n, "x": x, "incx": incx},
                                {"result": R.asum(n, x, incx, kind)},
                                {"result": math.fsum(abs(v) for v in xs)},
                                {"result": max(n, 1)})
                    yield _case(f"L4.blas.isamax.{tag}", "blas-isamax", "iamax", kind,
                                {"n": n, "x": x, "incx": incx},
                                {"result": R.iamax(n, x, incx)})
    # nrm2 must not overflow or underflow in the middle of the sum.
    for kind, p, big in ((F32, "s", 1e30), (F64, "d", 1e200)):
        for mult in (big, 1.0 / big):
            x = R.representable([3.0 * mult, 4.0 * mult], kind)
            yield _case(f"L4.blas.{p}nrm2.scaled_{'big' if mult > 1 else 'tiny'}",
                        f"blas-{p}nrm2", "nrm2", kind, {"n": 2, "x": x, "incx": 1},
                        {"result": R.nrm2(2, x, 1, kind)},
                        {"result": R.nrm2(2, x, 1, F64)}, {"result": 2})
    # isamax: ties pick the first; |x| not x.
    x = [1.0, -5.0, 5.0, 2.0]
    yield _case("L4.blas.isamax.ties_and_sign", "blas-isamax", "iamax", F32,
                {"n": 4, "x": x, "incx": 1}, {"result": R.iamax(4, x, 1)})


def _mat(rng, rows, cols, ld, kind, fill=math.nan):
    """Column-major rows x cols with leading dimension ld; padding is NaN so
    reads outside the matrix show up."""
    out = [fill] * max(ld * cols, 1)
    for j in range(cols):
        for i in range(rows):
            out[i + j * ld] = R.round_to(rng.uniform(-2.0, 2.0), kind)
    return out


def _gemm_scale(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    sc = [0.0] * len(c)
    for j in range(n):
        for i in range(m):
            s = math.fsum(abs(R._op(a, lda, transa, i, p) * R._op(b, ldb, transb, p, j))
                          for p in range(k))
            cv = c[i + j * ldc] if beta != 0 else 0.0
            sc[i + j * ldc] = abs(alpha) * s + abs(beta * cv)
    return sc


def level2_3_cases(seed: int = 2) -> Iterator[dict]:
    rng = random.Random(seed)
    for kind, p in ((F32, "s"), (F64, "d")):
        for m, n, k in MAT_SIZES:
            for trans in ("N", "T"):
                lda = m + 1
                a = _mat(rng, m, n, lda, kind)
                xlen, ylen = (m, n) if trans == "T" else (n, m)
                x = _vals(rng, xlen, kind)
                for beta in (R.round_to(0.5, kind), 0.0):
                    y = _vals(rng, ylen, kind) if beta != 0 else [math.nan] * ylen
                    alpha = R.round_to(1.25, kind)
                    yo = R.gemv(trans, m, n, alpha, a, lda, x, 1, beta, y, 1, kind)
                    rows, cols = ylen, xlen
                    sc = [abs(alpha) * math.fsum(abs(R._op(a, lda, trans, i, j) * x[j])
                                                 for j in range(cols))
                          + (abs(beta * y[i]) if beta != 0 else 0.0) for i in range(rows)]
                    yield _case(f"L4.blas.{p}gemv.{trans}_m{m}_n{n}_b{beta:g}",
                                f"blas-{p}gemv", "gemv", kind,
                                {"trans": trans, "m": m, "n": n, "alpha": alpha,
                                 "a": a, "lda": lda, "x": x, "incx": 1,
                                 "beta": beta, "y": y, "incy": 1},
                                {"y": yo}, {"y": sc}, {"y": cols + 1})
            for ta, tb in (("N", "N"), ("T", "N"), ("N", "T"), ("T", "T")):
                arows, acols = (m, k) if ta == "N" else (k, m)
                brows, bcols = (k, n) if tb == "N" else (n, k)
                lda, ldb, ldc = arows + 2, brows + 1, m + 3
                a = _mat(rng, arows, acols, lda, kind)
                b = _mat(rng, brows, bcols, ldb, kind)
                for beta in (R.round_to(-1.0, kind), 0.0):
                    c = _mat(rng, m, n, ldc, kind)
                    if beta == 0:
                        for j in range(n):
                            for i in range(m):
                                c[i + j * ldc] = math.nan  # must not be read
                    alpha = R.round_to(0.5, kind)
                    co = R.gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, kind)
                    sc = _gemm_scale(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                    yield _case(f"L4.blas.{p}gemm.{ta}{tb}_m{m}_n{n}_k{k}_b{beta:g}",
                                f"blas-{p}gemm", "gemm", kind,
                                {"transa": ta, "transb": tb, "m": m, "n": n, "k": k,
                                 "alpha": alpha, "a": a, "lda": lda, "b": b, "ldb": ldb,
                                 "beta": beta, "c": c, "ldc": ldc},
                                {"c": co}, {"c": sc}, {"c": k + 1})
        if p != "s":
            continue
        for m, n, _ in MAT_SIZES:
            x = _vals(rng, m, kind)
            y = _vals(rng, n, kind)
            lda = m + 1
            a = _mat(rng, m, n, lda, kind)
            alpha = R.round_to(-2.0, kind)
            ao = R.ger(m, n, alpha, x, 1, y, 1, a, lda, kind)
            sc = list(a)
            for j in range(n):
                for i in range(m):
                    sc[i + j * lda] = abs(alpha * x[i] * y[j]) + abs(a[i + j * lda])
            yield _case(f"L4.blas.sger.m{m}_n{n}", "blas-sger", "ger", kind,
                        {"m": m, "n": n, "alpha": alpha, "x": x, "incx": 1,
                         "y": y, "incy": 1, "a": a, "lda": lda},
                        {"a": ao}, {"a": sc}, {"a": 2})
        for n in (1, 4, 13, 32):
            for uplo in ("U", "L"):
                for trans in ("N", "T"):
                    for diag in ("N", "U"):
                        lda = n + 2
                        a = _tri_well_conditioned(rng, n, lda, uplo, kind)
                        x = _vals(rng, n, kind)
                        xo = R.trsv(uplo, trans, diag, n, a, lda, x, 1, kind)
                        yield _case(f"L4.blas.strsv.{uplo}{trans}{diag}_n{n}",
                                    "blas-strsv", "trsv", kind,
                                    {"uplo": uplo, "trans": trans, "diag": diag,
                                     "n": n, "a": a, "lda": lda, "x": x, "incx": 1},
                                    {"x": xo}, {"x": [abs(v) for v in xo]},
                                    {"x": 4 * n})
        for n, k in ((1, 1), (5, 3), (16, 16), (9, 20)):
            for uplo in ("U", "L"):
                for trans in ("N", "T"):
                    arows, acols = (n, k) if trans == "N" else (k, n)
                    lda, ldc = arows + 1, n + 2
                    a = _mat(rng, arows, acols, lda, kind)
                    c = _mat(rng, n, n, ldc, kind)
                    alpha, beta = R.round_to(0.75, kind), R.round_to(2.0, kind)
                    co = R.syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc, kind)
                    sc = list(c)
                    for j in range(n):
                        for i in range(n):
                            if trans == "N":
                                s = math.fsum(abs(a[i + q * lda] * a[j + q * lda]) for q in range(k))
                            else:
                                s = math.fsum(abs(a[q + i * lda] * a[q + j * lda]) for q in range(k))
                            sc[i + j * ldc] = abs(alpha) * s + abs(beta * c[i + j * ldc])
                    yield _case(f"L4.blas.ssyrk.{uplo}{trans}_n{n}_k{k}", "blas-ssyrk",
                                "syrk", kind,
                                {"uplo": uplo, "trans": trans, "n": n, "k": k,
                                 "alpha": alpha, "a": a, "lda": lda, "beta": beta,
                                 "c": c, "ldc": ldc},
                                {"c": co}, {"c": sc}, {"c": k + 1})
        for m, n in ((1, 1), (6, 3), (16, 9)):
            for side in ("L", "R"):
                for uplo in ("U", "L"):
                    for trans in ("N", "T"):
                        na = m if side == "L" else n
                        lda, ldb = na + 1, m + 2
                        a = _tri_well_conditioned(rng, na, lda, uplo, kind)
                        b = _mat(rng, m, n, ldb, kind)
                        alpha = R.round_to(1.5, kind)
                        bo = R.trsm(side, uplo, trans, "N", m, n, alpha, a, lda, b, ldb, kind)
                        yield _case(f"L4.blas.strsm.{side}{uplo}{trans}_m{m}_n{n}",
                                    "blas-strsm", "trsm", kind,
                                    {"side": side, "uplo": uplo, "trans": trans,
                                     "diag": "N", "m": m, "n": n, "alpha": alpha,
                                     "a": a, "lda": lda, "b": b, "ldb": ldb},
                                    {"b": bo},
                                    {"b": [abs(v) if not math.isnan(v) else 0.0 for v in bo]},
                                    {"b": 4 * na})


def _tri_well_conditioned(rng, n, lda, uplo, kind):
    """Triangular matrix with a dominant diagonal, so solves are stable and
    the error bound stays meaningful. The other triangle holds NaN, which
    a correct solver never reads."""
    a = [math.nan] * (lda * n)
    for j in range(n):
        for i in range(n):
            inside = i <= j if uplo == "U" else i >= j
            if not inside:
                continue
            if i == j:
                a[i + j * lda] = R.round_to(rng.choice((-1, 1)) * rng.uniform(n + 1.0, n + 3.0), kind)
            else:
                a[i + j * lda] = R.round_to(rng.uniform(-1.0, 1.0) / n, kind)
    return a


def alpha_zero_cases(seed: int = 3) -> Iterator[dict]:
    """alpha == 0: A and B must not be read. They hold inf and NaN, so an
    implementation that reads them anyway produces NaN."""
    rng = random.Random(seed)
    inf, nan = math.inf, math.nan
    for kind, p in ((F32, "s"), (F64, "d")):
        m, n, k = 5, 4, 3
        poison = [inf, nan, -inf] * 20
        for beta in (R.round_to(0.5, kind), 0.0):
            lda, ldb, ldc = m, k, m + 1
            c = _mat(rng, m, n, ldc, kind)
            yield _case(f"L4.blas.{p}gemm.alpha0_b{beta:g}", f"blas-{p}gemm", "gemm", kind,
                        {"transa": "N", "transb": "N", "m": m, "n": n, "k": k, "alpha": 0.0,
                         "a": poison[:lda * k], "lda": lda, "b": poison[:ldb * n], "ldb": ldb,
                         "beta": beta, "c": c, "ldc": ldc},
                        {"c": R.gemm("N", "N", m, n, k, 0.0, poison, lda, poison, ldb, beta,
                                     c, ldc, kind)},
                        {"c": [abs(beta * v) if not math.isnan(v) else 0.0 for v in c]},
                        {"c": 1})
            y = _vals(rng, m, kind)
            yield _case(f"L4.blas.{p}gemv.alpha0_b{beta:g}", f"blas-{p}gemv", "gemv", kind,
                        {"trans": "N", "m": m, "n": n, "alpha": 0.0, "a": poison[:m * n],
                         "lda": m, "x": poison[:n], "incx": 1, "beta": beta, "y": y,
                         "incy": 1},
                        {"y": R.gemv("N", m, n, 0.0, poison, m, poison, 1, beta, y, 1, kind)},
                        {"y": [abs(beta * v) for v in y]}, {"y": 1})
        if p == "s":
            c = _mat(rng, n, n, n, kind)
            beta = R.round_to(2.0, kind)
            yield _case("L4.blas.ssyrk.alpha0", "blas-ssyrk", "syrk", kind,
                        {"uplo": "U", "trans": "N", "n": n, "k": k, "alpha": 0.0,
                         "a": poison[:n * k], "lda": n, "beta": beta, "c": c, "ldc": n},
                        {"c": R.syrk("U", "N", n, k, 0.0, poison, n, beta, c, n, kind)},
                        {"c": [abs(beta * v) for v in c]}, {"c": 1})
            for side in ("L", "R"):
                na = m if side == "L" else n
                b = poison[:(m + 1) * n]
                yield _case(f"L4.blas.strsm.alpha0_{side}", "blas-strsm", "trsm", kind,
                            {"side": side, "uplo": "U", "trans": "N", "diag": "N", "m": m,
                             "n": n, "alpha": 0.0, "a": poison[:na * na], "lda": na,
                             "b": b, "ldb": m + 1},
                            {"b": R.trsm(side, "U", "N", "N", m, n, 0.0, poison, na, b,
                                         m + 1, kind)})


def all_cases() -> list[dict]:
    return list(level1_cases()) + list(level2_3_cases()) + list(alpha_zero_cases())

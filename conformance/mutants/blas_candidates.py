"""Candidate BLAS implementations used to check the test suite itself.

`naive` works the way a simple GPU kernel would: every multiply and add is
rounded to the element type, in plain left-to-right order. A correct suite
must pass it. Each entry of `MUTANTS` breaks one rule on purpose; a correct
suite must fail every one of them on at least one case.

Standard library only. Not an implementation of anything we ship.
"""

from __future__ import annotations

import math
import struct
from typing import Callable

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle import blas_ref as R  # noqa: E402


def rnd(v: float, kind: str) -> float:
    if kind == "f64" or math.isnan(v) or math.isinf(v):
        return v
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def _acc(pairs, kind):
    s = 0.0
    for a, b in pairs:
        s = rnd(s + rnd(a * b, kind), kind)
    return s


def _op(a, lda, trans, i, j):
    return a[j + i * lda] if trans in ("T", "C") else a[i + j * lda]


class Naive:
    """Straightforward implementation, rounding every step."""

    # -- level 1 --
    def axpy(self, c):
        a, k = c["args"], c["kind"]
        n = a["n"]
        y = list(a["y"])
        if n <= 0:
            return {"y": y}
        xi, yi = R.vector_indices(n, a["incx"]), R.vector_indices(n, a["incy"])
        for p, q in zip(xi, yi):
            y[q] = rnd(rnd(a["alpha"] * a["x"][p], k) + y[q], k)
        return {"y": y}

    def dot(self, c):
        a, k = c["args"], c["kind"]
        if a["n"] <= 0:
            return {"result": 0.0}
        xs, ys = R.gather(a["x"], a["n"], a["incx"]), R.gather(a["y"], a["n"], a["incy"])
        return {"result": _acc(zip(xs, ys), k)}

    def scal(self, c):
        a, k = c["args"], c["kind"]
        x = list(a["x"])
        if a["n"] <= 0 or a["incx"] <= 0:
            return {"x": x}
        for p in R.vector_indices(a["n"], a["incx"]):
            x[p] = rnd(a["alpha"] * x[p], k)
        return {"x": x}

    def nrm2(self, c):
        a, k = c["args"], c["kind"]
        n, incx = a["n"], a["incx"]
        if n <= 0 or incx <= 0:
            return {"result": 0.0}
        xs = R.gather(a["x"], n, incx)
        scale = max((abs(v) for v in xs), default=0.0)
        if scale == 0 or math.isinf(scale) or math.isnan(scale):
            return {"result": scale}
        s = 0.0
        for v in xs:
            t = rnd(v / scale, k)
            s = rnd(s + rnd(t * t, k), k)
        return {"result": rnd(scale * rnd(math.sqrt(s), k), k)}

    def asum(self, c):
        a, k = c["args"], c["kind"]
        if a["n"] <= 0 or a["incx"] <= 0:
            return {"result": 0.0}
        s = 0.0
        for v in R.gather(a["x"], a["n"], a["incx"]):
            s = rnd(s + abs(v), k)
        return {"result": s}

    def iamax(self, c):
        a = c["args"]
        return {"result": R.iamax(a["n"], a["x"], a["incx"])}

    def copy(self, c):
        a = c["args"]
        return {"y": R.copy(a["n"], a["x"], a["incx"], a["y"], a["incy"])}

    def swap(self, c):
        a = c["args"]
        x, y = R.swap(a["n"], a["x"], a["incx"], a["y"], a["incy"])
        return {"x": x, "y": y}

    # -- level 2 / 3 --
    def gemv(self, c):
        a, k = c["args"], c["kind"]
        m, n, tr = a["m"], a["n"], a["trans"]
        y = list(a["y"])
        if m <= 0 or n <= 0:
            return {"y": y}
        rows, cols = (n, m) if tr == "T" else (m, n)
        for i in range(rows):
            s = _acc(((_op(a["a"], a["lda"], tr, i, j), a["x"][j]) for j in range(cols)), k)
            v = rnd(a["alpha"] * s, k)
            if a["beta"] != 0:
                v = rnd(v + rnd(a["beta"] * y[i], k), k)
            y[i] = v
        return {"y": y}

    def ger(self, c):
        a, k = c["args"], c["kind"]
        out = list(a["a"])
        for j in range(a["n"]):
            for i in range(a["m"]):
                q = i + j * a["lda"]
                out[q] = rnd(rnd(rnd(a["alpha"] * a["x"][i], k) * a["y"][j], k) + out[q], k)
        return {"a": out}

    def gemm(self, c):
        a, k = c["args"], c["kind"]
        out = list(a["c"])
        for j in range(a["n"]):
            for i in range(a["m"]):
                s = _acc(((_op(a["a"], a["lda"], a["transa"], i, p),
                           _op(a["b"], a["ldb"], a["transb"], p, j)) for p in range(a["k"])), k)
                v = rnd(a["alpha"] * s, k)
                q = i + j * a["ldc"]
                if a["beta"] != 0:
                    v = rnd(v + rnd(a["beta"] * out[q], k), k)
                out[q] = v
        return {"c": out}

    def syrk(self, c):
        a, k = c["args"], c["kind"]
        n, kk, lda, ldc = a["n"], a["k"], a["lda"], a["ldc"]
        out = list(a["c"])
        for j in range(n):
            rows = range(0, j + 1) if a["uplo"] == "U" else range(j, n)
            for i in rows:
                if a["trans"] == "N":
                    pairs = ((a["a"][i + p * lda], a["a"][j + p * lda]) for p in range(kk))
                else:
                    pairs = ((a["a"][p + i * lda], a["a"][p + j * lda]) for p in range(kk))
                s = _acc(pairs, k)
                q = i + j * ldc
                out[q] = rnd(rnd(a["alpha"] * s, k) + rnd(a["beta"] * out[q], k), k)
        return {"c": out}

    def _tri_solve(self, get, n, lower, unit, rhs, k):
        x = list(rhs)
        order = range(n) if lower else range(n - 1, -1, -1)
        for i in order:
            s = x[i]
            cols = range(0, i) if lower else range(i + 1, n)
            for j in cols:
                s = rnd(s - rnd(get(i, j) * x[j], k), k)
            x[i] = s if unit else rnd(s / get(i, i), k)
        return x

    def trsv(self, c):
        a, k = c["args"], c["kind"]
        n, lda, tr = a["n"], a["lda"], a["trans"]
        upper = a["uplo"] == "U"
        lower_op = (not upper) if tr == "N" else upper

        def get(i, j):
            return a["a"][i + j * lda] if tr == "N" else a["a"][j + i * lda]

        xs = R.gather(a["x"], n, a["incx"])
        sol = self._tri_solve(get, n, lower_op, a["diag"] == "U", xs, k)
        return {"x": R.scatter(list(a["x"]), n, a["incx"], sol)}

    def trsm(self, c):
        a, k = c["args"], c["kind"]
        m, n, lda, ldb, tr = a["m"], a["n"], a["lda"], a["ldb"], a["trans"]
        upper = a["uplo"] == "U"
        out = list(a["b"])
        unit = a["diag"] == "U"
        if a["side"] == "L":
            lower_op = (not upper) if tr == "N" else upper

            def get(i, j):
                return a["a"][i + j * lda] if tr == "N" else a["a"][j + i * lda]

            for j in range(n):
                col = [rnd(a["alpha"] * out[i + j * ldb], k) for i in range(m)]
                sol = self._tri_solve(get, m, lower_op, unit, col, k)
                for i in range(m):
                    out[i + j * ldb] = sol[i]
        else:
            # X * op(A) = alpha*B, solved as op(A)^T X^T = alpha B^T
            lower_opt = upper if tr == "N" else (not upper)

            def get(i, j):  # element (i, j) of op(A)^T
                return a["a"][j + i * lda] if tr == "N" else a["a"][i + j * lda]

            for i in range(m):
                row = [rnd(a["alpha"] * out[i + j * ldb], k) for j in range(n)]
                sol = self._tri_solve(get, n, lower_opt, unit, row, k)
                for j in range(n):
                    out[i + j * ldb] = sol[j]
        return {"b": out}

    def __call__(self, case):
        return getattr(self, case["fn"])(case)


# ---- mutants: each breaks one rule ----------------------------------------

def _mutant(base: type, name: str, fn: str, impl: Callable) -> tuple[str, Naive]:
    cls = type(f"Mutant_{name}", (base,), {fn: impl})
    return name, cls()


def _iamax_zero_based(self, c):
    r = Naive.iamax(self, c)["result"]
    return {"result": r - 1 if r > 0 else 0}


def _iamax_signed(self, c):
    a = c["args"]
    if a["n"] <= 0:
        return {"result": 0}
    xs = R.gather(a["x"], a["n"], a["incx"])
    return {"result": xs.index(max(xs)) + 1}


def _nrm2_naive(self, c):
    a, k = c["args"], c["kind"]
    if a["n"] <= 0:
        return {"result": 0.0}
    s = 0.0
    for v in R.gather(a["x"], a["n"], a["incx"]):
        s = rnd(s + rnd(v * v, k), k)
    return {"result": rnd(math.sqrt(s), k)}


def _axpy_ignores_negative_inc(self, c):
    c2 = dict(c)
    args = dict(c["args"])
    args["incx"], args["incy"] = abs(args["incx"]), abs(args["incy"])
    c2["args"] = args
    return Naive.axpy(self, c2)


def _gemv_ignores_trans(self, c):
    c2 = dict(c)
    args = dict(c["args"])
    if args["trans"] == "T" and args["m"] == args["n"]:
        args["trans"] = "N"
    elif args["trans"] == "T":
        return {"y": list(args["y"])}
    c2["args"] = args
    return Naive.gemv(self, c2)


def _gemm_reads_c_when_beta_zero(self, c):
    a, k = c["args"], c["kind"]
    out = list(a["c"])
    for j in range(a["n"]):
        for i in range(a["m"]):
            s = _acc(((_op(a["a"], a["lda"], a["transa"], i, p),
                       _op(a["b"], a["ldb"], a["transb"], p, j)) for p in range(a["k"])), k)
            q = i + j * a["ldc"]
            out[q] = rnd(rnd(a["alpha"] * s, k) + rnd(a["beta"] * out[q], k), k)
    return {"c": out}


def _gemm_ignores_ld(self, c):
    c2 = dict(c)
    args = dict(c["args"])
    args["lda"] = args["m"] if args["transa"] == "N" else args["k"]
    c2["args"] = args
    return Naive.gemm(self, c2)


def _syrk_both_triangles(self, c):
    out = Naive.syrk(self, c)["c"]
    a = c["args"]
    n, ldc = a["n"], a["ldc"]
    for j in range(n):
        for i in range(n):
            if (a["uplo"] == "U" and i > j) or (a["uplo"] == "L" and i < j):
                out[i + j * ldc] = out[j + i * ldc]
    return {"c": out}


def _trsv_ignores_unit_diag(self, c):
    c2 = dict(c)
    args = dict(c["args"])
    args["diag"] = "N"
    c2["args"] = args
    return Naive.trsv(self, c2)


def _scal_drops_last(self, c):
    c2 = dict(c)
    args = dict(c["args"])
    args["n"] = max(args["n"] - 1, 0)
    c2["args"] = args
    return Naive.scal(self, c2)


def _dot_low_precision(self, c):
    a, k = c["args"], c["kind"]
    if a["n"] <= 0:
        return {"result": 0.0}
    s = 0.0
    for x, y in zip(R.gather(a["x"], a["n"], a["incx"]), R.gather(a["y"], a["n"], a["incy"])):
        s = s + x * y
        if s != 0 and not math.isinf(s):
            e = math.frexp(s)[1]
            s = round(s / 2.0 ** (e - 12)) * 2.0 ** (e - 12)  # 12-bit mantissa
    return {"result": rnd(s, k)}


def _trsm_ignores_side(self, c):
    c2 = dict(c)
    args = dict(c["args"])
    if args["side"] == "R":
        args["side"] = "L"
        if args["m"] != args["n"]:
            return {"b": list(args["b"])}
    c2["args"] = args
    return Naive.trsm(self, c2)


def _asum_signed(self, c):
    a, k = c["args"], c["kind"]
    if a["n"] <= 0:
        return {"result": 0.0}
    s = 0.0
    for v in R.gather(a["x"], a["n"], a["incx"]):
        s = rnd(s + v, k)
    return {"result": s}


def _swap_one_way(self, c):
    a = c["args"]
    return {"x": list(a["x"]), "y": R.copy(a["n"], a["x"], a["incx"], a["y"], a["incy"])}


MUTANTS = [
    _mutant(Naive, "iamax_zero_based", "iamax", _iamax_zero_based),
    _mutant(Naive, "iamax_signed", "iamax", _iamax_signed),
    _mutant(Naive, "nrm2_overflows", "nrm2", _nrm2_naive),
    _mutant(Naive, "axpy_ignores_negative_inc", "axpy", _axpy_ignores_negative_inc),
    _mutant(Naive, "gemv_ignores_trans", "gemv", _gemv_ignores_trans),
    _mutant(Naive, "gemm_reads_c_when_beta_zero", "gemm", _gemm_reads_c_when_beta_zero),
    _mutant(Naive, "gemm_ignores_lda", "gemm", _gemm_ignores_ld),
    _mutant(Naive, "syrk_writes_both_triangles", "syrk", _syrk_both_triangles),
    _mutant(Naive, "trsv_ignores_unit_diag", "trsv", _trsv_ignores_unit_diag),
    _mutant(Naive, "scal_drops_last", "scal", _scal_drops_last),
    _mutant(Naive, "dot_low_precision", "dot", _dot_low_precision),
    _mutant(Naive, "trsm_ignores_side", "trsm", _trsm_ignores_side),
    _mutant(Naive, "asum_signed", "asum", _asum_signed),
    _mutant(Naive, "swap_one_way", "swap", _swap_one_way),
]

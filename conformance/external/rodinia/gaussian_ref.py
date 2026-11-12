"""Gaussian elimination reference in double precision, with partial pivoting.

Writes the Rodinia gaussian input (the matrix create_matrix() builds in the
program, 10*exp(-0.01*|i-j|), and a right-hand side of ones) to <input>, and
the solution of the float32 values the program reads to <reference>.
The program does elimination in float without pivoting, so the two are
different algorithms.
Usage: gaussian_ref.py <n> <input> <reference>"""
import math, struct, sys

n = int(sys.argv[1])
a = [[10 * math.exp(-0.01 * abs(i - j)) for j in range(n)] for i in range(n)]
b = [1.0] * n
with open(sys.argv[2], "w") as f:
    f.write(f"{n}\n\n")
    for row in a:
        f.write("\t".join("%.9g" % v for v in row) + "\n")
    f.write("\n" + "\t".join("%.9g" % v for v in b) + "\n")


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


# the values the program reads: decimal text parsed to float32
lines = open(sys.argv[2]).read().split("\n")
rows = [[f32(float(v)) for v in lines[2 + i].split("\t")] for i in range(n)]
rhs = [f32(float(v)) for v in lines[2 + n + 1].split("\t")]
M = [rows[i][:] + [rhs[i]] for i in range(n)]
for k in range(n):
    p = max(range(k, n), key=lambda r: abs(M[r][k]))
    M[k], M[p] = M[p], M[k]
    for r in range(k + 1, n):
        factor = M[r][k] / M[k][k]
        if factor:
            for c in range(k, n + 1):
                M[r][c] -= factor * M[k][c]
x = [0.0] * n
for i in range(n - 1, -1, -1):
    x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
with open(sys.argv[3], "w") as f:
    f.write("".join("%.9g\n" % v for v in x))

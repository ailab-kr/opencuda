#!/usr/bin/env python3
"""Small generated inputs for the Parboil programs (fixed seeds).

    gen.py <dir>

Writes: sgemm_{a,b,bt}.txt (A 128x64, B 64x64, column-major text, B^T file),
mriq.bin (K=256 samples, X=1024 voxels), stencil.bin (16x16x16 floats),
cutcp.pqr (300 atoms), histo.bin (image 1024x64 of 32-bit values, histogram 1024x48).
"""
import os, random, struct, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rnd = random.Random(20261113)


def colmajor(path, rows, cols, vals):
    with open(path, "w") as f:
        f.write(f"{rows} {cols} ")
        f.write(" ".join(repr(v) for v in vals))
        f.write("\n")


M, K, N = 128, 64, 64  # the kernel needs M a multiple of 128, N of 16, K of 8
A = [[round(rnd.uniform(-1, 1), 4) for _ in range(K)] for _ in range(M)]
B = [[round(rnd.uniform(-1, 1), 4) for _ in range(N)] for _ in range(K)]
colmajor(f"{out}/sgemm_a.txt", M, K, [A[i][j] for j in range(K) for i in range(M)])
colmajor(f"{out}/sgemm_b.txt", K, N, [B[i][j] for j in range(N) for i in range(K)])
# the program reads B^T as an N x K matrix in column-major order
colmajor(f"{out}/sgemm_bt.txt", N, K, [B[i][j] for i in range(K) for j in range(N)])


def fl(n, lo, hi):
    return [rnd.uniform(lo, hi) for _ in range(n)]


nk, nx = 256, 1024
with open(f"{out}/mriq.bin", "wb") as f:
    f.write(struct.pack("<ii", nk, nx))
    for arr in (fl(nk, -.5, .5), fl(nk, -.5, .5), fl(nk, -.5, .5),
                fl(nx, -.5, .5), fl(nx, -.5, .5), fl(nx, -.5, .5),
                fl(nk, -1, 1), fl(nk, -1, 1)):
        f.write(struct.pack(f"<{len(arr)}f", *arr))

with open(f"{out}/stencil.bin", "wb") as f:
    f.write(struct.pack("<4096f", *fl(4096, 0, 1)))

iw, ih, hw, hh = 1024, 64, 1024, 48  # the program sizes the histogram by the image width and works in 24576-bin ranges
with open(f"{out}/histo.bin", "wb") as f:
    f.write(struct.pack("<4I", iw, ih, hw, hh))
    # values cluster, so some bins collect many hits
    vals = [min(hw * hh - 1, int(abs(rnd.gauss(0, hw * hh / 6)))) for _ in range(iw * ih)]
    f.write(struct.pack(f"<{len(vals)}I", *vals))

with open(f"{out}/cutcp.pqr", "w") as f:
    for i in range(300):
        x, y, z = (rnd.uniform(0, 24) for _ in range(3))
        f.write(f"ATOM  {i + 1:5d}  C   RES {1:5d}    {x:8.3f}{y:8.3f}{z:8.3f} {rnd.uniform(-1, 1):6.3f} 1.500\n")

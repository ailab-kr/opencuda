#!/usr/bin/env python3
"""Small generated inputs for the Parboil programs (fixed seeds).

    gen.py <dir>

Writes: sgemm_{a,b,bt}.txt (A 128x64, B 64x64, column-major text, B^T file),
mriq.bin (K=256 samples, X=1024 voxels), stencil.bin (16x16x16 floats),
cutcp.pqr (300 atoms), histo.bin (image 1024x64 of 32-bit values, histogram 1024x48),
bfs_graph.txt (4,096 nodes), sad_{ref,cur}.bin (128x96 frames), spmv_vec.bin.
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

# bfs: a connected random graph in the suite's text format (node count;
# "first-edge edge-count" per node; source; edge count; "target cost" per
# edge). Every edge costs 1: the suite's CPU version (src/base) ignores the
# costs and computes BFS levels.
nb = 4096
adj = [[] for _ in range(nb)]
for v in range(1, nb):
    u = rnd.randrange(v)
    c = 1
    adj[u].append((v, c))
    adj[v].append((u, c))
for _ in range(nb * 3):
    u, v = rnd.randrange(nb), rnd.randrange(nb)
    if u != v:
        c = 1
        adj[u].append((v, c))
        adj[v].append((u, c))
with open(f"{out}/bfs_graph.txt", "w") as f:
    f.write(f"{nb}\n")
    start = 0
    for v in range(nb):
        f.write(f"{start} {len(adj[v])}\n")
        start += len(adj[v])
    f.write(f"\n0\n\n{start}\n")
    for v in range(nb):
        for w, c in adj[v]:
            f.write(f"{w} {c}\n")

# sad: a reference frame and a current frame (16-bit little-endian width,
# height, then 16-bit pixels), 128x96 = 8x6 macroblocks. The current frame is
# the reference moved by a few pixels plus noise, so the best matches are
# inside the search window and some search positions fall outside the frame.
sw, sh = 128, 96
ref = [[rnd.randrange(256) for _ in range(sw)] for _ in range(sh)]
cur = [[min(255, max(0, ref[min(sh - 1, max(0, y - 3))][min(sw - 1, max(0, x + 5))] +
                        rnd.randint(-4, 4))) for x in range(sw)] for y in range(sh)]
for name, img in (("sad_ref.bin", ref), ("sad_cur.bin", cur)):
    with open(f"{out}/{name}", "wb") as f:
        f.write(struct.pack("<HH", sw, sh))
        f.write(struct.pack(f"<{sw * sh}h", *[p for row in img for p in row]))

# spmv: the input vector (the program reads as many values as the matrix has
# rows; fidapm05.mtx has 42),
# raw little-endian floats.
with open(f"{out}/spmv_vec.bin", "wb") as f:
    f.write(struct.pack("<64f", *fl(64, -1, 1)))

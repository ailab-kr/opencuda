#!/usr/bin/env python3
"""cmp.py <program> <file a> <file b>: compare two Parboil outputs.

Prints "SAME ..." when equal within the tolerance, else "DIFF ...".
sgemm writes numbers as text; mri-q and stencil write a uint32 count then
float32 values; lbm writes raw float32 velocities; histo writes a bitmap,
compared byte for byte.
"""
import struct
import sys

prog, a, b = sys.argv[1:4]
RTOL, ATOL = 1e-3, 1e-4


def floats(path):
    data = open(path, "rb").read()
    if prog == "sgemm":
        return [float(x) for x in data.split()[2:]]
    if prog != "lbm":
        data = data[4:]
    return list(struct.unpack(f"<{len(data) // 4}f", data[:len(data) // 4 * 4]))


try:
    if prog == "histo":
        da, db = open(a, "rb").read(), open(b, "rb").read()
        d = sum(1 for x, y in zip(da, db) if x != y) + abs(len(da) - len(db))
        print(("SAME" if d == 0 else "DIFF") + f" {len(da)} bytes, {d} differ")
        sys.exit(0)
    va, vb = floats(a), floats(b)
except Exception as e:  # noqa: BLE001
    print(f"DIFF cannot read outputs: {e}")
    sys.exit(0)
if len(va) != len(vb):
    print(f"DIFF length {len(va)} vs {len(vb)}")
    sys.exit(0)
bad = 0
worst = 0.0
for x, y in zip(va, vb):
    lim = ATOL + RTOL * max(abs(x), abs(y))
    if not abs(x - y) <= lim:
        bad += 1
    worst = max(worst, abs(x - y) / lim)
print(("SAME" if bad == 0 else "DIFF") + f" {len(va)} values, {bad} outside tolerance, worst error/limit {worst:.2f}")

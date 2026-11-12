"""Independent check of Rodinia nn: distances in float32 and the program's
own selection loop (whose record swap repeats records), compared with the
program's printed lines.  Usage: nn_check.py <filelist> <k> <lat> <lng> <out>"""
import math
import struct
import sys


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


files, k, lat0, lng0, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
recs = []
for name in open(files).read().split():
    for line in open(name, errors="replace"):
        line = line.rstrip("\n")
        if len(line) >= 38:
            recs.append(line)
dist = []
for r in recs:
    lat, lng = f32(float(r[28:33])), f32(float(r[33:38]))
    dist.append(f32(math.sqrt(f32(f32(lat - lat0) ** 2) + f32(f32(lng - lng0) ** 2))))
for i in range(k):  # the program's loop: records[minLoc] receives the new records[i]
    m = i
    for j in range(i, len(dist)):
        if dist[j] < dist[m]:
            m = j
    recs[i] = recs[m]
    recs[m] = recs[i]
    dist[i], dist[m] = dist[m], dist[i]
want = [(recs[i].split()[5], "%f" % dist[i]) for i in range(k)]
got = []
for line in open(out):
    head, _, tail = line.partition("-->")
    got.append((head.split()[5], tail.strip().split("=")[-1]))
same = sum(1 for a, b in zip(want, got) if a == b)
print(("SAME" if same == k and len(got) == k else "DIFF") + f" ({same}/{k} nearest records, {len(recs)} records)")

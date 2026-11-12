"""Compare the numbers in two text files.
Usage: numcmp.py <a> <b> <relative tolerance>; prints SAME or DIFF."""
import re, sys
def nums(p):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", open(p, errors="replace").read())]
a, b, tol = nums(sys.argv[1]), nums(sys.argv[2]), float(sys.argv[3])
if len(a) != len(b):
    print(f"MISMATCH count {len(a)} vs {len(b)}"); sys.exit()
bad = [i for i, (x, y) in enumerate(zip(a, b)) if abs(x - y) > tol * max(1.0, abs(y))]
worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
print(("SAME" if not bad else f"DIFF {len(bad)}") + f" ({len(a)} values, max diff {worst:.3g}, tol {tol})")

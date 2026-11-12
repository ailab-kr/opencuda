"""Inputs for Rodinia b+tree: a tree file (record count, then the keys) and a
command file of batch searches ("k <count>"). Keys are distinct random
integers with a fixed seed.
Usage: bpt_gen.py <records> <tree file> <command file> <queries...>"""
import random, sys

n = int(sys.argv[1])
random.seed(5)
keys = random.sample(range(n), n // 2)  # queries draw keys from [0, n)
with open(sys.argv[2], "w") as f:
    f.write(f"{n}\n" + "".join(f"{k}\n" for k in keys))
with open(sys.argv[3], "w") as f:
    for q in sys.argv[4:]:
        f.write(f"k {int(q)}\n")

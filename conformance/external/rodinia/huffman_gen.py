"""Input for Rodinia huffman: raw bytes with a skewed distribution (so the
code lengths vary), fixed seed. The program takes its element count from the
file size (4 bytes per element), so the size is a multiple of 1024 bytes.
Usage: huffman_gen.py <bytes> <out file>"""
import random, sys

n = int(sys.argv[1])
assert n % 1024 == 0
random.seed(9)
data = bytes(min(255, int(random.expovariate(0.05))) for _ in range(n))
open(sys.argv[2], "wb").write(data)

"""Random connected graph in the Rodinia bfs input format.
Usage: graphgen.py <nodes> <out file>"""
import random, sys
n = int(sys.argv[1]); random.seed(11)
adj = [[] for _ in range(n)]
for v in range(1, n):
    u = random.randrange(v); adj[u].append(v); adj[v].append(u)
for _ in range(n * 2):
    u, v = random.randrange(n), random.randrange(n)
    if u != v: adj[u].append(v); adj[v].append(u)
out = [str(n)]; start = 0
for v in range(n):
    out.append(f"{start} {len(adj[v])}"); start += len(adj[v])
out += ["", "0", "", str(start)]
for v in range(n):
    for w in adj[v]: out.append(f"{w} 1")
open(sys.argv[2], "w").write("\n".join(out) + "\n")

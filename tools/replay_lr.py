#!/usr/bin/env python3
# Can features beyond q predict landing well enough to beat the floor/mass policy? Logistic model per candidate
# (features: log q, selector score s, rank j, log q0, top-16 entropy, landed share over the last 8 rounds), fitted on
# even rounds, evaluated on odd rounds with the expected-value rule: offer j iff p_j * GAIN > ROW.
import sys, math
sys.path.insert(0, sys.path[0])
from replay import load, BASE, ROW, TOKENS
GAIN = 14.0

def feats(q, s, j, hist):
    ent = -sum(x * math.log(x + 1e-12) for x in q)
    return [1.0, math.log(q[j] + 1e-6), s[j], j / 15.0, math.log(q[0] + 1e-6), ent, hist]

def fit(X, y, iters=300, lr=0.5, l2=1e-3):
    w = [0.0] * len(X[0]); n = len(X)
    for _ in range(iters):
        g = [0.0] * len(w)
        for x, t in zip(X, y):
            z = sum(a * b for a, b in zip(w, x)); p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            for i in range(len(w)): g[i] += (p - t) * x[i]
        w = [wi - lr * (gi / n + l2 * wi) for wi, gi in zip(w, g)]
    return w

def prob(w, x):
    z = sum(a * b for a, b in zip(w, x)); return 1 / (1 + math.exp(-max(-30, min(30, z))))

sets = {p.split('/')[-1][:-4]: load(p) for p in sys.argv[1:]}
X, y = [], []
for rs in sets.values():
    hist = []
    for i, (r, o, rows, m, q, s) in enumerate(rs):
        h = sum(hist[-8:]) / max(1, len(hist[-8:]))
        if i % 2 == 0:
            for j in range(16): X.append(feats(q, s, j, h)); y.append(1.0 if r == j else 0.0)
        hist.append(m)
w = fit(X, y)
print('weights', [round(v, 3) for v in w])
for th_scale in [0.7, 1.0, 1.4]:
    tot_d = []
    for name, rs in sets.items():
        hist = []; ms_c = tok_c = ms_p = tok_p = 0
        for i, (r, o, rows, m, q, s) in enumerate(rs):
            h = sum(hist[-8:]) / max(1, len(hist[-8:])); hist.append(m)
            if i % 2 == 0: continue
            ms_c += BASE + ROW * rows; tok_c += 1 + m
            chosen = {0} | {j for j in range(1, 16) if prob(w, feats(q, s, j, h)) * GAIN > ROW * th_scale}
            land = 1 if r in chosen else 0
            ms_p += BASE + ROW * (len(chosen) + 1); tok_p += 1 + land
        dc = TOKENS * ms_c / tok_c; dp = TOKENS * ms_p / tok_p
        tot_d.append(dp - dc)
        print(f"scale {th_scale} {name:10s} current {dc:8.1f}  model {dp:8.1f}  delta {dp-dc:+6.2f} ms/128")
    print(f"scale {th_scale} mean delta {sum(tot_d)/len(tot_d):+.2f} ms/128\n")

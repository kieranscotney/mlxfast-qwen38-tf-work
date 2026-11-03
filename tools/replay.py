#!/usr/bin/env python3
"""Offline replay of depth-one sibling policies on TR traces (trace-apply.py).

Each traced round: the target token's rank r in the selector's order of the drafter's top-16 (16 = absent), the
siblings offered o, rows verified, landed m, and per candidate (in selector order) the in-top-16 drafter probability q
and the selector log-score s. A policy picks k siblings (a prefix of the selector order); the round lands iff r < k.
Cost: round_ms = BASE + ROW * (k + 1). Time to 128 tokens = 128 * E[round_ms] / E[tokens], tokens = 1 + landed.
Usage: replay.py traces/*.txt
"""
import sys, math, itertools

ROW = 0.19      # ms a verified row costs (S14/S15 floor tests)
BASE = 27.85 - ROW * 2.95  # public: 27.85 ms at 2.95 rows/round
TOKENS = 128


def load(path):
    rounds = []
    for line in open(path):
        f = line.split()
        head = dict(x.split('=') for x in f[:4])
        qs = [tuple(map(float, t.split('/'))) for t in f[4:20]]
        rounds.append((int(head['r']), int(head['o']), int(head['rows']), int(head['m']), [q for q, _ in qs], [s for _, s in qs]))
    return rounds


def floor_mass(f, mass, children=7):
    def pick(q, s):
        chosen, tot = [], 0.0
        for j in range(16):
            if len(chosen) == children: break
            if chosen and q[j] < f: continue
            chosen.append(j)
            tot += q[j]
            if tot >= mass: break
        return set(chosen)
    return pick


def ev_rule(threshold, children=16):
    """Offer the selector prefix while each next candidate's probability exceeds `threshold` (calibrated landing)."""
    def pick(q, s):
        return {0} | {j for j in range(1, children) if q[j] >= threshold}
    return pick


def score(rounds, pick):
    ms = tok = land = 0
    for r, o, rows, m, q, s in rounds:
        k = pick(q, s)
        landed = 1 if r in k else 0
        ms += BASE + ROW * (len(k) + 1)
        tok += 1 + landed
        land += landed
    n = len(rounds)
    return TOKENS * ms / tok, land / n, ms / n


def main():
    sets = {p.split('/')[-1][:-4]: load(p) for p in sys.argv[1:]}
    sets = {k: v for k, v in sets.items() if v}
    # sanity: the traced policy reproduces its own landings
    for name, rs in sets.items():
        bad = sum(1 for r, o, rows, m, q, s in rs if (r < o) != bool(m))
        print(f"{name:10s} rounds {len(rs):5d}  landed {sum(x[3] for x in rs)/len(rs):.4f}  rows/round {sum(x[2] for x in rs)/len(rs):.2f}  mismatches {bad}")
    print()
    policies = {'traced (o)': lambda q, s, _o=None: None}
    base = {}
    print(f"{'policy':26s}" + ''.join(f"{n[:9]:>10s}" for n in sets) + "   mean dms/128")
    def row(label, pick, traced=False):
        out, deltas = [], []
        for name, rs in sets.items():
            if traced:
                ms = sum(BASE + ROW * x[2] for x in rs); tok = sum(1 + x[3] for x in rs)
                t = TOKENS * ms / tok
                base[name] = t
            else:
                t = score(rs, pick)[0]
            out.append(t); deltas.append(t - base[name])
        print(f"{label:26s}" + ''.join(f"{t:10.1f}" for t in out) + f"   {sum(deltas)/len(deltas):+7.2f}")
    row('traced (current)', None, traced=True)
    # oracle: exact
    out, deltas = [], []
    for name, rs in sets.items():
        ms = sum(BASE + ROW * ((r + 1 if r < 16 else 1) + 1) for r, o, rows, m, q, s in rs)
        tok = sum(1 + (1 if r < 16 else 0) for r, o, rows, m, q, s in rs)
        t = TOKENS * ms / tok; out.append(t); deltas.append(t - base[name])
    print(f"{'oracle exact':26s}" + ''.join(f"{t:10.1f}" for t in out) + f"   {sum(deltas)/len(deltas):+7.2f}")
    for f, mass in itertools.product([0.005, 0.01, 0.015, 0.02, 0.03, 0.05], [0.9, 0.95, 0.98, 1.01]):
        row(f'floor {f} mass {mass}', floor_mass(f, mass))
    for th in [0.005, 0.01, 0.014, 0.02, 0.03, 0.05, 0.08]:
        row(f'ev q>={th}', ev_rule(th))
    # calibration: landing rate of the candidate at rank j by its q (how good q is as a landing probability)
    print('\ncalibration (all sets): q bin -> landing share of that candidate')
    bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]
    cnt = [0] * (len(bins) - 1); hit = [0] * (len(bins) - 1)
    for rs in sets.values():
        for r, o, rows, m, q, s in rs:
            for j in range(16):
                b = next(i for i in range(len(bins) - 1) if q[j] < bins[i + 1])
                cnt[b] += 1; hit[b] += (r == j)
    for i in range(len(cnt)):
        if cnt[i]: print(f"  q in [{bins[i]:.3f},{bins[i+1]:.3f})  n={cnt[i]:6d}  land={hit[i]/cnt[i]:.4f}")


if __name__ == '__main__':
    main()

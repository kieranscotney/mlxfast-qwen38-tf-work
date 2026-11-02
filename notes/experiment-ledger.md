# Optimisation backlog (persistent)

Target: leader ed4ee746 OFFICIAL 1.74675 (lucky control leg; its tree ~1.732 real). Bar ~1.7599. Real gap ~+1.54%.
Score = prefill_gain^0.25 x decode_gain^0.75; decode s/tok = (seed 1065-token prompt + 128-step window) / 128.
Decode-only route: ~2.06% faster decode time = ~0.57-0.89 ms/round (window share ~70%). Round 28.85 ms LOCAL.

## Best valid composite

| name | branch | contents | LOCAL | exact | est. score |
|---|---|---|---|---|---|
| C1 | opus-c1 (on pr556 = ed4ee746) | floor 0.02 + 8-part sparse prefix | decode +0.17% (x3) | 3x1024 EXACT | ~+0.15-0.3% |
| **C2b (best)** | opus-c2 (on pr589 = terrapinelf c153ffc9, OFFICIAL 1.73791) | 8-part sparse prefix + floor 0.02 | decode +0.18% vs base (x3) | 3x1024 EXACT | ~1.740 |

## Ledger (hypothesis -> predicted -> result -> decision)

| # | hypothesis | predicted | result | decision |
|---|---|---|---|---|
| 1 | sibling floor 0.02 (calibrated) | +0.3% | held-out tokens unchanged, rows -0.7; decode +0.26% | KEEP (C1) |
| 2 | 8-part sparse decode prefix | +0.1% | decode +0.15% | KEEP (C1) |
| 3 | oracle lattice selection | <=+0.2% | 1.1% in-lattice misses; equal-budget oracle ~+0.18% | CLOSED |
| 4 | DARTree | depth>=2 | contract pins depth 1 | CLOSED |
| 5 | persistent drafter kernel | <=0.7 ms | barrier bubble ~1.9 us; <=0.15 ms | CLOSED |
| 6 | drafter K splits (2-bit) | 0.3 ms | gu 120->125 us | NEGATIVE |
| 7 | low-rank head screen | 0.24 ms | top-1 recall 0.32-0.41 | NEGATIVE |
| 8 | partial-K head screen | 0.07 ms | top-7 recall 0.94-0.96: no net | NEGATIVE |
| 9 | 8x32 tensor destination | 0.25 ms | M=8 slower than M=16 (0.82 vs 0.93 T w/s) | NEGATIVE |
| 10 | 2-row SIMD qmv (2-bit) | 0.2 ms | naive 527 us vs 120 us tensor | NEGATIVE |
| 11 | sibling attention sharing | 0.07 ms | per-row scores inherent; chunking is arithmetic | CLOSED (low value) |
| 12 | NGU / f53f7f95 counted waits | 0.3-0.6 ms | starvation / slower | NEGATIVE (spinning waiters) |
| 13 | drafter split caps 4 / 2 (longer chains) | 0.1 ms | frame 2.80 -> 2.83 / 2.99 ms | NEGATIVE (split rule right) |
| 14 | 2-bit unpack vs tensor-op decomposition | - | MMA-only = full (1.31 T w/s); unpack hidden | layout changes cannot help |
| 15 | kquant-style verify_qmv (2-bit, 2 rows) | 0.15 ms | ~600 us vs 120; their own M3 numbers ~0.6 T w/s | NEGATIVE |
| 16 | TF 1.2.0 unroll on our decode loops | small | +0.08% (noise) | NEGATIVE |
| 17 | rigorous two-stage head screen | 0.1 ms | bound ~67 sigma: keeps 100% tiles | NEGATIVE |
| 18 | drafter norm/RoPE/conv fusion | 0.05 ms | already fused upstream (normRope, convNormSums) | CLOSED |
| 19 | drafter attention split chunk 256/32/16 vs 64 | 0.1 ms | 256 -0.9%, 32 flat, 16 -0.4% | CLOSED (64 optimal) |
| 20 | 2048-row prompt chunks (TF #1478) | 3-4% prefill | already one 1065-row chunk (max 2048) | ALREADY PRESENT |
| 21 | multi-projection prompt launches (TF #1439) | 8-11% (GB10) | principle present; residual merges <0.5% of GEMMs | CLOSED |
| 22 | drafter/verify overlap | up to 2.4 ms | two-queue overlap hides 18%; block phase needs bonus token | append-only ~+0.1-0.3% |
| 23 | prompt M-tail padding | up to 7.6% | 7/1072 rows = 0.65%, 16-row tiles native | CLOSED (<=+0.3%, unrecoverable) |
| 24 | k|v prompt projection on TN2 | ~3 ms | ThreadgroupMemoryLimit at sk 8; smaller sk inexact | REJECTED |
| 25 | long reg dispatch (TF #1738) 256/512/2048 | ~10% prefill (Ultra) | -0.2..-0.4% prefill on M5 Max (matches #435 OFFICIAL -0.3%) | REJECTED |
| 26 | banding + long dispatch on C2b | - | -0.2% (band) / -0.3% (band+long) | REJECTED |
| 27 | k|v split-storage TN2 | ~2 ms | estimate ~+0.1% before extra traffic | NOT BUILT (too small) |
| 28 | lossless compression of 4-bit codes + metadata | ~8% bytes ideal | practical ~5%; decode ~40 ms of ALU a round vs 26.4 ms verify | REJECTED (no kernel) |
| 29 | GDN prompt scan headroom (C2b, CC=4) | <=+1.0% (scan halved) | 37.2 ms/48 layers = 728 ns/row; 384 SIMD groups/layer (~10/core); ~300 instr/lane/row of which 256 are irreducible FP (mul, 3 fma per state word) -> issue floor ~20 ms if fully issue-bound; next-row input prefetch (exact) -0.55% prefill, -0.3% decode (registers); CC=8 spills; rows serial | CLOSED (<=+0.2% realistic) |
| 30 | 2-bit verify-head sieve (kshitij-hash 63a5493e, OFFICIAL 1.75276; same stack w/o sieve 913de13 1.73967) | +0.75% | statistical (kappa=10 spreads): token can differ from full head; rigorous bounds lose sqrt(K), ~70 sigma (#17) | NOT PORTED (relaxes exactness) |
| 31 | drafter per-linear precision recheck on C2b (fc q4 / o q4 / down 2-bit) | ~+0.3% | public 2 reps: fc4 -0.1/-0.4%, o4 -0.35%, d2 +0.1/+0.6% decode (noise band); indep prompts don't predict official acceptance (journal 3-bit test: 2-bit won locally, lost OFFICIAL); Arthur 48d75d22: down 2-bit rounds ~= bytes | CLOSED (current mapping kept) |
| 32 | prompt scan 32x1 threadgroups (ercumentyildirim 8438335, exact, OFFICIAL 1.74667) | ~+0.1-0.3% prefill | on C2b 3 pairs: prefill -0.2% (1256.7 vs 1259.1), decode +0.05% | REJECTED (net slightly negative) |
| 33 | da328474 discrepancy (official legs from validation PRs) | - | ran on laptop-1 (degrading: ctrl pre 998.7->983.2, cand pre 1259->1243); 2-1 stable; official = 1 pair, formula reproduces scores | RUNNER BOX. Isolation 5 rounds on 292d7fda: prefix +0.12%, floor +0.10%, both +0.20% decode (sd ~0.08), prefill -0.03% -> ~+0.10% score |
| 34 | GDN state write every other round (CUDA 75b941ee) | ~0.9% round on M5 | MLX already defers write-back to every 4th round (gdn_pool.defer_rounds = 4); 8 would save ~0.03 ms | PRESENT |
| 35 | drafter 2-bit Lloyd LS refit (CUDA adeef663 idea) + down at 2 bits, on a5b756b6 | ~0.5% window | public 5 passes: z1 refit -0.08%, z2 refit+down2 +0.30% (prefill flat); indep code/article/story z2 +0.19/+0.34/+0.33%; zf (z2+prefix+floor) +0.18% public, code -0.13% | z2 POSITIVE, not enough alone (~1.758 at 2-1 median control) |
| 36 | drafter down on 2 bits WITHOUT refit (zd) + 8-part prefix (zdp) on a5b756b6 | ~0.8% window | public: zd +0.61%, zdp +0.85%; code prompt zd +0.23%, zdp +0.35%; refit variant z2 only +0.21%; q/k/v 2-bit -0.15% | FILED 254ee549 |

## Drafter frame (S18, serialized 2757 us; real 2.42 ms)

gu 2-bit x5 599 | down q4 x5 449 | head 341 (screen 274, top-k 22, tiles 8, rescore 44) | attention x10 295 |
q-group 4096x5120 x5 194 (38.8 each vs ~18 at bandwidth) | norms x27 175 | fc 97 | k/v groups ~205 | rope/conv/slices ~157

## Next, ranked (estimated full-score gain)

1. (CLOSED #18) Drafter norm+RoPE and conv+norm+sums are already fused (normRope via held norms, convNormSums).
2. (CLOSED #19) Drafter attention chunking already optimal at 64.
3. SIMD-FMA 2-row path only if a kernel beats 1.48 T w/s in a microbench first (none found; deprioritised).
4. NEW BASE CANDIDATE: terrapinelf c153ffc9 (1.73791, prompt scan with running row offsets + reduce-scatter sums + two
   SIMD groups, on our 4-column scan). ercumentyildirim 94329066 (1.73651) and jungjipdo eebac6f2 (1.73620) found the
   sibling floor independently (0.015). Next: C2 = c153ffc9 (or its best descendant) + floor 0.02 + 8-part prefix;
   paired A/B, tapes; file only if it credibly reaches ~1.7599.

Distance: C1 ~1.735-1.737 (leader tree ~1.732 + ~0.2%); bar 1.7599 -> ~1.3% short. Ranked items sum to <~0.3%.
Checkpoint: branch opus-c1 (exact), binary .build/experiments/c1, tapes c1.*.json; diag patches in this folder.

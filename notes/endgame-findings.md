# Endgame findings, 2–3 Nov 2026: where the last 1% is (and isn't)

The last two days on `GumbiiDigital/mlxfast-qwen38-27b-tf`, from the pr867 (`cb4d3130`, 1.78748) and pr921 (`22b390be`,
1.79565, promoted) trees on an Apple M5 Max. The aim was one more promotion. Nothing was filed, because no exact build
could credibly clear the live bar: 1.79565 × 1.0075 = **1.80912**. This note covers what we measured along the way, most
of it negative. We publish it so nobody has to measure it again.

Labels: **LOCAL** = measured on our M5 Max with counterbalanced A/B (rotated order, warm-up run, per-round timers);
**OFFICIAL** = the ranked runner's score or legs; **CALC** = arithmetic on measured inputs.

## 1. Where a decode round's time goes at the frontier

Per-round GPU and host timestamps (`tools/tl-apply.py`, `tools/tlh-apply.py`), opus-b (pr867 + mass stop), public prompt,
median round. LOCAL:

| Piece | Time |
|---|---|
| Verify (10 command buffers, no gaps between them) | 25.59 ms |
| Drafter frame (starts 1 µs after the verify ends) | 2.36 ms |
| GPU idle before the next verify | 0.094 ms |
| **Round** | **27.81 ms** (freeform 28.10) |

The GPU is busy **~99.7%** of the round. The 94 µs idle gap breaks down as follows: the host notices the frame end 15 µs
late, the lattice and selector take 10, encoding layer 0 takes 15, settling the keep takes 20, and commit-to-GPU-start
takes 34. Moving the settle onto the keep's own shared event (find 12) saves 7 µs a round. The rest is signal and
submission latency. terrapinelf's `22b390be` removed most of this gap by building the next verify's first command buffer
and attention plan off the host path.

**Consequence.** At ~64 rounds per 128-token window, 0.23% of score is about 8.3 ms per official sample, or about
128 µs a round. Host overhead is already below that. What remains is GPU kernel time.

## 2. The M5 is saturated during the verify: overlap doesn't hide work

We built two overlaps and measured both cleanly:

- **Append overlap** ([patch](../patches/negative/x2-append-overlap-NEUTRAL.patch)). The verify signals an event after
  target layer 61. A staging drafter frame on a second queue computes every verify row's context K/V beside layers 62–63
  and the head. The next early frame waits for it and gathers the admitted rows. It works (exact; drafts are taken from
  the staged rows every round). Result: neutral (public +0.03%, freeform 0.0%). The append's ~0.15 ms is hidden, but
  contention on the verify's tail gives it all back.
- **Staged prompt catch-up** (find 13 below). 8 ms of drafter catch-up runs on the staging queue beside round 1's verify,
  and the round gets only 1.2 ms shorter.

Even the "memory-bound" verify kernels occupy every GPU core on this chip, so extra compute queued beside them competes
rather than fills gaps. We expect any prefetch, overlap or second-queue idea on this track to behave the same way.

## 3. The start of decoding: a cost local runs dilute 8×

- The drafter's catch-up on the 1065 prompt taps (`Model.flush`) costs **9.4 ms per request** (~41 TFLOPS, ~62% of
  tensor peak). Round 1 takes 38.6 ms against 27.8 ms steady. LOCAL (`tools/early-apply.py`).
- The official decode leg is (seed pass + one 128-token window)/128, so this cost is ~0.4% of the official decode leg.
  Local `score.json` measures a 1024-token window with no seed, so the same cost is diluted about 8×. It never shows in a
  local A/B.
- **Cold drafter context** (absorb only the last N prompt rows): N = 512/256/128 brings the flush to 4.6/2.4/1.4 ms. But
  copy-like prompts lose tokens for the whole run: 497/496/498 of 512 tokens landed, against 505. Rejected.
- **Cold for round 1 only, full context from round 2** (find 13): round 1 loses no tokens (63/64 early landings
  unchanged), but saves only 1.2 ms (see §2).
- **One tall launch per projection** for the catch-up ([patch](../patches/negative/x3-tall-absorb-SLOWER.patch)) is
  2.6 ms slower. Each 16-row block re-streams the projection's weights, whereas 128-row chunks keep both x and the weights
  in cache. A real speed-up needs the target's tiled prompt GEMM on drafter weights converted at load. CALC ceiling
  ~6.5 ms against 9.4 ms (+~0.08%). Not built.
- **Dependency check:** a prompt row's drafter K/V depends only on that row's taps from layers 5/19/33/47/61, with no
  cross-row dependence. A row can be absorbed as soon as it passes layer 61, so catch-up during the prompt pass is legal.
  We expect it to hit the same contention wall as §2.

## 4. Round count: the sibling policy is at its optimum

At depth 1 a round yields 1 or 2 tokens, so 128 tokens need at least 64 rounds. Measured rounds per 128 tokens
(`tools/trace-apply.py`, LOCAL): public 64.4, math 64.4, translate 64.1, freeform 65.8, poem 65.9.

- An offline replay (`tools/replay.py`) reproduces the traced landings to within 0–2 mismatches per ~520 rounds.
- No floor, mass-stop or threshold variant beats the current policy (floor 0.02, mass 0.95) by more than ±0.03 ms per 128
  tokens across 5 prompts.
- A logistic land-probability model (`tools/replay_lr.py`) uses q, sibling score, rank, q0, entropy and recent landing,
  trained on even rounds and tested on odd. It does worse, by +1.1 to +1.9 ms per 128 tokens.
- A perfect oracle saves 22.5 ms per 128 tokens. That gap is information the pinned drafter does not have at draft time.
- About half the misses have the target token in the drafter's top 16 but below the floor. Offering those costs more rows
  (~0.19 ms each, on every round) than the tokens it recovers. The rest are outside the top 16.

## 5. Verify kernels

- **K-split is not exact.** Changing `m5Split`/`sk` for the z/b/a projections changes each output's summation order. It
  fails the timed run with "free-run decode token mismatch". Treat split factors as fixed. Tile batch (`TF_TB`) was already
  swept with no gain.
- **Per-row verify cost** is +0.03 ms per row in dense layers. VPROF compared ~2 rows against ~8 rows back to back, and no
  shape stood out. An earlier figure of 0.17 ms per row came from runs under machine drift. The hypothesis that x is
  reloaded per tile is falsified, so the two-tile kernel was not built (`microbench/rowsx.swift`).

## 6. Drafter kernels (from 1–2 Nov, consolidated)

- The 2-bit reg kernels are bound by tensor-op issue rate, at ~1.48 T weights/s; the q4 down projection is
  bandwidth-bound at ~0.99 T weights/s. Measured with cache-resident weights (`microbench/dec2.swift`):
  - unpack plus matrix multiply-accumulate: 548 µs
  - multiply-accumulate alone: 546 µs
  - unpack alone: 266 µs

  The unpack is fully hidden.
- 8-row tensor destinations are slower than 16-row ones (0.82 vs 0.93 T w/s, `microbench/mm8.swift`), because the M5
  tensor op runs 16-row tiles natively.
- SIMD GEMV routes for 2 rows (naive, and mlx-kquant's geometry: `microbench/qmv2.swift`, `qmv3.swift`) are 4–5× slower
  than the tensor path.
- Drafter K-split caps: 2804 µs → 2834 µs (cap 4) → 2987 µs (cap 2). The current split rule is right.
- Drafter attention `split_chunk` 64 is optimal: 32, 16 and 256 are all slower.
- Narrowing the draft vocab span loses: public −0.65%, Chinese −17% (out-of-span tokens).
- The head screen is at its trade-off point. A low-rank head misses top-1 60–70% of the time. A 3/4-K screen saves ~68 µs
  but loses sibling recall (`tools/screen_study*.py`). A rigorous two-stage 1-bit bound keeps 100% of tiles
  (`tools/bound_study.py`).

## 7. Prompt pass

- **Stagger segments:** locally, 5 segments gain +1.25% prefill (1242.7 → 1261.5 rows/s)
  ([patch](../patches/negative/x4-five-stagger-segments-LOCAL-ONLY.patch)). Officially, 3 → 4 segments reads −0.1%
  (terrapinelf `faab2e87` vs `5819eca6` on box 2-1), and ~−1% on laptop-1. Going from 2 to 3 segments did transfer
  (+0.21% official vs +0.27% local). Our hypothesis: extra queues win GPU share from the local window compositor, and the
  quiet runner has no compositor to beat. Local prefill also runs ~2.5% below the official runner's.
- Wave quantisation (e-man07's CUDA tail split) does not apply here: prompt launches are ~37k small threadgroups on 40
  cores, so the tail wave is far below 0.1%.

## 8. Platform facts (M5 Max, macOS 27)

- GPU counters give stage-boundary timestamps only: no per-dispatch sampling and no bandwidth counter sets.
- Metal 4 command submission is no faster: idle-GPU round trip is 123 µs (classic) vs 120 µs (MTL4) at the median
  (`microbench/submit.swift`).
- A dependent compute stage costs about 2.2 µs. Every fusion we tried that removed stages lost more in kernel efficiency.

## 9. Measurement

- A browser running (Chrome helpers sharing the GPU) adds about ±1.7% run-to-run noise and up to 2.5% drift. With the
  machine quiet, the counterbalanced per-round harness (`tools/abrt2.sh` + `tools/rt-apply.py`) has a noise floor of
  about ±0.05%.
- The live minimum improvement is 75 bps (`yukon benchmark show`), not the 200 in the repo's `benchmark.json`.
- **Official variance is larger than most remaining gains.** One public tree read 1.7917, 1.7875, 1.7863, 1.7816 and
  1.7808 officially (kaankolcu). Promotion depends heavily on the control draw. `22b390be`'s candidate legs are about
  +0.3% over pr867. Locally it is +0.50% (public) and +0.44% (freeform), close to its official window gain of +0.41%. That
  puts its real level on box 2-1 at about 1.7925.

## 10. Small exact finds (patches 11–13)

| # | Find | Effect | On the record tree |
|---|---|---|---|
| 11 | Siblings stop once the offered drafter mass reaches 0.95 | rows 2198 → 2165 per 1024 tokens; +0.05% on pr867 | null on pr921 |
| 12 | Settle waits on the keep's own shared event; failure check deferred one settle | −7 µs a round | not measured |
| 13 | Two-state first round: prompt taps except the last 128 absorbed on the staging queue; round 1 drafts cold | round 1 −1.2 ms, no tokens lost | not measured |

Patch 13 includes the staging-queue infrastructure from the append-overlap experiment.

We also tested kaankolcu's `99b1b0c7` drafter-frame merges on top of `22b390be` (branch `opus-n`). Result: null.

## Score budget at the end

From pr867 (1.78748 OFFICIAL), CALC:

| Step | Effect | Running level |
|---|---|---|
| Mass stop | +0.05% | |
| Two-state round 1 | +0.03% | |
| Keep event | +0.01% | ~1.7894 |
| `22b390be`'s tree | | ~1.7925 real |

Against a bar of 1.80912, that leaves about 0.9% to find. Every remaining idea above is worth ≤ 0.1–0.3% even in theory,
and the field's own variance is ±0.5%. We stopped there rather than file entries we could not justify.

## Reproducing

The tools are patch-appliers for diagnostics (Python scripts that insert timers into `generation.zig` and friends) plus a
counterbalanced A/B runner. They assume a checkout of the challenge repo and the binaries under
`.build/experiments/<name>`. Diagnostic patches under `patches/diagnostic/` are not meant to ship. Patches under
`patches/negative/` are kept so their results can be checked.

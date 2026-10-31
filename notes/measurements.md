# Measurements and dead ends

All on one Apple M5 Max (128 GB). "pf" = prefill tok/s, "dec" = decode tok/s, both local, from alternating A/B runs.
"Exact" means the target's tokens are identical to the base (checked on the challenge gate and on 3×1024-token free-run tapes).
Bases are Yukon submission IDs on `GumbiiDigital/mlxfast-qwen38-27b-tf`.

## What the score actually rewards

- Score = prefill_gain^0.25 × decode_gain^0.75. But the official **decode time includes the seed prompt pass**
  (~28% of it: 24.3 ms/token = 7.4 seed + 17.0 window), so prefill effectively weighs about 0.46 and the decode window about 0.54.
- Sensitivity on base `2bcc4866`: 0.1 ms per decode round = +0.16% score; 0.47 ms per round = +0.75% (the promotion bar);
  1 ms off the prompt pass = +0.05%; 15 ms off it = +0.75%.
- The live promotion bar is **75 bps** (server), though the repo's `benchmark.json` says 200. Promotions at +1.02% and +1.33% confirm 75.
- Box-to-box variance on the ranked runner is about 1–2% of score, and local A/B noise is about ±0.3%. A real gain under ~1.5% is close to a lottery.
  ArthurOstapenko's star kernel scored +0.45% officially; the same idea measured +0.05% / −0.12% here.
- The pair-1 vs pair-2 decode gap in official runs (~100 ms) is **not** a first-use cost: the pairs use different prompts,
  and DFlash speed depends on per-prompt acceptance. Compiling the drafter head at load did not remove it.
- Seed audit: local seed = reset 0.95 + prompt 925.8 + token 0.13 ms, matching prefill. The +14 ms official seed-vs-prefill gap is outside the engine.

## Hardware ceilings (Swift microbenchmarks)

| Measure | Value |
|---|---|
| Peak GPU read bandwidth | 579 GB/s |
| Decode GEMV, big shapes | ~550 GB/s (95%) |
| Tensor-op peak (16×32×64, 32×32×64, 16×64×64) | 66 TFLOPS |
| Prompt gate/up | ~59 TFLOPS (90%); other prompt shapes 50–58 |
| GPU system-level cache | holds ≥ ~50 MB (re-running a launch: out-proj −36%, zba −26%, down −22%) |

## Where a decode round goes (base `2bcc4866`)

- Round 32.0 ms = verify 28.16 ms (9 command buffers, GPU never idle) + 3.7 ms drafter and gaps.
- Verify bandwidth floor ~23.6 ms (13.6 GB at 579 GB/s). The 4.6 ms over it is ~2.4 ms of latency-bound kernels plus ~2.3 ms of dense below peak.
- Each norm+barrier stage costs ~3.2 µs; 129 per verify = ~0.42 ms (1.3% decode), the largest measured removable overhead (but see norm fusion below).
- Host-side work on the critical path is < 0.3 ms per round; selector 0.02 ms, host prep 0.008 ms.
- Drafter frame (153 ops, 3.64 ms busy): gate/up group 1338 µs, other projections 1117 (fusion 5120×25600 at ~270 GB/s), head 475,
  attention 239, norms 156, top-k 97, conv 75, rope 65, append 43. GPU-bound, but not dispatch-bound and not byte-bound.
- Verify by width: 2 rows 27.33 ms, +~0.22 ms per extra sibling row.
- Acceptance on independent 1065-token prompts: 1.94–2.00 tokens per round, against a ceiling of 2.0 at depth 1. Little room left in selection.
- The decode ROWS8 loop is bound by dependent-load latency, not bytes: e-man07 prefetching the (scale, bias) index one group ahead gave +4.7% decode,
  while cutting 25% of index loads made it slower.

## Prompt pass

- Prompt GEMM is ~93% tensor-op bound: skipping unpack gives +5% pf, skipping unpack and epilogue +7%.
- Prompt pass: 9 command buffers back to back, GPU busy 909.9 of 911.6 ms. No idle time to fill.
- Grid order: tile-fastest (the leader's) is best. Rows-fastest −6.2% pf, bands of 256 tiles −6.5%, bands of 64 −18%.

## Dead ends (exact unless noted)

| Idea | Base | Result |
|---|---|---|
| Row-block unpack reuse in `tf_lane_reg` (TF_RB=2) | 7f3c0d09 | pf 1020 → 873 |
| Two-groups-ahead prefetch in `tf_lane_reg` | ff408317 | pf 858, dec 51.6 (register pressure) |
| Leader-lane `simd_shuffle` for scale/bias and XS loads | 9b7f90ee | dec 47.8 / 49.0 vs 51.5 |
| SLC "touch" prefetch of next out-proj weights during mixer phases | ff408317 | null; ceiling ~0.5% |
| Selector `edge` coefficient tuning | 9b7f90ee | edge changes the draft in 1.7% of rounds; upper bound +0.75% tokens/round |
| No codes in flight on prompts (LEAN0) / two groups (LEAN2) | 2d06a685 | −5.4% pf / 0 |
| Prompt serial split-K ×2 / ×4 | ea8f8271, e75967a0 | 0 / failed correctness |
| Deeper index or table prefetch | 7edba466 | −0.74% to −1.0%; one group ahead is optimal |
| Drafter adapters on the prefetched index (SBI) | 7edba466 | null |
| Drafter adapter dictionary; drafter rows8 | 2c41feb1 | null |
| GDN state fold into verify (as on the CUDA track) | 7edba466 | 151 MB state write costs −1.2% decode |
| Fused star attention (siblings through the prefix kernel) | 7edba466 | +0.13% but **not exact**; dropped |
| Deferred keep replay beside the next GDN scan | 2bcc4866 | −1.5% dec (competes with the latency-bound scan) |
| GDN prepare fused into the pair scan | 2bcc4866 | −3.0% (per simdgroup) / −1.2% (threadgroup-shared) |
| Norm fused into the gate/up prologue (proxy) | 2bcc4866 | −3.6% dec (~18 µs per call vs 3.2 µs per stage) |
| Prompt row-pair kernel (2 row blocks per simdgroup) | dd191b5d | −0.8% pf |
| Prompt scan next-row prefetch | dd191b5d | −0.55% pf |
| Drafter fusion split-K 16; drafter attention lane-per-key | 2bcc4866, 09862c94 | null |
| 8-bit hot-index format | 09862c94 | 19–25% larger than 12-bit |
| GDN prep fused in QKV/ZBA epilogues (TensorFold #360) | 07fd50da | dec 64.07 vs 64.25 on M5 |
| Two-tile fused SwiGLU prompt kernel (my attempt) | 07fd50da | pf 1170 vs 1182. ercumentyildirim later built a working version (+98 bps) |
| `#pragma unroll`, verify commit interval 8, max-threads attribute, TB 4/16 | various | null |
| Apple int4/fp8 quantized tensors (WWDC26 #330) | n/a | checkpoint's bf16 affine group-64 scales aren't representable without changing the math |

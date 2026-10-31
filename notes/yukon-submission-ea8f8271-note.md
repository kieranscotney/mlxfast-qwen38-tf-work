Model: Claude Opus 5.5
Harness: Claude Code

# Lean prefetch on prompt projections too, on terrapinelf's 2d06a685 with the drafter head compiled at load

Effort: Claude Opus 5.5, max effort. One Claude Code session.

## Summary

This entry is our `f49e2ed7-95cd-411b-8439-60874c4c1446` (still validating when this was built) plus a one-line change in `families/qwen27/quant_gpu.zig`: prompt-row reg projections now select kshitij-hash's lean prefetch variant of `tf_lane_reg` (`TF_REG_LEAN`), which until now was enabled only for decoded rows.

`f49e2ed7` is terrapinelf's `2d06a685` (LEAN2 on ROWS8 and lean prefetch, on the promoted `ff408317`), copied from its public submission branch, plus our pre-compile of the drafter head's lane pipeline at engine open (first submitted as our `1afa673b`).

## Why the prompt matters twice

In the official metrics of `ff408317`, `candidate_decode_seconds_per_token` is about 28.1-28.9 ms against a decode window of 20.0-20.8 ms per token over 128 tokens. The difference, about 1.04 s a pair, is the seed prompt pass of the ~1065-token prompt, which is inside the scored decode time. A faster prompt pass therefore lifts the prefill factor (weight 0.25) and about 28% of the decode factor (weight 0.75).

On the M5 Max the prompt pass is 93% dense 4-bit projections through `tf_lane_reg` (profiled on a near-identical stack: gate/up 34816x5120 453 ms, down 237 ms of ~1083 ms GPU per 1065-row prompt).

## The change

`laneEntry` builds the projection `Key`. Before: `.lean = reg and !k.prompt`. Now: `.lean = reg`. Everything else in the key is unchanged:

- `.rows8` stays `reg and !k.prompt and rows <= 8`, so prompt launches keep every lane's second row.
- `.reg_lean2` stays `lean and rows8`, so prompts do not take LEAN2.

With `TF_REG_LEAN` and without `TF_ROWS8`, a SIMD group keeps only the next 64-wide K group's codes (two `uint4`) in flight across the loop. That group's two metadata quads and both row sums are loaded at the top of its own iteration. The original prompt kernel held a whole prefetched `tf_reg_load` (codes, metadata quads and two row sums) live beside the current one. The lean variant holds ten fewer live registers per lane, so more SIMD groups fit per core.

## Why outputs cannot change

The lean variant reads the same words from the same addresses and feeds the same values to the same operations, in the same order. It unpacks the same codes into the same fp16 right-input registers, runs the same `matmul2d_descriptor(16, 32, 64)` `op.run` on the same A slice, and folds the same `fma(scale, P, fma(bias, sum, C))` chain in group order with the same split-K reduction. Only when the metadata and sums are loaded changes. kshitij-hash's note for `b3757a3b` makes the same argument for decoded rows, and that path has passed the official correctness gate in several entries since. DFlash depth 1, the pinned drafter and `mtp-head.manifest.json` are unchanged.

## Measurements (local, Apple M5 Max 128 GB Mac Studio; not official)

Round-robin, three rounds, each binary built from its own tree, standard `yukon run` (public long-copy prompt, local mode). Prefill and decode-window tokens/s; every run passed the local correctness gate:

| round | `f49e2ed7` (base) | + lean prompts (this entry) | + LEAN2 prompts (not submitted) |
|---|---|---|---|
| 1 | 1026.3 / 56.19 | 1058.4 / 56.28 | 1034.3 / 56.30 |
| 2 | 1024.9 / 56.33 | 1047.9 / 56.28 | 1029.7 / 56.29 |
| 3 | 1034.3 / 56.17 | 1056.2 / 56.25 | 1028.4 / 56.27 |
| mean | 1028.5 / 56.23 | 1054.2 / 56.27 | 1030.8 / 56.29 |

Lean prompts gave +2.5% prefill in every round. Decode window was unchanged, as expected: the local run's decode window does not include the seed prompt pass. LEAN2 on prompts gave nothing measurable. The exact submitted tree was rebuilt with the track's sandboxed `setup.sh` and passed one more local run: 1044.0 prefill / 56.06 decode tok/s, correctness and timing passed.

Expected official effect over `f49e2ed7`, if the prefill gain holds: about +0.6% from the prefill factor and about +0.5% from the seed's share of scored decode time, roughly +1.1% in total.

Earlier in this session, `f49e2ed7`'s tree measured +2.07% (about +1.2% excluding one slow baseline run) over the promoted `ff408317` locally.

## Reproduce

```bash
git fetch origin refs/heads/submissions/2d06a685-ce13-460d-ac64-e203be611c27
git checkout FETCH_HEAD -- Vendor/TensorFold mtp-head.manifest.json   # on main f9c564b
# apply this archive's quant_gpu.zig and dflash/runtime_backend.zig
./tools/fetch-benchd.sh && ./setup.sh
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
```

## Limitations

- Local prefill varies by about 1% run to run; the gain here is about 2.5% and was present in every round.
- The official score also depends on the box's live control leg. Since `ff408317`, entries with real local gains have scored below it on other boxes.
- Neither `f49e2ed7` nor this entry had an official score when this was submitted.

## Provenance

- Base: our `f49e2ed7` = terrapinelf's `2d06a685` (via `e2100b71`, kshitij-hash's `b3757a3b` and the promoted `ff408317` with its sources) + our drafter-head pre-compile (`1afa673b`).
- The lean prefetch kernel path is kshitij-hash's (`b3757a3b`), unchanged. Selecting it for prompt rows, and measuring that, is ours.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside the permitted inputs.


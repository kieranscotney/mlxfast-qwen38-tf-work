Model: Claude Opus 5.5
Harness: Claude Code

# Decoded rows read (scale, bias) through a per-matrix u16 index and table, on terrapinelf's 2c41feb1, with ercumentyildirim's event waits

Effort: Claude Opus 5.5, max effort. One Claude Code session.

## Summary

This entry is terrapinelf's promoted `2c41feb1-52ab-40b1-9bca-71668d74e8ab` (main `9340ae1`, official 1.53945) with two additions.

1. **Dictionary-coded decode metadata (ours, on kshitij-hash's pair index from `e59458da`).** Every 4-bit g64 target projection keeps its `[group][column]` u32 (scale, bias) words for prompt rows. At load it also gets a u16 index into a table of that matrix's distinct words, built in first-seen order by kshitij-hash's `indexPairs` (unchanged). Decoded-row launches (ROWS8, the verify's up-to-8 rows) read two `ushort4` index quads and look the words up in the table, instead of reading two `uint4` quads. That halves the metadata bytes of every decode projection, about 6% of a verify's weight bytes. On this checkpoint all 321 target matrices fit, with 2,227-4,289 distinct words each; the limit is 16,384.
2. **ercumentyildirim's GPU-signalled event waits (`7d41e661`), ported unchanged.** Polled verifies and drafter frames return on a GPU-signalled event. It scored 1.54200 on this same base (+0.17%).

`e59458da` (kshitij-hash, 30 Oct) tried a table on a much older kernel, where metadata and row sums were prefetched one group ahead, and measured no gain. In the current lean loop, a decoded row's metadata loads at the top of its own group's iteration and is not needed until after the unpack and the tensor op. That leaves slack for the extra dependent lookup.

## Files

- `kernels/metal/core/lane_projection.metal`: under `TF_DICT` (decoded rows only), the indexed loop reads `ushort4` index quads from the pairs' slot and the table from `buffer(7)`. The prompt pointer loop (`TF_PROMPT_SR`) is untouched.
- `src/core/lane_projection.zig`: `Layout.dict` (needs `reg`, `rows8`, `reg_lean`), `#define TF_DICT`, `Projection.encodeDict`. The table is bound at index 7 in `encodeImpl` and `encodeSwiglu`, so the decode SwiGLU epilogue uses it too.
- `src/families/qwen27/pair_index.zig`: kshitij-hash's file, unchanged.
- `src/families/qwen27/projection.zig`: `Linear.index` / `Linear.table`, built by `indexTable` in `prepare` (owned buffers, so they join the weights' residency set).
- `src/families/qwen27/quant_gpu.zig`: the decode `Key` and `Layout` carry `dict` when the matrix has a table; `gateUpActivated` binds it.
- `src/families/qwen27/dflash/runtime_head.zig`: the drafter's vocabulary-span head rewrites its own pairs, so it clears the inherited index and reads its pairs.
- `src/core/segments.zig`, `src/families/qwen27/dflash/runtime_frame.zig`: `7d41e661`'s hunks.

## Why outputs cannot change

The table holds exactly the u32 words the pairs hold, and the index maps each (group, column) to its own word: `table[index[i]] == pairs[i]` for every i, checked by `pair_index.zig`'s tests. The kernel feeds those words to the same scale/bias extraction, the same fma chain, the same tensor op and the same split-K reduction. Prompt rows still read the pairs. The event waits change host-side waiting only. DFlash depth 1, the pinned drafter and `mtp-head.manifest.json` are unchanged. The target decides every token.

## Measurements (local, Apple M5 Max 128 GB Mac Studio; not official)

Three alternating rounds, standard `yukon run`. Prefill / decode-window tokens/s; every run passed the local correctness gate:

| round | `2c41feb1` | + dictionary | + dictionary + event waits (this entry) |
|---|---|---|---|
| 1 | 1140.1 / 61.00 | 1139.9 / 61.36 | 1140.3 / 61.33 |
| 2 | 1140.4 / 61.07 | 1141.2 / 61.33 | 1140.4 / 61.45 |
| 3 | 1140.2 / 61.06 | 1140.9 / 61.46 | 1140.6 / 61.49 |
| mean | 1140.2 / 61.04 | 1140.7 / 61.38 | 1140.4 / 61.42 |

The dictionary gives +0.56% decode window; the event waits add little locally (their ranked +0.17% is the better guide). The exact submitted tree was rebuilt with the track's sandboxed `setup.sh` and passed a final local run: 1140.3 / 61.46, correctness and timing passed.

## Measured and not included

- The same dictionary for the drafter's prepared adapters (with ROWS8 pipelines): -0.07%. Drafter ROWS8 alone: +0.03%.
- Removing the drafter's five SwiGLU dispatches entirely (wrong outputs, timing probe): -0.017 ms a frame. The drafter frame (3.55 ms GPU of 3.66 ms) is not dispatch-bound.
- Prompt GEMM probes (wrong outputs, timing only): skipping the unpack gives +5% prefill, and skipping unpack and epilogue +7%. The prompt GEMM is about 93% tensor-op bound, so row-block sharing of the unpack cannot win (terrapinelf's `f9daa39d` measured the same).
- Decode probe: skipping the unpack saves 4% of verify dense GPU time. Decode projections at 8 rows are latency- and occupancy-bound, which is why halving metadata bytes gives about 0.6%, not 6%.

## Limitations

- Expected official effect over `2c41feb1` is about +0.3% to +0.5%. This may fall short of the 75-bip margin; the entry also serves as a ranked measurement of dictionary-coded metadata on Apple.
- The index adds about 0.8 GB of resident buffers (u16 per metadata word), within the wired budget.

## Provenance

- Base: terrapinelf's `2c41feb1` (which carries our prompt-row SwiGLU, head pre-compile and one-pass catch-up, and kshitij-hash's prompt-loop addressing).
- kshitij-hash's `pair_index.zig` and table idea (`e59458da`); ercumentyildirim's event waits (`7d41e661`).
- Ours: the TF_DICT decode path on the current lean ROWS8 and SwiGLU kernels, load-time wiring, the head guard and the measurements.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside the permitted inputs.


# In-place ring-cache attention for the DFlash drafter, on terrapinelf's 0231208a

## Summary

This entry starts from terrapinelf's `0231208a-97bd-4dee-a733-0d4251f18d74` (official score 1.33136866238078, rejected 0.13% under the promotion margin). The tree restored by `yukon reset 0231208a` is carried byte for byte except for three small edits in two areas. All of the base's work, and everything it declares, is credited to its authors through that dependency: the promoted `4ccaa92c`, fkiene's verify-tail picks and barrier-free drafter frame, GordoAR's draft input sums, 64-key draft attention chunks and serial split-K for the fused gate/up projection, e-man07's chain keep, and the upstream TensorFold ports the chain credits.

The one performance change of this entry is ours: the drafter's attention reads its rolling K/V cache in place when the cache does not wrap, instead of gathering it into scratch buffers every round and layer.

`mtp-head.manifest.json` is unchanged: DFlash depth 1 with the pinned drafter. No selector, tau, edge coefficient, kernel arithmetic, checkpoint or harness file is touched.

## The change (`dflash/runtime_context.zig`, `attendRows`)

Before: for every draft layer, in every round, `attendRows` allocated two scratch tensors of context_rows + rows keys, copied the whole sliding window of cached keys and values into them (`gather`, one or two copy dispatches each), copied the query rows' own keys and values behind them, then attended over the scratch.

Now: the cache holds `window + 16` rows. When the context does not wrap, that is when `begin % window + context_rows <= window`, its keys and values already stand contiguously in the cache at slot `begin % window`. The rows after the context are free (live slots are exactly the context's), and when the query rows still fit within `window + 16` slots, `attendRows` copies the query rows' own keys and values into the free slots right behind the context and runs the same attention kernels over the cache itself. When the context wraps, or the rows would not fit, the original gather path runs unchanged.

Per layer and round this removes two gather copies of the whole window (about 3 MB each way for a 2,000-row window) and two scratch allocations. The attention kernels, the mask, the scale, the key order and every operand value are the same, so the drafter's scores, candidates and chosen draft token are bit-identical.

## Why the draft and the target's tokens cannot change

- The attention kernel reads the same keys and values in the same order; only their address changes.
- The slots written (the query rows' own keys and values) are past the live context, so they never overwrite a live row. They are overwritten by the genuine appends of the next round, which are encoded before that round's attention.
- Wrapping contexts and any case where the rows would not fit within `window + 16` slots take the original path.
- The batched-stream path uses each stream's own caches the same way and is otherwise unchanged.
- The target still verifies every drafted token and decides every emitted token.

Checked, not just argued: a rolling FNV hash over every drafted token and every emitted token, taken every 100 rounds of the public long-copy run, is identical between the base and this tree (rounds 100 to 500, up to position about 2,009). The public correctness gate passes token for token through the full 1,023 steps, which runs the seed (1,065 tokens) past the 2,048-slot window boundary, so both the in-place path before it and the gather path after it are exercised.

## Other edits

- `bench/resident.zig`, `bench/main.zig`: the resident's keepalive hold is passed as a small function-pointer struct (`resident.Hold`) instead of importing `engine_api` into `resident.zig`. The base's `resident.zig` imported `engine_api`, which its host-test module (`test-bench-resident`) does not provide, so that test no longer compiled. With this edit it compiles and passes. Runtime behaviour is unchanged: the same `begin`/`end` calls bracket each request.

## Measurements (local, M5 Max, alternating runs)

These are local `yukon run` numbers with `score: null`. They are not official scores. Local decode accounting differs from the official seed-inclusive 128-step windows.

| Build | Prefill tok/s | Decode tok/s |
|---|---|---|
| 0231208a, 3 runs | 1034.0 / 1034.1 / 1034.0 | 54.33 / 54.36 / 54.42 |
| this entry, 3 alternating runs | 1033.8 / 1034.0 / 1033.9 | 54.56 / 54.68 / 54.76 |

That is about +0.55% decode with prefill unchanged. By the 0.25/0.75 score weighting it is worth about +0.4% of score. We are not claiming a score; the expected margin over the promotion threshold is small, and the official paired measurement decides.

## Tried and not kept

Each was compared against the same base in alternating runs and was within noise or worse: a later verify commit interval of 8 (54.73 vs 54.72 tok/s); declaring the register projection's maximum threadgroup size; unrolling pragmas on the register lane kernel; leader-lane shuffles for the register kernel's metadata and group sums (slower, 47.8-49.0 vs 51.5); a two-group prefetch in the register kernel (slower); tiling the drafter attention's key loop (no change); removing drafter barriers (no change, the drafter encoder is serial); command-buffer part thresholds for the drafter frame (no change, and no split at all was 0.5 ms slower).

## Validation

- Setup and `yukon run`: pass, correctness and timing, on the exact tree submitted.
- Host tests: all `test-qwen27-*-host` steps, `check-qwen27-speed-host`, `test-bench-protocol` and `test-bench-resident` pass.
- Repository mock suite (`tools/test-*.sh`): all pass except `test-orphan-guard.sh` (fails) and `test-sandboxed-build.sh` (times out after 200 s here). Both fail the same way on an unmodified checkout of the base on this host, so they are environmental; neither tool was touched.
- Editable-surface archive size check passes.

## Attribution

The base, `0231208a`, and through it `4ccaa92c`, are declared as material dependencies. The in-place attention change and the host-test import fix are ours. No code from other solvers beyond the base tree was imported. Model: Claude Sonnet 5.5, harness: Claude Code.

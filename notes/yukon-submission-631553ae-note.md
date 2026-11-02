Model: Claude Opus 5.5
Harness: Claude Code

# A four-column DeltaNet prompt scan and a-github-name's decode-attention accumulators, on ercumentyildirim's b490cd9f

Effort: Claude Opus 5.5, max effort. One Claude Code session on an Apple M5 Max 128 GB Mac Studio.

## Objective

Clear the promotion bar over e-man07's promoted `4311baf2` (OFFICIAL 1.70011, bar about 1.71286) by joining two
pieces that act on different phases: one of ours on the prompt pass and one of a-github-name's on the verify's
attention. The base is ercumentyildirim's `b490cd9f` (OFFICIAL 1.71053), the best scored tree on the record when this
entry was built. Numbers are labelled OFFICIAL (the ranked runner), LOCAL (this M5 Max) or ESTIMATE.
`mtp-head.manifest.json` is unchanged (DFlash depth 1, the pinned drafter). Every emitted token is the target's.

## Why the prompt pass

The decode leg's time is the seed window plus the 128-step decode window, and the seed is a full 1065-token prompt pass
(`seed_tokens` in the track fixture). In the record's official run the seed was about 0.85 s of about 2.87 s of decode
time. So a prompt pass that is 1% faster moves the prefill gain by 1% (weight 0.25) and the decode gain by about 0.3%
(weight 0.75): about +0.47% of score for each 1% of prompt pass.

LOCAL prompt profile of the record (1065 rows, per-class GPU time with per-launch timestamps):

| class | ms |
|---|---|
| dense projections | 862 |
| GDN scan (`tf_delta_gdn_chain_wide`, 48 layers) | 37.7 |
| attention | 9.0 |
| GDN prepare, norms, gate | 10.2 |

The scan does about 200 instructions a lane a row and holds about 19 SIMD groups a core, which matches its measured
~0.74 us a row (about 1000 cycles): it is bound by issue, not by loads. A one-row-ahead load prefetch, which we built
first, made it slower (37.7 to 40.8 ms LOCAL), so the lever is the instruction count, not latency.

## Change 1 (ours): the prompt scan keeps four value columns a lane

`tf_delta_gdn_chain_wide_body` gains a column-count parameter `CC`. Each lane still owns the same 16 key channels of
its columns (`sub = lane & 7`) and each column keeps its own arithmetic in the same order:

- the decay multiply and the in-order 16-channel key product, then the xor 1, 2, 4 shuffle tree;
- the correction `(value - remembered) * mixing`;
- the state update and the in-order 16-channel query product, then the same shuffle tree;
- the bf16 output store by the lane with `sub == 0`.

With `CC = 4` a lane converts each row's 16 key and 16 query words once for four columns instead of two, and a SIMD
group covers 16 value columns. `scanChain` dispatches `dv / 16` SIMD groups a head (`gdn.zig`), and the wide path
requires `dv % 16 == 0` (`gdn_pool_gpu.zig`; dv is 128 here). The prompt-marks kernel keeps `CC = 2` and its own
`dv / 8` grid, so its path is the base's byte for byte. Decode scans do not use this kernel.

Why it is exact: a column's words, operations and their order do not depend on how many columns share its lane. Only
the grouping of columns into lanes and SIMD groups changes.

## Change 2 (a-github-name's `6691bd03`): decode-attention accumulator pressure

Its delta over its parent `edff1a49` (`attention_mpp.metal`, `attention.zig`):

- the ordered merge loads two chunk records ahead instead of eight;
- the split tail keeps only the first eight query rows between PV steps (rows 8 to 15 are padding);
- a sparse prefix grid (`sg == 1`, fewer than 128 groups) runs a four-part instantiation that owns 64 output columns a
  SIMD group.

`b490cd9f` carries `edff1a49`'s prompt operand but not its counted-wait failure words, so we ported the delta onto
`b490cd9f`'s counted waits: the prefix keeps `b490cd9f`'s `q27_wait_count` call and its 32-byte chain words, and gains
the `PARTS` parameter, the `OD`/`OE` sizes and the `q27_attn_prefix_quarter_1` pipeline. Its note documents the
exactness of each part: the merge still combines partials in key order; the tail's padded rows are rebuilt as zeros
before the same PV operation; the four-part prefix runs the same scores and softmax in every part.

## Exactness

- Free-run tapes (`record-correctness-golden --benchmark-free-run --benchmark-steps 1024 --benchmark-spec dflash:1`) on
  three independent 1065-token prompts (endings in code, article and story text) equal the record lineage's tapes
  token for token over 1024 steps, with the teacher-forced cases.
- Every local run below passed the public correctness gate.
- `tools/test-editable-surface-size.sh`, `test-submission-security.sh`, `test-dflash-depth-guard.sh`,
  `test-spec-declaration.sh`, `test-vendored-subset.sh` and `test-participant-guard-docs.sh` pass.

## Measurements (LOCAL, M5 Max, alternating, standard `yukon run`)

| run | b490cd9f prefill / decode | + change 2 | + changes 1 and 2 (this entry) |
|---|---|---|---|
| 1 | 1244.2 / 68.82 | 1244.3 / 69.15 | 1252.8 / 69.19 |
| 2 | 1243.8 / 68.80 | 1244.6 / 69.22 | 1252.9 / 69.06 |

Change 2: decode +0.5%, prefill flat. Change 1: prefill +0.7% (the prompt pass), decode flat on this machine's
1023-step local window, where the seed is a small share; on the ranked 128-step window the seed is about 30% of
decode time.

ESTIMATE over `b490cd9f`: change 2 about +0.37% of score (decode); change 1 about +0.17% (prefill) and +0.15% (seed),
about +0.7% in all. Over the record's tree family (`b490cd9f` 1.71053, `5e33a2cf` 1.71109, `4f8c7fb0` 1.70685) that
is about 1.715 to 1.723.

## Also measured and left out

- terrapinelf's two-round deferred keep (`201049a5`) with jungjipdo's parallel GDN K slices and 16-layer commits
  (`002828e6`) on `b490cd9f`: decode 68.83 to 68.60 LOCAL over three pairs, so left out.
- ercumentyildirim's `5e33a2cf` delta over `b490cd9f` on this entry: prefill +0.15%, decode -0.3% LOCAL, so left out.
- A one-row-ahead load prefetch in the prompt scan: scan 37.7 to 40.8 ms LOCAL.

## Reproduce

```bash
git fetch origin pull/520/head && git checkout FETCH_HEAD -- Vendor/TensorFold mtp-head.manifest.json  # b490cd9f
# apply this archive's files
./tools/fetch-benchd.sh && ./setup.sh
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
```

## Environment and commands

- Hardware: Apple M5 Max (40-core GPU), 128 GB unified memory, Mac Studio; macOS 27.2. Pinned Zig from the track's
  `tools/zig-bin.sh`; the worker built with `zig build bench-worker -Doptimize=ReleaseFast`, staged with
  `tools/stage-bench-worker.sh`, run through `yukon run` (local mode, public prompt, DFlash depth 1).
- A/B: binaries alternate, one full `yukon run` each, on the same idle machine; prefill and decode tokens/s from
  `score.json`'s seconds-per-token fields.
- Profile: a diagnostic build (not submitted) that wraps the prompt chunk in the engine's GPU profiler and logs each
  class's time and each projection shape's time.

## Limitations

- Local A/B on this machine cannot resolve differences under about 0.3%.
- The ranked runner's control leg varies run to run by up to about 0.6%, so a single official score carries that noise.

## Provenance

- ercumentyildirim `b490cd9f` and what it carries (e-man07's record `4311baf2`, a-github-name's `edff1a49`,
  ArthurOstapenko's `c618f399` and `48d75d22`, ercumentyildirim's `f6a2bd2a`, jungjipdo's `486a9ac5`, and the chain
  under them); a-github-name `6691bd03`.
- Ours: the prompt profile and the scan's issue-bound diagnosis, the four-column scan, the port of `6691bd03` onto
  `b490cd9f`, the M5 Max measurements and the tapes.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside
  the permitted inputs.


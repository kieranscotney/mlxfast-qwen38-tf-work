# mlx.fast Qwen3.8-27B on TensorFold: optimisation work

Work by Kieran Scotney on the [Yukon](https://www.yukon.org/mlxfast?benchmark=qwen38-27b-tf&platform=mlx) mlx.fast
challenge `GumbiiDigital/mlxfast-qwen38-27b-tf` for [TensorFold](https://github.com/ashhart/TensorFold): make Qwen3.8-27B
(4-bit MLX, DFlash2 drafter at depth 1) prefill and decode faster on an Apple M5 Max, with the target's output token-for-token exact.

Done with Claude Code (Claude Sonnet 5.5 on 30 Oct, Claude Opus 5.5 from 31 Oct). All engine code this builds on is public; see Attribution.

**Headline:** submission [`07fd50da`](notes/yukon-submission-07fd50da-note.md) was **promoted** at 1.63957
(+76 bps over the previous leader). Three earlier finds (prompt lean, prompt-row SwiGLU, one-pass drafter catch-up) were
carried, with credit, in leader `2c41feb1` and every leader after it.

## My finds

Patches are relative to the TensorFold root (`zig/...`) and apply to the base submission listed. The same patches, one commit
each, are on the [`yukon-qwen38-finds` branch](https://github.com/kieranscotney/TensorFold/tree/yukon-qwen38-finds/yukon/qwen38-27b-tf)
of my TensorFold fork.

| # | Find | Base | Local effect | Official |
|---|---|---|---|---|
| A | In-place ring-cache attention for the drafter ([patch](patches/tensorfold-upstream-inplace-ring-attention.patch)) | 0231208a | drafter 4.60 → 4.47 ms, dec +0.5% | merged upstream, ashhart/TensorFold#1727 |
| B | Drafter head lane pipeline compiled at load | ff408317 | first draft round 190 → 29 ms on a loaded machine, dec +0.75% | ashhart/TensorFold#1734 |
| 1 | Prompt projections take the lean prefetch ([patch](patches/01-prompt-lean.patch)) | 2d06a685 | pf +2.5% | in `ea8f8271` 1.37849; carried in leaders |
| 2 | Prompt-row SwiGLU in the gate/up epilogue ([patch](patches/02-prompt-row-swiglu.patch)) | e75967a0 | pf +2.1% | in `23d85725` 1.52942; carried in leaders |
| 3 | Drafter catches up on a prompt in 1024-row passes ([patch](patches/03-drafter-flush-1024.patch)) | 5317b10e | dec window +0.15% | in `f3f8f654` 1.38132; carried in leaders |
| 4 | Drafter top-k as a SIMD-shuffle tournament ([patch](patches/04-drafter-simd-topk.patch)) | 38c11ee6 | small, exact | in promoted `07fd50da` |
| 5 | Two-tile prompt inputs for the mixer projections ([patch](patches/05-two-tile-prompt-inputs.patch)) | 38c11ee6 | prompt GPU −10.2 ms, pf +0.95% | **promoted** `07fd50da` 1.63957 |
| 6 | Depth-one star GDN verify kernel ([patch](patches/06-gdn-star-verify.patch)) | 7edba466 | 8-row verify −0.31 ms, dec +0.05% | not submitted; same idea by ArthurOstapenko scored +0.45% |
| 7 | Four-column prompt DeltaNet scan ([patch](patches/07-four-column-prompt-scan.patch)) | b490cd9f | prompt scan 4 columns a lane | in `631553ae` 1.71545; carried in `c1d09a0f` and every leader since |
| 8 | Eight-part sparse decode attention prefixes ([patch](patches/08-eight-part-decode-prefix.patch)) | c153ffc9 / 292d7fda | dec +0.12% (5 rounds) | in `da328474`, `254ee549` |
| 9 | Sibling floor 0.02, calibrated on independent prompts ([patch](patches/09-sibling-floor-0.02.patch)) | 292d7fda | dec +0.10% (5 rounds) | in `da328474` |
| 10 | Drafter MLP down projection on its 2-bit copy ([patch](patches/10-drafter-down-on-2-bits.patch)) | a5b756b6 | dec +0.61%; with 8: +0.85% | in `bf5555a3` 1.75523 (official −1.4% vs its base: did not transfer) |
| 11 | Siblings stop once the offered drafter mass reaches 0.95 ([patch](patches/11-sibling-mass-stop-0.95.patch)) | cb4d3130 | rows −1.5%, dec +0.05% | null on `22b390be` |
| 12 | Keep settle waits on its own shared event ([patch](patches/12-keep-event-settle.patch)) | cb4d3130 + 11 | −7 µs a round | not submitted |
| 13 | Two-state first round: catch-up staged beside round 1's verify ([patch](patches/13-two-state-first-round.patch)) | cb4d3130 + 11 | round 1 −1.2 ms, no tokens lost | not submitted |

Notes on what each one changes:

1. `Key.lean` went from `reg and !k.prompt` to `reg`, so prompt-row reg projections use kshitij-hash's codes-only prefetch
   (`TF_REG_LEAN`) that decode already used. Exact by construction. No codes in flight loses 5.4%; two groups gains nothing.
2. Extends fkiene's decode-only `TF_SWIGLU` fused epilogue to prompt rows (16-row blocks, split-K 1, row offsets) with the
   same bf16 rounding and `h` expression, so it stays exact. The patch also carries the `prepareDecode` helper from B.
3. The drafter absorbed a prompt's tap rows 128 at a time (nine frames and waits for 1065 tokens). Now 1024 per pass.
4. Each rank's tournament runs inside a simdgroup with `simd_shuffle_xor`, leaving one threadgroup step per rank. Tapes identical.
5. e-man07's generalised multi-tile prompt kernel (`e6c09968`), merged with `38c11ee6`'s fp16-subnormal unpack, applied to FFN
   down and, my addition, to the 12288×5120 and 10240×5120 mixer inputs and the unfused prompt gate/up. **The patch carries
   e-man07's kernel**; my part is the merge and the extra shapes.
6. With sibling drafts at depth 1 the verify tree is a star. `tf_delta_gdn_star` pairs value columns per simdgroup and runs
   independent children two at a time instead of the generic tree-slot path. Exact on 3×1024-token tapes.

7. Each lane of the prompt DeltaNet scan carries four value columns instead of two, sharing the key and query conversions.
   terrapinelf later added running row offsets and reduce-scatter sums on top (`a8112696`, `c153ffc9`).
8. `q27_attn_prefix<1, 8>` instantiated and dispatched in place of the quarter form on tensor-unit devices: the verify's
   full-attention prefixes split over twice as many SIMD groups. Same merge code, so exact.
9. A sibling the drafter rates under 2% is not verified. Measured on independent prompts; fkiene's 0.7% floor on the same
   base read lower officially (`9f65dcb3` 1.75200 against `292d7fda` 1.75637).
10. The drafter's 2-bit reg kernels run at the tensor units' rate (~1.48 T weights/s on M5 Max) while its q4 down
    projections are bandwidth-bound, so down on its 2-bit copy shortens the drafter frame (~449 to ~300 us) for a small
    acceptance cost. Checked on three independent prompts (+0.2 to +0.35%). A least-squares refit of the 2-bit copies
    ([negative patch](patches/x1-drafter-lloyd-refit-NEGATIVE.patch)) lowered weight error but halved the gain.

## Endgame (2–3 Nov): why we stopped

The frontier reached 1.79565 (`22b390be`), which makes the bar 1.80912. Our best exact build sat at about 1.789–1.7925.
Nothing was filed. [`notes/endgame-findings.md`](notes/endgame-findings.md) covers what was measured on the way. In short:

- The GPU is busy ~99.7% of a decode round. Host gaps are ~90 µs, and the rest is kernel time.
- On M5 Max, overlapping work with the verify doesn't hide it: a second-queue append and a staged prompt catch-up both
  give back almost everything to contention.
- The drafter's 9.4 ms prompt catch-up sits in every official sample (~0.4% of the decode leg). Local 1024-token windows
  dilute it 8×.
- Depth-1 sibling selection is at its optimum. The remaining misses need information the pinned drafter doesn't have.
- Changing a projection's K split breaks exactness. Five stagger segments gain locally but not on the official runner.
- New tools ([`tools/`](tools)), microbenchmarks ([`microbench/`](microbench)), negative results
  ([`patches/negative/`](patches/negative)) and diagnostic patches ([`patches/diagnostic/`](patches/diagnostic)).

## Measurement finds

The full write-up, with a table of 22 dead ends, is in [`notes/measurements.md`](notes/measurements.md). The main points:

- **The official decode time includes the seed prompt pass** (~28%), so prefill effectively weighs ~0.46 of the score, not 0.25.
- The live promotion bar is 75 bps, not the 200 in `benchmark.json`.
- Ceilings on M5 Max: 579 GB/s read (decode GEMV reaches ~95%), 66 TFLOPS tensor ops (prompt gate/up ~90%).
  Prompt GEMM is ~93% tensor-op bound; acceptance is ~98% of its depth-1 ceiling.
- The decode loop is bound by dependent-load latency, not bytes, so prefetching beats shrinking.
- **Official legs are public**: each validation PR comment carries both legs and the box. One pair is scored; the formula
  `(ctrl_prefill/cand_prefill)^0.25 x (ctrl_decode/cand_decode)^0.75` reproduces every score to 1e-5
  ([script](tools/official_legs.py), [all entries](notes/official-legs-all-entries.txt)). Box `m5-max-128gb-2-1` has a very
  stable control leg; `laptop-1` drifted ~1.5% slower through the evening of 1 Nov, which explains `da328474`.
- Curated state of the work: [`notes/summary.md`](notes/summary.md); every experiment with its result:
  [`notes/experiment-ledger.md`](notes/experiment-ledger.md).
- Ranked-box variance (~1–2%) and local noise (~±0.3%) mean small exact gains are worth banking even when local A/B can't see them.

## Yukon submissions

| Submission | What | Official | Result |
|---|---|---|---|
| [`7ea68997`](notes/yukon-submission-7ea68997-note.md) | `0231208a` + A | 1.33430 | rejected, 0.013% behind an entry with the same idea found independently |
| [`1afa673b`](notes/yukon-submission-1afa673b-note.md) | `ff408317` + B | 1.35639 | rejected |
| [`ea8f8271`](notes/yukon-submission-ea8f8271-note.md) | `2d06a685` + B + 1 | 1.37849 | rejected |
| [`5317b10e`](notes/yukon-submission-5317b10e-note.md) | `ea8f8271` + ported polling and small adapters + drafter lean | 1.38217 | rejected, 0.08% short |
| [`f3f8f654`](notes/yukon-submission-f3f8f654-note.md) | `5317b10e` + 3 | 1.38132 | rejected |
| [`23d85725`](notes/yukon-submission-23d85725-note.md) | `e75967a0` + 2 + B + 3 | 1.52942 | rejected, 0.08% short |
| [`ca8506c9`](notes/yukon-submission-ca8506c9-note.md) | `2c41feb1` + ported dictionary metadata and event waits | 1.54015 | rejected |
| [`07fd50da`](notes/yukon-submission-07fd50da-note.md) | `38c11ee6` + 5 + 4 + ported wait skip | **1.63957** | **promoted** |
| [`c903ef78`](notes/yukon-submission-c903ef78-note.md) | `7a2d34f1` + three ported pieces (composition only) | — | runner fault, no score |
| [`631553ae`](notes/yukon-submission-631553ae-note.md) | `b490cd9f` + 6691 attention + 7 | 1.71545 | rejected; 7 carried by `c1d09a0f` (promoted) |
| [`da328474`](notes/yukon-submission-da328474-note.md) | `292d7fda` + 8 + 9 | 1.74109 | rejected (laptop-1 box) |
| [`254ee549`](notes/yukon-submission-254ee549-note.md) | `a5b756b6` + 10 + 8 | — | cancelled; refiled as `bf5555a3` |
| `bf5555a3` | `a5b756b6` + 10 + 8 | 1.75523 | rejected |

## Attribution

This work builds on public submissions to the same challenge and on upstream TensorFold (Apache-2.0). Ported or built-upon work
is named where used: terrapinelf, e-man07, kshitij-hash, fkiene, ercumentyildirim, ArthurOstapenko, GordoAR, i34-9, jungjipdo, a-github-name, otarkhan, mrv777,
akki-rock05 (CUDA drafter refit idea), kaankolcu and the TensorFold authors. Each submission note lists its exact dependencies.

# Yukon mlxfast-qwen38-27b-tf: curated summary

Refreshed 2026-11-01 (late) from JOURNAL.md and BACKLOG.md; JOURNAL.md stays the raw log, BACKLOG.md the experiment
ledger (#1-30+). The previous version is kept as SUMMARY-2026-11-01-early.md. Labels:

- **OFFICIAL**: the ranked Yukon runner's sealed score.
- **LOCAL**: measured on our M5 Max (alternating `yukon run` A/B, PPROF/VSPAN/PGPU timers, tapes).
- **CALC**: arithmetic from measured inputs; not itself measured.
- **UNVERIFIED**: an assumption or inference not measured directly.

## State now

- Leader: mrv777 `ed4ee746`, OFFICIAL 1.74675. Live minimum improvement 75 bps (`yukon benchmark show`), so the bar is
  about **1.7599**. UNVERIFIED/CALC: its tree's real level is about 1.732 (promoted on a slow control leg).
- Highest scored entry: kshitij-hash `63a5493e`, OFFICIAL 1.75276 (+34 bps, not promoted). Its gain over its own exact
  stack (`913de13`, OFFICIAL 1.73967) comes from a statistical 2-bit verify-head sieve. Not exact by construction, so we do
  not port it (ledger #30).
- Exact stacks cluster at OFFICIAL 1.739-1.741: terrapinelf 685e2856 1.74009, dca2274 1.74103; ercumentyildirim
  927b5d7b 1.74104; e-man07 80d0abf 1.74092; kshitij-hash 913de13 1.73967; terrapinelf c153ffc9 1.73791.
- Our protected baseline: **C2b**, branch `opus-c2` = terrapinelf c153ffc9 + 8-part attention prefix + sibling floor 0.02.
  Binary `.build/experiments/c2b`; tapes EXACT. LOCAL ~1259 prefill / ~68.7-69.1 decode tok/s. CALC estimate ~1.740.
- Gap: about +1.1% exact real gain needed over the exact cluster (CALC). Nothing filed; one submission at a time.
- Watcher: `watch.sh` (240 s); exits on frontier change, an entry above the leader, or our own result; restart after.

## Our recent submissions (OFFICIAL)

| Entry | Content | Result |
|---|---|---|
| 631553ae (S10) | b490 + 6691 attention + our 4-column scan | 1.71545, rejected (faf6648e promoted first) |
| bf1977d3 (S11) | - | cancelled after c1d09a0f (carries our scan, credited) promoted at 1.73175 |

## Current-lineage measurements (LOCAL unless noted)

- Round 28.85 ms: verify ~26.4 ms (CALC floor 23.5 ms at 580 GB/s), drafter frame 2.42 ms in one command buffer;
  1.983 tokens/round; 3.65 rows/round public, 4.39 on independent prompts.
- Prompt pass (C2b PPROF, 1065 rows): dense 844.8 ms (~65 TFLOPS vs 66 peak), GDN scan 37.2 ms, attention 8.4,
  GDN prepare 4.1, norm 3.4, GDN gate 2.6.
- GDN scan (#29): 728 ns per row-step, 384 SIMD groups/layer, ~300 instr/lane/row with 256 irreducible FP. Exact
  next-row prefetch: prefill -0.55%. CC=8 spills. CALC realistic ceiling <= +0.2% score. Closed.
- Positive and kept: sibling floor 0.02 (calibrated; others found 0.015), 8-part attention prefix.
- Negative or closed: oracle selection, DARTree (needs depth > 1), persistent drafter kernel, drafter splits/caps,
  low-rank and partial-K head screens, M=8 tiles, SIMD qmv, TF 1.2.0 unroll, rigorous head screen (~70 sigma), drafter
  fusions (present), attention chunk (64 optimal), #1478/#1439 (present), two-queue overlap (hides 18%), M-tail (0.65%),
  k|v TN2, long reg dispatch (-0.2..-0.4%, matches #435 OFFICIAL -0.3%), banding on C2b (-0.2/-0.3%), lossless code
  compression (decode ALU ~40 ms/round). Details and numbers: BACKLOG.md.
- Drafter precision already selective in the base: gate/up/o/fc 2-bit, q/k/v/conv/down q4, head 2-bit screen + q4
  rescore, adaptive q4 frames when the sibling share drops below 0.95 (ArthurOstapenko's per-linear runs). See #31.

## Research loop (replaces the old "do nothing" re-entry rule)

1. Hypothesis with a CALC ceiling; skip if the ceiling is under ~+0.3% score.
2. Smallest exact prototype on C2b; alternating A/B (ab3.sh) plus independent prompts (abindep.sh) where acceptance moves.
3. Tapes EXACT for anything kept. Record every result in BACKLOG.md, immediately.
4. Watch competitors; port exact mechanisms with attribution onto the strongest exact base.
5. File immediately once a tree credibly reaches the live bar (75 bps over the leader).

---

# History: earlier lineage (7a2d34f1 / 0ee904e8 era, superseded)

The section below is the earlier summary, kept for its measurements and tools. Its leader, bar and "state" lines are
out of date.

## State at that time

- Leader: a-github-name `0ee904e8`, OFFICIAL 1.66912. The bar to beat it is about 1.6816 (75 bps).
- Our best candidate: **S7b** `c4f54a38`, OFFICIAL 1.67192 (+17 bps). Branch `opus-s7` at 4516056, patch
  `S7b-on-0ee9.patch`, binary `.build/experiments/s7b`, note `../outputs/opus-s7-note.md`, deps `../outputs/opus-s7-deps.json`.
  - Contents: 0ee904e8, plus 16559cea's five files, plus 068da9eb's residual-stream store.
  - Tapes: 3 × 1024 tokens EXACT.
  - Keep this unchanged.
- Our promotion: `07fd50da`, OFFICIAL 1.63957 (+76 bps over 38c11ee6), 2026-10-31.
- Watcher: `watch.sh` polls `yukon benchmark show` and `yukon submissions` every 240 s. It submits nothing. It exits on
  a frontier change, a newly scored entry above the leader, or a result for one of our own entries; restart it after
  every exit.

## Our submissions (OFFICIAL unless noted)

| Entry | Content | Result |
|---|---|---|
| 07fd50da (C4) | two-tile prompt inputs etc. on 38c11ee6 | 1.63957, PROMOTED |
| 2dbf20e2 | stack on 09862c94 | cancelled (frontier moved) |
| 9bb46730 (S3) | d739d71b + b47ac1ca + conv keep on 7a2d34f1 | cancelled for S5 |
| c903ef78 (S5) | 16559cea + 398dd495 + 3b698102 on 7a2d34f1 | FAILED: runner fault (SERIAL-CONTROL-LEG-OUTSIDE-BAND); no score |
| cda0b247 (S6) | S5 + 97bc5fb8 | cancelled for S7b |
| c4f54a38 (S7b) | 0ee904e8 + 16559cea + 068da9eb | 1.67192 (+17 bps over leader), not promoted |

LOCAL vs OFFICIAL calibration:

| Entry | LOCAL estimate | OFFICIAL |
|---|---|---|
| S7b | about +20 bps | +17 bps |
| C4 | +50–60 bps | +76 bps |

Local A/B cannot resolve gains under about 30 bps.

## Other entries' OFFICIAL scores on the 7a2d34f1 lineage, used as evidence

| Entry | Score | Note |
|---|---|---|
| 0ee904e8 | 1.66912 | PROMOTED |
| 6e709db5 | 1.66787 | |
| 068da9eb | 1.66662 | +11 bps over 16559cea |
| 16559cea | 1.66484 | |
| 91adefb2 | 1.66460 | |
| 97bc5fb8 | 1.66382 | |
| e009b685 | 1.66375 | |
| 01ba1875 | 1.66360 | |
| f57b1770 | 1.66266 | |
| d296634a | 1.66153 | z split; **below** 16559cea, so negative |
| 985eae44 | 1.66161 | |
| b1279077 | 1.67080 | S7b-like plus gate/up lockstep; **below** S7b, so lockstep is negative |

## Measured results (LOCAL) and their outcomes

### Rejected because slower or null (do not redo without new evidence)

**Metadata and indexing**
- f32 metadata (5b6cb063): decode-row table alone 64.95 -> 65.03 tok/s (noise); with prompt pairs, prefill 1207 -> 1175
  (-2.7%).
- Hot index: an index-load-skip proxy was slower (verify 27.7 -> 28.8 ms), so the loop is not sensitive to index bytes.
- Code-entropy palette: H = 3.72 bits; no group has 8 or fewer distinct codes.
- Drafter dictionary, drafter SBI, drafter rows8 builds: null.

**Gate/up lockstep**
- Ours: decode -3%. ercumentyildirim's b1279077 kernel: decode -0.7% (OFFICIAL also below S7b).

**GDN**
- Prepare folded into the scan: -3.0% decode per SIMD group, -1.2% threadgroup-shared form.
- #360 prepare folded into the projection epilogue: verify span 27.40 -> 27.56 ms.
- Deferred replay beside the scan: decode -1.5%.
- f2f0fb9a one-dispatch keep: flat.
- GDN state fold (write test): -1.2%.

**Norm fusion**
- Norm folded into a projection (proxy): decode -3.6%.

**Prefetch and cache**
- Two-group-ahead prefetch, deeper table or row-sum prefetch: losses.
- Cache "touch" prefetch in mixer windows (`touch-prefetch-experiment.patch`): null.

**Prompt kernels**
- Fused two-tile SwiGLU (ours), four-tile mixers and down projection, row-block reuse, row-pair kernel, Morton or banded
  grid orders, 256-row launches: slower or neutral.

**K-split changes**
- Change the summation order, so they break exactness. Serial SK4 failed correctness (`serial4-FAILED.patch`).

### Kept (shipped in our entries or carried by the leader)

- Prompt two-tile inputs (C4): prompt GPU 910.0 -> 899.8 ms.
- conv keep and #357 batched keep: verify span 27.26 ms, decode +0.6% on C4.
- 068da9eb on 0ee904e8: prefill +0.55%, decode +0.1%.
- 16559cea remainder on 0ee904e8: flat; carried for completeness.

### Probes (LOCAL, exact or diagnostic)

- GDN stage cost: running prepare twice per layer moved the verify span 27.19 -> 27.30 ms, so about 0.1 ms per round, or
  about 2.2 µs per layer-stage.
- Norm stage: about 3.2 µs each, 129 per verify, about 0.42 ms in all (on 2bcc).
- Verify width: 2 rows 27.33 ms, plus about 0.22 ms per extra sibling row (8 rows 28.83 ms).
- Warm-cache re-run of the same launch:

  | Projection | Change |
  |---|---|
  | attention out | -36% |
  | GDN z\|b\|a | -26% |
  | down | -22% |
  | attention k/v | -29% |
  | gate/up, vocabulary head | about 0 |

  So the GPU system cache holds at least about 50 MB.
- Removing attention computation entirely left the verify wall unchanged, so attention is off the critical path.
- The verify keeps the GPU busy with zero idle time (9 command buffers). Host gap about 0.2 ms.
- Hardware microbench: peak GPU read 579 GB/s; tensor-op peak 66 TFLOPS.
- Acceptance on independent prompts: 1.94–2.00 tokens per round (maximum 2.0); 3–5 rows per round after the sibling
  floor.

## Calculated ceilings (CALC)

- **Scoring sensitivity:** 0.47 ms per decode round ≈ +0.75%; 1 ms of prompt pass ≈ +0.05%; 15 ms of prompt ≈ +0.75%.
- **Verify weight bytes:** about 13.6 GB per round, logical bytes from matrix sizes (codes plus indices). At 579 GB/s
  that is 23.5 ms. With about 3.4 ms of non-dense work in a 27.3 ms verify, dense ≈ 23.9 ms ≈ 98% of peak.
  Reaching 100% recovers about 0.5 ms per round, ≈ +0.75%, and only at perfect bandwidth. Gate failed.
- **Hot index:** top-255 coverage 64–73%, so an 8-bit hot format is 19–25% *larger* than 12-bit. A 10-bit index with
  escapes saves about 0.7% of bytes, at most about +30 bps. Rejected.
- **Cross-round reuse:** a cache of at least about 50 MB against a 13.6 GB stream, with the drafter running between
  verifies, gives about 0 reuse. Even pinning 50–100 MB is at most 0.7% of bytes, about +25 bps, and there is no pinning
  API. Rejected.
- **GDN stage fusion:** removing prepare entirely is at most about +16 bps; prepare plus gate at most about +30 bps,
  before the fused kernel's cost. Rejected.
- **Projection prompt pass:** about 57.5 TFLOP at about 65 TFLOPS, near the tensor peak.

## Unverified assumptions

- **DRAM traffic is inferred, not measured.** No hardware memory counters were read (public Metal has none, and
  Instruments GPU counters were not run). The inference: each code word is loaded once per verify (rows8, 2 rows), plus
  about 0.3 GB of GDN state and 0.13 GB of KV. That is CALC from shapes and the GDN probe.
- 579 GB/s is our microbench peak; the true sustainable peak could differ slightly.
- The per-shape GB/s figures in `vprof*.txt` include per-command-buffer profiling overhead, so small shapes are
  understated. Use them for ranking only.
- The official box's control leg occasionally falls out of band (S5's failure). We assume this is runner-side.

## Reusable tools

- **Build:** `qb.sh <name>` builds and stages bench-worker and saves `.build/experiments/<name>`.
- **A/B:**
  - `ab3.sh <reps> <bins...>`: alternating `yukon run`, printing passed, prefill and decode tok/s.
  - `abspan.sh <reps> <bins...>`: the same plus the VSPAN verify span (needs a VSPAN build).
- **Tapes:** `../accept/tape/tape.sh <bin>` via `python3 locked.py`. It records 3 × 1024-token free-run tapes; compare
  `expected_decode_tokens` and `cases[0].expected_tokens` against `bw-7edb.*.json`.
- **Diagnostics:**

  | Patch | What it measures |
  |---|---|
  | `VSPAN-diag.patch` | verify span per round (logged in the resident log) |
  | `PGPU-diag.patch` | prompt GPU busy time |
  | `pprof-diag.patch` | per-class and per-shape verify profile (VPROF lines) |
  | `opus-diag-and-warm-on-7f3c.patch`, `star-with-diag.patch` | older diagnostics |

- **Submit:** `s6-submit.sh` is a template. Check the tree is clean, run `check-editable-surface-size.sh`, move
  BOTH `zig-out` and `Vendor/TensorFold/.zig-cache` aside (static review refuses any file over 2.5 MB), then `yukon submit` with the deps and note. Dependencies must be already-scored entries; unscored
  ones return "submission not found".
- **Branches:** `opus-s7` (S7b), `opus-s6`, `opus-s5`, `opus-on-38c1` (C4/C5), `opus-on-7a2d` (S3), `opus-on-07fd`, plus
  older lineage branches.
- **Patches:**
  - Rejected experiments, kept for reference: `*-SLOWER.patch`, `*-NULL.patch`, `*-FAILED.patch`, `*-INEXACT.patch`,
    `S7c-lock-on-s7b.patch` (our lockstep), `touch-prefetch-experiment.patch`.
  - Others' deltas, ready to re-apply: `pr380-attn.patch`, `pr383-on-16559.patch`, `pr385-on-16559.patch`,
    `pr386-on-16559.patch`, `pr395-on-0ee9.patch`, `pr396-kernel.patch`.


Model: Claude Opus 5.5
Harness: Claude Code

# kshitij-hash's 16559cea with fkiene's in-place GDN conv keep and rows8 drafter launches, on ercumentyildirim's 7a2d34f1

Effort: Claude Opus 5.5, max effort. One Claude Code session on an Apple M5 Max 128 GB Mac Studio.

## Objective

Reach the promotion bar over the record `7a2d34f1` (ercumentyildirim, OFFICIAL 1.65568, main `e539c41`, itself built on
our `07fd50da`) by composing three exact decode-side pieces that have not been measured together. Numbers are labelled
OFFICIAL (the ranked runner), LOCAL (this M5 Max) or ESTIMATE. `mtp-head.manifest.json` is unchanged (DFlash depth 1, the
pinned drafter). Every emitted token is the target's, and the target's output words are bitwise the base's.

## Composition

| # | Piece | Source | Evidence |
|---|---|---|---|
| 1 | Decoded gate/up on two SIMD groups (SwiGLU from each lane's registers), every reg scale pre-folded by 2^24, the drafter span head on the eight-tile interleave | kshitij-hash `16559cea` (with e-man07's `d739d71b`), unchanged | OFFICIAL 1.66484 on 7a2d34f1 (+0.55%) |
| 2 | In-place GDN conv keep: a round's keep drops its 48 `publish_conv` launches and the fence before them | fkiene `398dd495`, four files | OFFICIAL 1.62883 on 38c11ee6 (+0.10%) |
| 3 | Rows8 drafter decode launches: the drafter's prepared q4 linears carry an eight-row build that skips each lane's padding row | fkiene `3b698102`, unchanged | LOCAL below |

Piece 2 is taken as its diff from `98632cc` in `deltanet.metal`, `deltanet_kernels.zig`, `decode_round.zig` and
`gdn_pool_gpu.zig`; its `quant_gpu.zig` and `lane_projection.zig` hunks (256-row prompt launches, LOCAL neutral on this
M5 Max) and its `runtime_head.zig` (overlapping piece 1's head) are not carried. Piece 3's `affine4_lane.zig` merges with
piece 1's (the folded scales and the adapter index stay; the rows8 build `small` and `forRows` are added beside them).

## Exactness

- Piece 1: each decoded gate and up word rounds to bf16 as the unfused store does, the activation and group sums follow
  the existing epilogue's order; the 2^24 fold applies the subnormal unpack's scale to the stored scales once at load,
  the same IEEE products (16559cea's note).
- Piece 2: the keep's conv windows land in place: the same words, without the separate publish pass.
- Piece 3: the drafter's projections run the rows8 build on the same words and metadata; the drafter only proposes,
  the target decides every token.

Correctness beyond the public gate: free-run tapes (`record-correctness-golden --benchmark-free-run --benchmark-steps 1024
--benchmark-spec dflash:1`) on three independent 1065-token prompts (endings in code, article and story text) equal the
record lineage's tapes token for token over 1024 steps, with the teacher-forced cases. Every run below passed the public
correctness gate.

## Measurements (LOCAL, M5 Max, alternating, standard `yukon run`)

| run | 7a2d34f1 prefill / decode | this entry |
|---|---|---|
| 1 | 1210.6 / 64.35 | 1199.2 / 64.78 |
| 2 | 1212.7 / 64.40 | 1208.7 / 65.23 |

Decode +1.0%, prefill within this machine's noise. Piece 3 alone, on top of `d739d71b` + `b47ac1ca` + piece 2: decode
65.38 -> 65.58 tok/s (+0.3%) in two alternating pairs. ESTIMATE about +0.7% to +0.8% composite from piece 1's official
+0.55% plus pieces 2 and 3.

Also measured on this record and the previous one and left out: a two-tile fused SwiGLU prompt kernel of ours (slower
than `7a2d34f1`'s form), `f8ac43bc`'s GDN prepare in the projection epilogues (verify span 27.40 -> 27.56 ms, slower), the
256-row prompt launches (neutral) and `67d6e7ff`'s batched keep (within noise on this lineage).

## Reproduce

```bash
git fetch origin main && git checkout origin/main -- Vendor/TensorFold mtp-head.manifest.json   # e539c41 = 7a2d34f1
# apply this archive's files
./tools/fetch-benchd.sh && ./setup.sh
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
```

## Environment and commands

- Hardware: Apple M5 Max (40-core GPU), 128 GB unified memory, Mac Studio; macOS 27.2. Pinned Zig 0.17.0 from the
  track's `tools/zig-bin.sh`; the worker built with `zig build bench-worker -Doptimize=ReleaseFast`, staged with
  `tools/stage-bench-worker.sh`, and run through `yukon run` (local mode, public long-copy prompt, DFlash depth 1).
- A/B: the base and candidate binaries alternate, one full `yukon run` each, on the same idle machine; prefill and
  decode tokens/s are read from `score.json`'s seconds-per-token fields.
- Tapes: `benchd-bin/record-correctness-golden --worker-bin <bench-worker> --weights reference_weights/... --prompt-tokens
  <prompt>.tokens.json --steps 64 --benchmark-free-run --benchmark-steps 1024 --benchmark-spec dflash:1`, one resident
  booted at depth 1 for each binary, and the `expected_decode_tokens` compared between binaries.

## Next steps

- The decoded gate/up's group-sum hand-off and the verify's remaining dependent small kernels (norms, GDN prepare and
  gate) are the next decode costs we measure; each dependent stage costs about 3 us on this machine.

## Limitations

- Local decode varies by about 0.5% run to run here; the estimate leans on piece 1's official result.
- Piece 2's official evidence is on 38c11ee6; piece 3 has no official result yet.

## Provenance

- Base: ercumentyildirim's promoted `7a2d34f1` on our promoted `07fd50da`, and everything they carry.
- kshitij-hash `16559cea` (with e-man07 `d739d71b`); fkiene `398dd495` and `3b698102`. Ours: the composition, the
  affine4_lane merge, the exclusions above, the M5 Max measurements and the tapes.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside the
  permitted inputs.


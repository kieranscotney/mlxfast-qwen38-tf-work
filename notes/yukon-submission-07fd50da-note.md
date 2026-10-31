Model: Claude Opus 5.5
Harness: Claude Code

# Two tiles a SIMD group on every 5120-deep prompt input (the MLP gate/up unfused for it), on e-man07's 38c11ee6

Effort: Claude Opus 5.5, max effort. One Claude Code session on an Apple M5 Max 128 GB Mac Studio.

## Summary

This entry is e-man07's promoted `38c11ee6` (main `98632cc`, OFFICIAL 1.62714) with four changes. Numbers are labelled
OFFICIAL (the ranked runner), LOCAL (this M5 Max) or ESTIMATE. `mtp-head.manifest.json` is unchanged (DFlash depth 1, the
pinned drafter). Every emitted token is the target's, and the target's output words are bitwise the base's.

| # | Piece | Source |
|---|---|---|
| 1 | Two tiles a SIMD group on the prompt pass's 5120-deep inputs: attention query+gate (12288x5120), GDN qkv (10240x5120) and MLP gate/up (34816x5120) | ours |
| 2 | Prompt MLP gate/up runs as that two-tile projection plus the activation pass, not the fused one-tile SwiGLU epilogue | ours |
| 3 | The generalised two/four-tile prompt kernel piece 1 runs on, merged with 38c11ee6's fp16-subnormal unpack; the MLP down stays at two tiles | e-man07 `e6c09968` |
| 4 | Polled-pass chunk bookkeeping (earlier verify chunks already complete skip the blocking wait) | fkiene `78e984af` |
| 5 | The drafter's bf16 top-k tournaments on SIMD-group shuffles (one barrier a rank instead of eight) | ours |

## Pieces 1 to 3: the prompt pass's 5120-deep inputs

In the base, the prompt two-tile kernel (`TF_PROMPT_TN`: one 16x64 activation slice in cooperative registers feeding
adjacent tiles) serves only the 5120-wide outputs on a two-way prompt split. The inputs that read the 5120-wide normed rows,
the attention query+gate, the GDN qkv and the MLP gate/up, run one tile a SIMD group and reload each activation slice for
every tile. They are most of the prompt's matrix work.

e-man07's `e6c09968` generalised the tile kernel to two or four tiles of one code batch. Its diff conflicts with
`38c11ee6`'s fp16-subnormal unpack in the kernel's tile macro; the merge keeps `e6c09968`'s structure (one code pointer
and one destination stepping through the batch) with `38c11ee6`'s unpack (`word & 0x000F000F` straight into the tensor op)
and its 2^24 scale. `promptTiles` returns two for `k == 5120` and `n` in {12288, 10240, 34816} on whole code batches, and
keeps two for the 17408- and 6144-deep 5120-wide projections as the base does. Four tiles measured slower here on both the
inputs (LOCAL 907 vs 901 ms) and the down (914.6 vs 901 ms).

The prompt MLP gate/up had run through `gateUpActivatedPrompt`, a one-tile kernel with SwiGLU in its epilogue. On two tiles
the plain projection followed by the existing activation pass (`mlp`) is faster (LOCAL 896.8 vs 901.0 ms a prompt pass),
so prompt rows return false from `gateUpActivated` and take that path; decoded rows keep the fused rows8 kernel.

Exactness: each tile's codes unpack into the same register words and run the same tensor ops, scale/bias FMA chain, split
reduction and stores as the one-tile kernel; only which SIMD group computes a tile and where its activation operand comes
from change. The unfused gate/up writes the same bf16 gate and up words the fused epilogue rounded, and the activation
pass applies the same expression and group sums the fused epilogue reproduces (that equivalence is what made the fused
path exact). Output words are bitwise equal.

## Piece 4 and 5

fkiene's `78e984af` change to `segments.zig`, taken alone as the diff from its parent `b0622d89`: a polled pass's earlier
chunks that already report completion get only their failure check, not a blocking `wait()`.

The drafter top-k: each rank takes each SIMD group's best candidate by xor shuffles in the same `tf_topk_before` order
(score key, then lower token id), then the eight group bests through threadgroup memory, one barrier a rank with the
rank's parity choosing the scratch half. The non-finite mask is OR-reduced once up front, as the first rank's check did.
The selected tokens, scores and order are unchanged.

## Measurements

LOCAL, alternating with the base binary on the same machine; the prompt-pass GPU time is the sum of the prompt's command
buffers (about +-1 ms run to run), prefill and decode from `yukon run`:

| run | base prompt GPU / prefill / decode | this entry |
|---|---|---|
| 1 | 908.8 ms / 1166.8 / 64.51 | 900.8 ms / 1172.1 / 64.10 |
| 2 | 912.9 ms / 1160.9 / 64.27 | 901.1 ms / 1176.4 / 64.35 |
| 3 | 908.5 ms / 1167.3 / 64.25 | 897.7 ms / 1179.8 / 64.52 |

Prompt GPU time -1.1%, prefill +0.95%, decode within noise. ESTIMATE: the prompt pass counts in prefill and in the seed
window of decode, so roughly +0.5% to +0.7% composite.

Correctness beyond the public gate: free-run tapes (`record-correctness-golden --benchmark-free-run --benchmark-steps 1024
--benchmark-spec dflash:1`) on three independent 1065-token prompts equal the base lineage's tapes token for token over
1024 steps, with the teacher-forced cases and the seed token.

## Reproduce

```bash
git fetch origin main && git checkout origin/main -- Vendor/TensorFold mtp-head.manifest.json   # 98632cc = 38c11ee6
# apply this archive's files
./tools/fetch-benchd.sh && ./setup.sh
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
```

## Limitations

- The prompt gain is measured on an M5 Max Mac Studio; the ranked box is an M5 Max too, but its prefill may differ.
- The gain is mostly prompt-side; decode moves only through the seed window's shorter prompt pass.

## Provenance

- Base: e-man07's promoted `38c11ee6` and everything it carries (kshitij-hash, fkiene, ArthurOstapenko, i34-9,
  ercumentyildirim, terrapinelf and earlier work, including our prompt SwiGLU, drafter head pre-compile and dictionary
  metadata).
- e-man07: `e6c09968` generalised prompt tile kernel. fkiene: `78e984af` polled-pass chunk bookkeeping.
- Ours: two tiles on every 5120-deep prompt input, the unfused prompt gate/up, the merge with the subnormal unpack, the
  SIMD-shuffle drafter top-k and the measurements above.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside the
  permitted inputs.


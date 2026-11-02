# The drafter's MLP down projection on its 2-bit copy and eight-part decode prefixes, on a5b756b6

Effort: max. One Claude Code session.

## Summary

Base: ercumentyildirim's `a5b756b6-30d2-473a-8d59-af898fa70d1d` (OFFICIAL 1.75755 on box 2-1), carried byte for byte
except three files. That tree is terrapinelf's `292d7fda` (kshitij-hash's 2-bit verify-head sieve stack on mrv777's
record `ed4ee746`) plus paired cross-lane star sums, parallel K slices for the wide decode projections and 16-layer
verify commits. Every submission it declares is declared here too (our own `631553ae` is left off to fit the
20-entry limit).

This entry adds two pieces:

1. **The drafter's MLP down projection drafted on its 2-bit copy** (`dflash/runtime_backend.zig`, `draftBits`). The
   drafter already drafts its MLP gate and up, attention o and the fusion projection fc on 2-bit copies made at
   engine
   open from the pinned bf16 checkpoint; down stayed q4. On M5 Max the drafter's 2-bit reg kernels run at the tensor
   units' rate (about 1.48 T weights a second), while its q4 down projections are bandwidth-bound: the five down
   projections (445 M weights, 223 MB at q4) take about 449 us of a 2.42 ms drafter frame, against about 300 us on the
   2-bit copy. Frames that fall back to precise drafting (sibling landing share below 0.95) still read the q4 copy, so a
   stretch where the 2-bit copies miss more siblings drafts as before.
2. **Eight-part sparse decode attention prefixes** (`attention_mpp.metal`, `attention.zig`): an instantiation
   `q27_attn_prefix_eighth_1` (template `<1, 8>`) dispatched in place of the quarter form on tensor-unit devices, so the
   verify's full-attention prefixes split over twice as many SIMD groups.

`mtp-head.manifest.json` is unchanged: DFlash depth 1 with the pinned drafter; its checkpoint is unchanged. The target,
its weights, the verify and the sieve are unchanged.

## Evidence (LOCAL, M5 Max, alternating `yukon run`, all binaries interleaved in each pass)

On `a5b756b6` (z0 = base):

| build | change | decode vs base (public prompt) |
|---|---|---|
| zd | down on 2 bits | +0.50% (2 passes; 70.61 vs 70.26 tok/s) |
| z2 | down on 2 bits with a least-squares refit of the 2-bit copies | +0.21% |
| z3 | down and q/k/v on 2 bits | -0.15% (q/k/v cost acceptance, as ArthurOstapenko found) |

Prefill was flat in every build (within 0.05%). On three independent prompts (code, article and story continuations,
not the benchmark prompt) the down-on-2-bit drafter gained +0.19%, +0.34% and +0.33% decode in its first form, so the
change is not tuned to the benchmark prompt; the final tree's independent-prompt pass is in our run log.

The eight-part prefix alone, measured on `292d7fda` in five alternating rounds: +0.12% decode (per-binary sd about
0.08 tok/s), prefill flat.

Expected (CALC from the official leg model, score = pre^0.25 x dec^0.75 on one pair, decode including the prompt seed):
about +0.3% over the base's level, i.e. about 1.761 against box 2-1's median control leg; the bar is 1.75985.

## Why our previous entry read lower

`da328474` (292d7fda + the prefix and a floor change) read 1.74109 on box laptop-1, whose control and candidate legs
were slowing through the evening (control prefill 998.7 -> 983.2 tok/s over its last four runs; candidate prefill
1259 -> 1243). Locally the same pieces measured +0.2% decode over 292d7fda in five alternating rounds.

## Exactness

- Piece 1 changes only which copy of the drafter's down weights the drafter reads; the drafter only proposes. Every
  emitted token is the verify's pick.
- Piece 2 splits the same key ranges into more parts and merges the partial softmax sums with the same merge code.
- The verify head's sieve is the base's (statistical, kappa 10), unchanged.

## Files

| File | Change |
|---|---|
| `zig/src/families/qwen27/dflash/runtime_backend.zig` | `draftBits`: `mlp.down_proj.weight` joins the 2-bit set |
| `zig/kernels/metal/qwen27/attention_mpp.metal` | instantiation `q27_attn_prefix_eighth_1` (`<1, 8>`) |
| `zig/src/families/qwen27/attention.zig` | tensor-unit prefix pipeline `q27_attn_prefix_eighth_1`, grid 64 x parts |
| every other file | `a5b756b6`'s, unchanged |

## What we tried and dropped

- A least-squares (Lloyd) refit of the drafter's 2-bit copies, after the CUDA track's drafter refit: lower weight error,
  but -0.08% decode alone and it halved the down projection's gain, so the plain clip search stays.
- q/k/v on 2 bits: -0.15% decode; acceptance falls faster than the bytes save.
- A 0.02 sibling floor on top of the 2-bit down: no further gain and -0.13% on the code prompt; the floor stays 0.015.

## Rules and attribution

- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs; no reads outside the permitted inputs.
- No benchmark-specific outputs, fixture use or timing changes.
- Credit: ercumentyildirim (base), terrapinelf, kshitij-hash, fkiene, a-github-name, otarkhan, jungjipdo, e-man07,
  ArthurOstapenko (whose per-linear acceptance runs set the drafter's bit widths), mrv777 and the earlier record holders
  through the declared dependencies.

## Limitations

- The drafter's precision moves acceptance, which depends on the prompt; the official prompt is private. We checked
  three independent prompts, not the official one.
- The sieve's exactness is statistical (the base's property).
- Scores depend on the runner box (laptop-1 has read about 1% lower than 2-1 for the same trees this evening).

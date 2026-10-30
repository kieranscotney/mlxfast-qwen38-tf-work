# mlx.fast Qwen3.8-27B on TensorFold: optimisation work

Work by Kieran Scotney on the [Yukon](https://www.yukon.org/mlxfast?benchmark=qwen38-27b-tf&platform=mlx) mlx.fast
challenge for [TensorFold](https://github.com/ashhart/TensorFold): make Qwen3.8-27B (4-bit MLX, with the DFlash2 drafter at
depth 1) decode and prefill faster on an Apple M5 Max, with the target's output bit-exact.

Done with Claude Code (Claude Sonnet 5.5). All engine code this builds on is public; see Attribution.

## The contribution: in-place ring-cache attention for the DFlash drafter

`attendRows` in `zig/src/families/qwen27/dflash/runtime_context.zig` used to gather the drafter's whole sliding-window K/V
into scratch for every draft layer, every round. When the context does not wrap, it already stands contiguously in the
cache (which holds `window + 16` rows), so the query rows' own K/V are written into the free slots behind it and attention
reads the cache in place. A wrapping context takes the unchanged gather path. The kernels, mask, scale and key order are
unchanged, so the draft is bit-identical and the target still verifies every token.

- Drafter proposal: 4.60 ms to 4.47 ms per round (M5 Max).
- Decode: about +0.5% locally, prefill unchanged.
- Checked: a rolling hash of every drafted and emitted token matches the unmodified build round for round, and the
  exact-token gate passes past the 2,048-slot window boundary.

Patches:

- [`patches/tensorfold-upstream-inplace-ring-attention.patch`](patches/tensorfold-upstream-inplace-ring-attention.patch):
  against upstream TensorFold `main` (`01e25d3a`), `git am` format, with a CHANGELOG entry. `zig build test -Dcpu=apple_m1` exits 0.
- [`patches/challenge-inplace-ring-attention.patch`](patches/challenge-inplace-ring-attention.patch): the same change
  against the Yukon challenge tree.
- [`notes/upstream-pr.md`](notes/upstream-pr.md): the pull-request text.

## Yukon result

Submission `7ea68997` (terrapinelf's `0231208a` plus this change) scored **1.33430** officially, prefill +3.5%, decode +57.0%.
It missed promotion by about 0.013%: `e836f316` (1.33447) had been promoted a few minutes earlier, and it contains the same
in-place idea, found independently. See [`notes/yukon-submission-7ea68997-note.md`](notes/yukon-submission-7ea68997-note.md).

## What did not help (all bit-exact, measured with alternating runs)

Loop-unroll pragmas on the register lane kernel; leader-lane `simd_shuffle` for the register kernel's scale/bias and
group-sum loads (slower, 47.8 vs 51.5 tok/s); a two-group prefetch (slower); tiling the drafter attention's key loop;
removing drafter barriers (the encoder is serial, so no change); command-buffer split thresholds for the drafter frame;
a later verify commit interval; the register kernel's maximum-threadgroup attribute. Profiling showed the verify matmuls
run at roughly 85% of memory bandwidth, and that acceptance on the long-copy fixture is about 65% in the first 150 rounds
and about 100% afterwards.

## Attribution

This work builds on public submissions to the same challenge and on upstream TensorFold (Apache-2.0):
terrapinelf (`0231208a`, `4ccaa92c`), ercumentyildirim (`9b7f90ee`, `f379bc8c`, `e836f316`), fkiene, GordoAR and e-man07,
and the TensorFold authors. The patches here are only the changes made in this work.

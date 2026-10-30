## qwen27 DFlash draft: attend the rolling K/V cache in place when it does not wrap

**What.** The drafter's `attendRows` gathered the whole sliding-window K and V into scratch for every draft layer, every round. When the context does not wrap, it already stands contiguously in the cache (which holds `window + 16` rows), so the query rows' own K/V are written into the free slots behind it and attention reads the cache in place. A wrapping context, or rows that would not fit, takes the unchanged gather path.

**Exactness.** Same kernels, mask, scale and key order, so the draft's scores, candidates and chosen token are bit-identical, and the target still verifies every drafted token. Scratch slots lie past the live context and are overwritten by the next round's committed appends, which are encoded before that round's attention. Only the non-streamed and per-stream attention call sites that already used `attendRows` change.

**Evidence.**
- Rolling FNV hash over every drafted and emitted token: identical to the unmodified build every 100 rounds up to position ~2,009 of a 1,065-token-prompt, 1,023-step long-copy run; the exact-token gate passes through position ~2,088 (past the 2,048-slot boundary, where the gather path resumes).
- M5 Max, DFlash2 drafter, depth 1: drafter proposal 4.60 -> 4.47 ms per round; decode +0.4% to +0.6%; prefill unchanged. Alternating runs against the unmodified build.
- `zig build test -Dcpu=apple_m1` on an M5 Max: exits 0 (the log shows the suite's negative-control FAIL lines, as usual). `test-golden` was not run.

**Not in this PR.** No kernel, arithmetic, reduction-order or precision change; no new family or backend.

Found while working on the Yukon mlx.fast Qwen3.8-27B TensorFold challenge (benchmark 82b5f52e-2f2e-4281-96ee-aef75ff965c4), built on public work by terrapinelf, fkiene, GordoAR, e-man07 and ercumentyildirim.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

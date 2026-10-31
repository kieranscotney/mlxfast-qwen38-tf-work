Model: Claude Opus 5.5
Harness: Claude Code

# 5317b10e with the drafter's prompt catch-up in one pass instead of nine

Effort: Claude Opus 5.5, max effort. One Claude Code session.

## Summary

This entry is our `5317b10e-8931-4336-b493-5f29640c6a5d` (official 1.38217 on `m5-max-128gb-laptop-1`, 0.08% under the 75-bip margin over `ff408317`), carried byte for byte except `families/qwen27/dflash/runtime_model.zig`.

The change: when the drafter's first proposal after a prompt pass catches up on that prompt's committed taps (`Model.flush`), it absorbs them in passes of up to 1,024 rows instead of 128. A ~1,067-row seed prompt therefore takes two drafter frames, each with one commit and one host wait, instead of nine. The work in each frame is the same per-row work: the fusion projection over the taps, the hidden norm, and each layer's K/V projection, key norm, rotary and cache append.

The increment is small and we say so plainly: about 7-8 ms of host round trips inside the first timed decode round. That round sits inside the scored decode time.

## Where the time was

From diagnostic timers on this stack (removed again), the first proposal after a prompt cost about 14.5 ms, against about 3.7 ms for every later proposal. Of that, 10.9 ms was `Model.flush`: nine 128-row `Session.absorb` calls at 1.27-1.41 ms each (41 rows for the last). Each call opens a drafter frame, encodes, commits and waits. Most of each call's time is the frame round trip, not GPU work. With 1,024-row passes the catch-up is two frames.

## Why outputs cannot change

- The target is untouched. It decides every emitted token, and the drafter is still run at DFlash depth 1 on every round.
- Each operation in a catch-up pass is row-independent:
  - the reg-adapter projections run in 16-row blocks, 128 rows a launch (`encodePrepared`'s existing loop);
  - the RMS norms, rotary and cache appends are per row.
- Splitting the same rows into fewer, larger passes writes the same keys and values to the same cache slots in the same order. The drafter's state after catch-up is identical, so its proposals are too.
- `mtp-head.manifest.json`, the pinned drafter, the selector and every harness file are unchanged.
- The pass is still capped by the drafter's sliding window (`window - slot`), so a catch-up never wraps the ring inside one pass.

## Measurements (local, Apple M5 Max 128 GB Mac Studio; not official)

Alternating rounds, each binary built from its own tree, standard `yukon run` (public long-copy prompt, local mode). Prefill / decode-window tokens/s; every run passed the local correctness gate:

| round | `5317b10e` | + one-pass catch-up (this entry) |
|---|---|---|
| 1 | 1061.7 / 57.51 | 1062.0 / 57.50 |
| 2 | 1062.4 / 57.35 | 1062.6 / 57.55 |
| 3 | 1062.2 / 57.45 | 1060.6 / 57.54 |
| mean | 1062.1 / 57.44 | 1061.7 / 57.53 |

That is +0.15% decode window. The exact submitted tree was rebuilt with the track's sandboxed `setup.sh` and passed one more local run: 1061.7 prefill / 57.52 decode tok/s, correctness and timing passed.

## Official context for the stack

| entry | box | prefill speedup | decode speedup | score |
|---|---|---|---|---|
| `ff408317` (promoted) | `m5-max-128gb-2-1` | 1.0413 | 1.5056 | 1.37301 |
| our `ea8f8271` | `laptop-1` | 1.0626 | 1.5035 | 1.37849 |
| our `5317b10e` | `laptop-1` | 1.0621 | 1.5091 | 1.38217 |

On `laptop-1` the serial control leg decodes about 1.5-3% faster than on `-2-1` (42.08-42.22 vs 42.45-43.45 ms/token), which lowers a candidate's speedup there. In absolute terms `5317b10e`'s candidate decode (28.323 / 27.552 ms/token) was 2.0-2.1% faster than `ff408317`'s (28.912 / 28.148).

## Other things measured in this session and not included

- No codes in flight for prompt-row reg projections (load each group's codes at the top of its own iteration): prefill 1004.6 against about 1062 tok/s (-5.4%). One group in flight, as `5317b10e` has, is the optimum on M5.
- Serial split-K for the prompt's two-way-split FFN down and mixer out projections: no prefill change.
- fkiene's in-place GDN conv commit (`4e0a5cea`) on this stack: -0.1% (the keep runs behind the drafter, off the critical path).
- LEAN2 for the small-row drafter adapters: no gain.

## Provenance

- Base: our `5317b10e` = our `ea8f8271` (terrapinelf's `2d06a685` via `e2100b71`, kshitij-hash's `b3757a3b`, the promoted `ff408317` and its sources, our head pre-compile `1afa673b` and prompt-lean selection) + ercumentyildirim's polled decode waits (`b8cea221`) + ArthurOstapenko's small-row drafter adapters (`c5ac45a3`) + our lean drafter adapters.
- The one-pass catch-up is ours.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside the permitted inputs.

## Reproduce

```bash
# base: our 5317b10e source (opus-iter2 lineage), then this archive's runtime_model.zig
./tools/fetch-benchd.sh && ./setup.sh
MLXFAST_SCORE_PATH=score.json ./tools/qwen38-27b-tf-yukon-run.sh
```

The A/B above alternated two prebuilt `bench-worker` binaries, staged with `tools/stage-bench-worker.sh` before each `yukon run`. Each binary was built from its own tree with `zig build bench-worker -Doptimize=ReleaseFast` (Zig 0.17.0).

## Limitations

- The gain is a one-off per prompt pass, so it is small next to run-to-run noise (about 0.3% in decode window locally). It was positive on average over three alternating rounds, not in every round.
- The official score also depends on the box's live control leg. The table above shows how much that moved our earlier entries.
- A larger single-pass catch-up raises peak drafter frame scratch for that one frame (about 1,024 rows of fusion and hidden activations). The frame allocates it on demand and reuses it, as it already does for other pass sizes.


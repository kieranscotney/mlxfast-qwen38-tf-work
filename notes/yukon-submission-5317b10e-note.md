Model: Claude Opus 5.5
Harness: Claude Code

# ea8f8271 (prompt lean, +6.3% prefill officially) with polled decode waits, small-row drafter adapters and lean drafter adapters

Effort: Claude Opus 5.5, max effort. One Claude Code session.

## Summary

This entry is our `ea8f8271-3274-4503-bf69-6a9564138eeb` (official 1.37849, +0.40% over the promoted `ff408317`), carried byte for byte except four files. It adds three decode-side pieces:

1. **ercumentyildirim's polled decode waits (`b8cea221`), ported unchanged.** `core/segments.zig` gains an optional `polls()` hook. When a family returns true, the host spins on its last command buffers' status (`std.atomic.spinLoopHint`) before the blocking `wait()`. `families/qwen27/forward.zig` returns true for decoded rows (`rows <= 16`): a verify's 2 and a serial step's 1. That removes the OS wake-up latency from every round's critical path. `b8cea221` itself scored on `e2100b71`, which lacks LEAN2, prompt lean and our head pre-compile.
2. **ArthurOstapenko's small-row drafter adapters (`c5ac45a3`), ported unchanged.** `core/affine4_lane.zig` builds a second `rows8` pipeline next to each reg adapter's projection, sharing its prepared words and metadata. `forRows(rows)` picks it for windows of at most 8 rows, the drafter's decode blocks. `dflash/runtime_backend.zig` builds reg adapters with it, `dflash/runtime_callbacks.zig` encodes through `forRows`, and `residency.zig` keeps the extra pipeline resident.
3. **Lean prefetch for the drafter's reg adapters (ours).** In the same `Layout`, `.reg_lean = order != null`, so the drafter's prepared projections use kshitij-hash's codes-only prefetch path of `tf_lane_reg` (`TF_REG_LEAN`, from `b3757a3b`), as the target's decoded rows already do. LEAN2 for the small-row adapters was measured and is not included.

## Why ea8f8271 was close

`ea8f8271`'s official metrics on box `m5-max-128gb-laptop-1`:

- prefill speedup **1.0626**, against `ff408317`'s 1.0413 (+2.0%), from selecting the lean prefetch for prompt rows;
- candidate decode **28.426 / 27.658 ms/token**, against `ff408317`'s 28.912 / 28.148 on box `-2-1` (about 1.7% faster in absolute terms);
- but that box's serial control decoded at 42.153 ms/token against 43.454 / 42.453 on `-2-1`, so the decode speedup was 1.5035 against 1.5056.

The decode-side additions here target the remaining 0.35% to the 0.75% margin.

## Why outputs cannot change

- **Polling:** host scheduling only. The same command buffers are committed in the same order. `wait()` still runs after the poll, so completion, failure checks and buffer reuse are unchanged.
- **Small-row adapters:** each 16-row block's arithmetic is the reg kernel's, without the padding row's accumulators, sums and stores (terrapinelf's ROWS8, already on the target's decoded rows in this base). Same tensor op, same group-order fma chain, same split-K reduction for every live row.
- **Lean drafter adapters:** codes-only prefetch. Metadata quads and row sums load inside their own group's iteration instead of one iteration ahead. Same words, same order, same operations.
- The target decides every token. DFlash depth 1, the pinned drafter and `mtp-head.manifest.json` are unchanged.

## Measurements (local, Apple M5 Max 128 GB Mac Studio; not official)

Round-robin, each binary built from its own tree, standard `yukon run` (public long-copy prompt, local mode), prefill / decode-window tokens/s; every run passed the local correctness gate.

Ports on `ea8f8271` (two rounds each):

| binary | round 1 | round 2 | mean |
|---|---|---|---|
| `ea8f8271` | 1062.9 / 56.97 | 1063.0 / 56.99 | 1063.0 / 56.98 |
| + polled waits | 1061.4 / 57.33 | 1063.4 / 57.23 | 1062.4 / 57.28 |
| + small-row adapters | 1061.0 / 57.14 | 1061.7 / 57.11 | 1061.3 / 57.12 |
| + both | 1060.6 / 57.11 | 1062.3 / 57.42 | 1061.4 / 57.27 |

Repeat of `ea8f8271` against both ports: 1062.4 / 56.98 and 1062.6 / 56.98 against 1061.8 / 57.48 and 1061.2 / 57.46, so the ports give **+0.86% decode window**.

Drafter adapter lean on top of both ports: 1061.8 / 57.55 and 1060.6 / 57.38, against 1061.7 / 57.45 and 1060.1 / 57.34 (+0.1-0.2%). LEAN2 on the small adapters: 1060.3 / 57.41 (no gain, not included).

The exact submitted tree was rebuilt with the track's sandboxed `setup.sh` and passed one more local run: 1061.5 prefill / 57.54 decode tok/s, correctness and timing passed.

Also measured and dropped: serial split-K for the prompt's two-way-split FFN down (5120x17408) and mixer out (5120x6144) projections changed prefill by -0.02%.

## Expected official effect

About +1.0% decode window. The window is about 72% of scored decode time, so roughly +0.7% decode and +0.55% score over `ea8f8271` on the same box: about 1.386 on `laptop-1`, more on `-2-1`.

## Limitations

- Local decode varies by about 0.5% run to run. The port gain was present in every pairing.
- The official score depends on the box's live control leg (laptop-1 vs -2-1 differ by about 1.5% in control decode speed).

## Provenance

- Base: our `ea8f8271` = terrapinelf's `2d06a685` (via `e2100b71`, kshitij-hash's `b3757a3b`, the promoted `ff408317` and its sources) + our drafter-head pre-compile (`1afa673b`) + our prompt-lean selection.
- Ported unchanged: ercumentyildirim's polled decode waits (`b8cea221`) and ArthurOstapenko's small-row drafter adapters (`c5ac45a3`).
- Ours: selecting the lean prefetch for the drafter's reg adapters, and the measurements above.
- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs, no environment variables, no reads outside the permitted inputs.


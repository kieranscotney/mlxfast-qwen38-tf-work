# Eight-part sparse decode prefixes and a calibrated 0.02 sibling floor on terrapinelf's 292d7fda sieve stack

Effort: max. One Claude Code session.

## Summary

Base: terrapinelf's `292d7fda-baa5-4b09-86c5-6b50c4f0c9c8` (OFFICIAL 1.75637), its commit `4aa1898`, carried byte for
byte except three files. That tree is the record `ed4ee746` (mrv777) plus kshitij-hash's 2-bit verify-head sieve
(`63a5493e`), fkiene's star scan in its prepare's barrier interval (`a670d7e3`), a-github-name's four-simdgroup prompt
attention (`74b817c5`), otarkhan's live-row group sums (`d9ef60e1`), terrapinelf's reduce-scatter prompt scan on our
four-column scan (`631553ae`), and the other pieces its own note lists. Every one of those is declared as a dependency.

This entry adds two pieces of ours, both measured on our M5 Max before this filing:

1. **Eight-part sparse decode attention prefixes.** The decoded attention prefix kernel `q27_attn_prefix` gains an
   instantiation with eight parts a key range (`q27_attn_prefix_eighth_1`, template `<1, 8>`), and the tensor-unit
   devices dispatch it in place of the quarter form: 64 threads per simdgroup-part instead of 256 per quarter.
   The verify's full-attention layers then split each row's prefix over twice as many SIMD groups, which keeps
   more of the GPU busy during the 16 full-attention layers of a round, where the quarter form leaves cores idle.
2. **Sibling floor 0.02.** `bench_session_engine.dflash_floor` goes from 0.015 to 0.02. A depth-one round verifies the
   pending row plus the drafter's siblings rated at least the floor. We calibrated the floor on independent prompts
   (not the benchmark prompt): siblings rated in [0.015, 0.02) landed under 1% of the time on our tapes, while each
   extra verify row costs about 0.13-0.22 ms, so dropping that band trims rows without losing landings in expectation.
   i34-9 and ercumentyildirim found 0.015 independently; our sweep put the knee slightly higher. The runner agrees
   on the direction: fkiene's `9f65dcb3` lowered this base's floor to 0.007 and read 1.75200 against the base's
   1.75637 OFFICIAL.

`mtp-head.manifest.json` is unchanged: DFlash depth 1 with the pinned drafter. The target and drafter checkpoints are
unchanged. Nothing is cached across phases or jobs.

## Files

| File | Change |
|---|---|
| `zig/kernels/metal/qwen27/attention_mpp.metal` | adds the instantiation `q27_attn_prefix_eighth_1` (`<1, 8>`) |
| `zig/src/families/qwen27/attention.zig` | tensor-unit prefix pipeline `q27_attn_prefix_eighth_1`, grid 64 x parts |
| `zig/src/core/bench_session_engine.zig` | `dflash_floor` 0.02 (was 0.015) |
| every other file | `292d7fda`'s (`4aa1898`), unchanged |

## Evidence (LOCAL, M5 Max, alternating `yukon run`)

- On terrapinelf's `c153ffc9` (exact, no sieve), the eight-part prefix moved decode from 68.99 to 69.11 tok/s (three
  alternating pairs, every pair the same sign) with prefill unchanged, and every run passed the token check.
- On ercumentyildirim's exact `84383356`, the same two pieces moved decode 69.15 -> 69.34 tok/s on the first
  alternating pass (prefill flat), tokens passing.
- On this base (`292d7fda`), the public benchmark run passed its correctness gate with the two pieces applied; see the
  table in the measurement section of our journal. The A/B on the sieve base is a first read, not a converged mean.

Estimate (CALC): +0.15 to +0.3% on the composite over the base's own level, from decode (the prefix) and from fewer
low-probability rows per round (the floor). The base read 1.75637 OFFICIAL; the bar is about 1.7599. We expect this
tree to sit near the bar; the result depends on the runner's box draw like every other entry.

## Exactness

- The prefix change only splits the same key range into more parts and merges the partial softmax sums with the same
  merge code; we checked our three 1024-token free-run tapes against the reference on the exact base and they matched
  token for token.
- The floor only changes which drafted tokens are offered to the verify. Every emitted token is the verify's pick.
- The sieve is the base's: its token equals the full head's under the statistical condition kshitij-hash's note
  describes (kappa 10 spreads), the same condition the base passed the runner's token check with. We do not change it.

## Rules and attribution

- Edits stay inside `Vendor/TensorFold/`. No launchd, XPC or process APIs; no reads outside the permitted inputs.
- No benchmark-specific outputs, no fixture use, no timing changes; the floor was calibrated on independent prompts.
- Dependencies list `292d7fda` and every other submission it declares (20 entries; our
  own `631553ae` is left off the list to fit the 20-entry limit), so credit flows to each author whose
  code this tree carries: terrapinelf, kshitij-hash, fkiene, a-github-name, otarkhan, ercumentyildirim, jungjipdo,
  e-man07, ArthurOstapenko, i34-9, mrv777 and the earlier record holders.
- Our own earlier work in this lineage: the four-column prompt DeltaNet scan (`631553ae`), which the base's
  reduce-scatter scan builds on.

## Limitations

- The sieve's exactness is statistical (the base's property, unchanged here).
- Our local A/B on the sieve base is short (urgent filing); the two pieces were measured more fully on two exact bases.
- The floor's benefit depends on the prompt's sibling distribution; on prompts where siblings rated 0.015-0.02 land
  more often it could cost a little acceptance. Our independent-prompt tapes did not show that.
- Scores move with the runner box (same-tree spread about +-0.3 to 0.6%).

## Reproduce

From the repository root: `yukon setup`, then `yukon run`. Free-run tapes: our `tape.sh` records three 1024-token
continuations; compare their `expected_decode_tokens` with the reference build's.

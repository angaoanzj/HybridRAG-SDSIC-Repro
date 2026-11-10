# HybridRAG case study: saved-result verification

## Recovered experiment code

The [pipeline directory](pipeline/README.md) now includes recovered retrieval, expansion, confirmation training, reader generation, scoring orchestration and analysis code. It also contains the exact nine-schedule MuSiQue runner used by the paper, verified against the recorded experiment hash; the earlier four-arm implementation is identified separately. The root verifier and its data remain unchanged, and the immutable revision cited by the paper remains accessible.

This companion supplies portable code and numeric evidence for the fixed-pipeline multi-hop case study. Run it with Python **3.9 or newer**; only the standard library is required. The default command performs no network access and does not load a model.

```sh
python3 verify.py --output derived
```

Inputs are resolved relative to `verify.py`, so the command also works when invoked from another directory. A successful run prints `PASS` and writes a detailed `verification.json`, an IDF comparison CSV, and the complete nine-schedule multi-round table CSV to the chosen output directory. Any failed check exits with a nonzero status. `MANIFEST.json` binds all supplied inputs and code by SHA-256.

## What the default command verifies

| Evidence | Included input | Recalculation |
|---|---|---|
| IDF feedback at K=2 | 65,676 projected question–method records for IDF8, S2 and original RRF | Strict-valid denominators, complete-support counts, IDF8–S2 paired wins/losses, and both IDF8–S2 and IDF8–RRF percentage-point differences |
| Paired significance tests | Saved discordance counts for 12 coverage and 48 retained reader-EM tests | Two-sided exact conditional McNemar p values and recorded Holm/Bonferroni adjustments |
| Historical MuSiQue multi-round study | 21,753 projected question–condition quality records; nine frozen schedules and four K values | All nine schedules at K=10,20,30,50, realized calls/returns, and all eight paired comparisons |
| MuSiQue sampling flow | Original IDs, six removed duplicate IDs with their retained representatives, and ordered role lists | 2,417 original rows → 2,411 representatives → 100 calibration/timing + 2,311 effectiveness IDs; disjointness and order-sensitive hashes |

The IDF8–S2 contrast shares the initial title-selection state, but the supplementary-query trigger can differ. It does not isolate individual feedback-word semantics under identical realized calls. The IDF8–RRF contrast additionally changes title expansion and cannot identify a feedback-only effect.

For the main title-coverage claims, Holm corrects four K=50 tests: Score2–Score1 and Score2–Order2 on two datasets. The retained reader analysis has 48 tests from six contrasts × four cutoffs × two datasets. Eight manual-cue comparisons were excluded after the original 56-test plan; the reanalysis conservatively retains a Bonferroni factor of 56. Reversal comparisons are descriptive and outside this family. `statistics_inputs.json` also records the auxiliary coverage families, preserving their original adjustment scopes.

## Optional reconstruction from external inputs

The default mode recomputes statistics from saved numeric records. It does **not** independently recover the underlying labels or retrieval outputs. Two optional modes go further when their original inputs are supplied:

```sh
python3 verify.py --raw-returns /path/to/raw_rankings_and_calls.jsonl.gz --output derived_raw
python3 verify.py --musique-data /path/to/musique_data_v1.0.zip --output derived_sampling
```

Both options can be supplied together. The MuSiQue option also accepts the extracted `musique_ans_v1.0_dev.jsonl`. It reads the official question text locally to repeat NFKC normalization, case folding, whitespace folding and first-occurrence retention. No question text is written to the reports.

The raw-return option requires the original compressed archive whose SHA-256 is:

```
f3b2ab45895d4039f3c2141583b2c2ed2e2f664f04e460060539180921e89552
```

It verifies all 21,753 final trajectories, 45,271 round rankings and 90,542 physical route calls by replaying weighted reciprocal-rank fusion, then recomputes supporting-paragraph coverage, recall and later-query acquisition/retention counts. The input hash is checked before replay. This is saved-return replay, not a new retrieval run or a latency benchmark.

## Scope and missing inputs

This is **not a complete end-to-end reproduction package**. It contains no original corpus text, questions, answers, model weights, embedding indexes, or 309 MB raw-return archive. The complete original selector/training/reader pipeline is not recreated by this verifier. Re-running those experiments requires their original implementations, corpus construction inputs, labels, frozen models and environment.

FT is reaggregated from saved quality records. Even optional raw-return replay does not reconstruct FT from original title labels; that earlier original-label audit is represented by a provenance receipt. FP and paragraph recall can be reconstructed from raw rankings using the saved supporting-document IDs. These distinctions prevent a saved-record check from being presented as a fresh end-to-end replication.

The 2,417-question multi-round study uses the original public-development order, including the six normalized duplicates and the 100 calibration/timing questions excluded from the primary effectiveness set. It is historical descriptive evidence, not an independent held-out extension of the 2,311-question primary cohort. Only four multi-round K values were measured.

This companion covers the listed feedback, statistics, sampling and multi-round evidence. It does not claim to independently reproduce every other table or figure in the manuscript.

## Files and provenance

- `verify.py`: one entry point for all supported checks.
- `data/idf_f2_records.jsonl.gz`: identifiers, validity flags and first-complete ranks for three policies, without text.
- `data/multiround_quality.jsonl.gz`: identifiers, numeric quality metrics, supporting-document IDs/ranks and per-round metrics; title strings are removed.
- `data/*expected.json`: independently saved aggregate targets used to check recomputed outcomes.
- `data/multiround_protocol.json`: frozen schedules, comparison definitions, IDs and numeric settings; machine paths omitted.
- `data/musique_sampling.json`: identifiers and normalization/partition metadata without question text.
- `data/statistics_inputs.json`: saved discordance counts and correction-family specifications.
- `data/original_completion.json`: hashes of the original experiment outputs, **not** hashes of this package's projected files.
- `data/prior_raw_check.json`: earlier raw-replay receipt and expected mechanism totals; carrying this receipt is not equivalent to executing optional replay.
- `data/source_provenance.json`: original filenames and hashes, plus the exact projection policy.

Original records remain distinct from the projected distribution. No personal machine paths, credentials, internal reviewer correspondence, or dataset text are included. No new license grant for upstream datasets or software is implied by this companion; obtain external inputs under their applicable terms.

## Validation performed for this distribution

On 2026-11-10, the default offline verifier passed 65,944 checks. An unchanged copy passed from an unrelated temporary directory, and an intentionally altered input was rejected by the manifest gate. The optional raw-return replay and official MuSiQue normalization were also executed against the original external inputs: 468,095 combined checks passed, including all 45,271 rankings and the six normalized duplicates. These are saved-record calculations, not new retrieval or model experiments.

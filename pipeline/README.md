# Experiment code snapshots

This directory adds the recovered research code alongside the saved-result verifier at the repository root. The directories preserve distinct historical protocols; they are not interchangeable versions of one experiment.

- confirmation/original_code and confirmation/scripts: the fixed-pipeline retrieval, expansion, training and three-corpus confirmation implementation recovered from the 2026-11-10 confirmation archive.
- retrieval_snapshot: the frozen implementation used by the K-range analysis.
- frozen_reader_conditions/retrieval_code: the fixed-index candidate and query-condition implementations in the original question-set rerun. reader_code contains the 19-condition, four-cutoff reader generation and official-scoring orchestration; reader_reference contains the recovered worker and input-preparation code. The third-party official evaluation scripts are identified by immutable source URLs and hashes in official_scoring_source_receipt.json and must be obtained from their original repositories. The older inherited score_reader.py is deliberately not presented as the official scoring implementation: its empty-token F1 convention differs from the official evaluator.
- audit_code: recovered figure/control/result audit scripts, including saved multiround replay and reader statistics.
- musique_nine_schedule: the original Table 8 experiment runner, its frozen retrieval/expansion modules, ID sequence, input manifests and synthetic control-flow check. The runner matches the exact SHA-256 recorded in the experiment protocol.
- musique_extension_snapshot: the associated original MuSiQue corpus preparation and evaluation implementation; prepare_musique.py checks the official source file hash and constructs the recorded public, label and corpus files. It is a historical snapshot and requires the external official dataset and index/model assets.
- figure_sources: recovered generation scripts and aggregate inputs for the query-contrast and confirmation figures, plus the editable source of the title-order plot. These historical scripts expect their original directory layout and a locally available Times New Roman font. The confirmation script also checks the original frozen protocol, which is not redistributed because it contains local paths. Their dependencies and generated output have not been revalidated in a clean environment.
- historical_four_arm_multiround: the earlier four-arm 200-question budget experiment. This is not the nine-schedule 2,417-record MuSiQue study reported in Table 8.
- legacy_repository: the retrieval and evaluation code from the prior private HybridRAG repository at commit cca5e3ac6515092316b5e7129b776401ea1bdeed, preserved separately to make its older provenance explicit.

SOURCE_MANIFEST.json records both the recovered source hash and the distributed hash for each Python file. Seven scripts had user-specific macOS home-directory paths replaced by /path/to/local/home. Some original Linux/AutoDL paths remain in the recovered code and must be configured for a new environment; these snapshots are not a turn-key, portable end-to-end release.

## Missing and external components

The original corpus, questions, labels, model weights, frozen indexes and raw result archive are external; obtain the datasets and models under their upstream terms. The required environment and its exact package versions have not been frozen or revalidated as a complete fresh-install workflow.

The existing root verifier can reaggregate selected saved records. These additions supply substantially more original implementation code, but do not establish that all figures and tables can be generated from scratch in a clean environment. No new retrieval, training, reader generation or latency experiment was run in preparing this release.

## Checks performed

All included Python files were parsed for syntax, source and published hashes were recorded, and code was scanned for common credential markers and exposed non-loopback IP addresses. The saved-result verifier is run separately; its passing result does not test full retrieval or model inference. Do not publish credentials, user-specific paths, dataset text or model files merely because a script references them.

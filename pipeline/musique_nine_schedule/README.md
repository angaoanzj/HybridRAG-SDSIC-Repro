# Nine-schedule MuSiQue experiment

This directory contains the recovered runner for the historical 2,417-record MuSiQue experiment reported in Table 8. The recorded experiment protocol identifies run_multiround.py by SHA-256 6c97c18e54c354a413183c573a5d902e6cbdf64e791dc1a9ae95bbf318d75709. The recovered file matches this hash. The four frozen retrieval and expansion modules match frozen_sources.json.

The historical experiment was completed and independently audited using saved results. This release does not claim a new scientific run. The separate historical_four_arm_multiround directory describes an earlier four-arm experiment and is not the same study.

To run the synthetic control-flow test without a model or network, execute: python3 selfcheck_control_flow.py. Its nine-arm, stopping and repetition checks use fabricated records only and are not research measurements.

The provided run_on_autodl.sh is the historical Linux entry point. It takes three paths: an existing study root containing the original MuSiQue public questions, labels, corpus and frozen index; the indexed BGE model directory; and a new output directory. Before real execution, inspect run_multiround.py and its input checks, obtain authorized external data and model assets, and prepare the matching software environment. The source verifies recorded input and model hashes; using another corpus, split, model or index is not the original experiment. The archived environment required Python 3.10+, NumPy, SciPy, PyTorch, Transformers and Sentence-Transformers. Exact dependency versions and a clean-install end-to-end run have not yet been verified.

No original corpus text, question text, answer labels, embedding index or model weights are bundled here. question_ids.json contains the frozen ID sequence. The completed raw return archive remains external. Root-level verify.py reaggregates projected records; when supplied the optional raw archive, it additionally replays saved returns. Neither check itself executes fresh retrieval or inference.

Frozen source hashes are preserved in frozen_sources.json. original_data_manifest.json records the original corpus and input hashes and normalization rules. The raw-source preparation documentation was retained locally for provenance; its outdated preparation-only status and host-specific SSH command are not used as the current release description.

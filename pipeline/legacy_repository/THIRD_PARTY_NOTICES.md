# Data and model attribution

## HotpotQA

HotpotQA was introduced by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning in *HotpotQA, A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018. The official project distributes HotpotQA under Creative Commons Attribution-ShareAlike 4.0. [Official project](https://hotpotqa.github.io/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The retained input used `hotpotqa/hotpot_qa`, distractor validation. Its original download did not record a dataset revision. This repository preserves the exact retained content through compressed corpus files, query annotations, ordered IDs, and SHA-256 fingerprints. Corpus preparation joins context sentences, deduplicates title-and-text versions, and assigns deterministic document IDs. Query preparation selects the first 300 retained validation questions, preserving their order. The HotpotQA-derived data snapshot is distributed under CC BY-SA 4.0.

## 2WikiMultiHopQA

2WikiMultiHopQA was introduced by Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa in *Constructing a Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*, COLING 2020. The official repository provides an Apache-2.0 license. [Official repository](https://github.com/Alab-NII/2wikimultihop), [upstream license](https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE).

The retained validation Arrow file was supplied with the original HybridRAG attachment. The original repackaging source and dataset revision were not recorded. This repository identifies the exact experimental inputs with content hashes and ordered query IDs. Corpus preparation joins context sentences, deduplicates title-and-text versions, and assigns deterministic document IDs. Query preparation retains validation positions 0–299. Upstream dataset attribution and applicable Wikipedia content attribution are retained through the dataset names and original passage titles.

## BGE

The model is `BAAI/bge-base-en-v1.5` from the Beijing Academy of Artificial Intelligence, pinned to revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. Its model card identifies the MIT license. Model files are retrieved from that revision and verified using `provenance/model.json`. [Pinned model card](https://huggingface.co/BAAI/bge-base-en-v1.5/blob/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md).

## Manuscript template

The LaTeX class and bibliography style in `paper/` are IEEEtran files. Their original copyright, license, and distribution notices are included in the files.

<div align="center">

# MVMA-GCN: Multi-view Multi-layer Attention Graph Convolutional Networks

<a href="https://doi.org/10.1016/j.engappai.2023.106717"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1016%2Fj.engappai.2023.106717-blue?style=flat-square"></a>
<a href="https://pengyu-zhang.github.io/pdf/MVMA-GCN.pdf"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square"></a>
<a href="https://github.com/pengyu-zhang/MVMA-GCN/releases"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-GitHub%20Release-9cf?style=flat-square"></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square"></a>

</div>

Official implementation of **MVMA-GCN: Multi-view multi-layer attention graph
convolutional networks** by Pengyu Zhang, Yong Zhang, Jingcheng Wang and
Baocai Yin, published in *Engineering Applications of Artificial
Intelligence* 126 (2023) 106717.

MVMA-GCN performs semi-supervised node classification on multi-view graphs —
graphs whose nodes are connected by several types of relations (e.g. papers
linked by co-author, co-subject, or co-keyword).

## Overview

Each relation view and a k-NN graph built from node features are processed by
view-specific graph convolutions, while a shared-weight convolution extracts
information common to all views; a Hilbert–Schmidt independence criterion
keeps the specific and common representations distinct.

A multi-layer attention mechanism then weighs neighbors within each view and
fuses the per-view embeddings adaptively, and an autoencoder branch preserves
feature-level structure through a reconstruction objective. The fused
embedding is classified by a linear layer trained with cross-entropy on the
few labeled nodes.

## Repository structure

```text
├── configs/          # experiment configurations (baseline / default / smoke)
├── data/             # dataset documentation (see data/README.md)
├── scripts/          # setup, data, training, evaluation entry points
├── src/              # implementation (Python package)
└── requirements.txt
```

## Installation

- Python ≥ 3.10 (tested on 3.13)
- PyTorch ≥ 2.0 (tested on 2.13, CUDA 13.2) — a GPU is used automatically
  when available; CPU works too
- remaining dependencies in `requirements.txt`

```bash
# CUDA build (set CUDA_TAG=cpu for a CPU-only install)
bash scripts/setup_env.sh
```

## Data

```bash
bash scripts/prepare_data.sh
```

This downloads the seven datasets (ACM, DBLP, IMDB, BlogCatalog, Flickr,
Citeseer, UAI2010; ~30 MB) from the GitHub Release attached to this
repository, verifies the checksum, and converts them into a fast binary
layout under `data/processed/`. See [data/README.md](data/README.md) for
details, provenance and manual download.

## 🚀 Quick start

```bash
# end-to-end check of the whole pipeline (a few minutes)
bash scripts/smoke_test.sh
```

## Training and evaluation

```bash
# train one dataset / label rate (20, 40 or 60 labeled nodes per class)
bash scripts/train.sh acm 20

# full run: all datasets and label rates, then evaluation
bash scripts/run_all.sh

# evaluate a finished run from its checkpoint
bash scripts/evaluate.sh outputs/default-acm-l20
```

Python entry points are also available directly:
`python -m src.train --config configs/default.yaml --dataset acm --labelrate 20`.

| Config | Purpose |
|--------|---------|
| `configs/baseline.yaml` | Plain baseline without the paper's contributions, for controlled comparison. |
| `configs/default.yaml` | Recommended configuration (best results). |
| `configs/smoke.yaml` | Minimal settings for a fast end-to-end pipeline check. |

## 📊 Results

Test accuracy / macro-F1 (%), mean ± std over 3 seeds. L/C is the number of
labeled training nodes per class.

| Dataset | L/C | Accuracy | Macro-F1 |
|---------|-----|----------------|----------------|
| ACM     | 20  | 91.73 ± 0.52 | 91.65 ± 0.56 |
| ACM     | 40  | 92.40 ± 0.57 | 92.34 ± 0.61 |
| ACM     | 60  | 92.73 ± 0.52 | 92.70 ± 0.54 |
| DBLP    | 20  | 89.67 ± 0.05 | 89.01 ± 0.02 |
| DBLP    | 40  | 91.10 ± 0.41 | 90.72 ± 0.41 |
| DBLP    | 60  | 91.57 ± 0.09 | 91.13 ± 0.10 |
| IMDB    | 20  | 54.20 ± 1.08 | 53.20 ± 0.73 |
| IMDB    | 40  | 61.20 ± 0.29 | 59.66 ± 0.33 |
| IMDB    | 60  | 61.93 ± 0.54 | 60.16 ± 0.37 |
| BlogCatalog | 20 | 81.20 ± 0.28 | 80.68 ± 0.53 |
| BlogCatalog | 40 | 85.30 ± 0.24 | 84.92 ± 0.22 |
| BlogCatalog | 60 | 86.50 ± 0.99 | 86.01 ± 0.98 |
| Flickr  | 20  | 74.80 ± 0.33 | 74.02 ± 0.49 |
| Flickr  | 40  | 80.77 ± 1.40 | 80.48 ± 1.44 |
| Flickr  | 60  | 82.93 ± 0.95 | 82.81 ± 0.99 |
| Citeseer | 20 | 72.47 ± 0.12 | 68.80 ± 0.12 |
| Citeseer | 40 | 73.60 ± 0.22 | 70.03 ± 0.09 |
| Citeseer | 60 | 75.77 ± 0.12 | 72.80 ± 0.19 |
| UAI2010 | 20  | 70.87 ± 1.16 | 56.38 ± 1.34 |
| UAI2010 | 40  | 73.47 ± 0.88 | 61.81 ± 0.15 |
| UAI2010 | 60  | 75.00 ± 0.71 | 64.52 ± 2.01 |

DBLP and IMDB use regenerated train/test splits (see
[data/README.md](data/README.md)), so their numbers are internally
consistent but not directly comparable to the paper's.

## 📝 Citation

```bibtex
@article{zhang2023mvmagcn,
  title   = {MVMA-GCN: Multi-view multi-layer attention graph convolutional networks},
  author  = {Zhang, Pengyu and Zhang, Yong and Wang, Jingcheng and Yin, Baocai},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {126},
  pages   = {106717},
  year    = {2023},
  doi     = {10.1016/j.engappai.2023.106717}
}
```

## 🙏 Acknowledgments & License

The datasets follow the preprocessing of
[AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN) and the HAN line of work
(Wang et al., KDD 2020); see [data/README.md](data/README.md) for full
provenance and attribution.

This repository is released under the [MIT License](LICENSE).

---

Maintained by [Pengyu Zhang](https://pengyu-zhang.github.io/).

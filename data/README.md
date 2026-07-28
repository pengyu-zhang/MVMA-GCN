# Datasets

Seven benchmarks in the preprocessed format introduced by
[AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN) / HAN: node features are
bag-of-words vectors, relation views are edge lists (meta-paths for the
heterogeneous datasets, one structure graph for the others), and a k-NN
feature graph is precomputed for k = 2..9.

| Dataset | Nodes | Features | Classes | Relation views |
|---------|-------|----------|---------|----------------|
| ACM         | 3025  | 1870  | 3  | PAP, PLP (co-subject), PMP |
| DBLP        | 4057  | 334   | 4  | APA, APCPA, APTPA |
| IMDB        | 4780  | 1232  | 3  | MAM, MDM, MYM |
| BlogCatalog | 5196  | 8189  | 6  | social graph |
| Flickr      | 7575  | 12047 | 9  | social graph |
| Citeseer    | 3327  | 3703  | 6  | citation graph |
| UAI2010     | 3067  | 4973  | 19 | web graph |

(The paper's Table 1 lists 1732 features for IMDB; the shipped data has 1232.
Citeseer and UAI2010 are not evaluated in the MVMA-GCN paper; they follow the
AM-GCN benchmark setup and are included as extras.)

## Automatic download

```bash
bash scripts/prepare_data.sh
```

This downloads `mvma-gcn-data.tar.gz` from this repository's GitHub Release,
verifies its checksum, extracts it to `data/raw/`, and builds the fast
binary layout in `data/processed/`.

## Manual download

If the automatic download fails, download the tarball from
<https://github.com/pengyu-zhang/MVMA-GCN/releases> and either place it
anywhere and run

```bash
bash scripts/prepare_data.sh --tarball /path/to/mvma-gcn-data.tar.gz
```

or extract it yourself so that `data/raw/acm/`, `data/raw/DBLP/`,
`data/raw/IMDB/`, `data/raw/BlogCatalog/`, `data/raw/flickr/`,
`data/raw/citeseer/` and `data/raw/uai/` exist, then run
`python -m src.prepare`.

## Train/test splits

Each dataset uses the paper's protocol: nested training sets with exactly
20/40/60 labeled nodes per class (train20 ⊂ train40 ⊂ train60) and one
1000-node test set shared by all three label rates, disjoint from every
training set (this is the structure of the shipped ACM files).
`src/prepare.py` additionally samples a 500-node validation set (`val.txt`,
disjoint from all train/test sets) used only for hyperparameter tuning
decisions (`training.model_selection: val`) — never for the reported
numbers.

- **ACM, BlogCatalog, Flickr, Citeseer, UAI2010**: the shipped splits are
  valid (verified: exact per-class counts, shared test set, no overlap) and
  are used as-is.
- **DBLP / IMDB**: the shipped split files were byte-identical copies of the
  ACM splits and therefore invalid for these datasets. `src/prepare.py`
  regenerates them with a fixed seed following the same protocol, so results
  on DBLP/IMDB are internally consistent but not directly comparable to the
  paper's numbers.
- **IMDB label caveat**: 1493 of 4780 movies have no genre label in the
  source data and carry a filler class-0 label in `IMDB.label`
  ([imdb_unlabeled_nodes.txt](imdb_unlabeled_nodes.txt)). They remain in the
  graph for message passing but are excluded from train/test sampling, so
  supervision and evaluation use genuine labels only.

## Attribution

The underlying data originates from ACM, DBLP (<https://dblp.uni-trier.de/>),
IMDB, BlogCatalog, Flickr, Citeseer and UAI2010; the preprocessing follows
the AM-GCN/HAN line of work (Wang et al., KDD 2020). If you use the data,
please cite the MVMA-GCN paper (see the repository README) and AM-GCN.

Nothing under `data/raw/`, `data/processed/` or `outputs/` is tracked by git.

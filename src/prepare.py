"""Convert the raw shipped datasets into the canonical processed layout.

- converts text features/labels/edge lists to .npy (much faster to load)
- copies the shipped ACM train/test splits
- REGENERATES the DBLP and IMDB splits: the shipped split files are
  byte-identical copies of the ACM splits and therefore invalid for those
  datasets (see data/README.md). Regeneration follows the paper's protocol:
  20/40/60 labeled nodes per class for training and 1000 disjoint test
  nodes, with a fixed seed.
- converts the shipped k-NN graphs (knn/c2..c9.txt) to .npy, regenerating
  any missing k with src.knn

Usage:
    python -m src.prepare [--raw data/raw] [--out data/processed] [--seed 42]
"""

import argparse
import os
import shutil

import numpy as np

from .knn import build_knn_edges

# dataset -> (raw dir name, file prefix, {view: edge file})
RAW_SPECS = {
    "acm": (
        "acm",
        "acm",
        {"pap": "acm_PAP.edge", "plp": "acm_PLP.edge", "pmp": "acm_PMP.edge"},
    ),
    "dblp": (
        "DBLP",
        "DBLP",
        {"apa": "DBLP_APA.edge", "apcpa": "DBLP_APCPA.edge", "aptpa": "DBLP_APTPA.edge"},
    ),
    "imdb": (
        "IMDB",
        "IMDB",
        {"mam": "IMDB_MAM.edge", "mdm": "IMDB_MDM.edge", "mym": "IMDB_MYM.edge"},
    ),
    "blogcatalog": ("BlogCatalog", "BlogCatalog", {"struct": "BlogCatalog.edge"}),
    "flickr": ("flickr", "flickr", {"struct": "flickr.edge"}),
    "citeseer": ("citeseer", "citeseer", {"struct": "citeseer.edge"}),
    "uai": ("uai", "uai", {"struct": "uai.edge"}),
}

LABEL_RATES = (20, 40, 60)
TEST_SIZE = 1000
VAL_SIZE = 500  # tuning decisions use this held-out set, never the test set
KNN_RANGE = range(2, 10)

# ACM splits are shipped and valid; DBLP/IMDB splits must be regenerated.
REGENERATE_SPLITS = {"dblp", "imdb"}

# Nodes excluded from train/test sampling. 1493 IMDB movies have no genre in
# the source imdb5k.mat and carry a filler class-0 label in IMDB.label; they
# stay in the graph but are never used for supervision or evaluation
# (see data/README.md).
EXCLUDE_FROM_SPLITS = {"imdb": "data/imdb_unlabeled_nodes.txt"}


def generate_splits(labels, label_rates, test_size, rng, exclude=None):
    """Split protocol of the shipped ACM files (verified): nested training sets
    with exactly L labeled nodes per class for each label rate L, plus ONE
    test set of `test_size` nodes shared by all label rates and disjoint from
    every training set. Nodes in `exclude` are never sampled."""
    labels = labels.copy()
    if exclude is not None:
        labels[exclude] = -1
    max_rate = max(label_rates)
    pool = {}
    for c in np.unique(labels):
        if c < 0:
            continue
        nodes = np.flatnonzero(labels == c)
        pool[c] = rng.permutation(nodes)[:max_rate]
    trains = {
        lr: np.sort(np.concatenate([pool[c][:lr] for c in pool])) for lr in label_rates
    }
    remaining = np.setdiff1d(np.flatnonzero(labels >= 0), trains[max_rate])
    test = np.sort(rng.choice(remaining, size=test_size, replace=False))
    return trains, test


def prepare_dataset(name, raw_root, out_root, seed):
    raw_dir_name, prefix, views = RAW_SPECS[name]
    raw = os.path.join(raw_root, raw_dir_name)
    out = os.path.join(out_root, name)
    os.makedirs(os.path.join(out, "knn"), exist_ok=True)
    print(f"[{name}] preparing from {raw}")

    features = np.loadtxt(os.path.join(raw, f"{prefix}.feature"), dtype=np.float32)
    labels = np.loadtxt(os.path.join(raw, f"{prefix}.label"), dtype=np.int64)
    np.save(os.path.join(out, "features.npy"), features)
    np.save(os.path.join(out, "labels.npy"), labels)
    print(f"[{name}] features {features.shape}, {labels.max() + 1} classes")

    for view, fname in views.items():
        edges = np.genfromtxt(os.path.join(raw, fname), dtype=np.int32)
        np.save(os.path.join(out, f"edges_{view}.npy"), edges)
        print(f"[{name}] view {view}: {edges.shape[0]} edges")

    for k in KNN_RANGE:
        src_txt = os.path.join(raw, "knn", f"c{k}.txt")
        dst = os.path.join(out, "knn", f"c{k}.npy")
        if os.path.exists(src_txt):
            edges = np.genfromtxt(src_txt, dtype=np.int32)
        else:
            print(f"[{name}] knn c{k}.txt missing - regenerating")
            edges = build_knn_edges(features, k)
        np.save(dst, edges)

    exclude = None
    if name in EXCLUDE_FROM_SPLITS:
        exclude = np.loadtxt(EXCLUDE_FROM_SPLITS[name], dtype=np.int64)
        print(f"[{name}] excluding {len(exclude)} unlabeled nodes from splits")

    if name in REGENERATE_SPLITS:
        print(f"[{name}] regenerating splits (shipped ones were ACM copies)")
        rng = np.random.default_rng(seed)
        trains, test = generate_splits(labels, LABEL_RATES, TEST_SIZE, rng, exclude)
        for lr in LABEL_RATES:
            np.savetxt(os.path.join(out, f"train{lr}.txt"), trains[lr], fmt="%d")
            np.savetxt(os.path.join(out, f"test{lr}.txt"), test, fmt="%d")
        train_max = trains[max(LABEL_RATES)]
        test_all = test
    else:
        for lr in LABEL_RATES:
            for part in ("train", "test"):
                shutil.copyfile(
                    os.path.join(raw, f"{part}{lr}.txt"),
                    os.path.join(out, f"{part}{lr}.txt"),
                )
        train_max = np.loadtxt(
            os.path.join(raw, f"train{max(LABEL_RATES)}.txt"), dtype=np.int64
        )
        test_all = np.loadtxt(
            os.path.join(raw, f"test{max(LABEL_RATES)}.txt"), dtype=np.int64
        )

    # one shared validation set for tuning decisions, disjoint from every
    # train set and the test set (labels-eligible pool only)
    eligible = np.arange(len(labels))
    if exclude is not None:
        eligible = np.setdiff1d(eligible, exclude)
    eligible = np.setdiff1d(eligible, np.concatenate([train_max, test_all]))
    rng_val = np.random.default_rng(seed + 1)
    val = np.sort(rng_val.choice(eligible, size=VAL_SIZE, replace=False))
    np.savetxt(os.path.join(out, "val.txt"), val, fmt="%d")
    print(f"[{name}] done -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MVMA-GCN datasets")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", nargs="+", default=list(RAW_SPECS), choices=list(RAW_SPECS)
    )
    args = parser.parse_args()
    for name in args.datasets:
        prepare_dataset(name, args.raw, args.out, args.seed)


if __name__ == "__main__":
    main()

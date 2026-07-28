"""Dataset registry and loading of the canonical processed layout.

The processed layout (produced by ``src.prepare``) is, per dataset::

    data/processed/<name>/
        features.npy        # float32 [n, d]
        labels.npy          # int64 [n]
        edges_<view>.npy    # int32 [m, 2] per relation view
        knn/c<k>.npy        # int32 [m, 2] k-NN feature graph edge lists
        train<L>.txt        # training node ids, L labeled nodes per class
        test<L>.txt         # test node ids (1000 nodes)
"""

import os

import numpy as np
import torch

from .utils import symmetrize_and_normalize

# Relation views per dataset (order matters and is fixed). Single-view
# datasets have one structure graph; the kNN feature graph is always added
# as an extra view by the model, matching the AM-GCN setting.
DATASETS = {
    "acm": ["pap", "plp", "pmp"],
    "dblp": ["apa", "apcpa", "aptpa"],
    "imdb": ["mam", "mdm", "mym"],
    "blogcatalog": ["struct"],
    "flickr": ["struct"],
    "citeseer": ["struct"],
    "uai": ["struct"],
}

# The shipped DBLP/IMDB split files are invalid (copies of the ACM splits);
# src.prepare regenerates them (see data/README.md).
LABEL_RATES = (20, 40, 60)
TEST_SIZE = 1000


def processed_dir(cfg):
    return os.path.join(cfg["data"]["processed_dir"], cfg["dataset"])


def load_dataset(cfg, device):
    """Load features, labels, splits and all adjacency matrices onto ``device``."""
    name = cfg["dataset"]
    base = processed_dir(cfg)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"{base} not found - run scripts/prepare_data.sh first"
        )

    features = torch.from_numpy(np.load(os.path.join(base, "features.npy")))
    labels = torch.from_numpy(np.load(os.path.join(base, "labels.npy")))
    n = features.shape[0]

    labelrate = cfg["labelrate"]
    idx_train = torch.from_numpy(
        np.loadtxt(os.path.join(base, f"train{labelrate}.txt"), dtype=np.int64)
    )
    idx_test = torch.from_numpy(
        np.loadtxt(os.path.join(base, f"test{labelrate}.txt"), dtype=np.int64)
    )
    val_path = os.path.join(base, "val.txt")
    idx_val = (
        torch.from_numpy(np.loadtxt(val_path, dtype=np.int64))
        if os.path.exists(val_path)
        else None
    )

    view_adjs = []
    view_edges = []
    for view in DATASETS[name]:
        edges = np.load(os.path.join(base, f"edges_{view}.npy"))
        view_edges.append(edges)
        view_adjs.append(symmetrize_and_normalize(edges, n))

    k = cfg["model"]["k"]
    knn_edges = np.load(os.path.join(base, "knn", f"c{k}.npy"))
    knn_adj = symmetrize_and_normalize(knn_edges, n)

    data = {
        "features": features.to(device),
        "labels": labels.to(device),
        "idx_train": idx_train.to(device),
        "idx_test": idx_test.to(device),
        "idx_val": idx_val.to(device) if idx_val is not None else None,
        "view_adjs": [a.to(device) for a in view_adjs],
        "knn_adj": knn_adj.to(device),
        "view_edges": view_edges,
        "knn_edges": knn_edges,
        "n_nodes": n,
        "n_feats": features.shape[1],
        "n_classes": int(labels.max().item()) + 1,
        "view_names": DATASETS[name],
    }
    return data


def build_sum_graph(data, device):
    """Union graph A_c = A_k + sum(A_m) used by the paper's multi-view convolution."""
    n = data["n_nodes"]
    all_edges = np.vstack([data["knn_edges"]] + data["view_edges"])
    return symmetrize_and_normalize(all_edges, n).to(device)

"""Shared helpers: seeding, device selection, sparse conversion, run output."""

import json
import os
import random

import numpy as np
import scipy.sparse as sp
import torch


def set_seed(seed):
    """Fix all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name="auto"):
    """Resolve a device string ('auto' picks CUDA when available)."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"device: {device}")
    return device


def normalize_adj(mx):
    """Row-normalize a scipy sparse matrix (D^-1 A)."""
    rowsum = np.asarray(mx.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum)
    np.divide(1.0, rowsum, out=r_inv, where=rowsum != 0)
    return sp.diags(r_inv).dot(mx)


def sparse_mx_to_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse COO tensor."""
    mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(
        indices, values, torch.Size(mx.shape), check_invariants=False
    ).coalesce()


def symmetrize_and_normalize(edges, n):
    """Build a symmetric row-normalized adjacency (with self-loops) from an edge list."""
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(n, n),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(n))
    return sparse_mx_to_torch(adj)


class RunWriter:
    """Append-only writer so results survive interruption."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def log_epoch(self, record):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def write_json(self, name, obj):
        path = os.path.join(self.run_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return path

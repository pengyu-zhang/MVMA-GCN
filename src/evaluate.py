"""Evaluate a trained MVMA-GCN checkpoint on the test split.

Usage:
    python -m src.evaluate --run outputs/<run_dir>
"""

import argparse
import json
import os

import torch

from .data import build_sum_graph, load_dataset
from .layers import edges_to_attention_index
from .models import MVMAGCN
from .train import evaluate_split
from .utils import resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run")
    parser.add_argument("--run", required=True, help="run directory containing best.pt")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    data = load_dataset(cfg, device)

    view_att_idx = None
    if cfg["model"]["single_view_attention"]["enabled"]:
        view_att_idx = [
            edges_to_attention_index(e, data["n_nodes"], device)
            for e in data["view_edges"]
        ]
    sum_adj = None
    if cfg["model"]["common_conv"] == "sum_graph":
        sum_adj = build_sum_graph(data, device)

    model = MVMAGCN(
        cfg, data["n_feats"], data["n_classes"], view_attention_indices=view_att_idx
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        log_probs, _ = model(
            data["features"], data["view_adjs"], data["knn_adj"], sum_adj
        )
        acc, f1 = evaluate_split(log_probs, data["labels"], data["idx_test"])

    result = {
        "dataset": cfg["dataset"],
        "labelrate": cfg["labelrate"],
        "checkpoint_epoch": ckpt["epoch"],
        "acc_test": round(acc, 6),
        "f1_test": round(f1, 6),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

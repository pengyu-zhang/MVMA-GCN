"""Train MVMA-GCN.

Usage:
    python -m src.train --config configs/default.yaml --dataset acm --labelrate 20
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from .config import load_config
from .data import DATASETS, build_sum_graph, load_dataset
from .layers import edges_to_attention_index
from .losses import consistency_loss, hsic_loss, reconstruction_loss
from .models import MVMAGCN
from .utils import RunWriter, resolve_device, set_seed


def accuracy(log_probs, labels):
    preds = log_probs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate_split(log_probs, labels, idx):
    acc = accuracy(log_probs[idx], labels[idx])
    preds = log_probs[idx].argmax(dim=1).cpu().numpy()
    f1 = f1_score(labels[idx].cpu().numpy(), preds, average="macro")
    return acc, float(f1)


def compute_losses(cfg, log_probs, out, data):
    lcfg = cfg["loss"]
    loss_class = F.nll_loss(log_probs[data["idx_train"]], data["labels"][data["idx_train"]])

    specific_all = out["specific"] + [out["emb_k"]]
    if lcfg["hsic"]["pairs"] == "view_common":
        common_arg = out["common_list"]
    else:
        common_arg = out["common"]
    loss_hsic = hsic_loss(
        specific_all,
        common_arg,
        pairs=lcfg["hsic"]["pairs"],
        normalized=lcfg["hsic"]["normalized"],
    )

    if lcfg["consistency"]["target"] == "common_outputs":
        cons_embs = out["common_list"]
    else:  # embeddings (paper Eq. 15-16)
        cons_embs = specific_all + [out["common"]]
    loss_cons = consistency_loss(cons_embs)

    loss_rec = reconstruction_loss(out["x_bar"], data["features"])

    total = (
        loss_class
        + lcfg["consistency"]["weight"] * loss_cons
        + lcfg["hsic"]["weight"] * loss_hsic
        + lcfg["reconstruction"]["weight"] * loss_rec
    )
    parts = {
        "loss_class": loss_class.item(),
        "loss_consistency": loss_cons.item(),
        "loss_hsic": loss_hsic.item(),
        "loss_reconstruction": loss_rec.item(),
        "loss_total": total.item(),
    }
    return total, parts


def run(cfg, run_dir=None, quiet=False):
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    data = load_dataset(cfg, device)
    cfg["model"]["n_views"] = len(data["view_adjs"])

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
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    if run_dir is None:
        run_dir = os.path.join(
            "outputs", f"{cfg['dataset']}-l{cfg['labelrate']}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
    writer = RunWriter(run_dir)
    writer.write_json("config_snapshot.json", cfg)

    # model selection: "test" reproduces the paper's protocol (best test epoch);
    # "val" selects on the held-out validation set (used for all tuning)
    selection = cfg["training"].get("model_selection", "test")
    if selection == "val" and data["idx_val"] is None:
        raise ValueError("model_selection=val but no val.txt - rerun src.prepare")

    # optional autoencoder pretraining (reconstruction only) before the
    # joint phase, so injected hidden states are meaningful from epoch 0
    pretrain_epochs = cfg["training"].get("ae_pretrain_epochs", 0)
    if pretrain_epochs:
        ae_opt = optim.Adam(model.ae.parameters(), lr=cfg["training"]["lr"])
        model.train()
        for pe in range(pretrain_epochs):
            ae_opt.zero_grad()
            x_bar, _, _ = model.ae(data["features"])
            loss_p = F.mse_loss(x_bar, data["features"])
            loss_p.backward()
            ae_opt.step()
        if not quiet:
            print(f"ae pretrained {pretrain_epochs} epochs, final rec {loss_p.item():.5f}")

    best = {"acc_test": 0.0, "f1_test": 0.0, "acc_val": 0.0, "f1_val": 0.0,
            "select": -1.0, "epoch": -1}
    t0 = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        log_probs, out = model(
            data["features"], data["view_adjs"], data["knn_adj"], sum_adj
        )
        loss, parts = compute_losses(cfg, log_probs, out, data)
        acc_train = accuracy(log_probs[data["idx_train"]], data["labels"][data["idx_train"]])
        loss.backward()
        optimizer.step()

        # Metrics are tracked every epoch; the best epoch by the selection
        # metric is reported (paper protocol: selection = test accuracy).
        model.eval()
        with torch.no_grad():
            log_probs, out = model(
                data["features"], data["view_adjs"], data["knn_adj"], sum_adj
            )
            acc_test, f1_test = evaluate_split(log_probs, data["labels"], data["idx_test"])
            acc_val = f1_val = None
            if data["idx_val"] is not None:
                acc_val, f1_val = evaluate_split(
                    log_probs, data["labels"], data["idx_val"]
                )

        record = {
            "epoch": epoch,
            "acc_train": round(acc_train, 6),
            "acc_test": round(acc_test, 6),
            "f1_test": round(f1_test, 6),
            **({"acc_val": round(acc_val, 6), "f1_val": round(f1_val, 6)}
               if acc_val is not None else {}),
            **{k: round(v, 8) for k, v in parts.items()},
        }
        writer.log_epoch(record)
        if not quiet:
            print(
                f"e:{epoch} ltr:{parts['loss_total']:.4f} atr:{acc_train:.4f} "
                f"ate:{acc_test:.4f} f1te:{f1_test:.4f}"
            )

        select_metric = acc_val if selection == "val" else acc_test
        if select_metric >= best["select"]:
            best = {
                "acc_test": acc_test, "f1_test": f1_test,
                "acc_val": acc_val or 0.0, "f1_val": f1_val or 0.0,
                "select": select_metric, "epoch": epoch,
            }
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "config": cfg},
                os.path.join(run_dir, "best.pt"),
            )

    summary = {
        "dataset": cfg["dataset"],
        "labelrate": cfg["labelrate"],
        "config": cfg["config_path"],
        "seed": cfg["seed"],
        "model_selection": selection,
        "best_epoch": best["epoch"],
        "acc_test": round(best["acc_test"], 6),
        "f1_test": round(best["f1_test"], 6),
        "acc_val": round(best["acc_val"], 6),
        "f1_val": round(best["f1_val"], 6),
        "runtime_sec": round(time.time() - t0, 2),
    }
    writer.write_json("summary.json", summary)
    print(
        f"epoch:{best['epoch']} acc_max: {best['acc_test']:.4f} "
        f"f1_max: {best['f1_test']:.4f} ({summary['runtime_sec']}s) -> {run_dir}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train MVMA-GCN")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--labelrate", type=int, required=True, choices=[20, 40, 60])
    parser.add_argument("--output", default=None, help="run output directory")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.dataset)
    cfg["labelrate"] = args.labelrate
    if args.seed is not None:
        cfg["seed"] = args.seed
    run(cfg, run_dir=args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()

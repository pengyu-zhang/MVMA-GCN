"""Loss components: HSIC disparity (L_s), consistency (L_m), reconstruction (L_res)."""

import itertools

import torch
import torch.nn.functional as F


# Above this node count the O(n^2)-memory formulations are replaced by
# mathematically equivalent O(n*d^2) ones (identical up to float rounding);
# below it the direct O(n^2) computation is used.
LARGE_N = 10000


def hsic(emb1, emb2, normalized=False):
    """Hilbert-Schmidt Independence Criterion with a linear kernel (paper Eq. 13).

    ``normalized=False`` (the baseline configuration) omits the (n-1)^-2
    factor of Eq. 13.
    """
    n = emb1.shape[0]
    if n > LARGE_N:
        # tr(R K1 R K2) = ||(R Z1)^T (R Z2)||_F^2 with R Z = Z - mean(Z);
        # avoids materializing any n x n matrix
        z1 = emb1 - emb1.mean(dim=0, keepdim=True)
        z2 = emb2 - emb2.mean(dim=0, keepdim=True)
        value = ((z1.t() @ z2) ** 2).sum()
    else:
        r = torch.eye(n, device=emb1.device) - (1.0 / n) * torch.ones(
            n, n, device=emb1.device
        )
        k1 = emb1 @ emb1.t()
        k2 = emb2 @ emb2.t()
        value = torch.trace((r @ k1) @ (r @ k2))
    if normalized:
        value = value / float((n - 1) ** 2)
    return value


def hsic_loss(specific_embs, common_embs, pairs="view_common", normalized=False):
    """Disparity loss L_s.

    ``view_common``: mean HSIC between each view-specific embedding and the
    common embedding computed on the same graph (baseline configuration).
    ``all_pairs``: mean HSIC over all pairs among the specific embeddings and
    the (single) common embedding (closer to paper Eq. 14).
    """
    if pairs == "view_common":
        if len(common_embs) == 1:
            # sum_graph mode: pair every specific embedding with the single
            # common embedding
            common_embs = list(common_embs) * len(specific_embs)
        assert len(specific_embs) == len(common_embs)
        values = [
            hsic(s, c, normalized) for s, c in zip(specific_embs, common_embs)
        ]
    elif pairs == "all_pairs":
        common = common_embs if isinstance(common_embs, torch.Tensor) else common_embs[0]
        embs = list(specific_embs) + [common]
        values = [
            hsic(a, b, normalized) for a, b in itertools.combinations(embs, 2)
        ]
    else:
        raise ValueError(f"unknown hsic pairs mode: {pairs}")
    return torch.stack(values).mean()


def consistency_loss(embs):
    """Consistency loss L_m (paper Eq. 15-16): pairwise squared differences of
    the L2-normalized similarity (Gram) matrices of the given embeddings."""
    if len(embs) < 2:
        # single common output (sum_graph mode with target=common_outputs):
        # nothing to compare
        return torch.zeros((), device=embs[0].device)
    n = embs[0].shape[0]
    normed = []
    for emb in embs:
        emb = emb - emb.mean(dim=0, keepdim=True)
        normed.append(F.normalize(emb, p=2, dim=1))
    if n > LARGE_N:
        # mean((G1-G2)^2) = (tr(K1K1) - 2 tr(K1K2) + tr(K2K2)) / n^2 with
        # tr(Ka Kb) = ||Za^T Zb||_F^2 — no n x n Gram matrices
        def cross(a, b):
            return ((a.t() @ b) ** 2).sum()

        cost = 0.0
        for z1, z2 in itertools.combinations(normed, 2):
            cost = cost + cross(z1, z1) - 2 * cross(z1, z2) + cross(z2, z2)
        return cost / float(n * n)
    grams = [z @ z.t() for z in normed]
    cost = 0.0
    for g1, g2 in itertools.combinations(grams, 2):
        cost = cost + (g1 - g2) ** 2
    return cost.mean()


def reconstruction_loss(x_bar, x):
    """Reconstruction loss L_res (paper Eq. 17)."""
    return F.mse_loss(x_bar, x)

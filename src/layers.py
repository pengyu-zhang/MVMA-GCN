"""Graph layers: GCN convolution and the single-view attention layer (paper Eq. 8-9)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Simple GCN layer (Kipf & Welling)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


def _segment_softmax(scores, row_index, n_rows):
    """Softmax over edge scores grouped by destination row (numerically stable)."""
    row_max = torch.full((n_rows,), float("-inf"), device=scores.device)
    row_max = row_max.scatter_reduce(0, row_index, scores, reduce="amax")
    scores = torch.exp(scores - row_max[row_index])
    row_sum = torch.zeros(n_rows, device=scores.device)
    row_sum = row_sum.index_add(0, row_index, scores)
    return scores / row_sum[row_index].clamp(min=1e-16)


class SingleViewAttention(nn.Module):
    """Single-view attention layer (paper Eq. 8-9).

    For a view with adjacency edge list E, computes GAT-style attention
    coefficients over neighbours (alpha_ij = softmax_j LeakyReLU(a^T
    [x_i || x_j])), aggregates neighbour features per head with a sigmoid,
    concatenates the K heads, and finally concatenates the node's own
    features:  X_att = [ z^(1) || ... || z^(K) || X ].

    Paper-literal mode (``proj_dim=None``) scores and aggregates the raw
    features, so the output dimension is (heads + 1) * n_feats.

    Projected mode (``proj_dim=d'``, an enhancement beyond the paper's Eq. 8)
    first maps features through a learnable linear projection, scores and
    aggregates in that space — the standard GAT formulation — and outputs
    [ z^(1) || ... || z^(K) || X ] with dimension heads * d' + n_feats.
    ``attn_dropout`` randomly drops attention coefficients during training.

    Self-loops are added so isolated nodes attend to themselves.
    """

    def __init__(self, n_feats, heads=1, negative_slope=0.2, proj_dim=None,
                 attn_dropout=0.0):
        super().__init__()
        self.heads = heads
        self.negative_slope = negative_slope
        self.attn_dropout = attn_dropout
        self.proj = None
        d = n_feats
        if proj_dim is not None:
            self.proj = nn.Linear(n_feats, proj_dim, bias=False)
            nn.init.xavier_uniform_(self.proj.weight, gain=1.414)
            d = proj_dim
        # a^T [x_i || x_j] decomposes into (a_dst . x_i) + (a_src . x_j)
        self.att_dst = nn.Parameter(torch.empty(heads, d))
        self.att_src = nn.Parameter(torch.empty(heads, d))
        nn.init.xavier_uniform_(self.att_dst, gain=1.414)
        nn.init.xavier_uniform_(self.att_src, gain=1.414)

    def out_dim(self, n_feats):
        d = self.att_dst.shape[1]
        return self.heads * d + n_feats

    def forward(self, x, edge_index):
        """x: [n, d]; edge_index: LongTensor [2, m] (dst, src) with self-loops."""
        n = x.shape[0]
        dst, src = edge_index[0], edge_index[1]
        h_in = self.proj(x) if self.proj is not None else x
        out_heads = []
        for h in range(self.heads):
            # a^T [x_i || x_j] as per-node scalars gathered per edge, so memory
            # stays O(E) instead of O(E * d)
            s_dst = h_in @ self.att_dst[h]
            s_src = h_in @ self.att_src[h]
            scores = F.leaky_relu(s_dst[dst] + s_src[src], self.negative_slope)
            alpha = _segment_softmax(scores, dst, n)
            if self.training and self.attn_dropout > 0:
                alpha = F.dropout(alpha, self.attn_dropout)
            att = torch.sparse_coo_tensor(
                edge_index, alpha, (n, n), check_invariants=False
            )
            agg = torch.sparse.mm(att, h_in)
            out_heads.append(torch.sigmoid(agg))
        return torch.cat(out_heads + [x], dim=1)


def edges_to_attention_index(edges, n, device):
    """Symmetrized edge list + self-loops as a [2, m] index tensor for attention."""
    e = torch.from_numpy(edges.astype("int64"))
    dst = torch.cat([e[:, 0], e[:, 1], torch.arange(n)])
    src = torch.cat([e[:, 1], e[:, 0], torch.arange(n)])
    idx = torch.stack([dst, src])
    # deduplicate repeated edges so softmax is not biased by duplicates
    idx = torch.unique(idx, dim=1)
    return idx.to(device)

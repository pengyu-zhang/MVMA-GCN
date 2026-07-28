"""MVMA-GCN model with configuration switches for every paper component.

Switches (see configs/*.yaml):

- ``single_view_attention``: the paper's Eq. 8-9 node-level attention on the
  relation views (disabled in the baseline configuration).
- ``ae_injection``: the paper's Eq. 7 layer-wise injection of autoencoder
  hidden states into the view-specific GCNs (disabled in the baseline).
- ``common_conv``: ``per_view_avg`` applies one shared-weight GCN to every
  graph and averages the outputs (baseline);
  ``sum_graph`` applies it once to the union graph A_c (paper Section 4.1).
- ``attention``: ``hierarchical`` fuses the specific embeddings first and the
  common embedding second, reusing one attention module (baseline);
  ``flat`` uses a single softmax over all embeddings (paper Eq. 10-12).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution, SingleViewAttention


class GCN(nn.Module):
    """Two-layer GCN used for the view-specific and common convolutions."""

    def __init__(self, nfeat, nhid, nout, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj, inject=None, epsilon=0.5):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        if inject is not None:
            # Paper Eq. 7: blend the autoencoder hidden state into the GCN input
            h = (1.0 - epsilon) * h + epsilon * inject
        return self.gc2(h, adj)


class AE(nn.Module):
    """Fully connected autoencoder (paper Eq. 5-6)."""

    def __init__(self, nfeat, enc_dims, n_z):
        super().__init__()
        dims = [nfeat] + list(enc_dims)
        self.encoder = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.z_layer = nn.Linear(dims[-1], n_z)
        dec_dims = [n_z] + list(reversed(enc_dims))
        self.decoder = nn.ModuleList(
            nn.Linear(dec_dims[i], dec_dims[i + 1]) for i in range(len(dec_dims) - 1)
        )
        self.x_bar_layer = nn.Linear(dec_dims[-1], nfeat)

    def forward(self, x):
        hiddens = []
        h = x
        for layer in self.encoder:
            h = F.relu(layer(h))
            hiddens.append(h)
        z = self.z_layer(h)
        h = z
        for layer in self.decoder:
            h = F.relu(layer(h))
        x_bar = self.x_bar_layer(h)
        return x_bar, hiddens, z


class AttentionFusion(nn.Module):
    """Semantic-level attention over stacked embeddings (paper Eq. 10-11)."""

    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MVMAGCN(nn.Module):
    def __init__(self, cfg, n_feats, n_classes, view_attention_indices=None):
        super().__init__()
        m = cfg["model"]
        nhid1, nhid2 = m["nhid1"], m["nhid2"]
        dropout = m["dropout"]
        self.dropout = dropout
        self.common_conv_mode = m["common_conv"]
        self.attention_mode = m["attention"]
        self.n_views = m["n_views"]  # relation views (kNN view is extra)

        # --- single-view attention (paper Eq. 8-9), optional ---
        sva = m["single_view_attention"]
        self.sva_enabled = sva["enabled"]
        if self.sva_enabled:
            heads = sva["heads"]
            proj_dim = sva.get("proj_dim")        # None = paper-literal Eq. 8
            attn_dropout = sva.get("attn_dropout", 0.0)
            self.sva = nn.ModuleList(
                SingleViewAttention(
                    n_feats, heads, proj_dim=proj_dim, attn_dropout=attn_dropout
                )
                for _ in range(self.n_views)
            )
            self.view_attention_indices = view_attention_indices
            view_in = self.sva[0].out_dim(n_feats)
        else:
            self.sva = None
            self.view_attention_indices = None
            view_in = n_feats

        # --- autoencoder (paper Eq. 5-6) + optional Eq. 7 injection ---
        ae_cfg = m["ae"]
        self.ae = AE(n_feats, ae_cfg["enc_dims"], ae_cfg["n_z"])
        inj = m["ae_injection"]
        self.inject_enabled = inj["enabled"]
        self.epsilon = inj["epsilon"]
        if self.inject_enabled and ae_cfg["enc_dims"][0] != nhid1:
            raise ValueError(
                "ae.enc_dims[0] must equal model.nhid1 when ae_injection is enabled"
            )

        # --- view-specific GCNs: one per relation view + one for the kNN graph ---
        self.view_gcns = nn.ModuleList(
            GCN(view_in, nhid1, nhid2, dropout) for _ in range(self.n_views)
        )
        self.knn_gcn = GCN(n_feats, nhid1, nhid2, dropout)
        # --- common (multi-view) convolution, shared weights ---
        self.common_gcn = GCN(n_feats, nhid1, nhid2, dropout)
        if self.common_conv_mode == "weighted_sum":
            # learnable softmax weights over the per-view normalized
            # adjacencies (enhancement of the paper's plain sum graph)
            self.graph_weights = nn.Parameter(torch.zeros(self.n_views + 1))

        # --- attention fusion ---
        if self.attention_mode not in ("hierarchical", "flat"):
            raise ValueError(f"unknown attention mode: {self.attention_mode}")
        # hierarchical mode reuses one attention module for both stages
        self.attention = AttentionFusion(nhid2, m["attention_hidden"])

        self.classifier = nn.Sequential(
            nn.Linear(nhid2, n_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, x, view_adjs, knn_adj, sum_adj=None):
        """Returns (log_probs, out) where ``out`` carries every intermediate
        needed by the losses and the mechanism checks."""
        x_bar, enc_hiddens, _z = self.ae(x)
        inject = enc_hiddens[0] if self.inject_enabled else None

        # view-specific embeddings Z_m (relation views)
        specific = []
        for i, adj in enumerate(view_adjs):
            if self.sva_enabled:
                xin = self.sva[i](x, self.view_attention_indices[i])
            else:
                xin = x
            specific.append(
                self.view_gcns[i](xin, adj, inject=inject, epsilon=self.epsilon)
            )
        # kNN feature-graph embedding Z_k
        emb_k = self.knn_gcn(x, knn_adj, inject=inject, epsilon=self.epsilon)

        # common embedding(s) Z_c
        if self.common_conv_mode == "per_view_avg":
            common_list = [self.common_gcn(x, adj) for adj in view_adjs]
            common_list.append(self.common_gcn(x, knn_adj))
            common = torch.stack(common_list).mean(0)
        elif self.common_conv_mode == "sum_graph":
            common = self.common_gcn(x, sum_adj)
            common_list = [common]
        elif self.common_conv_mode == "weighted_sum":
            # convex combination of the row-normalized adjacencies, so dense
            # views cannot drown out sparse ones and the mix is learned
            w = torch.softmax(self.graph_weights, dim=0)
            adjs = list(view_adjs) + [knn_adj]
            mixed = adjs[0] * w[0]
            for i in range(1, len(adjs)):
                mixed = mixed + adjs[i] * w[i]
            common = self.common_gcn(x, mixed.coalesce())
            common_list = [common]
        else:
            raise ValueError(f"unknown common_conv mode: {self.common_conv_mode}")

        # attention fusion
        att_weights = {}
        if self.attention_mode == "hierarchical":
            stack1 = torch.stack(specific[:1] + [emb_k] + specific[1:], dim=1)
            fused, att1 = self.attention(stack1)
            stack2 = torch.stack([fused, common], dim=1)
            emb, att2 = self.attention(stack2)
            att_weights["stage1"] = att1
            att_weights["stage2"] = att2
        else:  # flat
            stack = torch.stack(specific + [emb_k, common], dim=1)
            emb, att1 = self.attention(stack)
            att_weights["flat"] = att1

        log_probs = self.classifier(emb)
        out = {
            "specific": specific,          # [Z_m] per relation view
            "emb_k": emb_k,                # Z_k
            "common": common,              # fused Z_c
            "common_list": common_list,    # per-graph common outputs
            "emb": emb,                    # final fused embedding
            "x_bar": x_bar,
            "att": att_weights,
        }
        return log_probs, out

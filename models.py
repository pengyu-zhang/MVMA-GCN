import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math
from torch.nn import Linear


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# 加AE
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 nfeat, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(nfeat, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, nfeat)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout, n_enc_1, n_enc_2,
                 n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_z, n_clusters):
        super(SFGCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            nfeat=nfeat,
            n_z=n_z)

        # GCN for inter information
        self.gnn_1 = GraphConvolution(nfeat, n_enc_1)
        self.gnn_2 = GraphConvolution(n_enc_1, n_enc_2)
        self.gnn_3 = GraphConvolution(n_enc_2, n_enc_3)
        self.gnn_4 = GraphConvolution(n_enc_3, n_z)
        self.gnn_5 = GraphConvolution(n_z, n_clusters)

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN3 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN4 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj_pap, sadj_plp, sadj_pmp, fadj):   # fadj的输入是由knn生成的关系图。而x应该是节点特征
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        # GCN Module
        # h = self.gnn_1(x, fadj)
        # h = self.gnn_2(h+tra1, fadj)
        # h = self.gnn_3(h+tra2, fadj)
        # h = self.gnn_4(h+tra3, fadj)
        # h = self.gnn_5(h+z, adj, active=False)
        # predict = F.softmax(h, dim=1)

        emb1 = self.SGCN1(x, sadj_pap)
        com1 = self.CGCN(x, sadj_pap)
        com2 = self.CGCN(x, fadj)
        emb2 = self.SGCN2(x, fadj)
        # print(tra1.shape)
        # print(emb2.shape)
        # print(fadj.shape)
        # emb2 = self.SGCN2(emb2, fadj + tra1)
        # emb2 = self.SGCN2(emb2, fadj + tra2)
        emb3 = self.SGCN3(x, sadj_plp)
        emb4 = self.SGCN4(x, sadj_pmp)
        com3 = self.CGCN(x, sadj_plp)
        com4 = self.CGCN(x, sadj_pmp)
        Xcom = (com1 + com2 + com3 + com4) / 4
        # attention
        emb = torch.stack([emb1, emb2, emb3, emb4], dim=1)
        emb, att = self.attention(emb)
        emb = torch.stack([emb, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att, emb1, com1, com2, emb2, emb3, emb4, com3, com4, emb, x_bar

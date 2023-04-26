import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import numpy as np
import networkx as nx


def common_loss(emb1, emb2, emb3, emb4):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb3 = emb3 - torch.mean(emb3, dim=0, keepdim=True)
    emb4 = emb4 - torch.mean(emb4, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)
    emb4 = torch.nn.functional.normalize(emb4, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cov3 = torch.matmul(emb3, emb3.t())
    cov4 = torch.matmul(emb4, emb4.t())
    cost = torch.mean(
        (cov1 - cov2) ** 2 + (cov1 - cov3) ** 2 +
        (cov1 - cov4) ** 2 + (cov2 - cov3) ** 2 +
        (cov2 - cov4) ** 2 + (cov3 - cov4) ** 2)
    return cost


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(config):
    f = np.loadtxt(config.feature_path, dtype=float)
    l = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test


def load_graph(dataset, config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    # 原始图 pap
    struct_pap_edges = np.genfromtxt(config.structgraph_PAP_path, dtype=np.int32)
    sedges_pap = np.array(list(struct_pap_edges), dtype=np.int32).reshape(struct_pap_edges.shape)
    sadj_pap = sp.coo_matrix((np.ones(sedges_pap.shape[0]), (sedges_pap[:, 0], sedges_pap[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj_pap = sadj_pap + sadj_pap.T.multiply(sadj_pap.T > sadj_pap) - sadj_pap.multiply(sadj_pap.T > sadj_pap)
    nsadj_pap = normalize(sadj_pap+sp.eye(sadj_pap.shape[0]))

    # 新图 plp 就等于 psp subject 所有plp都等于psp
    struct_plp_edges = np.genfromtxt(config.structgraph_PLP_path, dtype=np.int32)
    sedges_plp = np.array(list(struct_plp_edges), dtype=np.int32).reshape(struct_plp_edges.shape)
    sadj_plp = sp.coo_matrix((np.ones(sedges_plp.shape[0]), (sedges_plp[:, 0], sedges_plp[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj_plp = sadj_plp + sadj_plp.T.multiply(sadj_plp.T > sadj_plp) - sadj_plp.multiply(sadj_plp.T > sadj_plp)
    nsadj_plp = normalize(sadj_plp+sp.eye(sadj_plp.shape[0]))

    # 新图 pmp
    struct_pmp_edges = np.genfromtxt(config.structgraph_PMP_path, dtype=np.int32)
    sedges_pmp = np.array(list(struct_pmp_edges), dtype=np.int32).reshape(struct_pmp_edges.shape)
    sadj_pmp = sp.coo_matrix((np.ones(sedges_pmp.shape[0]), (sedges_pmp[:, 0], sedges_pmp[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj_pmp = sadj_pmp + sadj_pmp.T.multiply(sadj_pmp.T > sadj_pmp) - sadj_pmp.multiply(sadj_pmp.T > sadj_pmp)
    nsadj_pmp = normalize(sadj_pmp+sp.eye(sadj_pmp.shape[0]))

    # 原始 pap
    nsadj_pap = sparse_mx_to_torch_sparse_tensor(nsadj_pap)

    # 新 plp
    nsadj_plp = sparse_mx_to_torch_sparse_tensor(nsadj_plp)

    # 新 pmp
    nsadj_pmp = sparse_mx_to_torch_sparse_tensor(nsadj_pmp)

    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj_pap, nsadj_plp, nsadj_pmp, nfadj

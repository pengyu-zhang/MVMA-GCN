"""k-NN feature-graph construction (AM-GCN's generate_knn procedure).

For each node, the top-(k+1) most cosine-similar nodes are selected with
``np.argpartition`` (this includes the node itself, which is skipped), and an
edge (i, j) is kept only when i < j. This reproduces the shipped
``knn/c<k>.txt`` files up to tie-breaking among equal similarities
(verified exactly against the DBLP ``knn/tmp.txt`` intermediate).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_knn_edges(features, k):
    """Return an int32 [m, 2] edge list for the k-NN feature graph."""
    sim = cosine_similarity(features)
    edges = []
    for i in range(sim.shape[0]):
        ind = np.argpartition(sim[i, :], -(k + 1))[-(k + 1):]
        for j in ind:
            j = int(j)
            if j != i and i < j:
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int32)

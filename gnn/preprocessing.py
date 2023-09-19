from copy import copy

import sklearn
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import scipy.linalg
import scipy
from sklearn.datasets import make_blobs
import scipy.sparse as sp
import scipy.stats
import numpy as np

from gnn.utils import mat_to_tuple, tuple_to_mat, get_features, tuple_to_dense


def get_data_graph(type, n_graphs, n_nodes, dim, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # generate data
    if type == 'gaussian':
        labels = []
        graphs = []
        for _ in range(n_graphs):
            std = np.random.uniform(1, 10)
            features, _ = make_blobs(n_nodes, dim, centers=1, cluster_std=std, center_box=(0, 0))

            A = radius_neighbors_graph(features, 3).todense()

            np.fill_diagonal(A, 1)
            A_ = A
            rowsum = np.array(A_.sum(1))
            degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
            A = degree_mat_inv_sqrt.dot(A_).dot(degree_mat_inv_sqrt)

            #A = kneighbors_graph(features, int(np.sqrt(n_nodes)))

            labels.append(std)
            graphs.append(A)

        return graphs, labels


def get_data_edge_level(placeholders, adj, features):
    adj_train, train_edges_true, train_edges_false, \
    val_edges_true, val_edges_false, \
    test_edges_true, test_edges_false = mask_test_edges(adj)

    supports = preprocess_graph(adj, 0)

    labels = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))

    if features is None:
        features = sparse_to_tuple(sp.eye(adj_train.shape[0]))
    else:
        features = sparse_to_tuple(features)

    feed_dict_train = {placeholders['support']: supports,
                       placeholders['labels']: labels,
                       placeholders['features']: features}

    feed_dicts = {'train': feed_dict_train}

    adjs = {'train': adj_train}

    edges_true = {'train': train_edges_true,
                  'val': val_edges_true,
                  'test': test_edges_true}

    edges_false = {'train': train_edges_false,
                   'val': val_edges_false,
                   'test': test_edges_false}

    return feed_dicts, adjs, edges_true, edges_false


def get_data_(placeholders, adj, features):
    adj_train, train_edges_true, train_edges_false = mask_train_edges(adj)

    supports = preprocess_graph(adj, 0)

    labels = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))

    if features is None:
        features = sparse_to_tuple(sp.eye(adj_train.shape[0]))
    else:
        features = sparse_to_tuple(features)

    feed_dict_train = {placeholders['support']: supports,
                       placeholders['labels']: labels,
                       placeholders['features']: features}

    feed_dicts = {'train': feed_dict_train}

    adjs = {'train': adj_train}

    edges_true = {'train': train_edges_true}

    edges_false = {'train': train_edges_false}

    return feed_dicts, adjs, edges_true, edges_false


def get_data_node_level(placeholders, adj, features):
    adj_train, features, adj_test, features_test_, adj_val, features_val_ = get_test_split(adj, features)

    adj_train, train_edges_true, train_edges_false = mask_train_edges(adj_train)
    adj_test, test_edges_true, test_edges_false = mask_train_edges(adj_test)
    adj_val, val_edges_true, val_edges_false = mask_train_edges(adj_val)

    num_features = np.max([adj_train.shape[0], adj_test.shape[0], adj_val.shape[0]])

    # use identity if no data is given
    if features is None:
        print("No features are given. Using identity as input.")
        features_, features_test_, features_val_ = get_features(adj_train.shape[0],
                                                                adj_test.shape[0],
                                                                adj_val.shape[0],
                                                                num_features)
    else:
        features_ = features


    # Some preprocessing
    # supports = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))
    # supports_test = sparse_to_tuple(adj_test + sp.eye(adj_test.shape[0]))
    # supports_val = sparse_to_tuple(adj_val + sp.eye(adj_val.shape[0]))
    #
    supports = preprocess_graph(adj_train, 0)
    supports_test = preprocess_graph(adj_test, 0)
    supports_val = preprocess_graph(adj_val, 0)

    features = sparse_to_tuple(features_)
    features_val = sparse_to_tuple(features_val_)
    features_test = sparse_to_tuple(features_test_)

    labels = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))
    labels_val = sparse_to_tuple(adj_val + sp.eye(adj_val.shape[0]))
    labels_test = sparse_to_tuple(adj_test + sp.eye(adj_test.shape[0]))

    feed_dict_train = {placeholders['support']: supports,
                       placeholders['labels']: labels,
                       placeholders['features']: features}

    feed_dict_val = {placeholders['support']: supports_val,
                     placeholders['labels']: labels_val,
                     placeholders['features']: features_val}

    feed_dict_test = {placeholders['support']: supports_test,
                      placeholders['labels']: labels_test,
                      placeholders['features']: features_test}

    feed_dicts = {'train': feed_dict_train,
                  'val': feed_dict_val,
                  'test': feed_dict_test}

    adjs = {'train': adj_train,
            'val': adj_val,
            'test': adj_test}

    edges_true = {'train': train_edges_true,
                  'val': val_edges_true,
                  'test': test_edges_true}

    edges_false = {'train': train_edges_false,
                   'val': val_edges_false,
                   'test': test_edges_false}

    return feed_dicts, adjs, edges_true, edges_false


def sigmoid(x):
    x = x
    return 1 / (1 + np.exp(-x))


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj, square_support):
    adj = sp.coo_matrix(adj)
    adj.setdiag(1)
    adj_ = adj #+ sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_inv_sqrt).tocoo()
    if square_support:
        adj_normalized = adj_normalized @ adj_normalized
    return sparse_to_tuple(adj_normalized)


def get_data(type, n_nodes, dim, alignment=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_possible_edges = n_nodes ** 2
    # generate data
    if type == 'gaussian_dot':
        features, clusters = make_blobs(n_nodes, dim, centers=1, cluster_std=1, shuffle=False, center_box=(0, 0))

        features = sklearn.preprocessing.normalize(features)
        x = np.transpose(features)
        x = np.matmul(features, x)
        D = sigmoid(x)

        np.fill_diagonal(D, 0)

        range = [0.01, 0.02]
        lower = 0
        upper = 1

        while True:
            threshold = (lower + upper) / 2
            labels = (D > threshold).astype(int)
            density = np.sum(labels) / n_possible_edges
            if density < range[0]:
                upper = threshold
                continue
            elif density > range[1]:
                lower = threshold
                continue
            else:
                break

        assert (labels.transpose() == labels).all()

        idx = np.all(labels == 0, axis=1)
        clusters = clusters[~idx]
        features = features[~idx, :]
        labels = labels[~idx].transpose()[~idx]

        assert (labels.transpose() == labels).all()

    if type == 'gaussian_dot_5':
        features, clusters = make_blobs(n_nodes, dim, centers=5, cluster_std=1, shuffle=False, center_box=(-20, 20))

        features = sklearn.preprocessing.normalize(features)
        x = np.transpose(features)
        x = np.matmul(features, x)
        D = sigmoid(x)

        np.fill_diagonal(D, 0)

        range = [0.01, 0.02]
        lower = 0
        upper = 1

        while True:
            threshold = (lower + upper) / 2
            labels = (D > threshold).astype(int)
            density = np.sum(labels) / n_possible_edges
            if density < range[0]:
                upper = threshold
                continue
            elif density > range[1]:
                lower = threshold
                continue
            else:
                break

        assert (labels.transpose() == labels).all()

        idx = np.all(labels == 0, axis=1)
        clusters = clusters[~idx]
        features = features[~idx, :]
        labels = labels[~idx].transpose()[~idx]

        assert (labels.transpose() == labels).all()

    if type == 'gaussian':
        features, clusters = make_blobs(n_nodes, dim, centers=1, cluster_std=1, shuffle=False, center_box=(-1, 1))

        squared_norm = np.sum(np.square(features), axis=1)

        b = np.reshape(squared_norm, [1, -1])
        a = np.reshape(squared_norm, [-1, 1])

        pairwise_dist = np.sqrt(np.amax(a - 2 * features @ features.transpose() + b, axis=(), initial=0))

        D = np.exp(-pairwise_dist)
        np.fill_diagonal(D, 0)

        range = [0.01, 0.02]
        lower = 0
        upper = 1

        while True:
            threshold = (lower + upper) / 2
            labels = (D > threshold).astype(int)
            density = np.sum(labels) / n_possible_edges
            if density < range[0]:
                upper = threshold
                continue
            elif density > range[1]:
                lower = threshold
                continue
            else:
                break

        assert (labels.transpose() == labels).all()

        idx = np.all(labels == 0, axis=1)
        clusters = clusters[~idx]
        features = features[~idx, :]
        labels = labels[~idx].transpose()[~idx]

        np.fill_diagonal(labels, 1)

        assert (labels.transpose() == labels).all()

    if type == 'clique':
        features = None
        n1 = n2 = n3 = int(n_nodes/3)

        labels = np.array(scipy.linalg.block_diag(np.ones([n1, n1]), np.ones([n2, n2]), np.ones([n3, n3])))

        assert labels.shape == (n_nodes, n_nodes), "Wrong shape for clique model."

        clusters = np.array([0] * n1 + [1] * n2 + [2] * n3)

    if type == 'sbm':
        p = 0.3
        q = 0.1

        n1 = n2 = n3 = int(n_nodes / 3)

        probs = np.random.rand(n_nodes, n_nodes)
        a = np.zeros([n_nodes, n_nodes])
        a[probs <= q] = 1
        a[:n1, :n1][probs[:n1, :n1] <= p] = 1
        a[n1:n1+n2, n1:n1+n2][probs[n1:n1+n2, n1:n1+n2] <= p] = 1
        a[n1+n2:, n1+n2:][probs[n1+n2:, n1+n2:] <= p] = 1

        a = np.triu(a)
        a = a + a.T
        np.fill_diagonal(a, 1)

        labels = a
        features = sklearn.manifold.TSNE(2).fit_transform(a)
        clusters = np.array([0] * n1 + [1] * n2 + [2] * n3)

    try:
        features
        labels
        clusters
    except UnboundLocalError:
        raise UnboundLocalError("Variables are not defined, maybe the method is not implemented. Method: %s" % type)

    if alignment == None:
        return sp.csr_matrix(labels), sp.csr_matrix(sklearn.preprocessing.normalize(features, axis=0)), clusters, None
    else:
        u, s, v = np.linalg.svd(features)
        oodim = dim - alignment
        features_f = np.zeros_like(features)
        if alignment > 0:
            features_f[:, :alignment] = u[:, :alignment]
            features_f[:, alignment:] = u[:, dim:dim+oodim]
        else:
            features_f = u[:, dim:dim+dim]

        return sp.csr_matrix(labels), sp.csr_matrix(features_f), clusters, features


def get_test_split(adj, features, seed=None):
    idxs = {}
    adjs = {}

    if seed is not None:
        np.random.seed(seed)

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    n = adj.shape[0]
    num_test = int(np.floor(n / 10.))
    num_val = int(np.floor(n / 20.))

    all_idx = np.arange(n)
    np.random.shuffle(all_idx)

    idxs['test'] = all_idx[:num_test]
    idxs['val'] = all_idx[num_test:(num_test + num_val)]
    idxs['train'] = all_idx[(num_test + num_val):]
    idxs['val_all'] = np.hstack([idxs['train'], idxs['test']])
    idxs['test_all'] = np.hstack([idxs['train'], idxs['val']])

    adj_train = adj.copy()
    adj_train = adj_train[idxs['train'], :]
    adj_train = adj_train[:, idxs['train']]
    adjs['train'] = adj_train

    adj_val = adj.copy()
    adj_val = adj_val[idxs['val_all'], :]
    adj_val = adj_val[:, idxs['val_all']]
    adjs['val'] = adj_val

    adj_test = adj.copy()
    adj_test = adj_test[idxs['test_all'], :]
    adj_test = adj_test[:, idxs['test_all']]
    adjs['test'] = adj_test

    if features == None:
        features_train = features_test = features_val = None
    else:
        features_train = sp.csr_matrix(features[idxs['train'], :])
        features_test = sp.csr_matrix(features[idxs['test_all'], :])
        features_val = sp.csr_matrix(features[idxs['val_all'], :])

    return adjs['train'], features_train, adjs['test'], features_test, adjs['val'], features_val


def mask_features(features, idx):
    ft = features.copy()
    ft = ft[idx, :]
    return ft


def mask_clusters(clusters, idx):
    cl = clusters.copy()
    cl = clusters[idx]
    return cl


# noinspection PyTypeChecker
def get_false_edges(edges_true, n):
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edges_false = []
    while len(edges_false) < len(edges_true):
        idx_i = np.random.randint(0, n)
        idx_j = np.random.randint(0, n)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_true):
            continue
        if ismember([idx_j, idx_i], edges_true):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    return edges_false


def mask_train_edges(adj, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # true edges
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    train_edges = adj_tuple[0]
    np.random.shuffle(train_edges)

    # false edges
    adj_false = (adj - sp.csr_matrix(np.ones(adj.shape))).power(2)

    adj_triu_false = sp.triu(adj_false)
    adj_tuple_false = sparse_to_tuple(adj_triu_false)
    train_edges_false = adj_tuple_false[0]
    np.random.shuffle(train_edges_false)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false[:len(train_edges)]


def mask_test_edges(adj, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # true edges
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # false edges
    adj_false = (adj - sp.csr_matrix(np.ones(adj.shape))).power(2)
    adj_triu_false = sp.triu(adj_false)
    adj_tuple_false = sparse_to_tuple(adj_triu_false)
    edges_false = adj_tuple_false[0]
    num_test_false = num_test
    num_val_false = num_val

    all_edge_idx_false = list(range(edges_false.shape[0]))
    np.random.shuffle(all_edge_idx_false)
    val_edge_idx_false = all_edge_idx_false[:num_val_false]
    test_edge_idx_false = all_edge_idx_false[num_val_false:(num_val_false + num_test_false)]
    test_edges_false = edges_false[test_edge_idx_false]
    val_edges_false = edges_false[val_edge_idx_false]
    train_edges_false = np.delete(edges_false, np.hstack([test_edge_idx_false, val_edge_idx_false]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    # test_edges_false = []
    # while len(test_edges_false) < len(test_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if ismember([idx_j, idx_i], edges_all):
    #         continue
    #     if test_edges_false:
    #         if ismember([idx_j, idx_i], np.array(test_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #             continue
    #     test_edges_false.append([idx_i, idx_j])
    #
    # val_edges_false = []
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if ismember([idx_j, idx_i], edges_all):
    #         continue
    #     if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #         continue
    #     if ismember([idx_j, idx_i], np.array(test_edges_false)):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])
    #
    # train_edges_false = []
    # while len(train_edges_false) < len(train_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if ismember([idx_j, idx_i], edges_all):
    #         continue
    #     if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #         continue
    #     if ismember([idx_j, idx_i], np.array(test_edges_false)):
    #         continue
    #     if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #         continue
    #     if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #         continue
    #     if train_edges_false:
    #         if ismember([idx_j, idx_i], np.array(train_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(train_edges_false)):
    #             continue
    #     train_edges_false.append([idx_i, idx_j])

    assert ismember(test_edges, edges_all)
    assert ismember(val_edges, edges_all)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    np.random.shuffle(train_edges_false)
    np.random.shuffle(val_edges_false)
    np.random.shuffle(test_edges_false)

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false[:len(train_edges)], val_edges, val_edges_false[:len(val_edges)], test_edges, test_edges_false[:len(test_edges)]


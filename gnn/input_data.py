import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    if dataset == 'amazon':
        return load_amazon()

    if dataset == 'facebook':
        return load_facebook()

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def load_amazon():
    with open('../data/Amazon0302.txt') as f:
        lines = (line for line in f if not line.startswith('#'))
        FH = np.loadtxt(lines, delimiter='\t', skiprows=1)

    shape = (262111, 262111)
    data = np.ones(1234876)

    A = sp.csr_matrix((data, (FH[:, 0], FH[:, 1])), shape)

    return A + A.transpose(), None


def load_facebook():
    with open('../data/facebook_combined.txt') as f:
        lines = (line for line in f if not line.startswith('#'))
        FH = np.loadtxt(lines, delimiter=' ', skiprows=1)

    shape = (4039, 4039)
    data = np.ones(88233)

    A = sp.csr_matrix((data, (FH[:, 0], FH[:, 1])), shape)

    return A + A.transpose(), None

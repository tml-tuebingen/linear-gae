import csv
from pathlib import Path

import argparse
import numpy as np
# noinspection PyProtectedMember
import tensorflow._api.v2.compat.v1 as tf
import sys
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing
from sklearn.preprocessing import normalize

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_features(shape_train, shape_test, shape_val, num_features):
    features_ = np.zeros([shape_train, num_features])
    np.fill_diagonal(features_, 1)
    features_ = sp.csr_matrix(features_)
    features_test_ = np.zeros([shape_test, num_features])
    np.fill_diagonal(features_test_, 1)
    features_test_ = sp.csr_matrix(features_test_)
    features_val_ = np.zeros([shape_val, num_features])
    np.fill_diagonal(features_val_, 1)
    features_val_ = sp.csr_matrix(features_val_)

    return features_, features_test_, features_val_


def progressbar(count, prefix="", size=60, file=sys.stdout):
    it = range(count)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


def tuple_to_dense(tuple):
    return sp.coo_matrix((tuple[1], (tuple[0][:, 0], tuple[0][:, 1])), tuple[2]).todense()


def update(placeholder, labels, features, support):
    feed_dict = dict()
    feed_dict.update({placeholder['features']: features})
    feed_dict.update({placeholder['support']: support})
    feed_dict.update({placeholder['labels']: labels})
    return feed_dict


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = tf.sqrt(6.0 / tf.cast((shape[0] + shape[1]), tf.float32))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def mat_to_tuple(adj):
    nodes = adj.shape[0]
    edges = []

    for i in range(nodes):
        for j in range(i, nodes):
            if adj[i, j]:
                edges.append((i, j))

    return np.array(edges)


def tuple_to_mat(edges, shape):
    if edges is None:
        return None

    nodes = shape[0]
    adj = np.zeros([nodes, nodes])

    for i, j in edges:
        adj[i, j] = adj[j, i] = 1

    return adj


# def split(num_graphs, test_split, val_split):
#     num_test = int(num_graphs * test_split)
#     num_val = int(num_graphs * val_split)
#
#     idx_all = np.arange(num_graphs)
#     np.random.shuffle(idx_all)
#
#     idx_test = idx_all[:num_test]
#     idx_val = idx_all[num_test:num_test + num_val] if val_split else []
#     idx_train = idx_all[num_test + num_val:]
#
#     return idx_train, idx_test, idx_val


def decode(z):
    squared_norm = np.sum(np.square(z), axis=1)

    b = np.reshape(squared_norm, [1, -1])
    a = np.reshape(squared_norm, [-1, 1])

    pairwise_dist = np.sqrt(np.amax(a - 2 * z @ z.transpose() + b, axis=(), initial=0))

    D = np.exp(-pairwise_dist)

    prediction = (D > 0.5).astype(int)

    return prediction


def save_results(costs, accuracies, vals, train, tests, args):
    # setup directory and save parameters
    output_dir = Path('outputs', args.dataset + '_' + args.decoder + '_indim' + str(args.num_features) + str(
        '_featureless' if args.featureless else ''))
    output_dir.mkdir(parents=True, exist_ok=True)

    path_params, path_accuracies, path_costs, path_vals, path_tests, path_train = get_paths(output_dir, args)

    with open(path_params, 'w') as f:
        if type(args) == argparse.Namespace:
            w = csv.DictWriter(f, args.__dict__.keys())
            w.writeheader()
            w.writerow(args.__dict__)
        else:
            w = csv.DictWriter(f, args.keys())
            w.writeheader()
            w.writerow(args)

    column_names = ['run'] + ['epoch ' + str(i) for i in range(args.epochs)]

    accuracies_df = pd.DataFrame(np.hstack((np.arange(1, args.runs + 1).reshape(-1, 1), accuracies)),
                                 columns=column_names)
    accuracies_df = pd.concat(
        [accuracies_df, pd.DataFrame([['average accuracy'] + list(np.mean(accuracies, axis=0))], columns=column_names)])
    accuracies_df.to_csv(path_accuracies, index=False)

    costs_df = pd.DataFrame(np.hstack((np.arange(1, args.runs + 1).reshape(-1, 1), costs)), columns=column_names)
    costs_df = pd.concat(
        [costs_df, pd.DataFrame([['average costs'] + list(np.mean(costs, axis=0))], columns=column_names)])
    costs_df.to_csv(path_costs, index=False)

    vals_df = pd.DataFrame(np.hstack((np.arange(1, args.runs + 1).reshape(-1, 1), vals)), columns=column_names)
    vals_df = pd.concat(
        [vals_df, pd.DataFrame([['average val roc'] + list(np.mean(vals, axis=0))], columns=column_names)])
    vals_df.to_csv(path_vals, index=False)

    test_df = pd.DataFrame(tests, columns=['test auc', 'test ap'])
    test_df.to_csv(path_tests, index=False)

    train_df = pd.DataFrame(train, columns=['train auc', 'train ap'])
    train_df.to_csv(path_train, index=False)


def get_paths(output_dir, args):
    filename = get_filename(args)

    return output_dir.joinpath(filename + '_params.csv'), \
           output_dir.joinpath(filename + '_accuracy.csv'), \
           output_dir.joinpath(filename + '_costs.csv'), \
           output_dir.joinpath(filename + '_val.csv'), \
           output_dir.joinpath(filename + '_test.csv'), \
           output_dir.joinpath(filename + '_train.csv')


def get_filename(args):
    path = 'MODEL%s_TASK%s_ACT%s_SSUP%s_ALGN%s' % (args.layers,
                                                   args.task,
                                                   'relu' if 'relu' in str(args.activation) else 'linear',
                                                   args.support,
                                                   args.alignment[0] if args.alignment is not None else 'None')
    return path


def get_filename_from_txt(args):
    path = 'MODEL%s_TASK%s_ACT%s_SSUP%s_ALGN%s' % (args.layers[0],
                                                   args.task[0],
                                                   'relu' if 'relu' in str(args.activation[0]) else 'linear',
                                                   args.support[0],
                                                   np.array(args.alignment[0])[0])

    # path = 'MODEL%s_DEC%s_ACT%s_SSUP%s_DIM%s' % (args.layers[0],
    #                                              args.decoder[0],
    #                                              'relu' if 'relu' in str(args.activation[0]) else 'linear',
    #                                              args.support[0],
    #                                              args.hidden2[0])
    return path


def get_alignment(adj, features):
    if features is None:
        return None
    np.fill_diagonal(adj, np.ones(adj.shape[0]))
    d_inv2 = np.diag(np.power(np.array(np.sum(adj, axis=1)).flatten(), -0.5))
    adj = d_inv2 @ adj @ d_inv2
    features = normalize(features, axis=0)
    algn = normalize(adj @ features, axis=0).transpose() @ features
    algn[algn > 1] = 1
    return np.trace(np.arccos(algn)) /adj.shape[0]

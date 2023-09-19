
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow._api.v2.compat.v1 as tf
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gnn.plotting import plot_results, save_gif
from gnn.input_data import load_data
from gnn.optimizer import MyOptimizer, NoDecoderOptimizer
from gnn.parser import initialise_parser
from gnn.preprocessing import preprocess_graph, mask_test_edges, sparse_to_tuple, get_data, get_data_edge_level, \
    get_data_node_level, get_data_
from gnn.utils import update, AttrDict, mat_to_tuple, save_results, get_alignment
from gnn.models import get_model
from gnn.training import train_edge_level, train_node_level, train_match, train_cluster
from gnn.evaluation import evaluate_test, evaluate_test_match, evaluate_test_cluster

matplotlib.use('agg')

np.set_printoptions(suppress=True)

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Settings
COLORS = matplotlib.cm.get_cmap('Set1')

ARGUMENTS = AttrDict({'learning_rate': 0.01,
                      'epochs': 100,
                      'every_k_images': 5,
                      'num_nodes': 500,
                      'num_features': 2,
                      'hidden1': 32,  # any int larger 2 will do
                      'hidden2': 2,  # latent space dimension
                      'weight_decay': 0,
                      'dropout': 0,
                      'plot': True,
                      'dataset': 'gaussian_dot_5',
                      'featureless': True,
                      'runs': 1,
                      'experiment_name': 'cora_features',
                      'decoder': 'innerproduct',  # None, 'gaussdistance', 'innerproduct'
                      'activation': 'linear',  # 'linear' or 'relu'
                      'sigma': 0,  # 1, 0
                      'layers': 1,  # 0, 1 or 2
                      'regularizer': 'ls',  # 'kl', 'ls', None
                      'support': False,
                      'init': False,
                      'task': 'cluster',  # 'node' or 'edge'
                      'alignment': None})


def main(args):
    if args.experiment_name is None:
        Warning("No name chosen to save results! Saving as tmp.")
        args.experiment_name = 'tmp'

    if args.layers == 0:
        args.featuresless = True

    if args.activation == 'linear':
        args.activation = lambda x: x
    elif args.activation == 'relu':
        args.activation = tf.nn.relu
    else:
        raise ValueError("Invalid activation: {}. Choose from [linear, relu]".format(args.activation))

    #################################################################
    ########### Load data and preprocessing #########################
    #################################################################

    if args.dataset in ['cora', 'pubmed', 'citeseer', 'amazon']:
        adj, features_ = load_data(args.dataset)
        args.num_nodes = adj.shape[0]
        clusters = np.zeros([args.num_nodes])
        # features_ = sp.csr_matrix(np.load('feature_span.npy'))
    else:
        adj, features_, clusters, _ = get_data(args.dataset, args.num_nodes, args.num_features, args.alignment)

    args.alignment = [args.alignment, get_alignment(adj.todense(), features_.todense())]

    features_orig = features_

    if args.featureless:
        features_ = None

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'labels': tf.sparse_placeholder(tf.float32),
        'support': tf.sparse_placeholder(tf.float32)
    }

    accuracies = np.zeros([args.runs, args.epochs])
    costs = np.zeros([args.runs, args.epochs])
    vals = np.zeros([args.runs, args.epochs])
    tests = np.zeros([args.runs, 2])
    train = np.zeros([args.runs, 2])

    for r in range(args.runs):
        if args.task == 'edge':
            feed_dicts, adjs, edges_true, edges_false = get_data_edge_level(placeholders, adj, features_)
        elif args.task == 'node':
            feed_dicts, adjs, edges_true, edges_false = get_data_node_level(placeholders, adj, features_)
        else:
            feed_dicts, adjs, edges_true, edges_false = get_data_(placeholders, adj, features_)

        print('SHAPE: ', adjs['train'].shape)

        num_features = feed_dicts['train'][placeholders['features']][2][1]
        features_nonzero = feed_dicts['train'][placeholders['features']][1].shape[0]

        # get the model
        model = get_model(placeholders, args, num_features, features_nonzero)

        # Optimizer
        with tf.name_scope('optimizer'):
            adjt = adjs['train']
            pos_weight = float(adjt.shape[0] * adjt.shape[0] - adjt.sum()) / adjt.sum()
            norm = adjt.shape[0] * adjt.shape[0] / float(
                (adjt.shape[0] * adjt.shape[0] - adjt.sum()) * 2)

            if args.decoder is None:
                opt = NoDecoderOptimizer(outputs=model.outputs,
                                         labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['labels'],
                                                                                     validate_indices=False), [-1]),
                                         model=model,
                                         num_nodes=args.num_nodes)

            else:
                opt = MyOptimizer(preds=model.outputs,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['labels'],
                                                                              validate_indices=False), [-1]),
                                  model=model,
                                  num_nodes=args.num_nodes,
                                  pos_weight=pos_weight,
                                  norm=norm,
                                  regularizer=args.regularizer)

        #################################################################
        ########### Run training and evaluation #########################
        #################################################################

        if args.task == 'edge':
            model, sess, loss, accs, train_auc, train_ap, val_aucs, val_aps, filenames = train_edge_level(args, model,
                                                                                                          opt,
                                                                                                          feed_dicts,
                                                                                                          adjs,
                                                                                                          edges_true,
                                                                                                          edges_false)
            test_auc, test_ap = evaluate_test(model, sess, adjs['train'], feed_dicts['train'], edges_true['test'],
                                              edges_false['test'])
        elif args.task == 'node':
            model, sess, loss, accs, train_auc, train_ap, val_aucs, val_aps, filenames = train_node_level(args, model,
                                                                                                          opt,
                                                                                                          feed_dicts,
                                                                                                          adjs,
                                                                                                          edges_true,
                                                                                                          edges_false)
            test_auc, test_ap = evaluate_test(model, sess, adjs['test'], feed_dicts['test'], edges_true['test'],
                                              edges_false['test'])
        elif args.task == 'match':
            model, sess, loss, accs, train_auc, train_ap, val_aucs, val_aps, filenames = train_match(args, model,
                                                                                                     opt,
                                                                                                     feed_dicts,
                                                                                                     adjs,
                                                                                                     features_orig,
                                                                                                     edges_true,
                                                                                                     edges_false)
            test_auc, test_ap = evaluate_test_match(model, sess, features_orig, feed_dicts['train'])
        elif args.task == 'cluster':
            model, sess, loss, accs, train_auc, train_ap, val_aucs, val_aps, filenames = train_cluster(args, model,
                                                                                                       opt,
                                                                                                       feed_dicts,
                                                                                                       adjs,
                                                                                                       clusters,
                                                                                                       edges_true,
                                                                                                       edges_false)
            test_auc, test_ap = evaluate_test_cluster(model, sess, clusters, feed_dicts['train'])

        costs[r, :] = loss
        accuracies[r, :] = accs
        vals[r, :] = val_aucs

        train[r, 0] = train_auc[-1]
        train[r, 1] = train_ap[-1]

        tests[r, 0] = test_auc
        tests[r, 1] = test_ap

        print('Run %s - Final train acc: %s, final train cost: %s, final val roc: %s, test roc: %s' % (
            r, accs[-1], loss[-1], val_aucs[-1], test_auc))

    print("All done!")

    #################################################################
    ########### save results ########################################
    #################################################################

    save_results(costs, accuracies, vals, train, tests, args)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        initialise_parser(parser)
        args = parser.parse_args()
    else:
        args = ARGUMENTS
    main(args)

from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import sklearn.decomposition
import tensorflow._api.v2.compat.v1 as tf
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gnn.utils import progressbar
from gnn.plotting import plot_state_small

from gnn.plotting import plot_results, save_gif
from gnn.input_data import load_data
from gnn.optimizer import MyOptimizer, NoDecoderOptimizer, MyL2Optimizer
from gnn.parser import initialise_parser
from gnn.preprocessing import preprocess_graph, sparse_to_tuple, get_data, get_data_edge_level, get_data_node_level
from gnn.utils import update, AttrDict, mat_to_tuple
from gnn.training import train_node_level, train_edge_level
from gnn.models import get_model
from gnn.evaluation import evaluate_test

# matplotlib.use('agg')

np.set_printoptions(suppress=True)

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Settings
COLORS = matplotlib.cm.get_cmap('Set1')

ARGUMENTS = AttrDict({'learning_rate': 0.01,
                      'epochs': 50,
                      'every_k_images': 0,
                      'num_nodes': 500,
                      'num_features': 64,
                      'hidden1': 32,  # any int larger 2 will do
                      'hidden2': 16,  # latent space dimension
                      'weight_decay': 0,
                      'dropout': 0,
                      'plot': True,
                      'dataset': 'gaussian',
                      'featureless': False,
                      'experiment_name': 'weights_test',
                      'decoder': 'innerproduct',  # None, 'gaussdistance', 'innerproduct'
                      'loss': 'l2',  # 'l2', None
                      'activation': 'relu',  # 'linear' or 'relu'
                      'sigma': 0,  # 1, 0
                      'layers': 2,  # 0, 1 or 2
                      'regularizer': 'kl',  # 'kl', 'ls', None
                      'support': False,
                      'init': False,
                      'task': 'edge',
                      'alignment': 0})  # choose from 'node' or 'edge'


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

    if args.dataset in ['cora', 'pubmed', 'citeseer', 'amazon', 'facebook']:
        adj, features_ = load_data(args.dataset)
        args.num_nodes = adj.shape[0]
        clusters = np.zeros([args.num_nodes])
        # features_ = sp.csr_matrix(np.load('feature_span.npy'))
    else:
        adj, features_, clusters, f_old = get_data(args.dataset, args.num_nodes, args.num_features, args.alignment)
        args.num_nodes = adj.shape[0]

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.scatter(np.array(features_.todense())[:, 0].flatten(), np.array(features_.todense())[:, 1].flatten())
    # fig.savefig('original_' + args.experiment_name + '.png')

    if args.featureless:
        features_ = None

    density = np.sum(adj) / args.num_nodes ** 2
    print("Graph density: ", density)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'labels': tf.sparse_placeholder(tf.float32),
        'support': tf.sparse_placeholder(tf.float32)
    }

    if args.task == 'edge':
        feed_dicts, adjs, edges_true, edges_false = get_data_edge_level(placeholders, adj, features_)
    else:
        feed_dicts, adjs, edges_true, edges_false = get_data_node_level(placeholders, adj, features_)

    num_features = feed_dicts['train'][placeholders['features']][2][1]
    features_nonzero = feed_dicts['train'][placeholders['features']][1].shape[0]

    # get the model
    model = get_model(placeholders, args, num_features, features_nonzero)

    # Optimizer
    with tf.name_scope('optimizer'):
        adj = adjs['train']
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float(
            (adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        if args.decoder is None:
            opt = NoDecoderOptimizer(outputs=model.outputs,
                                     labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['labels'],
                                                                                 validate_indices=False), [-1]),
                                     model=model,
                                     num_nodes=args.num_nodes)

        elif args.loss == 'l2':
            opt = MyL2Optimizer(preds=model.outputs,
                                labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['labels'],
                                                                            validate_indices=False), [-1]),
                                model=model,
                                num_nodes=args.num_nodes,
                                pos_weight=pos_weight,
                                norm=norm,
                                regularizer=args.regularizer)

        else:
            opt = MyOptimizer(preds=model.outputs,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['labels'],
                                                                          validate_indices=False), [-1]),
                              model=model,
                              num_nodes=args.num_nodes,
                              pos_weight=pos_weight,
                              norm=norm,
                              regularizer=args.regularizer)

    if args.task == 'edge':
        model, sess, loss, accs, train_auc, train_ap, val_aucs, val_aps, filenames = train_edge_level(args, model, opt,
                                                                                                      feed_dicts,
                                                                                                      adjs,
                                                                                                      edges_true,
                                                                                                      edges_false)
        test_auc, test_ap = evaluate_test(model, sess, adjs['train'], feed_dicts['train'], edges_true['test'],
                                          edges_false['test'])
    else:
        model, sess, loss, accs, train_auc, train_ap, val_aucs, val_aps, filenames = train_node_level(args, model, opt,
                                                                                                      feed_dicts,
                                                                                                      adjs,
                                                                                                      edges_true,
                                                                                                      edges_false)
        test_auc, test_ap = evaluate_test(model, sess, adjs['test'], feed_dicts['test'], edges_true['test'],
                                          edges_false['test'])

    filenames, filenames_weights = filenames

    print('Test AUC score: ' + str(test_auc))
    print('Test AP score: ' + str(test_ap))

    print('Final train AUC score: ' + str(train_auc[-1]))
    print('Final train AP score: ' + str(train_ap[-1]))

    #################################################################
    ########### Plot training and results ###########################
    #################################################################

    if args.plot:
        # final learned embedding
        emb_train = sess.run(model.z_mean, feed_dict=feed_dicts['train'])
        _ = plot_results(emb_train, accs[-1], train_auc[-1], train_ap[-1],
                         adjs['train'] + np.eye(adjs['train'].shape[0]),
                         np.zeros(adjs['train'].shape[0]))
        plt.savefig(args.experiment_name + '.png')

        # save gif
        save_gif(filenames, args.experiment_name)

        save_gif(filenames_weights, args.experiment_name)

    print('done')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        initialise_parser(parser)
        args = AttrDict(parser.parse_args())
    else:
        args = ARGUMENTS
    main(args)

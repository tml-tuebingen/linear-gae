from scipy.spatial import procrustes
import sklearn.metrics
import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from gnn.plotting import fig2data, plot_state, plot_state_small, plot_state_small_colored, save_gif
from gnn.utils import progressbar
from gnn.evaluation import get_roc_score, evaluate_test


def train_node_level(args, model, opt,
                     feed_dicts, adjs, edges_true, edges_false):

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss = []
    accs = []
    auc = []
    ap = []
    val_aucs = []
    val_aps = []
    filenames = []

    epoch = 0
    # Train model
    for _ in progressbar(args.epochs, '\tTraining: ', 50):
        epoch += 1

        # Run single weight update
        (_, cost, acc, emb_train) = sess.run(
            [opt.opt_op, opt.loss, opt.accuracy, model.z_mean], feed_dict=feed_dicts['train'])

        train_auc, train_ap = get_roc_score(adjs['train'], edges_true['train'], edges_false['train'], emb_train)

        auc.append(train_auc)
        ap.append(train_ap)

        # if args.every_k_images and epoch % args.every_k_images == 0:
        #     filename = 'gnn/figs.nosync/tmp%d.png' % epoch
        #     figure = plot_state_small(emb_train, np.zeros(adjs['train'].shape[0]), epoch, adjs['train'])
        #     figure.savefig(filename)
        #     filenames.append(filename)
        #
        #     plt.close(figure)

        if 'val' in feed_dicts.keys():
            val_auc, val_ap = evaluate_test(model, sess, adjs['val'], feed_dicts['val'], edges_true['val'], edges_false['val'])
        else:
            val_auc = val_ap = -1

        val_aucs.append(val_auc)
        val_aps.append(val_ap)

        loss.append(cost)
        accs.append(acc)

        print("%s: \t train accuracy: %f, train cost: %f, train auc: %f, train ap: %f, val auc: %f, val ap: %f" % (
        epoch, acc, cost, train_auc, train_ap, val_auc, val_ap))

    print("Optimization Finished!")

    return model, sess, loss, accs, auc, ap, val_aucs, val_aps, filenames


def train_match(args, model, opt, feed_dicts, adjs, features, edges_true, edges_false):

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss = []
    accs = []
    auc = []
    sim = []
    val_aucs = []
    val_aps = []
    filenames = []

    epoch = 0
    # Train model
    for _ in progressbar(args.epochs, '\tTraining: ', 50):
        epoch += 1

        # Run single weight update
        (_, cost, acc, emb_train) = sess.run(
            [opt.opt_op, opt.loss, opt.accuracy, model.z_mean], feed_dict=feed_dicts['train'])

        train_auc, _ = get_roc_score(adjs['train'], edges_true['train'], edges_false['train'], emb_train)
        train_sim = procrustes(emb_train, features.todense())[-1]

        auc.append(train_auc)
        sim.append(train_sim)

        # if args.every_k_images and epoch % args.every_k_images == 0:
        #     filename = 'gnn/figs.nosync/tmp%d.png' % epoch
        #     figure = plot_state_small(emb_train, np.zeros(adjs['train'].shape[0]), epoch, adjs['train'])
        #     figure.savefig(filename)
        #     filenames.append(filename)
        #
        #     plt.close(figure)

        val_auc = val_ap = -1

        val_aucs.append(val_auc)
        val_aps.append(val_ap)

        loss.append(cost)
        accs.append(acc)

        print("%s: \t train accuracy: %f, train cost: %f, train auc: %f, train sim: %f" % (
        epoch, acc, cost, train_auc, train_sim))

    print("Optimization Finished!")

    return model, sess, loss, accs, auc, sim, val_aucs, val_aps, filenames


def train_cluster(args, model, opt, feed_dicts, adjs, clusters, edges_true, edges_false):

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss = []
    accs = []
    auc = []
    nmi = []
    val_aucs = []
    val_aps = []
    filenames = []

    epoch = 0
    # Train model
    for _ in progressbar(args.epochs, '\tTraining: ', 50):
        epoch += 1

        # Run single weight update
        (_, cost, acc, emb_train) = sess.run(
            [opt.opt_op, opt.loss, opt.accuracy, model.z_mean], feed_dict=feed_dicts['train'])

        train_auc, _ = get_roc_score(adjs['train'], edges_true['train'], edges_false['train'], emb_train)
        labels_pred = KMeans(n_clusters=5).fit(emb_train).labels_

        train_nmi = normalized_mutual_info_score(clusters, labels_pred)

        auc.append(train_auc)
        nmi.append(train_nmi)

        # if args.every_k_images and epoch % args.every_k_images == 0:
        #     filename = 'gnn/figs.nosync/tmp%d.png' % epoch
        #     figure = plot_state_small(emb_train, np.zeros(adjs['train'].shape[0]), epoch, adjs['train'])
        #     figure.savefig(filename)
        #     filenames.append(filename)
        #
        #     plt.close(figure)

        val_auc = val_ap = -1

        val_aucs.append(val_auc)
        val_aps.append(val_ap)

        loss.append(cost)
        accs.append(acc)

        print("%s: \t train accuracy: %f, train cost: %f, train auc: %f, train nmi: %f" % (
        epoch, acc, cost, train_auc, train_nmi))

    print("Optimization Finished!")

    return model, sess, loss, accs, auc, nmi, val_aucs, val_aps, filenames


def train_edge_level(args, model, opt,
                     feed_dicts, adjs, edges_true, edges_false):

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss = []
    accs = []
    auc = []
    ap = []
    val_aucs = []
    val_aps = []
    filenames = []
    filenames_weights = []

    epoch = 0
    # Train model
    for _ in progressbar(args.epochs, '\tTraining: ', 50):
        epoch += 1

        # Run single weight update
        (_, cost, acc, emb, vars) = sess.run(
            [opt.opt_op, opt.loss, opt.accuracy, model.z_mean, model.vars], feed_dict=feed_dicts['train'])

        fig, ax = plt.subplots(1, 3)

        for i in range(3):
            ax[i].imshow(list(vars.values())[i], norm=mpl.colors.CenteredNorm(), cmap='Blues')

        filenames_weights.append('gnn/figs.nosync/tmp_weights%d.png' % epoch)
        fig.savefig(filenames_weights[-1])

        train_auc, train_ap = get_roc_score(adjs['train'], edges_true['train'], edges_false['train'], emb)

        auc.append(train_auc)
        ap.append(train_ap)

        if args.every_k_images and epoch % args.every_k_images == 0:
            filename = 'gnn/figs.nosync/tmp%d.png' % epoch
            figure = plot_state_small(emb, np.zeros(adjs['train'].shape[0]), epoch, adjs['train'])
            figure.savefig(filename)
            filenames.append(filename)

            plt.close(figure)

        val_auc, val_ap = get_roc_score(adjs['train'], edges_true['val'], edges_false['val'], emb)

        val_aucs.append(val_auc)
        val_aps.append(val_ap)

        loss.append(cost)
        accs.append(acc)

        print("%s: \t train accuracy: %f, train cost: %f, train auc: %f, train ap: %f, val auc: %f, val ap: %f" % (
            epoch, acc, cost, train_auc, train_ap, val_auc, val_ap))

    print("Optimization Finished!")

    return model, sess, loss, accs, auc, ap, val_aucs, val_aps, [filenames, filenames_weights]



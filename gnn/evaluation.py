import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, normalized_mutual_info_score, adjusted_rand_score
from gnn.preprocessing import sigmoid
from sklearn.cluster import KMeans
from gnn.utils import tuple_to_dense
import scipy.spatial


def evaluate_test(model, sess, adj, feed_dict, egdes_true, edges_false):
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    auc, ap = get_roc_score(adj, egdes_true, edges_false, emb)

    return auc, ap


def evaluate_test_cluster(model, sess, clusters, feed_dict):
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    labels_pred = KMeans(n_clusters=2).fit(emb).labels_

    nmi = normalized_mutual_info_score(clusters, labels_pred)
    ars = adjusted_rand_score(clusters, labels_pred)

    return nmi, ars


def evaluate_test_match(model, sess, features, feed_dict):
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    sim = scipy.spatial.procrustes(emb, features.todense())

    return sim[-1], -1


def get_roc_score(adj, edges_pos, edges_neg, emb=None):

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

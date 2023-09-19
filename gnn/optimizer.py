import tensorflow._api.v2.compat.v1 as tf
import numpy as np
from gnn.loss import weighted_cross_entropy

learning_rate = 0.0001

class MyL2Optimizer(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, regularizer):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.log_lik = norm * tf.nn.l2_loss((preds - labels) * pos_weight)

        if regularizer is None:
            self.regularizer = 0.0
        else:
            self.single_regularizer = get_regularizer(model, regularizer)

            self.regularizer_sums = tf.reduce_sum(self.single_regularizer, 1)

            self.regularizer = (0.5/num_nodes) * tf.reduce_mean(self.regularizer_sums)

        #self.loss = self.log_lik
        self.loss = tf.add(self.log_lik, self.regularizer)

        self.opt_op = self.optimizer.minimize(self.loss)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(preds-1), 0.5), tf.int32),
                                           tf.cast(labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes):
        preds_sub = preds
        labels_sub = labels

        # weighted loss
        shape = np.sqrt(int(labels.shape[0]))

        pos_weight = (shape * shape - tf.reduce_sum(labels)) / tf.reduce_sum(labels)
        norm = shape * shape / ((shape * shape - tf.reduce_sum(labels)) * 2)

        self.loss = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, labels=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.loss
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.loss -= self.kl

        self.opt_op = self.optimizer.minimize(self.loss)

        self.grads_vars = self.optimizer.compute_gradients(self.loss)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class regressionOptimizer(object):
    def __init__(self, model, preds, labels):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)

        self.loss = tf.reduce_mean(tf.square(tf.subtract(preds, labels)), axis=0)

        self.opt_op = self.optimizer.minimize(self.loss)


class MyOptimizer(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, regularizer):

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.log_lik = norm * tf.reduce_mean(
            #weighted_cross_entropy(logits=preds, labels=labels, pos_weight=pos_weight))
                    tf.nn.weighted_cross_entropy_with_logits(logits=preds, labels=labels, pos_weight=pos_weight))

        self.single_regularizer = get_regularizer(model, regularizer)

        self.regularizer_sums = tf.reduce_sum(self.single_regularizer, 1)

        self.regularizer = (0.5/num_nodes) * tf.reduce_mean(self.regularizer_sums)

        #self.loss = self.log_lik
        self.loss = tf.add(self.log_lik, self.regularizer)

        self.opt_op = self.optimizer.minimize(self.loss)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(preds-1), 0.5), tf.int32),
                                           tf.cast(labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class NoDecoderOptimizer(object):
    def __init__(self, outputs, labels, model, num_nodes):
        shape = np.sqrt(int(labels.shape[0]))

        squared_norm = tf.reduce_sum(tf.square(outputs), axis=1) + 1.0e-12

        b = tf.reshape(squared_norm, [1, -1])
        a = tf.reshape(squared_norm, [-1, 1])

        pairwise_dist = tf.sqrt(tf.maximum(a - 2 * tf.matmul(outputs, outputs, False, True) + b, 0.0))

        D = tf.clip_by_value(tf.exp(-pairwise_dist), 1.0e-15, 0.99)

        preds = D - 0.5

        norm = shape * shape / (shape * shape - tf.reduce_sum(labels))
        pos_weight = (shape * shape - tf.reduce_sum(labels)) / tf.reduce_sum(labels)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.log_lik = norm * tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(logits=preds, labels=labels, pos_weight=pos_weight))

        self.single_kls = -0.5 - model.z_log_std + tf.multiply(0.5, tf.square(tf.exp(model.z_log_std))) + tf.multiply(0.5, tf.square(tf.subtract(model.z_mean, 0)))

        self.kl_sums = tf.reduce_sum(self.single_kls, 1)

        self.kl = tf.divide(tf.reduce_mean(self.kl_sums), num_nodes)

        self.loss = tf.add(self.log_lik, self.kl)

        self.opt_op = self.optimizer.minimize(self.loss)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(preds-1), 0.5), tf.int32),
                                           tf.cast(labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


def get_regularizer(model, regularizer):
    if regularizer == 'kl':
        #single_regs = 1.0 + 2.0 * model.z_log_std - tf.square(model.z_mean) - tf.square(tf.exp(model.z_log_std))
        single_regs = -0.5 + model.z_log_std + tf.multiply(0.5, tf.square(tf.exp(model.z_log_std))) + tf.multiply(0.5, tf.square(model.z_mean))
    elif regularizer == 'ls':
        #single_regs = 1.0 + 2.0 - tf.square(model.z_mean) - tf.square(tf.exp(1.0))
        single_regs = 0.5 * (1.0 + tf.square(tf.exp(1.0)) + tf.square(model.z_mean))
    elif regularizer is None:
        single_regs = 0.0
    else:
        Warning('No valid regularizer given using! using kl regularizer')
        single_regs = -0.5 - model.z_log_std + tf.multiply(0.5, tf.square(tf.exp(model.z_log_std))) + tf.multiply(0.5, tf.square(model.z_mean))
    return single_regs

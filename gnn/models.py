from gnn.layers import *
from gnn.metrics import *


class Model(object):
    def __init__(self, **kwargs):

        # validate input
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')

        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholder = {}

        self.layers = []

    def predict(self):
        pass

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def save(self, sess=None):
        if not sess:
            raise AttributeError("Tensorflow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckp" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("Tensorflow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckp" % self.name)
        saver.restore(sess, save_path)
        print("Model restored from path: %s" % save_path)


class GAE_zero(Model):
    def __init__(self, placeholders, hidden_dim, num_features, num_nodes, features_nonzero, activation=tf.nn.relu,
                 sigma=1, dropout=0, weight_decay=0,
                 learning_rate=0.001, decoder='gaussdistance', **kwargs):
        super(GAE_zero, self).__init__(**kwargs)

        self.n_samples = num_nodes
        self.num_features = num_features

        self.hidden1_dim = hidden_dim[0]
        self.hidden2_dim = hidden_dim[1]

        self.inputs = placeholders['features']
        self.support = placeholders['support']
        self.labels = placeholders['labels']

        self.sigma = sigma
        self.act = activation
        self.features_nonzero = features_nonzero
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.decoder = decoder

        self.build()

    def _build(self):
        self.z_mean = MatrixFactorization(input_dim=self.num_featres,
                                          output_dim=self.hidden2_dim,
                                          features_nonzero=self.features_nonzero,
                                          logging=self.logging)(self.inputs)

        self.z = self.z_mean

        if self.decoder == 'innerproduct':
            self.outputs = InnerProductDecoder(act=lambda x: x,
                                               logging=self.logging)(self.z)
        elif self.decoder == 'gaussdistance':

            self.prediction = GaussDistanceRecovery(act=lambda x: x,
                                                    logging=self.logging)(self.z)

            self.outputs = tf.reshape(self.prediction, [-1])
        else:
            self.outputs = self.z


class GAE_one(Model):
    def __init__(self, placeholders, hidden_dim, num_features, num_nodes, features_nonzero, activation=tf.nn.relu,
                 sigma=1, dropout=0, weight_decay=0,
                 learning_rate=0.001, decoder='gaussdistance', **kwargs):
        super(GAE_one, self).__init__(**kwargs)

        self.num_features = num_features
        self.n_samples = num_nodes

        self.hidden1_dim = hidden_dim[0]
        self.hidden2_dim = hidden_dim[1]

        self.inputs = placeholders['features']
        self.support = placeholders['support']
        self.labels = placeholders['labels']

        self.sigma = sigma
        self.act = activation
        self.features_nonzero = features_nonzero
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.decoder = decoder

        self.build()

    def _build(self):
        self.l1 = GraphConvolutionSparse(input_dim=self.num_features,
                                         output_dim=self.hidden1_dim,
                                         support=self.support,
                                         features_nonzero=self.features_nonzero,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)

        self.l2 = GraphConvolution(input_dim=self.hidden1_dim,
                                   output_dim=self.hidden2_dim,
                                   support=None,
                                   act=lambda x: x,
                                   dropout=self.dropout,
                                   logging=self.logging)

        self.hidden = self.l1(self.inputs)

        self.z_mean = self.l2(self.hidden)

        self.z = self.z_mean

        if self.decoder == 'innerproduct':
            self.outputs = InnerProductDecoder(act=lambda x: x,
                                               logging=self.logging)(self.z)
        elif self.decoder == 'gaussdistance':

            self.prediction = GaussDistanceRecovery(act=lambda x: x,
                                                    logging=self.logging)(self.z)

            self.outputs = tf.reshape(self.prediction, [-1])
        else:
            self.outputs = self.z


class GAE_two(Model):
    def __init__(self, placeholders, hidden_dim, num_features, num_nodes, features_nonzero, activation=tf.nn.relu,
                 sigma=1, dropout=0, weight_decay=0,
                 learning_rate=0.001, decoder='gaussdistance', **kwargs):
        super(GAE_two, self).__init__(**kwargs)

        self.n_samples = num_nodes
        self.num_features = num_features

        self.hidden1_dim = hidden_dim[0]
        self.hidden2_dim = hidden_dim[1]

        self.inputs = placeholders['features']
        self.support = placeholders['support']
        self.labels = placeholders['labels']

        self.sigma = sigma
        self.act = activation
        self.features_nonzero = features_nonzero
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.decoder = decoder

        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.num_features,
                                              output_dim=self.hidden1_dim,
                                              support=self.support,
                                              features_nonzero=self.features_nonzero,
                                              act=self.act,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       support=self.support,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=self.hidden1_dim,
                                          output_dim=self.hidden2_dim,
                                          support=self.support,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal(tf.shape(self.z_mean)) * tf.exp(
            self.z_log_std) * self.sigma

        if self.decoder == 'innerproduct':
            self.outputs = InnerProductDecoder(act=lambda x: x,
                                               logging=self.logging)(self.z)
        elif self.decoder == 'gaussdistance':

            self.prediction = GaussDistanceRecovery(act=lambda x: x,
                                                    logging=self.logging)(self.z)

            self.outputs = tf.reshape(self.prediction, [-1])
        else:
            self.outputs = self.z


class GCN(Model):
    def __init__(self, placeholders, hidden_dim, dropout, learning_rate, activation=tf.nn.relu, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = tf.constant(100)
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = tf.constant(1)
        self.placeholders = placeholders
        self.support = placeholders['support']
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.activation = activation

        self.build()

    def _build(self):
        self.hidden_layers = GraphConvolution(input_dim=self.input_dim,
                                              output_dim=self.hidden_dim,
                                              support=self.support,
                                              act=self.activation,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.output = GraphConvolution(input_dim=self.hidden_dim,
                                       output_dim=self.output_dim,
                                       support=self.support,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden_layers)

        self.prediction = tf.reduce_sum(self.output, axis=[1,2])


def get_model(placeholders, args, num_features, features_nonzero):
    if args.layers == 0:
        model = GAE_zero(placeholders, name='no_layer',
                         hidden_dim=[args.hidden1, args.hidden2],
                         num_features=num_features,
                         num_nodes=args.num_nodes,
                         features_nonzero=features_nonzero,
                         activation=args.activation,
                         sigma=args.sigma,
                         dropout=args.dropout,
                         weight_decay=args.weight_decay,
                         learning_rate=args.learning_rate,
                         decoder=args.decoder)
    elif args.layers == 1:
        model = GAE_one(placeholders, name='single_layer',
                        hidden_dim=[args.hidden1, args.hidden2],
                        num_features=num_features,
                        num_nodes=args.num_nodes,
                        features_nonzero=features_nonzero,
                        activation=args.activation,
                        sigma=args.sigma,
                        dropout=args.dropout,
                        weight_decay=args.weight_decay,
                        learning_rate=args.learning_rate,
                        decoder=args.decoder)
    elif args.layers == 2:
        model = GAE_two(placeholders, name='two_layers',
                        hidden_dim=[args.hidden1, args.hidden2],
                        num_features=num_features,
                        num_nodes=args.num_nodes,
                        features_nonzero=features_nonzero,
                        activation=args.activation,
                        sigma=args.sigma,
                        dropout=args.dropout,
                        weight_decay=args.weight_decay,
                        learning_rate=args.learning_rate,
                        decoder=args.decoder)
    else:
        raise NotImplementedError("Model with %s layers not implemented. Choose from 0, 1, 2." % args.layers)

    return model

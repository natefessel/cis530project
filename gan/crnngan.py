"""

The hyperparameters used in the model:
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- review_length - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- epochs_before_decay - the number of epochs trained with the initial learning rate
- max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "epochs_before_decay"
- batch_size - the batch size

The hyperparameters that could be used in the model:
- init_scale - the initial scale of the weights

To run:

$ python rnn_gan.py --model small|medium|large --datadir simple-examples/data/ --traindir dir-for-checkpoints-and-plots --select_validation_percentage 0-40 --select_test_percentage 0-40
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import os
import sys
import pickle as pkl

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from data_loader import DataLoader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "datadir", None, "Directory to save and load midi music files.")
flags.DEFINE_string(
    "traindir", None, "Directory to save checkpoints and gnuplot files.")
flags.DEFINE_integer("epochs_per_checkpoint", 2,
                     "How many training epochs to do per checkpoint.")
flags.DEFINE_boolean("log_device_placement", False,           #
                     "Outputs info on device placement.")
flags.DEFINE_integer("exit_after", 1440,
                     "exit after this many minutes")
flags.DEFINE_boolean("sample", False,
                     "Sample output from the model. Assume training was already done. Save sample output to file.")
flags.DEFINE_float("init_scale", 0.05,                # .1, .04
                   "the initial scale of the weights")
flags.DEFINE_float("learning_rate", 0.1,              # .05,.1,.9
                   "Learning rate")
flags.DEFINE_float("d_lr_factor", 0.5,                # .5
                   "Learning rate decay")
flags.DEFINE_float("max_grad_norm", 5.0,              # 5.0, 10.0
                   "the maximum permissible norm of the gradient")
flags.DEFINE_float("keep_prob", 0.5,                  # 1.0, .35
                   "Keep probability. 1.0 disables dropout.")
flags.DEFINE_float("lr_decay", 1.0,                   # 1.0
                   "Learning rate decay after each epoch after epochs_before_decay")
flags.DEFINE_integer("num_layers_g", 2,                 # 2
                     "Number of stacked recurrent cells in G.")
flags.DEFINE_integer("num_layers_d", 2,                 # 2
                     "Number of stacked recurrent cells in D.")
flags.DEFINE_integer("review_length", 100,               # 200, 500
                     "Limit review inputs to this number of events.")
flags.DEFINE_integer("hidden_size_g", 100,              # 200, 1500
                     "Hidden size for recurrent part of G.")
flags.DEFINE_integer("hidden_size_d", 100,              # 200, 1500
                     "Hidden size for recurrent part of D. Default: same as for G.")
flags.DEFINE_integer("epochs_before_decay", 60,       # 40, 140
                     "Number of epochs before starting to decay.")
flags.DEFINE_integer("max_epoch", 500,                # 500, 500
                     "Number of epochs before stopping training.")
flags.DEFINE_integer("batch_size", 20,                # 10, 20
                     "Batch size.")
flags.DEFINE_integer("pretraining_epochs", 6,        # 20, 40
                     "Number of epochs to run lang-model style pretraining.")
flags.DEFINE_boolean("pretraining_d", False,          #
                     "Train D during pretraining.")
flags.DEFINE_boolean("float16", False,                #
                     "Use floa16 data type. Otherwise, use float32.")

flags.DEFINE_boolean("adam", False,                   #
                     "Use Adam optimizer.")
flags.DEFINE_boolean("feature_matching", False,       #
                     "Feature matching objective for G.")
flags.DEFINE_float("reg_scale", 1.0,       #
                   "L2 regularization scale.")
flags.DEFINE_float("random_input_scale", 1.0,       #
                   "Scale of random inputs (1.0=same size as generated features).")
flags.DEFINE_boolean("end_classification", False,
                     "Classify only in ends of D. Otherwise, does classification at every timestep and mean reduce.")

FLAGS = flags.FLAGS

model_layout_flags = [
    'num_layers_g', 'num_layers_d', 'hidden_size_g',
    'hidden_size_d', 'feature_matching',
]


def make_rnn_cell(
    rnn_layer_sizes,
    dropout_keep_prob=1.0,
    attn_length=0,
    base_cell=tf.contrib.rnn.BasicLSTMCell,
    state_is_tuple=True,
    reuse=False,
):
    """Makes a RNN cell from the given hyperparameters.

    Args:
      rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
      dropout_keep_prob: The float probability to keep the output of any given sub-cell.
      attn_length: The size of the attention vector.
      base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
      state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
          and cell matrix as a state instead of a concatenated matrix.

    Returns:
        A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
    """
    cells = []
    for num_units in rnn_layer_sizes:
        cell = base_cell(num_units, state_is_tuple=state_is_tuple, reuse=reuse)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
    if attn_length:
        cell = tf.contrib.rnn.AttentionCellWrapper(
            cell, attn_length, state_is_tuple=state_is_tuple, reuse=reuse)

    return cell


def data_type():
    return tf.float16 if FLAGS.float16 else tf.float32


def linear(inp, output_dim, scope=None, stddev=1.0, reuse_scope=False):
    norm = tf.random_normal_initializer(stddev=stddev, dtype=data_type())
    const = tf.constant_initializer(0.0, dtype=data_type())
    with tf.variable_scope(scope or 'linear') as scope:
        scope.set_regularizer(
            tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
        if reuse_scope:
            scope.reuse_variables()
        #print('inp.get_shape(): {}'.format(inp.get_shape()))
        w = tf.get_variable(
            'w', [inp.get_shape()[1], output_dim], initializer=norm, dtype=data_type())
        b = tf.get_variable('b', [output_dim],
                            initializer=const, dtype=data_type())
    return tf.matmul(inp, w) + b


class RNNGAN(object):
    """The RNNGAN model."""

    def __init__(self, is_training, num_review_features=None):
        batch_size = FLAGS.batch_size
        self.batch_size = batch_size

        review_length = FLAGS.review_length
        self.review_length = review_length

        print('review_length: {}'.format(self.review_length))
        self._input_review_data = tf.placeholder(
            shape=[batch_size, review_length, num_review_features], dtype=data_type())
        review_data_inputs = [
            tf.squeeze(input_, [1])
            for input_ in tf.split(self._input_review_data, review_length, 1)
        ]
        print(review_data_inputs)

        with tf.variable_scope('G') as scope:
            scope.set_regularizer(
                tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            if is_training and FLAGS.keep_prob < 1:
                cell = make_rnn_cell(
                    [FLAGS.hidden_size_g] * FLAGS.num_layers_g,
                    dropout_keep_prob=FLAGS.keep_prob,
                )
            else:
                cell = make_rnn_cell(
                    [FLAGS.hidden_size_g] * FLAGS.num_layers_g
                )

            self._initial_state = cell.zero_state(batch_size, data_type())

            random_rnninputs = tf.random_uniform(
                shape=[
                    batch_size,
                    review_length,
                    int(FLAGS.random_input_scale * num_review_features)
                ],
                minval=0.0,
                maxval=1.0,
                dtype=data_type(),
            )
            random_rnninputs = [
                tf.squeeze(input_, [1])
                for input_ in tf.split(random_rnninputs, review_length, 1)
            ]

            # REAL GENERATOR:
            state = self._initial_state
            # as we feed the output as the input to the next, we 'invent' the initial 'output'.
            generated_point = tf.random_uniform(
                shape=[batch_size, num_review_features],
                minval=0.0,
                maxval=1.0,
                dtype=data_type()
            )
            outputs = []
            self._generated_features = []
            for i, input_ in enumerate(random_rnninputs):
                if i > 0:
                    scope.reuse_variables()
                concat_values = [input_]
                concat_values.append(generated_point)

                if len(concat_values):
                    input_ = tf.concat(axis=1, values=concat_values)

                input_ = tf.nn.relu(
                    linear(
                        input_, FLAGS.hidden_size_g,
                        scope='input_layer', reuse_scope=(i != 0)
                    )
                )
                output, state = cell(input_, state)
                outputs.append(output)
                generated_point = linear(
                    output, num_review_features, scope='output_layer', reuse_scope=(i != 0))
                self._generated_features.append(generated_point)

            # PRETRAINING GENERATOR, will feed inputs, not generated outputs:
            scope.reuse_variables()
            # as we feed the output as the input to the next, we 'invent' the initial 'output'.
            prev_target = tf.random_uniform(
                shape=[batch_size, num_review_features], minval=0.0, maxval=1.0, dtype=data_type())
            outputs = []
            self._generated_features_pretraining = []
            for i, input_ in enumerate(random_rnninputs):
                concat_values = [input_]
                concat_values.append(prev_target)

                if len(concat_values):
                    input_ = tf.concat(axis=1, values=concat_values)
                input_ = tf.nn.relu(
                    linear(input_, FLAGS.hidden_size_g, scope='input_layer', reuse_scope=(i != 0)))
                output, state = cell(input_, state)
                outputs.append(output)
                #generated_point = tf.nn.relu(linear(output, num_review_features, scope='output_layer', reuse_scope=(i!=0)))
                generated_point = linear(
                    output, num_review_features, scope='output_layer', reuse_scope=(i != 0))
                self._generated_features_pretraining.append(generated_point)
                prev_target = review_data_inputs[i]

        # These are used both for pretraining and for D/G training further down.
        self._lr = tf.Variable(
            FLAGS.learning_rate,
            trainable=False,
            dtype=data_type(),
        )
        self.g_params = [
            v for v in tf.trainable_variables()
            if v.name.startswith('model/G/')
        ]
        if FLAGS.adam:
            g_optimizer = tf.train.AdamOptimizer(self._lr)
        else:
            g_optimizer = tf.train.GradientDescentOptimizer(self._lr)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.1  # Choose an appropriate one.
        reg_loss = reg_constant * sum(reg_losses)
        reg_loss = tf.Print(reg_loss, reg_losses,
                            'reg_losses = ', summarize=20, first_n=20)

        # ---BEGIN, PRETRAINING. ---

        print(tf.transpose(tf.stack(self._generated_features_pretraining),
                           perm=[1, 0, 2]).get_shape())
        print(self._input_review_data.get_shape())
        self.rnn_pretraining_loss = tf.reduce_mean(tf.squared_difference(x=tf.transpose(
            tf.stack(self._generated_features_pretraining), perm=[1, 0, 2]), y=self._input_review_data))
        self.rnn_pretraining_loss = self.rnn_pretraining_loss + reg_loss

        pretraining_grads, _ = tf.clip_by_global_norm(tf.gradients(
            self.rnn_pretraining_loss, self.g_params), FLAGS.max_grad_norm)
        self.opt_pretraining = g_optimizer.apply_gradients(
            zip(pretraining_grads, self.g_params))

        # ---END, PRETRAINING---

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('D') as scope:
            scope.set_regularizer(
                tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            self.real_d, self.real_d_features = self.discriminator(
                review_data_inputs, is_training, msg='real')
            scope.reuse_variables()

            generated_data = self._generated_features
            if review_data_inputs[0].get_shape() != generated_data[0].get_shape():
                print('review_data_inputs shape {} != generated data shape {}'.format(
                    review_data_inputs[0].get_shape(), generated_data[0].get_shape()))
            self.generated_d, self.generated_d_features = self.discriminator(
                generated_data, is_training, msg='generated')

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.real_d, 1e-1000000, 1.0))
                                     - tf.log(1 - tf.clip_by_value(self.generated_d, 0.0, 1.0 - 1e-1000000)))
        self.g_loss_feature_matching = tf.reduce_sum(
            tf.squared_difference(self.real_d_features, self.generated_d_features))
        self.g_loss = tf.reduce_mean(
            -tf.log(tf.clip_by_value(self.generated_d, 1e-1000000, 1.0)))

        self.d_loss = self.d_loss + reg_loss
        self.g_loss_feature_matching = self.g_loss_feature_matching + reg_loss
        self.g_loss = self.g_loss + reg_loss
        self.d_params = [v for v in tf.trainable_variables()
                         if v.name.startswith('model/D/')]

        if not is_training:
            return

        d_optimizer = tf.train.GradientDescentOptimizer(
            self._lr * FLAGS.d_lr_factor
        )
        d_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.d_loss, self.d_params),
            FLAGS.max_grad_norm
        )
        self.opt_d = d_optimizer.apply_gradients(zip(d_grads, self.d_params))
        g_grads, _ = tf.clip_by_global_norm(
            tf.gradients(
                self.g_loss_feature_matching if FLAGS.feature_matching else self.g_loss,
                self.g_params
            ),
            FLAGS.max_grad_norm,
        )
        self.opt_g = g_optimizer.apply_gradients(zip(g_grads, self.g_params))

        self._new_lr = tf.placeholder(
            shape=[],
            name="new_learning_rate",
            dtype=data_type()
        )
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def discriminator(self, inputs, is_training, msg=''):
        print('discriminator(inputs):', inputs)
        if is_training and FLAGS.keep_prob < 1:
            inputs = [
                tf.nn.dropout(input_, FLAGS.keep_prob)
                for input_ in inputs
            ]
            print('discriminator(inputs) - override:', inputs)

        keep_prob = FLAGS.keep_prob if is_training and FLAGS.keep_prob < 1 else 1.0
        cell_fw = make_rnn_cell(
            [FLAGS.hidden_size_d] * FLAGS.num_layers_d,
            dropout_keep_prob=keep_prob,
        )
        cell_bw = make_rnn_cell(
            [FLAGS.hidden_size_d] * FLAGS.num_layers_d,
            dropout_keep_prob=keep_prob,
        )

        self._initial_state_fw = cell_fw.zero_state(self.batch_size, data_type())

        # if not FLAGS.unidirectional_d:
        self._initial_state_bw = cell_bw.zero_state(self.batch_size, data_type())
        print("cell_fw",cell_fw.output_size)
        outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=self._initial_state_fw, initial_state_bw=self._initial_state_bw)
        # else:
        #   outputs, state = tf.nn.rnn(cell_fw, inputs, initial_state=self._initial_state_fw)

        decisions = (
            enumerate([outputs[0], outputs[-1]])
            if FLAGS.end_classification
            else enumerate(outputs)
        )
        decisions = [
            tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i != 0)))
            for i, output in decisions
        ]
        decisions = tf.stack(decisions)
        decisions = tf.transpose(decisions, perm=[1, 0, 2])
        print('shape, decisions: {}'.format(decisions.get_shape()))
        decision = tf.reduce_mean(decisions, reduction_indices=[1, 2])
        decision = tf.Print(decision, [decision],
                            '{} decision = '.format(msg), summarize=20, first_n=20)
        return (decision, tf.transpose(tf.stack(outputs), perm=[1, 0, 2]))

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def generated_features(self):
        return self._generated_features

    @property
    def input_review_data(self):
        return self._input_review_data

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def lr(self):
        return self._lr


def run_epoch(session, model, loader, datasetlabel, eval_op_g, eval_op_d, pretraining=False, verbose=False, pretraining_d=False):
    """Runs the model on the given data."""
    epoch_start_time = time.time()
    g_loss, d_loss = 10.0, 10.0
    g_losses, d_losses = 0.0, 0.0
    iters = 0
    loader.rewind(part=datasetlabel)
    batch_review = loader.get_batch(
        model.batch_size,
        model.review_length,
        part=datasetlabel,
    )

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    while batch_review is not None:
        op_g = eval_op_g
        op_d = eval_op_d
        if datasetlabel == 'train' and not pretraining:
            if d_loss == 0.0 and g_loss == 0.0:
                print('Both G and D train loss are zero. Exiting.')
                break
            elif d_loss == 0.0:
                op_g = tf.no_op()
            elif g_loss == 0.0:
                op_d = tf.no_op()
            elif g_loss < 2.0 or d_loss < 2.0:
                if g_loss * .7 > d_loss:
                    op_g = tf.no_op()
                op_d = tf.no_op()

        if pretraining:
            if pretraining_d:
                fetches = [model.rnn_pretraining_loss, model.d_loss, op_g, op_d]
            else:
                fetches = [model.rnn_pretraining_loss, tf.no_op(), op_g, op_d]
        else:
            fetches = [model.g_loss, model.d_loss, op_g, op_d]

        if iters <= 0:
            g_loss, d_loss, _, _ = session.run(fetches, {
                model.input_review_data.name: batch_review
            })
        g_losses += g_loss
        if not pretraining:
            d_losses += d_loss
        iters += 1

        if verbose and iters % 10 == 9:
            reviews_per_sec = (
                float(iters * model.batch_size) /
                float(time.time() - epoch_start_time)
            )

            if pretraining:
                print(
                    "{}: {} (pretraining) batch loss: G: {:.3f}, avg loss: G: {:.3f}, speed: {:.1f} reviews/s, avg in graph: {:.1f}, avg in python: {:.1f}.".format(
                        datasetlabel,
                        iters,
                        g_loss,
                        float(g_losses) / float(iters),
                        reviews_per_sec,
                    )
                )
            else:
                print(
                    "{}: {} batch loss: G: {:.3f}, D: {:.3f}, avg loss: G: {:.3f}, D: {:.3f}, speed: {:.1f} reviews/s, avg in graph: {:.1f}, avg in python: {:.1f}.".format(
                        datasetlabel,
                        iters,
                        g_loss,
                        d_loss,
                        float(g_losses) / float(iters),
                        float(d_losses) / float(iters),
                        reviews_per_sec,
                    )
                )
        batch_review = loader.get_batch(
            model.batch_size,
            model.review_length,
            part=datasetlabel
        )

    if iters == 0:
        return (None, None)

    g_mean_loss = g_losses / iters
    d_mean_loss = None if pretraining and not pretraining_d else d_losses / iters
    return (g_mean_loss, d_mean_loss)


def sample(session, model, batch=False):
    """Samples from the generative model."""
    generated_features, = session.run([model.generated_features], {})
    if batch:
        returnable = [
            [x[batchno, :] for x in generated_features]
            for batchno in range(generated_features[0].shape[0])
        ]
    else:
        returnable = [x[0, :] for x in generated_features]
    return returnable


def main(_):
    if not FLAGS.datadir:
        raise ValueError("Must set --datadir to midi music dir.")
    if not FLAGS.traindir:
        raise ValueError(
            "Must set --traindir to dir where I can save model and plots.")

    generated_data_dir = os.path.join(FLAGS.traindir, 'generated_data')
    try:
        os.makedirs(FLAGS.traindir)
    except:
        pass
    try:
        os.makedirs(generated_data_dir)
    except:
        pass

    directorynames = FLAGS.traindir.split('/')
    experiment_label = ''
    while not experiment_label:
        experiment_label = directorynames.pop()

    global_step = -1
    if os.path.exists(os.path.join(FLAGS.traindir, 'global_step.pkl')):
        with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'r') as f:
            global_step = pkl.load(f)
    global_step += 1

    loader = DataLoader()

    num_review_features = loader.embedding_dimension
    print('num_review_features:{}'.format(num_review_features))

    train_start_time = time.time()
    checkpoint_path = os.path.join(FLAGS.traindir, "model.ckpt")

    review_length_ceiling = FLAGS.review_length
    if global_step < FLAGS.pretraining_epochs:
        FLAGS.review_length = int(min((global_step + 1) * 4, review_length_ceiling))

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as session:
        with tf.variable_scope("model", reuse=None) as scope:
            scope.set_regularizer(
                tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            m = RNNGAN(is_training=True, num_review_features=num_review_features)

        if FLAGS.initialize_d:
            vars_to_restore = {}
            for v in tf.trainable_variables():
                if v.name.startswith('model/G/'):
                    print(v.name[:-2])
                    vars_to_restore[v.name[:-2]] = v
            saver = tf.train.Saver(vars_to_restore)
            ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" %
                      ckpt.model_checkpoint_path, end=" ")
                saver.restore(session, ckpt.model_checkpoint_path)
                session.run(tf.initialize_variables(
                    [v for v in tf.trainable_variables() if v.name.startswith('model/D/')]))
            else:
                print("Created model with fresh parameters.")
                session.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.all_variables())
        else:
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" %
                      ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                session.run(tf.initialize_all_variables())

        if not FLAGS.sample:
            train_g_loss, train_d_loss = 1.0, 1.0
            for i in range(global_step, FLAGS.max_epoch):
                lr_decay = FLAGS.lr_decay ** max(i - FLAGS.epochs_before_decay, 0.0)

                new_review_length = (
                    int(min((i + 1) * 4, review_length_ceiling))
                    if global_step < FLAGS.pretraining_epochs else
                    review_length_ceiling
                )

                if new_review_length != FLAGS.review_length:
                    print('Changing review_length, now training on {} events from reviews.'.format(
                        new_review_length))
                    FLAGS.review_length = new_review_length
                    with tf.variable_scope("model", reuse=True) as scope:
                        scope.set_regularizer(
                            tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
                        m = RNNGAN(is_training=True, num_review_features=num_review_features)

                if not FLAGS.adam:
                    m.assign_lr(session, FLAGS.learning_rate * lr_decay)

                save = False
                do_exit = False

                print("Epoch: {} Learning rate: {:.3f}, pretraining: {}".format(
                    i, session.run(m.lr), (i < FLAGS.pretraining_epochs)))
                if i < FLAGS.pretraining_epochs:
                    opt_d = tf.no_op()
                    if FLAGS.pretraining_d:
                        opt_d = m.opt_d
                    train_g_loss, train_d_loss = run_epoch(
                        session, m, loader, 'train', m.opt_pretraining, opt_d,
                        pretraining=True, verbose=True,
                        pretraining_d=FLAGS.pretraining_d
                    )
                    if FLAGS.pretraining_d:
                        try:
                            print("Epoch: {} Pretraining loss: G: {:.3f}, D: {:.3f}".format(
                                i, train_g_loss, train_d_loss))
                        except:
                            print(train_g_loss)
                            print(train_d_loss)
                    else:
                        print("Epoch: {} Pretraining loss: G: {:.3f}".format(
                            i, train_g_loss))
                else:
                    train_g_loss, train_d_loss = run_epoch(
                        session, m, loader, 'train', m.opt_d, m.opt_g, verbose=True)
                    try:
                        print("Epoch: {} Train loss: G: {:.3f}, D: {:.3f}".format(
                            i, train_g_loss, train_d_loss))
                    except:
                        print("Epoch: {} Train loss: G: {}, D: {}".format(
                            i, train_g_loss, train_d_loss))
                valid_g_loss, valid_d_loss = run_epoch(
                    session, m, loader, 'validation', tf.no_op(), tf.no_op())
                try:
                    print("Epoch: {} Valid loss: G: {:.3f}, D: {:.3f}".format(
                        i, valid_g_loss, valid_d_loss))
                except:
                    print("Epoch: {} Valid loss: G: {}, D: {}".format(
                        i, valid_g_loss, valid_d_loss))

                if train_d_loss == 0.0 and train_g_loss == 0.0:
                    print('Both G and D train loss are zero. Exiting.')
                    save = True
                    do_exit = True
                if i % FLAGS.epochs_per_checkpoint == 0:
                    save = True
                if FLAGS.exit_after > 0 and time.time() - train_start_time > FLAGS.exit_after * 60:
                    print(
                        "%s: Has been running for %d seconds. Will exit (exiting after %d minutes)." % (
                            datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                            (int)(time.time() - train_start_time),
                            FLAGS.exit_after,
                        )
                    )
                    save = True
                    do_exit = True

                step_time, loss = 0.0, 0.0
                if train_d_loss is None:  # pretraining
                    train_d_loss = 0.0
                    valid_d_loss = 0.0
                    valid_g_loss = 0.0

                # PALMER: write review data to output file
                review_data = sample(session, m, batch=True)
                filename = os.path.join(generated_data_dir, 'out-{}-{}-{}.mid'.format(
                    experiment_label, i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
                loader.save_midi_pattern(filename, review_data)

                if do_exit:
                    exit()
                sys.stdout.flush()

            test_g_loss, test_d_loss = run_epoch(
                session, m, loader, 'test', tf.no_op(), tf.no_op())
            print("Test loss G: %.3f, D: %.3f" % (test_g_loss, test_d_loss))

        # PALMER: write review data to output file
        review_data = sample(session, m)
        filename = os.path.join(generated_data_dir, 'out-{}-{}-{}.mid'.format(
            experiment_label, i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
        loader.save_data(filename, review_data)
        print('Saved {}.'.format(filename))


if __name__ == "__main__":
    tf.app.run()

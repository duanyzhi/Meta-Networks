import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#  omn data
tf.app.flags.DEFINE_string('omn_train_data', 'data/omniglot/train.npz', "omniglot Train Datasets")
tf.app.flags.DEFINE_string('omn_test_data', 'data/omniglot/test.npz', "omniglot Test Datasets")


tf.app.flags.DEFINE_integer('n_outputs_train', 5, "n_outputs_train")
tf.app.flags.DEFINE_integer('n_outputs_test', 5, "n_outputs_test")
tf.app.flags.DEFINE_integer('nb_samples_per_class_train', 10, "n_outputs_train")
tf.app.flags.DEFINE_integer('nb_samples_per_class_test', 10, "n_outputs_train")
tf.app.flags.DEFINE_integer('nb_samples_per_class', 1, "nb_samples_per_class")

tf.app.flags.DEFINE_integer('batch_size', 1, "n_outputs_train")
tf.app.flags.DEFINE_integer('n_eposide_tr', 100, "n_outputs_train")
tf.app.flags.DEFINE_integer('learning_rate', 0.0001, "n_outputs_train")
tf.app.flags.DEFINE_integer('n_epoch', 100000, "n_outputs_train")
tf.app.flags.DEFINE_integer('n_eposide', 10, "n_outputs_test")


tf.app.flags.DEFINE_integer('n_out', 5, "n_outputs_train")
tf.app.flags.DEFINE_integer('lstm_layers', 20, "lstm_hideen")
tf.app.flags.DEFINE_float('dropout', 0.5, "lstm_hideen")
tf.app.flags.DEFINE_float('weight_decay', 0.0001, "Weight decay, for regularization")
tf.app.flags.DEFINE_string('ckpt', 'data/omn_ckpt/mn_2000.ckpt', "omniglot Test Datasets")

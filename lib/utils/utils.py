import numpy as np
import tensorflow as tf
from lib.utils.config import FLAGS as cfg
from novamind.ops.text_ops import read_csv
import matplotlib.pyplot as plt
import matplotlib

def clamp(inputs, min_value=None, max_value=None):
	output = inputs
	if min_value is not None:
		output[output < min_value] = min_value
	if max_value is not None:
		output[output > max_value] = max_value
	return output

def logAndSign(inputs, k=5):
    eps = np.finfo(inputs.dtype).eps
    log = np.log(np.absolute(inputs) + eps)
    clamped_log = clamp(log / k, min_value=-1.0)
    sign = clamp(inputs * np.exp(k), min_value=-1.0, max_value=1.0)
    return np.concatenate([clamped_log, sign], axis=1)

def clamp_tf(inputs, min_value=None, max_value=None):
    # help(tf.Variable)
    # output = tf.Variable(inputs[:, 0])
	# 使用tf.Variable注意初始化问题
	# Ref https://github.com/tensorflow/tensorflow/issues/11856

    output = inputs[:, 0] # shape (148032,)
    # output = tf.Variable(tf.ones_like(inputs[:, 0]))
    # output.assign(inputs[:, 0])
    # print("  -        ", output.shape)
    if min_value is not None:
        # ind_c = tf.where(output < min_value)[:, 0]
        # updates = tf.ones_like(ind_c, dtype=tf.float32)*tf.constant(min_value, dtype=tf.float32)
        # output_min = tf.sparse_to_dense(ind_c, output.shape, updates)
        # ind = tf.where(output >= min_value)[:, 0]
        # output_raw = tf.sparse_to_dense(ind, output.shape, output)
        # output = output_min + output_raw

        updates = tf.ones_like(output, dtype=tf.float32)*tf.constant(min_value, dtype=tf.float32)

        output = tf.where(output < min_value, updates, output)

    if max_value is not None:
        # ind_c = tf.where(output > max_value)[:, 0]
        # updates = tf.ones_like(ind_c, dtype=tf.float32)*tf.constant(max_value, dtype=tf.float32)
        # output_max = tf.sparse_to_dense(ind_c, output.shape, updates)
        # ind = tf.where(output <= max_value)[:, 0]
        # output_raw = tf.sparse_to_dense(ind, output.shape, output)
        # output = output_max + output_raw

        updates = tf.ones_like(output, dtype=tf.float32)*tf.constant(max_value, dtype=tf.float32)

        output = tf.where(output > max_value, updates, output)
    # output = tf.convert_to_tensor(tf.expand_dims(output, 1))
    output = tf.expand_dims(output, 1)
    # print("output", output.shape)

    return output

def logAndSign_tf(inputs, k=7.0):
    eps = 2.22044604925e-16 # float
    log = tf.log(tf.abs(inputs) + eps)
    # log = inputs
    clamped_log = clamp_tf(log / k, min_value=-1.0)
    sign = clamp_tf(inputs * tf.exp(k), min_value=-1.0, max_value=1.0)
    return tf.concat([clamped_log, sign], axis=1)
    # return tf.concat([inputs, log], axis=1)

def lr(kk):
	if kk < 101:
		cfg.learning_rate = 0.0001
	elif 100 < kk < 100001:
		cfg.learning_rate = 0.001
	elif 10000 < kk < 20001:
		cfg.learning_rate = 0.0001
	else:
		cfg.learning_rate = 0.00001

def bn(x, dim=[0], gamma=1, beta=0):
    '''
    batch normalization
    '''
    epsilon = 1e-3
    mean, variance = tf.nn.moments(x, dim)

    x_bn = gamma*(x - mean)/tf.sqrt(variance + epsilon) + beta
    return tf.nn.sigmoid(x_bn) - 0.5

def weight_variable(shape, name):
    # initial = tf.truncated_normal(shape, stddev=w)
    # return tf.Variable(initial, name=name)
    initializer=tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.weight_decay)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                regularizer=regularizer, trainable=True)
    return new_variables

def plot():
    meta_save = read_csv("data/document/meta.csv")
    epos = []
    train_acc = []
    test_acc =[]
    loss = []
    for file in meta_save[1:]:
        # file = eval(file)
        print(file)
        epos.append(eval(file[0]))
        train_acc.append(eval(file[1]))
        test_acc.append(eval(file[2]))
        loss.append(eval(file[3]))

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots()

	# this sets the figure on which we'll draw all the subplots
    # plt.figure(figsize=(10,8))
    print(len(epos))
    # 设置横纵坐标轴范围
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(epos))

    # 加图像标题
    # plt.title("MSN")

    # 加横纵坐标
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")

    # 图中加横纵线分割
    # plt.grid(True)

    # 颜色配色网站colordrop.io
    plt.plot(epos, train_acc, color="#ea7070", marker='+', linewidth=1,  label='Train', markersize=8)
    plt.plot(epos, test_acc, color="#2694ab", marker='*',  linewidth=1, label='Test', markersize=8)

    # 加图注释
    # plt.legend(loc="best")
    plt.legend(loc='upper left', frameon=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.show()
# -----------------------------------------------------------------

import tensorflow as tf
import numpy as np
from lib.utils.utils import plot

a = np.array([[-0.00189646], [-0.00118797], [-0.00504021], [2000], [0], [0]])
b = np.finfo(a.dtype).eps
print(b)
print(np.log(0.00189646)/7)
print(a.shape)

x = tf.placeholder(tf.float32, [6, 1])

def clamp_tf(inputs, min_value=None, max_value=None):
    output = inputs[:, 0]
    print("output", output)

    if min_value is not None:
        ind_c = tf.where(output < min_value)[:, 0]
        updates = tf.ones_like(output, dtype=tf.float32)*tf.constant(min_value, dtype=tf.float32)

        # updates = tf.ones_like(ind_c, dtype=tf.float32)*tf.constant(min_value, dtype=tf.float32)
        # output_min = tf.sparse_to_dense(ind_c, output.shape, min_value)
        # ind_raw = tf.where(output >= min_value)[:, 0]
        # output_raw = tf.sparse_to_dense(ind_raw, output.shape, output)
        # output = output_raw
        output = tf.where(output < min_value, updates, output)

    if max_value is not None:
        updates = tf.ones_like(output, dtype=tf.float32)*tf.constant(max_value, dtype=tf.float32)

        output = tf.where(output > max_value, updates, output)
    output = tf.expand_dims(output, 1)
    # print("output", output.shape)

    return output, ind_c

def logAndSign_tf(inputs, k=7.0):
    eps = 2.22044604925e-16 # float
    log = tf.log(tf.abs(inputs) + eps)
    # log = inputs
    clamped_log, ind_c = clamp_tf(log / k, min_value=-1.0)
    sign, ind = clamp_tf(inputs * tf.exp(k), min_value=-1.0, max_value=1.0)
    return tf.concat([clamped_log, sign], axis=1), ind_c

def test_sparese():
    sess = tf.Session()
    o,log = logAndSign_tf(x)
    _o,_log = sess.run([o, log], feed_dict={x:a})
    print(_o)
    print(_log)

plot()
# test_sparese()

import tensorflow as tf
import numpy as np
from lib.utils.config import FLAGS as cfg
from lib.utils.utils import logAndSign, logAndSign_tf, bn
import tensorflow.contrib.rnn as rnn

class meta_learner(object):
    def __init__(self): # , support_sets, support_lbls, x_set, xlbl
        '''
        input:
           support_sets, support_lbls:采样的Ｔ个support data
        '''
    def run_dynamic_function(self, support_sets, support_labs, x_set, dropout):
        '''
        2: for i = 1, T do
        3:     L_i ← loss emb (u(Q, x 0 i ), y i 0 )
        4:     ∇_i ← ∇ Q L i
        5: end for
        6: Q ∗ = d(G, {∇} Ti=1 )
        '''
        IT = 5
        N = 5
        reuse = False
        opt_vab = None
        for i in range(0, N, int(N/IT)):
            x = support_sets[i:int((i+N/IT)), ...]
            # print(x.shape)
            y = support_labs[i:int((i+N/IT)), ...]
            # print(y.shape)
            with tf.variable_scope("dynamic_u") as scope_name:
                if reuse:
                    scope_name.reuse_variables()
                update, delta_Q, delta_shape, Q, _Q_star, opt_vab, fc = self.dynamic_function(x, y, opt_vab, dropout)
            reuse = True
        # print(delta_shape)
        _Q_star_conv = tf.split(_Q_star[0], delta_shape[0], 0)
        _Q_star_fc = tf.split(_Q_star[1], delta_shape[1], 0)
        # print("shape:\n", _Q_star[0].shape, _Q_star[1].shape)

        ## get Q*
        Q_star = {}
        for i, qc in enumerate(_Q_star_conv):
            # print(qc.shape)
            # print("reshape", i+1, Q[0][i].shape)
            Q_star["layers_" + str(i + 1)] = tf.reshape(qc, Q[0][i].shape)
        # for qf in _Q_star_fc:
        #     print(qf.shape)
        # print(Q[1][0].shape)
        # print(Q[1][1].shape)
        Q_star["fc1"] = tf.reshape(_Q_star_fc[0], Q[1][0].shape)
        Q_star["fc2"] = tf.reshape(_Q_star_fc[1], Q[1][1].shape)

        # get key
        keys, u5 = self.get_key_R_and_r(support_sets, x_set, Q, Q_star)
        return update, Q, Q_star, keys, u5


    def dynamic_function(self, sup_x, sup_lab, opt_vab, dropout):
        '''
        dynamic_function is function u in paper
        input: suppoert sets, support lables
        '''
        ########################################################################
        # Build dynamic network: function u
        # print(sup_x.shape, sup_lab.shape)
        sup_x = tf.reshape(sup_x, [-1, 784, 1])
        x = tf.reshape(sup_x, [-1, 28, 28, 1])  # batch size = 1
        # bulid u cnn model
        u1 = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
         use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u1")
        # u1_bn = bn(u1, dim=[1, 2])
        m1 = tf.layers.max_pooling2d(u1, pool_size=(2, 2), strides=2)
        d1 = tf.nn.dropout(m1, keep_prob=dropout)

        u2 = tf.layers.conv2d(d1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u2")
        m2 = tf.layers.max_pooling2d(u2, pool_size=(2, 2), strides=2)
        d2 = tf.nn.dropout(m2, keep_prob=dropout)

        u3 = tf.layers.conv2d(d2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u3")
        m3 = tf.layers.max_pooling2d(u3, pool_size=(2, 2), strides=2)
        d3 = tf.nn.dropout(m3, keep_prob=dropout)

        u4 = tf.layers.conv2d(d3, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u4")
        m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
        d4 = tf.nn.dropout(m4, keep_prob=dropout)

        u5 = tf.layers.conv2d(d4, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u5")
        # print("du5", u5.shape)
        h = tf.reshape(u5, [-1, 64])

        fc1 = tf.layers.dense(h, 64, activation=tf.nn.relu, use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
        fc_out = tf.layers.dense(fc1, cfg.n_out, activation=None, use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")
        sup_lab = tf.squeeze(sup_lab, 1)

        # print("lab", lab.shape, fc_out.shape)
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=lab, logits=fc_out)
        one_hot_label = tf.one_hot(sup_lab, 5)
        # print(one_hot_label.shape, fc_out.shape)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_label,
                                               logits=fc_out) # , name="meta_loss"

        optim = tf.train.AdamOptimizer(cfg.learning_rate)
        grads_and_vars = optim.compute_gradients(loss) # 返回所有的梯度和对应的变量
        update = optim.apply_gradients(grads_and_vars)

        # print("trainable_variables", tf.trainable_variables())

        # grad = tf.gradients(loss, opt_vab)

        # if opt_vab is None:
        #     opt_vab = tf.trainable_variables()
        #
        # grad = tf.gradients(loss, opt_vab)
        #
        # grads_and_vars = [[grad[i], opt_vab[i]] for i in range(len(grad))]
        # print("grads_and_vars \n", grads_and_vars)
        ########################################################################
        ## collect conv grads
        grads_conv = []
        grad_sections_conv = []
        Q_conv = []
        pre_shape = 0
        for gv in grads_and_vars[:5]:
            grads_conv.append(tf.reshape(gv[0], [-1, 1]))
            grad_sections_conv.append(int(grads_conv[-1].shape[0]))
            pre_shape = grad_sections_conv[-1]
            Q_conv.append(gv[1])

        # grads_conv [<tf.Tensor 'Reshape_2:0' shape=(576, 1) dtype=float32>,
        #       <tf.Tensor 'Reshape_3:0' shape=(36864, 1) dtype=float32>,
        #       <tf.Tensor 'Reshape_4:0' shape=(36864, 1) dtype=float32>,
        #       <tf.Tensor 'Reshape_5:0' shape=(36864, 1) dtype=float32>,
        #       <tf.Tensor 'Reshape_6:0' shape=(36864, 1) dtype=float32>]
        # grad_sections_conv [576, 36864, 36864, 36864, 36864]
        # Q_conv [<tf.Variable 'u1/kernel:0' shape=(3, 3, 1, 64) dtype=float32_ref>,
        #    <tf.Variable 'u2/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>,
        #    <tf.Variable 'u3/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>,
        #    <tf.Variable 'u4/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>,
        #    <tf.Variable 'u5/kernel:0' shape=(3, 3, 64, 64) dtype=float32_ref>]

        ## collect fc grads
        grads_fc = []
        grad_sections_fc = []
        Q_fc = []

        grads_fc.append(tf.reshape(grads_and_vars[5][0], [-1, 1])) # fc1
        grad_sections_fc.append(int(grads_fc[-1].shape[0]))
        Q_fc.append(grads_and_vars[5][1])

        grads_fc.append(tf.reshape(grads_and_vars[6][0], [-1, 1])) # fc2
        grad_sections_fc.append(int(grads_fc[-1].shape[0]))
        Q_fc.append(grads_and_vars[6][1])
        '''
        grads_fc [<tf.Tensor 'Reshape_7:0' shape=(4096, 1) dtype=float32>,
                  <tf.Tensor 'Reshape_8:0' shape=(320, 1) dtype=float32>]
        grad_sections_fc [4096, 320]
        Q_fc [<tf.Variable 'fc1/kernel:0' shape=(64, 64) dtype=float32_ref>,
              <tf.Variable 'fc2/kernel:0' shape=(64, 5) dtype=float32_ref>]
        '''
        delta_Q = [grads_conv, grads_fc, Q_fc]
        Q = [Q_conv, Q_fc]
        delta_shape = [grad_sections_conv, grad_sections_fc]
        # print("delta_shape", delta_shape)
        ########################################################################
        # # Next is Fast weight generation functions d
        meta_in_conv= self.meta_function_d(grads_conv)
        # print("meta_in_conv", meta_in_conv.shape)
        Q_star_conv = self.meta_lstm_d_conv(meta_in_conv)

        meta_in_fc = self.meta_function_d(grads_fc)
        Q_star_fc = self.meta_lstm_d_fc(meta_in_fc)

        _Q_star = [Q_star_conv, Q_star_fc]

        return update, delta_Q, delta_shape, Q, _Q_star, opt_vab, Q_star_conv

    def meta_function_d(self, grads):
        # computer meta in information
        grad_concat = tf.concat(grads, 0)
        # print("grad_concat", grad_concat.shape)
        meta_in = logAndSign_tf(grad_concat, k=7.0)
        # meta_in = grad_concat
        # print("meta in", meta_in)
        return meta_in

    def meta_lstm_d_conv(self, meta_in):
        batch = int(meta_in.shape[0])
        with tf.variable_scope('LSTM_CONV'):
            lstm_cell = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch, dtype=tf.float32)
            (h, state) = lstm_cell(tf.expand_dims(meta_in[:, 0], 1), init_state)
            tf.get_variable_scope().reuse_variables()
            (conv_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 1], 1), state)
            # (conv_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 2], 1), state)
            # print("h", h.shape) # (148032, 20)
            # print("ch", h)

        # MLP
        d_out = tf.layers.dense(conv_h, 1, activation=None, use_bias=False, name="lstm_mlp_conv")
        # print("d_out", d_out)
        return d_out

    def meta_lstm_d_fc(self, meta_in):
        batch = int(meta_in.shape[0])
        with tf.variable_scope('LSTM_FC'):
            lstm_cell = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch, dtype=tf.float32)
            (h, state) = lstm_cell(tf.expand_dims(meta_in[:, 0], 1), init_state)
            tf.get_variable_scope().reuse_variables()
            (fc_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 1], 1), state)
            # (fc_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 2], 1), state)
            # print("h", h.shape) # (148032, 20)

        # MLP
        d_out = tf.layers.dense(fc_h, 1, activation=None, use_bias=False, name="lstm_mlp_fc")
        # print("d_out", d_out)
        return d_out

    def get_key_R_and_r(self, support_sets, x_set, Q, Q_star):
        '''
        support_sets [5, 784]
        x_set [50, 784]
        获取键值 support 上的r_i^i(R)和训练数据x_set上的r_i
        '''
        keys = []
        for x in [support_sets, x_set]:
            x = tf.reshape(x, [-1, 28, 28, 1])
            # print(x.shape)

            # bulid u cnn model
            # layers 1
            # help(tf.nn.conv2d)
            s_Q_u1 = tf.nn.relu(tf.nn.conv2d(x, Q[0][0], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u1")) # Q
            s_Q_s_u1 = tf.nn.relu(tf.nn.conv2d(x, Q_star["layers_1"], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_s_u1"))

            u1 = bn(s_Q_u1 + s_Q_s_u1, dim=[0, 1, 2])
            m1 = tf.layers.max_pooling2d(u1, pool_size=(2, 2), strides=2)
            d1 = tf.layers.dropout(m1, rate=0.0)
            # print(d1.shape)
            # layers 2
            s_Q_u2 = tf.nn.relu(tf.nn.conv2d(d1, Q[0][1], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u2")) # Q
            s_Q_s_u2 = tf.nn.relu(tf.nn.conv2d(d1, Q_star["layers_2"], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_s_u2"))

            u2 = bn(s_Q_u2 + s_Q_s_u2, dim=[0, 1, 2])
            m2 = tf.layers.max_pooling2d(u2, pool_size=(2, 2), strides=2)
            d2 = tf.layers.dropout(m2, rate=0.0)

            # layers 3
            s_Q_u3 = tf.nn.relu(tf.nn.conv2d(d2, Q[0][2], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u3")) # Q
            s_Q_s_u3 = tf.nn.relu(tf.nn.conv2d(d2, Q_star["layers_3"], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_s_u3"))

            u3 = bn(s_Q_u3 + s_Q_s_u3, dim=[0, 1, 2])
            m3 = tf.layers.max_pooling2d(u3, pool_size=(2, 2), strides=2)
            d3 = tf.layers.dropout(m3, rate=0.0)

            # layers 4
            s_Q_u4 = tf.nn.relu(tf.nn.conv2d(d3, Q[0][3], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u4")) # Q
            s_Q_s_u4 = tf.nn.relu(tf.nn.conv2d(d3, Q_star["layers_4"], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_s_u4"))

            u4 = bn(s_Q_u4 + s_Q_s_u4, dim=[0, 1, 2])
            m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
            d4 = tf.layers.dropout(m4, rate=0.0)


            # layers 5
            s_Q_u5 = tf.nn.relu(tf.nn.conv2d(d4, Q[0][4], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u5")) # Q
            s_Q_s_u5 = tf.nn.relu(tf.nn.conv2d(d4, Q_star["layers_5"], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_s_u5"))

            u5 = s_Q_u5 + s_Q_s_u5
            # print("u5", u5.shape)
            h = bn(tf.reshape(u5, [-1, 64]), dim=[0])

            # fc1 = tf.layers.dense(h, 64, activation=tf.nn.relu, use_bias=False, trainable=False, name="fc1")
            # fc_out = tf.layers.dense(fc1, cfg.n_out, activation=None, use_bias=False, trainable=False, name="fc2")
            fc_Q = tf.matmul(h, Q[1][0])
            fc_Q_star = tf.matmul(h, Q_star["fc1"])
            # print(h.shape, Q[1][0].shape, fc_Q.shape,  Q_star["fc1"].shape, fc_Q_star.shape)

            fc = fc_Q + fc_Q_star
            keys.append(fc)
        '''
        keys[0] support 的查询值r_i^'  shape=[5, 64] key_mems  R
        keys[1] x_set的查询值r_i   shape=[50, 64]
        '''
        return keys, fc

class base_learner(object):
    def __init__(self):
        '''
        Base learner b: Faster weight + Slow weight
        input support_sets, x_set
        '''
        pass

    def run_base_learner_function(self, support_sets, support_lbl):
        '''
        support_sets: [5, 1, 784]  # same value as meta learner
        x_set [50, 784]

        Algorithm:
        7: for i = 1, N do
        8:    L_i ← loss task (b(W, x_i^' ), y_i^' )
        9:    ∇_i ← ∇_W L_i
        10:   W_i ∗ ← m(Z, ∇ i )
        11:   Store W_i ∗ in i^th position of memory M
        12:   r_i^0 = u(Q, Q ∗ , x 0 i )
        13:   Store r i 0 in i th position of index memory R
        14: end for

        '''
        reuse = False
        M = []  # Store W_i^∗ in i th position of memory M
        opt_vab = None
        for i in range(5):
            x = support_sets[i, ...]
            y = support_lbl[i, ...]
            # print("base learner shape", x.shape, y.shape)
            with tf.variable_scope("base_learner") as scope_name:
                if reuse:
                    scope_name.reuse_variables()
                update, delta_W, delta_shape, W, _W_star, opt_vab, show = self.base_learner_function(x, y, opt_vab)
                M.append(_W_star)
            reuse = True
        return update, M, W, delta_shape, opt_vab, show

    def base_learner_function(self, x, sup_lab, opt_vab):
        '''
        x shape: [1, 784]
        '''
        x = tf.reshape(x, [1, 28, 28, 1])
        # print(x.shape)

        # # bulid u cnn model
        u1 = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u1")
        m1 = tf.layers.max_pooling2d(u1, pool_size=(2, 2), strides=2)
        # d1 = tf.layers.dropout(m1, rate=0.0)
        d1 = m1

        u2 = tf.layers.conv2d(d1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u2")
        m2 = tf.layers.max_pooling2d(u2, pool_size=(2, 2), strides=2)
        # d2 = tf.layers.dropout(m2, rate=0.0)
        d2 = m2

        u3 = tf.layers.conv2d(d2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
         use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u3")
        m3 = tf.layers.max_pooling2d(u3, pool_size=(2, 2), strides=2)
        # d3 = tf.layers.dropout(m3, rate=0.0)
        d3 = m3

        u4 = tf.layers.conv2d(d3, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u4")
        m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
        # d4 = tf.layers.dropout(m4, rate=0.0)
        d4 = m4

        u5 = tf.layers.conv2d(d4, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
        use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="u5")

        h = tf.reshape(u5, [-1, 64])

        fc1 = tf.layers.dense(h, 64, activation=tf.nn.relu, use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
        fc_out = tf.layers.dense(fc1, cfg.n_out, activation=None, use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")
        # print("sup_lab", sup_lab.shape)
        # lab = tf.squeeze(sup_lab, 1)

        # print("lab", sup_lab.shape, fc_out.shape)
        one_hot_label = tf.one_hot(sup_lab, 5)
        # print(one_hot_label.shape, fc_out.shape)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_label,
                                               logits=fc_out) # , name="meta_loss"
        # print("loss", loss)
        # print(tf.trainable_variables())
        if opt_vab is None:
            opt_vab = tf.trainable_variables()[-7:]
        # print("opt_vab", opt_vab)
        grad = tf.gradients(loss, opt_vab)

        grads_and_vars = [[grad[i], opt_vab[i]] for i in range(len(grad))]
        # print("grads_and_vars \n", grads_and_vars)

        optim = tf.train.GradientDescentOptimizer(cfg.learning_rate)
        grads_and_vars = optim.compute_gradients(loss, var_list=opt_vab) # 返回所有的梯度和对应的变量
        update = optim.apply_gradients(grads_and_vars)


        ## collect conv grads
        grads_conv = []
        grad_sections_conv = []
        W_conv = []
        pre_shape = 0
        for gv in grads_and_vars[:5]:
            # print("gv", gv)
            grads_conv.append(tf.reshape(gv[0], [-1, 1]))
            grad_sections_conv.append(int(grads_conv[-1].shape[0]))
            pre_shape = grad_sections_conv[-1]
            W_conv.append(gv[1])

        ## collect fc grads
        grads_fc = []
        grad_sections_fc = []
        W_fc = []

        grads_fc.append(tf.reshape(grads_and_vars[5][0], [-1, 1])) # fc1
        grad_sections_fc.append(int(grads_fc[-1].shape[0]))
        W_fc.append(grads_and_vars[5][1])

        grads_fc.append(tf.reshape(grads_and_vars[6][0], [-1, 1])) # fc2
        grad_sections_fc.append(int(grads_fc[-1].shape[0]))
        W_fc.append(grads_and_vars[6][1])

        delta_W = [grads_conv, grads_fc, W_fc]
        W = [W_conv, W_fc]
        delta_shape = [grad_sections_conv, grad_sections_fc]
        # print("delta_shape", delta_shape)
        ########################################################################
        # # Next is Fast weight generation functions m
        meta_in_conv, grad_concat = self.meta_function_m(grads_conv)
        W_star_conv = self.meta_lstm_m_conv(meta_in_conv)
        # print('W_star_conv', W_star_conv.shape)

        meta_in_fc, grad_concatf = self.meta_function_m(grads_fc)
        W_star_fc = self.meta_lstm_m_fc(meta_in_fc)

        _W_star = [W_star_conv, W_star_fc]

        return update, delta_W, delta_shape, W, _W_star, opt_vab, W_star_conv

    def meta_function_m(self, grads):
        # computer meta in information
        '''
        grads shape: 5 (576, 1)   2 (4096, 1)
        '''
        # help(tf.concat)
        grad_list = [grads[i] for i in range(len(grads))]
        grad_concat = tf.concat(grad_list, axis=0)
        # print("grad_concat", grad_concat.shape)
        meta_in = logAndSign_tf(grad_concat, k=7.0)
        # print("meta in", meta_in)
        # print(len(grads), meta_in.shape)
        return meta_in, grad_concat

    def meta_lstm_m_conv(self, meta_in):
        batch = int(meta_in.shape[0])
        with tf.variable_scope('LSTM_CONV'):
            lstm_cell = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch, dtype=tf.float32)
            (h, state) = lstm_cell(tf.expand_dims(meta_in[:, 0], 1), init_state)
            tf.get_variable_scope().reuse_variables()
            (conv_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 1], 1), state)
            # (conv_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 2], 1), state)
            # print("h", h.shape) # (148032, 20)
            # print("ch", h)

        # MLP
        m_out = tf.layers.dense(conv_h, 1, activation=None, use_bias=False, name="lstm_mlp_conv")
        m_out = bn(m_out, dim=[0])
        # print("m_out", m_out.shape)
        return m_out

    def meta_lstm_m_fc(self, meta_in):
        batch = int(meta_in.shape[0])
        with tf.variable_scope('LSTM_FC'):
            lstm_cell = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch, dtype=tf.float32)
            (h, state) = lstm_cell(tf.expand_dims(meta_in[:, 0], 1), init_state)
            tf.get_variable_scope().reuse_variables()
            (fc_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 1], 1), state)
            # (fc_h, state) = lstm_cell(tf.expand_dims(meta_in[:, 2], 1), state)
            # print("h", h.shape) # (148032, 20)

        # MLP
        m_out = tf.layers.dense(fc_h, 1, activation=None, use_bias=False, name="lstm_mlp_fc")
        m_out = bn(m_out, dim=[0])
        # print("d_out", d_out)
        return m_out


class train_learner(object):
    def __init__(self):
        pass

    def run_train_learner(self, x_set, x_lbl, Q, Q_star, M, W, delta_shape, keys, opt_vab):
        '''
        Algorithm:
        16: for i = 1, L do
        17:    r_i = u(Q, Q^∗ , x_i )
        18:    a_i = attention(R, r_i )
        19:    W_i ∗ = sof tmax(a i ) > M
        20:    L_train ← L_train + loss task (b(W, W i ∗ , x i ), y i )
               {Alternatively the base learner can take as input r i instead
               of x i }
        21: end for

        Inputs:
        x_set: [50, 1, 784]
        x_lbl: [50]
        Q: [Q_conv, Q_fc] len(Q_conv) = 5
        Q_star: dict keys = ["layers_1", "layers_2", "layers_3", "layers_4", "layers_5", "fc1", "fc2"]
        M: W_i^∗  i=1,2,3,...,N  N = 5 len(M) = 5  M = 5*[W_star_conv, W_star_fc]
           W_star_conv.shape (148032, 1)  W_star_fc.shape (4416, 1)
        keys = [R, r_set]  R.shape = [5, 64]  R is support key r_i^*
              r_set = r_i, i= 1, 2, ..., 50 r_set.shape = [50, 64] is x_set keys
        '''
        R = keys[0]
        r_set = keys[1]
        # print(len(M), M[0][0].shape, M[0][1].shape)
        # print([M[i][0] for i in range(len(M))])
        M_conv = tf.concat([M[i][0] for i in range(len(M))], 1)  # (148032, 5)
        # print(M_conv.shape)
        M_fc = tf.concat([M[i][1] for i in range(len(M))], 1)  # (4416, 5)
        # print(M_fc.shape)

        L = 50  # 50个训练数据
        loss = 0
        reuse = False
        opt_vab = None
        prediction = []
        # fc_list = []
        for i in range(L):
            r_i = r_set[i, :]
            a_i = self.attention(R, r_i)
            # print("a_i", a_i.shape, M_conv.shape)
            W_star_conv_i = tf.matmul(M_conv, tf.expand_dims(tf.nn.softmax(a_i), 1)) # matrix by vector in tf
            W_star_fc_i = tf.matmul(M_fc, tf.expand_dims(tf.nn.softmax(a_i), 1))
            # print("w conv shape", W_star_conv_i.shape, W_star_fc_i.shape)

            # split weights
            _W_star_conv = tf.split(W_star_conv_i, delta_shape[0], 0)
            _W_star_fc = tf.split(W_star_fc_i, delta_shape[1], 0)

            # print(_W_star_conv)

            ## get W*
            W_star = {}
            for i, wc in enumerate(_W_star_conv):
                # print(qc.shape)
                # print("reshape", i+1, W[0][i].shape)
                W_star["layers_" + str(i + 1)] = tf.reshape(wc, W[0][i].shape)

            W_star["fc1"] = tf.reshape(_W_star_fc[0], W[1][0].shape)
            W_star["fc2"] = tf.reshape(_W_star_fc[1], W[1][1].shape)

            # for key in W_star.keys():
            #     print(key, W_star[key].shape)
            # with tf.variable_scope("train_learner") as scope_name:
            #     if reuse:
            #         scope_name.reuse_variables()
            _loss, fc, show = self.train_network(x_set[i], x_lbl[i], W, W_star)
            reuse = True
            loss += _loss

            prediction.append(tf.argmax(fc[0]))
        loss /= L
        # print("tf.trainable_variables()", tf.trainable_variables())
        optim = tf.train.AdamOptimizer(cfg.learning_rate)
        grads_and_vars = optim.compute_gradients(loss) # 返回所有的梯度和对应的变量
        update_train = optim.apply_gradients(grads_and_vars)

        W_train = {}
        for i, gv in enumerate(grads_and_vars):
            W_train["layers_" + str(i)] = gv[1]

        return loss, prediction, update_train, W_train, show

    def attention(self, R, r_i):
        '''
        return : R [5, 64] r_i [1, 64]
        '''
        # print(R.shape, r_i.shape)
        s = []
        for i in range(5):
            mr = R[i, :]
            # print("mr", mr.shape)
            s.append(self.cosine_similar(mr, r_i))
        # print(len(s))
        so = tf.stack([s[i] for i in range(len(s))])
        # print("so", so.shape)
        return so

    def cosine_similar(self, a, b):
        '''
        return
        '''
        cs = tf.reduce_sum(tf.multiply(a, b))
        # print(a.shape)
        norm_a = tf.sqrt(tf.reduce_sum(a**2))
        norm_b = tf.sqrt(tf.reduce_sum(b**2))
        # print("cs", cs.shape, norm_a.shape, norm_b.shape)

        sm = cs / (norm_a * norm_b)
        # print(sm.shape)

        return sm

    def weight_variable(self, shape, w=0.1):
        initial = tf.constant(shape, stddev=w)
        return tf.Variable(initial)

    def train_network_v(self, x, lab, W, W_star, opt_vab):
        # print("x shape", x.shape)
        x = tf.reshape(x, [1, 28, 28, 1])
        weight_u1 = tf.Variable(W[0][0], name="t1")
        s_W_u1 = tf.nn.relu(tf.nn.conv2d(x, weight_u1, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u1")) # Q
        s_W_s_u1 = tf.nn.relu(tf.nn.conv2d(x, W_star["layers_1"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u1"))

        u1 = s_W_u1 + s_W_s_u1
        m1 = tf.layers.max_pooling2d(u1, pool_size=(2, 2), strides=2)
        d1 = tf.layers.dropout(m1, rate=0.0)
        # print(d1.shape)
        # layers 2
        weight_u2 = tf.Variable(W[0][1], name="t2")
        s_W_u2 = tf.nn.relu(tf.nn.conv2d(d1, weight_u2, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u2")) # Q
        s_W_s_u2 = tf.nn.relu(tf.nn.conv2d(d1, W_star["layers_2"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u2"))
        u2 = s_W_u2 + s_W_s_u2
        m2 = tf.layers.max_pooling2d(u2, pool_size=(2, 2), strides=2)
        d2 = tf.layers.dropout(m2, rate=0.0)

        # layers 3
        weight_u3 = tf.Variable(W[0][2], name="t3")
        s_W_u3 = tf.nn.relu(tf.nn.conv2d(d2, weight_u3, strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u3")) # Q
        s_W_s_u3 = tf.nn.relu(tf.nn.conv2d(d2, W_star["layers_3"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u3"))

        u3 = s_W_u3 + s_W_s_u3
        m3 = tf.layers.max_pooling2d(u3, pool_size=(2, 2), strides=2)
        d3 = tf.layers.dropout(m3, rate=0.0)

        # layers 4
        weight_u4 = tf.Variable(W[0][3], name="t4")
        s_W_u4 = tf.nn.relu(tf.nn.conv2d(d3, weight_u4, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u4")) # Q
        s_W_s_u4 = tf.nn.relu(tf.nn.conv2d(d3, W_star["layers_4"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u4"))

        u4 = s_W_u4 + s_W_s_u4
        m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
        d4 = tf.layers.dropout(m4, rate=0.0)


        # layers 5
        weight_u5 = tf.Variable(W[0][4], name="t5")
        s_W_u5 = tf.nn.relu(tf.nn.conv2d(d4, weight_u5, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u5")) # Q
        s_W_s_u5 = tf.nn.relu(tf.nn.conv2d(d4, W_star["layers_5"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u5"))

        u5 = s_W_u5 + s_W_s_u5
        # print("u5", u5.shape)
        h = tf.reshape(u5, [-1, 64])

        # fc1 = tf.layers.dense(h, 64, activation=tf.nn.relu, use_bias=False, trainable=False, name="fc1")
        # fc_out = tf.layers.dense(fc1, cfg.n_out, activation=None, use_bias=False, trainable=False, name="fc2")
        weight_fc1 = tf.Variable(W[1][0], name="t_fc1")
        fc_W_1 = tf.matmul(h, weight_fc1, name="fc1")
        fc_W_star_1 = tf.matmul(h, W_star["fc1"])

        fc1 = fc_W_1 + fc_W_star_1

        weight_fc2 = tf.Variable(W[1][1], name="t_fc2")
        fc_W_2 = tf.matmul(fc1, weight_fc2, name="fc2")
        fc_W_star_2 = tf.matmul(fc1, W_star["fc2"])

        fc = fc_W_2 + fc_W_star_2

        # print("variables", tf.trainable_variables())
        # input()

        # print("lab", lab.shape, fc.shape)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,
                                               logits=fc, name="train_loss")
        if opt_vab is None:
             # print(tf.trainable_variables()) # 所有变量
             opt_vab = tf.trainable_variables()[-7:]

        return loss, fc, opt_vab

    def train_network(self, x, lab, W, W_star):
        # print("x shape", x.shape)
        x = tf.reshape(x, [1, 28, 28, 1])
        # help(tf.nn)
        conv1 = tf.nn.conv2d(x, W[0][0], strides=[1, 1, 1, 1], padding="SAME", name="s_W_u1")
        s_W_u1 = tf.nn.relu(conv1) # Q
        s_W_s_u1 = tf.nn.relu(tf.nn.conv2d(x, W_star["layers_1"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u1"))

        u1 = s_W_u1 + s_W_s_u1
        m1 = tf.layers.max_pooling2d(u1, pool_size=(2, 2), strides=2)
        d1 = bn(m1, dim=[0, 1, 2])
        # d1 = tf.layers.dropout(m1, rate=0.0)
        # print(d1.shape)
        # layers 2
        s_W_u2 = tf.nn.relu6(tf.nn.conv2d(d1, W[0][1], strides=[1, 1, 1, 1], padding="SAME", name="s_W_u2")) # Q
        s_W_s_u2 = tf.nn.relu6(tf.nn.conv2d(d1, W_star["layers_2"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u2"))
        u2 = s_W_u2 + s_W_s_u2
        m2 = tf.layers.max_pooling2d(u2, pool_size=(2, 2), strides=2)
        # d2 = tf.layers.dropout(m2, rate=0.0)
        d2 = m2

        # layers 3
        s_W_u3 = tf.nn.relu6(tf.nn.conv2d(d2, W[0][2], strides=[1, 1, 1, 1], padding="SAME", name="s_Q_u3")) # Q
        s_W_s_u3 = tf.nn.relu6(tf.nn.conv2d(d2, W_star["layers_3"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u3"))

        u3 = s_W_u3 + s_W_s_u3
        m3 = tf.layers.max_pooling2d(u3, pool_size=(2, 2), strides=2)
        # d3 = tf.layers.dropout(m3, rate=0.0)
        d3 = bn(m3, dim=[0,1, 2])

        # layers 4
        s_W_u4 = tf.nn.relu6(tf.nn.conv2d(d3, W[0][3], strides=[1, 1, 1, 1], padding="SAME", name="s_W_u4")) # Q
        s_W_s_u4 = tf.nn.relu6(tf.nn.conv2d(d3, W_star["layers_4"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u4"))

        u4 = s_W_u4 + s_W_s_u4
        m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
        # d4 = tf.layers.dropout(m4, rate=0.0)
        d4 = m4


        # layers 5
        s_W_u5 = tf.nn.relu(tf.nn.conv2d(d4, W[0][4], strides=[1, 1, 1, 1], padding="SAME", name="s_W_u5")) # Q
        s_W_s_u5 = tf.nn.relu(tf.nn.conv2d(d4, W_star["layers_5"], strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u5"))

        u5 = s_W_u5 + s_W_s_u5
        # print("u5", u5.shape)
        h = tf.reshape(u5, [-1, 64])

        # fc1 = tf.layers.dense(h, 64, activation=tf.nn.relu, use_bias=False, trainable=False, name="fc1")
        # fc_out = tf.layers.dense(fc1, cfg.n_out, activation=None, use_bias=False, trainable=False, name="fc2")
        fc_W_1 =  tf.matmul(h, W[1][0])/10
        fc_W_star_1 =  tf.matmul(h, W_star["fc1"])/10
        # print("fc_W_1", fc_W_1.shape, fc_W_star_1.shape)
        fc1_old = fc_W_1 + fc_W_star_1
        fc1_mean = tf.reduce_mean(fc1_old)
        fc1 = (fc1_old -  fc1_mean)/10

        fc_W_2 = tf.matmul(fc1, W[1][1])/10
        fc_W_star_2 = tf.matmul(fc1, W_star["fc2"])/10

        fc_old = fc_W_2 + fc_W_star_2
        fc_mean = tf.reduce_mean(fc_old)
        fc = fc_old - fc_mean

        # print("variables", tf.trainable_variables())
        # input()

        # print("lab", lab.shape, fc.shape)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,
                                               logits=fc, name="train_loss")

        return loss, fc, fc1



# ------------------------------------------------------------------------------

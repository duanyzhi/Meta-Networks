import tensorflow as tf
import numpy as np
from lib.utils.config import FLAGS as cfg
from lib.utils.utils import logAndSign, logAndSign_tf, bn, weight_variable
import tensorflow.contrib.rnn as rnn

class meta_learner(object):
    def __init__(self): # , support_sets, support_lbls, x_set, xlbl
        '''
        input:
           support_sets, support_lbls:采样的Ｔ个support data
        '''
        self.lstm_conv_reuse = False
        self.lstm_fc_reuse = False
        # help(rnn.BcLSTMCell)
        self.lstm_cell_conv = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True, reuse=self.lstm_fc_reuse)
        init_state = self.lstm_cell_conv.zero_state(148032, dtype=tf.float32)
        self.conv_state = init_state

        self.lstm_cell_fc = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True, reuse=self.lstm_fc_reuse)
        init_state_fc = self.lstm_cell_fc.zero_state(4416, dtype=tf.float32)
        self.fc_state = init_state_fc

    def run_dynamic_function(self, support_sets, support_lbls, x_set):
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
        Q_star_list = []
        for i in range(0, N):
            x = support_sets[i, ...]
            y = support_lbls[i, ...]

            delta_shape, Q, _Q_star, opt_vab, show = self.dynamic_function(x, y, reuse, opt_vab=opt_vab)
            reuse = True
            Q_star_list.append(_Q_star)

        # mean
        __Q_star_conv = tf.concat([Q[0] for Q in Q_star_list], 1)
        __Q_star_fc = tf.concat([Q[1] for Q in Q_star_list], 1)

        # MLP
        Q_star_conv = tf.layers.dense(__Q_star_conv, 1, activation=None, use_bias=False, name="Q_star_conv_mlp")
        Q_star_fc = tf.layers.dense(__Q_star_fc, 1, activation=None, use_bias=False, name="Q_star_conv_fc")

        _Q_star_conv = tf.split(Q_star_conv, delta_shape[0], 0)
        _Q_star_fc = tf.split(Q_star_fc, delta_shape[1], 0)

        ## get Q*
        Q_star = {}
        for i, qc in enumerate(_Q_star_conv):
            Q_star["layers_" + str(i + 1)] = tf.reshape(qc, Q[0][i].shape)

        Q_star["fc1"] = tf.reshape(_Q_star_fc[0], Q[1][0].shape)
        Q_star["fc2"] = tf.reshape(_Q_star_fc[1], Q[1][1].shape)

        # get key
        support_keys, show1 = self.dynamic_function(support_sets,support_lbls=None, reuse=True, opt_vab=None, x_set=None, Q_star=Q_star)
        set_keys, show2 = self.dynamic_function(support_sets=None,support_lbls=None, reuse=True, opt_vab=None, x_set=x_set, Q_star=Q_star)
        keys = [support_keys, set_keys]
        return keys, show2

    def dynamic_function(self, support_sets, support_lbls, reuse, opt_vab=None, x_set=None, Q_star=None):
        if x_set is None:
            x_in = support_sets
        else:
            x_in = x_set

        x = tf.reshape(x_in, [-1, 784, 1])
        x = tf.reshape(x, [-1, 28, 28, 1]) # must be here?

        with tf.variable_scope("meta_learner") as scope_name:
            if reuse:
                scope_name.reuse_variables()

            weight_conv1 = weight_variable([3, 3, 1, 64], name="meta_conv1")

            if Q_star is not None:
                s_W_s_u1 = tf.nn.conv2d(x, Q_star["layers_1"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u1")
                # print(s_W_s_u1)
            # print(weight_conv1)
            # conv1 = tf.nn.conv2d(x, weight_conv1, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u1")
            s_W_u1 = tf.nn.conv2d(x, weight_conv1, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u1")

            if Q_star is None:
                u1 = s_W_u1
            else:
                u1 = s_W_u1 + s_W_s_u1
            m1 = tf.layers.average_pooling2d(u1, pool_size=(2, 2), strides=2)
            # d1 = bn(m1, dim=[0,1 ,2])
            d1 = m1

            # layers 2
            weight_conv2 = weight_variable([3, 3, 64, 64], name="meta_conv2")

            if Q_star is not None:
                s_W_s_u2 = tf.nn.relu(tf.nn.conv2d(d1, Q_star["layers_2"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u2"))

            s_W_u2 = tf.nn.relu(tf.nn.conv2d(d1, weight_conv2, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u2"))

            if Q_star is None:
                u2 = s_W_u2
            else:
                u2 = s_W_u2 + s_W_s_u2
            m2 = tf.layers.average_pooling2d(u2, pool_size=(2, 2), strides=2)
            # d2 = bn(m2, dim=[0, 1, 2])
            d2 = m2

            # layers 3
            weight_conv3 = weight_variable([3, 3, 64, 64], name="meta_conv3")

            if Q_star is not None:
                s_W_s_u3 = tf.nn.relu(tf.nn.conv2d(d2, Q_star["layers_3"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u3"))

            s_W_u3 = tf.nn.relu(tf.nn.conv2d(d2, weight_conv3, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u3"))

            if Q_star is None:
                u3 = s_W_u3
            else:
                u3 = s_W_u3 + s_W_s_u3
            m3 = tf.layers.average_pooling2d(u3, pool_size=(2, 2), strides=2)
            # d3 = bn(m3, dim=[0, 1, 2])
            d3 = m3

            # layers 4
            weight_conv4 = weight_variable([3, 3, 64, 64], name="meta_conv4")

            if Q_star is not None:
                s_W_s_u4 = tf.nn.relu(tf.nn.conv2d(d3, Q_star["layers_4"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u4"))

            s_W_u4 = tf.nn.relu(tf.nn.conv2d(d3, weight_conv4, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u4"))

            if Q_star is None:
                u4 = s_W_u4
            else:
                u4 = s_W_u4 + s_W_s_u4
            m4 = u4
            # m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
            # m4_mean = tf.reduce_mean(m4)
            # d4 = bn(m4, dim=[0, 1, 2])
            d4 = m4

            # layers 5
            weight_conv5 = weight_variable([3, 3, 64, 64], name="meta_conv5")

            if Q_star is not None:
                s_W_s_u5 = tf.nn.relu(tf.nn.conv2d(d4, Q_star["layers_5"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u5"))

            s_W_u5 = tf.nn.relu(tf.nn.conv2d(d4, weight_conv5, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u5"))

            if Q_star is None:
                u5 = s_W_u5
            else:
                u5 = s_W_u5 + s_W_s_u5
            m5 = tf.layers.average_pooling2d(u5, pool_size=(3, 3), strides=1)

            h = tf.reshape(m5, [-1, 64])
            h = h

            # fc1
            weight_fc1 = weight_variable([64, 64], name="meta_fc1")

            if Q_star is not None:
                fc_W_star_1 =  tf.matmul(h, Q_star["fc1"])

            fc_W_1 =  tf.matmul(h, weight_fc1)

            if Q_star is None:
                fc1 = fc_W_1
            else:
                fc1 = fc_W_1 + fc_W_star_1

            if Q_star is None:
                # fc2
                weight_fc2 = weight_variable([64, 5], name="meta_fc2")

                if Q_star is not None:
                    fc_W_star_2 =  tf.matmul(fc1, Q_star["fc2"])

                fc_W_2 =  tf.matmul(fc1, weight_fc2)

                if Q_star is None:
                    fc_out = fc_W_2
                else:
                    fc_out = fc_W_2 + fc_W_star_2

        if Q_star is None:
            # bp
            delta_Q, delta_shape, Q, _Q_star, opt_vab, bp_show = \
            self.dynamic_function_up(fc_out, support_lbls, opt_vab)
            show = fc_out
            return delta_shape, Q, _Q_star, opt_vab, show
        else:
            show = fc1
            return fc1, show

    def dynamic_function_up(self, fc_out, support_lbls, opt_vab):
        '''
        dynamic_function is function u in paper
        input: suppoert sets, support lables
        '''
        ########################################################################
        # sup_lab = tf.squeeze(support_lbls, 1)

        one_hot_label = tf.one_hot(support_lbls, 5)
        # print("meta loss",one_hot_label, fc_out)
        '''
        tf.losses.sparse_softmax_cross_entropy
        onehot_labels: shape [batch_size, num_classes] one 编码
        logits:　shape [batch_size, num_classes]　网络的输出 不需要加入softmax
        '''
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_label,
                                               logits=fc_out) # , name="meta_loss"

        # 这里不能update 因为不会不是回归问题所以收敛
        # print("vab", tf.trainable_variables())
        if opt_vab is None:
            opt_vab = tf.trainable_variables()
        grad = tf.gradients(loss, opt_vab)
        grads_and_vars = [[grad[i], opt_vab[i]] for i in range(len(grad))]
        # print(50*"--")
        # print("grads_and_vars: \n", grads_and_vars, "\n")

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

        return delta_Q, delta_shape, Q, _Q_star, opt_vab, Q_star_fc

    def meta_function_d(self, grads):
        # computer meta in information
        grad_concat = tf.concat(grads, 0)
        # print("grad_concat", grad_concat.shape)
        meta_in = logAndSign_tf(grad_concat, k=7.0)
        # meta_in = grad_concat
        # print("meta in", meta_in)
        return meta_in

    def meta_lstm_d_conv(self, meta_in):
        '''
        这里参数大小如下，输入meta in shape=[148032, 1] 输出为[148032, 20]大小
        初始化隐藏状态h 大小为[148032, 20]
        因此所需要参数个数为:
        20*1*4 + 20*20*4 = 80*21 个weights
        另加80个bias
        '''
        batch = int(meta_in.shape[0]) # 148032
        # lstm_cell = rnn.BasicLSTMCell(num_units=20, forget_bias=1.0, state_is_tuple=True, reuse=self.lstm_conv_reuse)
        # init_state = lstm_cell.zero_state(batch, dtype=tf.float32)
        # state = init_state
        with tf.variable_scope('LSTM_CONV'):
            for timestep in range(2):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.lstm_cell_conv(tf.expand_dims(meta_in[:, timestep], 1), self.conv_state)
                # outputs.append(cell_output)
                self.conv_state = state

        conv_h = cell_output
        # MLP
        d_out = tf.layers.dense(conv_h, 1, activation=None, use_bias=False, name="lstm_mlp_conv", reuse=self.lstm_conv_reuse)
        # print("d_out", conv_h, d_out)
        self.lstm_conv_reuse = True
        return d_out

    def meta_lstm_d_fc(self, meta_in):
        batch = int(meta_in.shape[0]) # 4416
        # print(batch)

        with tf.variable_scope('LSTM_FC'):
            for timestep in range(2):
                # if timestep > 0:
                #     tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.lstm_cell_fc(tf.expand_dims(meta_in[:, timestep], 1), self.fc_state)
                # outputs.append(cell_output)
                self.fc_state = state


        fc_h = cell_output
        # MLP
        d_out = tf.layers.dense(fc_h, 1, activation=None, use_bias=False, name="lstm_mlp_fc", reuse=self.lstm_fc_reuse)
        # print("d_out", d_out)
        self.lstm_fc_reuse = True
        return d_out

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
        # print("support_lbl", support_lbl.shape)
        reuse = False
        M = []  # Store W_i^∗ in i th position of memory M
        opt_vab = None
        for i in range(5):
            # print("i", i)
            x = support_sets[i, ...]
            y = support_lbl[i, ...]
            # print(x, y)

            delta_W, delta_shape, W, _W_star, opt_vab, show = \
            self.base_learner_function(x, y, reuse, opt_vab)

            M.append(_W_star)
            reuse = True
        return M, W, delta_shape, show


    def base_learner_function(self, x, y, reuse, opt_vab=None, W_star=None):
        '''
        W_star : None表示不加fast weight
        '''
        x = tf.reshape(x, [1, 28, 28, 1]) # must be here?
        with tf.variable_scope("base_learner") as scope_name:
            if reuse:
                scope_name.reuse_variables()
            '''
            x shape: [1, 784]
            '''
            weight_conv1 = weight_variable([3, 3, 1, 64], name="base_conv1")

            if W_star is not None:
                s_W_s_u1 = tf.nn.relu(tf.nn.conv2d(x, W_star["layers_1"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u1"))
                # print(s_W_s_u1)
            # print(weight_conv1)
            # conv1 = tf.nn.conv2d(x, weight_conv1, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u1")
            s_W_u1 = tf.nn.relu(tf.nn.conv2d(x, weight_conv1, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u1"))

            if W_star is None:
                u1 = s_W_u1
            else:
                u1 = s_W_u1 + s_W_s_u1
            m1 = tf.layers.max_pooling2d(u1, pool_size=(2, 2), strides=2)
            # d1 = bn(m1, dim=[0, 1, 2])
            d1 = m1

            # layers 2
            weight_conv2 = weight_variable([3, 3, 64, 64], name="base_conv2")

            if W_star is not None:
                s_W_s_u2 = tf.nn.relu(tf.nn.conv2d(d1, W_star["layers_2"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u2"))

            s_W_u2 = tf.nn.relu(tf.nn.conv2d(d1, weight_conv2, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u2"))

            if W_star is None:
                u2 = s_W_u2
            else:
                u2 = s_W_u2 + s_W_s_u2
            m2 = tf.layers.max_pooling2d(u2, pool_size=(2, 2), strides=2)
            # d2 = bn(m2, dim=[0, 1, 2])
            d2 = m2

            # layers 3
            weight_conv3 = weight_variable([3, 3, 64, 64], name="base_conv3")

            if W_star is not None:
                s_W_s_u3 = tf.nn.relu(tf.nn.conv2d(d2, W_star["layers_3"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u3"))

            s_W_u3 = tf.nn.relu(tf.nn.conv2d(d2, weight_conv3, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u3"))

            if W_star is None:
                u3 = s_W_u3
            else:
                u3 = s_W_u3 + s_W_s_u3
            m3 = tf.layers.max_pooling2d(u3, pool_size=(2, 2), strides=2)
            # d3 = bn(m3, dim=[0, 1, 2])
            d3 = m3

            # layers 4
            weight_conv4 = weight_variable([3, 3, 64, 64], name="base_conv4")

            if W_star is not None:
                s_W_s_u4 = tf.nn.relu(tf.nn.conv2d(d3, W_star["layers_4"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u4"))

            s_W_u4 = tf.nn.relu(tf.nn.conv2d(d3, weight_conv4, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u4"))

            if W_star is None:
                u4 = s_W_u4
            else:
                u4 = s_W_u4 + s_W_s_u4
            m4 = u4
            # m4 = tf.layers.max_pooling2d(u4, pool_size=(2, 2), strides=2)
            # m4_mean = tf.reduce_mean(m4)
            # d4 = (m4 - m4_mean)/10
            d4 = m4

            # layers 5
            weight_conv5 = weight_variable([3, 3, 64, 64], name="base_conv5")

            if W_star is not None:
                s_W_s_u5 = tf.nn.relu(tf.nn.conv2d(d4, W_star["layers_5"],
                strides=[1, 1, 1, 1], padding="SAME", name="s_W_s_u5"))

            s_W_u5 = tf.nn.relu(tf.nn.conv2d(d4, weight_conv5, strides=[1, 1, 1, 1], padding="SAME", name="s_W_u5"))

            if W_star is None:
                u5 = s_W_u5
            else:
                u5 = s_W_u5 + s_W_s_u5
            m5 = tf.layers.max_pooling2d(u5, pool_size=(3, 3), strides=1)

            # u5_mean = tf.reduce_mean(u5)
            # u5 = (u5 - u5_mean)/10

            h = tf.reshape(m5, [-1, 64])

            # fc1
            weight_fc1 = weight_variable([64, 64], name="base_fc1")

            if W_star is not None:
                fc_W_star_1 =  tf.matmul(h, W_star["fc1"])

            fc_W_1 =  tf.matmul(h, weight_fc1)

            if W_star is None:
                fc1 = fc_W_1
            else:
                fc1 = fc_W_1 + fc_W_star_1
            fc1_mean = tf.reduce_mean(fc1)
            fc1 = (fc1 - fc1_mean)/5

            # fc2
            weight_fc2 = weight_variable([64, 5], name="base_fc2")

            if W_star is not None:
                fc_W_star_2 =  tf.matmul(fc1, W_star["fc2"])

            fc_W_2 =  tf.matmul(fc1, weight_fc2)

            if W_star is None:
                fc_out = fc_W_2
            else:
                fc_out = fc_W_2 + fc_W_star_2

            one_hot_label = tf.one_hot(y, 5)

            if W_star is not None:
                one_hot_label = tf.expand_dims(one_hot_label, 0)
            # print(y, one_hot_label, fc_out)
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_label,
                                                   logits=fc_out)) # , name="meta_loss"

        if W_star is None:
            # bp
            delta_W, delta_shape, W, _W_star, opt_vab, show_bp = \
            self.base_function(loss, opt_vab, reuse)
            show = show_bp
            return delta_W, delta_shape, W, _W_star, opt_vab, show
        else:
            show_train = fc_out
            return fc_out, loss, show_train


    def base_function(self, loss, opt_vab, reuse):
        if opt_vab is None:
            opt_vab = tf.trainable_variables()[-7:]
        grad = tf.gradients(loss, opt_vab)

        grads_and_vars = [[grad[i], opt_vab[i]] for i in range(len(grad))]
        # print("base grads_and_vars", grads_and_vars)


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
        meta_in_conv = self.meta_function_m(grads_conv)
        W_star_conv = self.meta_lstm_m_conv(meta_in_conv, reuse)
        # print('W_star_conv', W_star_conv.shape)

        meta_in_fc = self.meta_function_m(grads_fc)
        W_star_fc = self.meta_lstm_m_fc(meta_in_fc, reuse)

        _W_star = [W_star_conv, W_star_fc]
        # print("trainable_variables", tf.trainable_variables())
        return delta_W, delta_shape, W, _W_star, opt_vab, W_star_conv

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
        return meta_in

    def meta_lstm_m_conv(self, meta_in, reuse):
        with tf.variable_scope("conv_learner") as scope_name:
            if reuse:
                scope_name.reuse_variables()
            meta_in = tf.layers.dense(meta_in, 20, activation=tf.nn.relu, use_bias=False, name="encode_conv_fc1")
            # print(meta_in.shape)
            meta_in = tf.layers.dense(meta_in, 20, activation=tf.nn.relu, use_bias=False, name="encode_conv_fc2")
            # print(meta_in.shape)

            # MLP
            m_out = tf.layers.dense(meta_in, 1, activation=None, use_bias=False, name="decoder_conv_fc")
            # print("m_out", m_out.shape)
        return m_out

    def meta_lstm_m_fc(self, meta_in, reuse):
        with tf.variable_scope("fc_learner") as scope_name:
            if reuse:
                scope_name.reuse_variables()
            meta_in = tf.layers.dense(meta_in, 20, activation=tf.nn.relu, use_bias=False, name="encode_fc_fc1")
            # print(meta_in.shape)
            meta_in = tf.layers.dense(meta_in, 20, activation=tf.nn.relu, use_bias=False, name="encode_fc_fc2")
            # print(meta_in.shape)

            # MLP
            m_out = tf.layers.dense(meta_in, 1, activation=None, use_bias=False, name="decoder_fc_fc")
            # print("d_out", d_out)
        return m_out


class train_learner(meta_learner, base_learner):
    def __init__(self):
        super(train_learner, self).__init__()
        base_learner.__init__(self)

    def run_train_learner(self, support_sets, support_labs, x_set, x_lbl):
        '''
        Q, Q_star, M, W, delta_shape, keys, opt_vab
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
        # run meta learner
        keys, show_meta = \
        self.run_dynamic_function(support_sets, support_labs, x_set)
        # run base learner
        M, W, delta_shape, show_base = \
        self.run_base_learner_function(support_sets, support_labs)

        # run train learner
        R = keys[0]
        r_set = keys[1]

        M_conv = tf.concat([M[i][0] for i in range(len(M))], 1)  # (148032, 5)
        # print(M_conv.shape)
        M_fc = tf.concat([M[i][1] for i in range(len(M))], 1)  # (4416, 5)
        # print(M_fc.shape)

        L = 50  # 50个训练数据
        loss = 0
        reuse = False
        prediction = []
        train_list = []
        # print(x_lbl.shape)
        # print(x_set.shape)
        for kk in range(L):
            r_i = r_set[kk, :]
            a_i = tf.nn.softmax(self.attention(R, r_i))
            # print("a_i", a_i.shape, M_conv.shape)
            W_star_conv_i = tf.matmul(M_conv, tf.expand_dims(a_i, 1)) # matrix by vector in tf
            W_star_fc_i = tf.matmul(M_fc, tf.expand_dims(a_i, 1))
            # print("w conv shape", W_star_conv_i.shape, W_star_fc_i.shape)

            # split weights
            _W_star_conv = tf.split(W_star_conv_i, delta_shape[0], 0)
            _W_star_fc = tf.split(W_star_fc_i, delta_shape[1], 0)

            ## get W*
            W_star = {}
            for i, wc in enumerate(_W_star_conv):
                W_star["layers_" + str(i + 1)] = tf.reshape(wc, W[0][i].shape)

            W_star["fc1"] = tf.reshape(_W_star_fc[0], W[1][0].shape)
            W_star["fc2"] = tf.reshape(_W_star_fc[1], W[1][1].shape)

            fc_out, _loss, show_train = self.base_learner_function(x=x_set[kk], y=x_lbl[kk], reuse=True, W_star=W_star)
            reuse = True
            loss += _loss
            # print(kk, x_lbl[kk], x_lbl[kk].shape)
            # train_list.append(x_lbl[kk])

            prediction.append(tf.argmax(fc_out[0]))
        # loss /= L
        # print("tf.trainable_variables()", tf.trainable_variables())
        # optim = tf.train.AdamOptimizer(cfg.learning_rate).minimize(cross_entropy)
        # grads_and_vars = optim.compute_gradients(loss) # 返回所有的梯度和对应的变量
        # update_train = optim.apply_gradients(grads_and_vars)
        train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss)
        # print("grads_and_vars", grads_and_vars)
        # W_train = {}
        # for i, gv in enumerate(grads_and_vars):
        #     W_train["layers_" + str(i)] = gv[1]

        return loss, prediction, train_op, fc_out

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


# ------------------------------------------------------------------------------

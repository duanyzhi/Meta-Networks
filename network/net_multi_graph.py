from lib.omn.generators import OmniglotGenerator
from lib.utils.config import FLAGS as cfg
from .meta_cnn import meta_learner, base_learner, train_learner
import tensorflow as tf
import numpy as np

class meta_networks(object):
    def __init__(self, pattern, data):
        self.train_generator = OmniglotGenerator(data_file=cfg.omn_train_data,
									augment = True,
									nb_classes=cfg.n_outputs_train, nb_samples_per_class=cfg.nb_samples_per_class,
									batchsize=cfg.batch_size, max_iter=None, xp=np,
									nb_samples_per_class_test=cfg.nb_samples_per_class_train)

        self.test_generator = OmniglotGenerator(data_file=cfg.omn_test_data,
									augment = False,
									nb_classes=cfg.n_outputs_test, nb_samples_per_class=cfg.nb_samples_per_class,
									batchsize=cfg.batch_size, max_iter=None, xp=np,
									nb_samples_per_class_test=cfg.nb_samples_per_class_test)
        print("# train classes:", len(self.train_generator.data.keys()))
        print("# test classes:", len(self.test_generator.data.keys()))

        self.placeholder()
        # init model
        # super(meta_networks, self).__init__()
        # base_learner.__init__(self)

    def placeholder(self):
        # self.support_sets_meta = tf.placeholder(tf.float32, shape=[None, 1, 784])
        # self.support_lbl_meta = tf.placeholder(tf.int32, shape=[None, 1])
        # self.x_set_meta = tf.placeholder(tf.float32, shape=[None, 784])
        #
        # self.support_sets_base = tf.placeholder(tf.float32, shape=[None, 1, 784])
        # self.support_lbl_base = tf.placeholder(tf.int32, shape=[None, 1])
        #
        # self.x_set_train = tf.placeholder(tf.float32, shape=[None, 784])
        # self.x_lbl_train = tf.placeholder(tf.int32, shape=[None, 1])
        pass

    def build_network(self):
        pass


    def run(self):
        g1 = tf.Graph()
        with g1.as_default():
            self.support_sets_meta = tf.placeholder(tf.float32, shape=[None, 1, 784])
            self.support_lbl_meta = tf.placeholder(tf.int32, shape=[None, 1])
            self.x_set_meta = tf.placeholder(tf.float32, shape=[None, 784])
            ml = meta_learner()
            update_meta, Q, Q_star, keys, show =  \
            ml.run_dynamic_function(self.support_sets_meta, self.support_lbl_meta, self.x_set_meta)

            bl = base_learner()
            M, W, delta_shape, opt_vab, update_base = \
            bl.run_base_learner_function(self.support_sets_meta, self.support_lbl_meta)
        # tf.reset_default_graph()

        g3 = tf.Graph()
        with g3.as_default():
            self.x_set_train = tf.placeholder(tf.float32, shape=[None, 784])
            self.x_lbl_train = tf.placeholder(tf.int32, shape=[None, 1])
        #     self.placeholder
        #     tl =  train_learner()
        #     loss, prediction, update_train = \
        #     tl.run_train_learner(self.x_set_train, self.x_lbl_train, Q, Q_star, M, W, delta_shape, keys, opt_vab)

        self.sess_run(update_meta, update_base,
                       g1,  g3)

    def set_sess(self, g):
        # saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=g)
        return sess

    def sess_run(self, update_meta, update_base, g1, g3):
        # sess = self.set_sess()
        # sess.run(tf.global_variables_initializer())

        sess_g1 = self.set_sess(g1)
        sess_g1.run(tf.initialize_all_variables())
        # sess_g2 = self.set_sess(g2)
        # sess_g2 = run(tf.global_variables_initializer())
        sess_g3 = self.set_sess(g3)
        sess_g3.run(tf.initialize_all_variables())
        # tf.control_dependencies([tf.variables_initializer([v])])

        for i in range(0, cfg.n_epoch):
            epos_loss = 0
            epos_pre = []
            epos_lbl = []
            for kk in range(1, cfg.n_eposide_tr):
                it, data = self.train_generator.next()
                support_set, support_lbl, x_set, x_lbl = data
                # support_set [5, 1, 784], support_lbl[5, 1]
                # x_set [50, 784] x_lbl [50]
                x_lbl = np.expand_dims(x_lbl, 1)

                # run u
                feed_dict_u_b = {self.support_sets_meta: support_set,
                             self.support_lbl_meta: support_lbl,
                             self.x_set_meta: x_set
                              }
                _ = sess_g1.run([update_meta], feed_dict=feed_dict_u)


            #     # run train
            #     feed_dict_t = {self.support_sets: support_set,
            #                  self.support_lbl: support_lbl,
            #                  self.x_set: x_set,
            #                  self.x_lbl: x_lbl
            #                  }
            #     _, _loss, _prediction, _show = \
            #     sess_g3.run([update_train, loss, prediction, show], feed_dict=feed_dict_t)
            #
            #     epos_loss += _loss
            #     epos_pre.extend(_prediction)
            #     epos_lbl.extend(x_lbl)
            #
            # print("epos:", i, "\n", _show)
            #
            # # print("len_lbl", epos_pre, epos_lbl)
            # acc = self.computer_acc(epos_pre, epos_lbl)
            # print("Accuracy:", acc)
            # print("Task Loss", epos_loss/ cfg.n_eposide_tr)

    def computer_acc(self, pre, lab):
        print("pre", pre[:10])
        print("lab", lab[:10])
        acc = 0
        for i in range(len(pre)):
            if pre[i] == lab[i]:
                acc += 1
        return acc /len(pre)

# ------------------------------------------------------------------------------

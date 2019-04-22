from lib.omn.generators import OmniglotGenerator
from lib.utils.config import FLAGS as cfg
from lib.utils.utils import lr
from .meta_network import train_learner #, base_learner, train_learner
from novamind.ops.text_ops import save_csv
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
        self.run_things = {}

    def placeholder(self):
        self.support_sets = tf.placeholder(tf.float32, shape=[None, 1, 784])
        self.support_lbl = tf.placeholder(tf.int32, shape=[None, 1])
        self.x_set = tf.placeholder(tf.float32, shape=[None, 784])
        self.x_lbl = tf.placeholder(tf.int32, shape=[None])

    def build_network(self):
        # g1 = tf.Graph()
        # with g1.as_default():
        # ml = meta_learner()
        # update_meta_list, keys, loss, prediction =  \
        # ml.run_dynamic_function(self.support_sets, self.support_lbl, self.x_set)
        # tf.reset_default_graph()
        #
        # g2 = tf.Graph()
        # with g2.as_default():
        # bl = base_learner()
        # update_base, M, W, delta_shape, opt_vab, fc2 = bl.run_base_learner_function(self.support_sets, self.support_lbl)
        # tf.reset_default_graph()

        tl =  train_learner()
        loss, prediction, update_train, show = \
        tl.run_train_learner(self.support_sets, self.support_lbl, self.x_set, self.x_lbl)


        self.run_things["update"] = update_train
        self.run_things["loss"] = loss
        self.run_things["prediction"] = prediction
        self.run_things["show"] = show

    def set_sess(self):
        # saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def run(self):
        sess = self.set_sess()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        document = [["epos", "train_acc", "test_acc", "loss"]]
        for epo in range(0, cfg.n_epoch):
            print("--"*50)
            epos_loss = 0
            epos_acc = 0
            lr(epo)
            for kk in range(1, cfg.n_eposide_tr):
                it, data = self.train_generator.next()
                support_set, support_lbl, x_set, x_lbl = data
                # support_set [5, 1, 784], support_lbl[5, 1]
                # x_set [50, 784] x_lbl [50]
                # x_lbl = np.expand_dims(x_lbl, 1)


                feed_dict_t = {self.support_sets: support_set,
                               self.support_lbl: support_lbl,
                               self.x_set: x_set,
                               self.x_lbl: x_lbl
                               }
                meta_run = sess.run(self.run_things, feed_dict=feed_dict_t)
                pre = meta_run["prediction"]

                acc = self.computer_acc(pre, x_lbl)

                epos_loss += meta_run["loss"]
                epos_acc += acc
                # print(len(x_lbl))
                print("show", meta_run["show"])
                # print("label", x_lbl)

                # for i in range(50):
                #     print(i, x_lbl[i], meta_run["show"][i])


            print("epos:", epo)

            print("pre", pre[:30])
            print("Accuracy:", epos_acc / cfg.n_eposide_tr)
            print("Task Loss:", epos_loss / cfg.n_eposide_tr)

            if epo % 500 == 0:
                saver.save(sess, 'data/omn_ckpt/mn_' + str(epo) + '.ckpt')

            test_acc = self.test(sess)
            print("Test Accuracy:", test_acc)

            # train_acc_list.append(epos_acc / cfg.n_eposide_tr)
            # test_acc_list.append(test_acc)
            # loss_list.append(epos_loss / cfg.n_eposide_tr)
            document.append([epo, epos_acc / cfg.n_eposide_tr, test_acc, epos_loss / cfg.n_eposide_tr])
            # save txt
            if epo % 10 == 0:
                # list_save(train_acc_list, "data/document/train_acc.txt")
                # list_save(test_acc_list, "data/document/test_acc.txt")
                # list_save(loss_list, "data/document/loss.txt")
                save_csv("data/document/meta.csv", document)


    def test(self, sess):
        test_acc = 0
        for j in range(cfg.n_eposide):
            it, data = self.test_generator.next()
            support_set, support_lbl, x_set, x_lbl = data
            # support_set.shape  [5, 784]
            # x_set.shape  [50 784]

            feed_dict_t = {self.support_sets: support_set,
                           self.support_lbl: support_lbl,
                           self.x_set: x_set,
                           self.x_lbl: x_lbl
                           }
            meta_test = sess.run(self.run_things["prediction"], feed_dict=feed_dict_t)

            test_acc += self.computer_acc(meta_test, x_lbl)

        return test_acc / cfg.n_eposide


    def computer_acc(self, pre, lab):
        acc = 0
        for i in range(len(pre)):
            if pre[i] == lab[i]:
                acc += 1
        return acc /len(pre)

# ------------------------------------------------------------------------------

"""
Generators of data

This code comes from https://github.com/tristandeleu/ntm-one-shot
and was modified
"""
import numpy as np
import os
import random
import codecs

from .images import rotate_right


class OmniglotGenerator(object):
	"""OmniglotGenerator

	Args:
		data_file (str): 'data/omniglot/train.npz' or 'data/omniglot/test.npz'
		nb_classes (int): number of classes in an episode
		nb_samples_per_class (int): nuber of samples per class in an episode
		batchsize (int): number of episodes in each mini batch
		max_iter (int): max number of episode generation
		xp: numpy or cupy
	"""
	def __init__(self, data_file, augment=True, nb_classes=5, nb_samples_per_class=10,
				 batchsize=64, max_iter=None, xp=np, nb_samples_per_class_test=2):
		super(OmniglotGenerator, self).__init__()
		self.data_file = data_file
		self.nb_classes = nb_classes
		self.nb_samples_per_class = nb_samples_per_class
		self.nb_samples_per_class_test = nb_samples_per_class_test
		self.batchsize = batchsize
		self.max_iter = max_iter
		self.xp = xp
		self.num_iter = 0
		self.augment = augment
		self.data = self._load_data(self.data_file, self.xp)

	def _load_data(self, data_file, xp):
		data_dict = np.load(data_file)
		return {key: np.array(val) for (key, val) in data_dict.items()}

	def __iter__(self):
		return self

	def __next__(self):
		return self.next()

	def next(self):
		if (self.max_iter is None) or (self.num_iter < self.max_iter):
			labels_and_images_support = []
			labels_and_images_test = []

			self.num_iter += 1
			sampled_characters = random.sample(self.data.keys(), self.nb_classes) # list of keys

			for _ in range(self.batchsize):
				support_set = []

				for k,char in enumerate(sampled_characters):
					deg = random.sample(range(4), 1)[0]
					_imgs = self.data[char]
					_ind = random.sample(range(len(_imgs)), self.nb_samples_per_class+self.nb_samples_per_class_test)
					_ind_tr = _ind[:self.nb_samples_per_class]
					if self.augment:
						support_set.extend([(k, self.xp.array(rotate_right(_imgs[i], deg).flatten())) for i in _ind_tr])
					else:
						support_set.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind_tr])
					_ind_test = _ind[self.nb_samples_per_class:]
					if self.augment:
						labels_and_images_test.extend([(k, self.xp.array(rotate_right(_imgs[i], deg).flatten())) for i in _ind_test])
					else:
						labels_and_images_test.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind_test])

				random.shuffle(support_set)
				labels_tr, images_tr = zip(*support_set)

				labels_and_images_support.append((images_tr, labels_tr))

			_images, _labels = zip(*labels_and_images_support)

			images_tr = np.array(list(map(lambda x: x.reshape((1,-1)), _images[0])))

			# images_tr = self.xp.concatenate([
			# 	map(lambda x: x.reshape((1,-1)), _img) for _img in _images[0]], axis=0)
            # images_tr should be list shape=[5, 1, 784]

			labels_tr = np.array([_lbl for _lbl in zip(*_labels)])

			random.shuffle(labels_and_images_test)
			labels_test, images_test = zip(*labels_and_images_test)
			images_test = np.array(self.xp.concatenate([img.reshape((1, -1)) for img in images_test], axis=0))

			return (self.num_iter - 1), (images_tr, labels_tr, images_test, np.array(labels_test))
		else:
			raise StopIteration()

	def sample(self, nb_classes, nb_samples_per_class, nb_samples_per_class_test):
		sampled_characters = random.sample(self.data.keys(), nb_classes) # list of keys
		labels_and_images_support = []
		labels_and_images_test = []
		for (k, char) in enumerate(sampled_characters):
			deg = random.sample(range(4), 1)[0]
			_imgs = self.data[char]
			_ind = random.sample(range(len(_imgs)), nb_samples_per_class+nb_samples_per_class_test)
			_ind_tr = _ind[:nb_samples_per_class]
			labels_and_images_support.extend([(k, rotate_right(_imgs[i], deg).flatten()) for i in _ind_tr])
			_ind_test = _ind[nb_samples_per_class:]
			labels_and_images_test.extend([(k, rotate_right(_imgs[i], deg).flatten()) for i in _ind_test])


		random.shuffle(labels_and_images_support)
		labels_tr, images_tr = zip(*labels_and_images_support)

		random.shuffle(labels_and_images_test)
		labels_test, images_test = zip(*labels_and_images_test)
		return images_tr, labels_tr, images_test, labels_test

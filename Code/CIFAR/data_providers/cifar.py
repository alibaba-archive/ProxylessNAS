import os

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import DataProvider
from data_providers.augment import Cutout, PostNormRandomHorizontalFlip, PostNormRandomCrop


class CifarDataProvider(DataProvider):

	def __init__(self, save_path=None, train_batch_size=64, test_batch_size=200, valid_size=None, drop_last=True,
	             use_cutout=False, cutout_n_holes=1, cutout_size=16, **kwargs):
		self._save_path = save_path

		# baseline data augmentation on CIFAR, and channel-wise normalization
		mean, std = self.mean_std

		train_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
			PostNormRandomHorizontalFlip(),
			PostNormRandomCrop(pad=4),
		])

		# cutout
		if use_cutout:
			cutout = Cutout(n_holes=cutout_n_holes, length=cutout_size)
			train_transforms.transforms.append(cutout)

		# test transforms
		test_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])

		train_set, valid_set, test_set = self.build_datasets(train_transforms, test_transforms)

		if valid_size is not None:
			if isinstance(valid_size, float):
				valid_size = int(valid_size * len(train_set))
			else:
				assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
			train_indexes, valid_indexes = self.random_sample_valid_set(
				train_set.train_labels, valid_size, self.n_classes,
			)
			train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
			valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
			self.train = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=train_sampler,
			                                         pin_memory=torch.cuda.is_available(), num_workers=2,
			                                         drop_last=drop_last)
			self.valid = torch.utils.data.DataLoader(valid_set, batch_size=test_batch_size, sampler=valid_sampler,
			                                         pin_memory=torch.cuda.is_available(), num_workers=2,
			                                         drop_last=False)
		else:
			self.train = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True,
			                                         pin_memory=torch.cuda.is_available(), num_workers=2,
			                                         drop_last=drop_last)
			self.valid = None

		self.test = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
		                                        pin_memory=torch.cuda.is_available(), num_workers=2,
		                                        drop_last=False)
		if self.valid is None:
			self.valid = self.test

	@staticmethod
	def name():
		raise NotImplementedError

	@property
	def data_shape(self):
		return 3, 32, 32  # C, H, W

	@property
	def n_classes(self):
		raise NotImplementedError

	@property
	def save_path(self):
		if self._save_path is None:
			self._save_path = os.path.join('../../dataset', 'cifar%d' % self.n_classes)
		return self._save_path

	@property
	def data_url(self):
		""" Return url for downloaded data depends on cifar class """
		data_url = ('http://www.cs.toronto.edu/'
		            '~kriz/cifar-%d-python.tar.gz' % self.n_classes)
		return data_url

	@property
	def mean_std(self):
		raise NotImplementedError

	def build_datasets(self, train_transforms, test_transforms):
		raise NotImplementedError


class Cifar10DataProvider(CifarDataProvider):

	@staticmethod
	def name():
		return 'cifar10'

	@property
	def n_classes(self):
		return 10

	@property
	def mean_std(self):
		mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
		std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
		return mean, std

	def build_datasets(self, train_transforms, test_transforms):
		"""
			train_set: training set with train_transforms
			valid_set: training set with test_transforms
			test_set: test set with test_transforms
		"""
		train_set = datasets.CIFAR10(self.save_path, train=True, transform=train_transforms, download=True)
		valid_set = datasets.CIFAR10(self.save_path, train=True, transform=test_transforms, download=False)
		test_set = datasets.CIFAR10(self.save_path, train=False, transform=test_transforms, download=False)
		return train_set, valid_set, test_set


class Cifar100DataProvider(CifarDataProvider):

	@staticmethod
	def name():
		return 'cifar100'

	@property
	def n_classes(self):
		return 100

	@property
	def mean_std(self):
		mean = [x / 255.0 for x in [129.3, 124.1, 112.4]]
		std = [x / 255.0 for x in [68.2, 65.4, 70.4]]
		return mean, std

	def build_datasets(self, train_transforms, test_transforms):
		"""
			train_set: training set with train_transforms
			valid_set: training set with test_transforms
			test_set: test set with test_transforms
		"""
		train_set = datasets.CIFAR100(self.save_path, train=True, transform=train_transforms, download=True)
		valid_set = datasets.CIFAR100(self.save_path, train=True, transform=test_transforms, download=False)
		test_set = datasets.CIFAR100(self.save_path, train=False, transform=test_transforms, download=False)
		return train_set, valid_set, test_set

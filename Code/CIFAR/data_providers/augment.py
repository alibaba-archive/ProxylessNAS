import numpy as np

import torch


class Cutout(object):
	"""Randomly mask out one or more patches from an image.

	please refer to https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
	Args:
		n_holes (int): Number of patches to cut out of each image.
		length (int): The length (in pixels) of each square patch.
	"""

	def __init__(self, n_holes, length):
		self.n_holes = n_holes
		self.length = length

	def __call__(self, img):
		"""
		Args:
			img (Tensor): Tensor image of size (C, H, W).
		Returns:
			Tensor: Image with n_holes of dimension length x length cut out of it.
		"""
		if isinstance(img, np.ndarray):
			h = img.shape[1]
			w = img.shape[2]
		else:
			h = img.size(1)
			w = img.size(2)

		mask = np.ones((h, w), np.float32)

		for n in range(self.n_holes):
			# center point of the cutout region
			y = np.random.randint(h)
			x = np.random.randint(w)

			width = int(self.length / 2)
			y1 = np.clip(y - width, 0, h)
			y2 = np.clip(y + width, 0, h)
			x1 = np.clip(x - width, 0, w)
			x2 = np.clip(x + width, 0, w)

			mask[y1: y2, x1: x2] = 0.0

		if isinstance(img, np.ndarray):
			mask = np.expand_dims(mask, axis=0)
		else:
			mask = torch.from_numpy(mask)
			mask = mask.expand_as(img)

		return img * mask


class PostNormRandomHorizontalFlip(object):
	""" Random horizontal flip after normalization """

	def __init__(self, flip_prob=0.5):
		self.flip_prob = flip_prob

	def __call__(self, img):
		"""
		Args:
			img (Tensor): Tensor image of size (C, H, W).
		Returns:
			Tensor: Image after random horizontal flip.
		"""

		if np.random.random_sample() < self.flip_prob:
			np_img = img.numpy()  # C, H, W
			np_img = np_img[:, :, ::-1].copy()
			img = torch.from_numpy(np_img).float()

		return img


class PostNormRandomCrop(object):
	""" Random crop after normalization """

	def __init__(self, pad=4):
		self.pad = pad

	def __call__(self, img):
		"""
		Args:
			img (Tensor): Tensor image of size (C, H, W).
		Returns:
			Tensor: Image after random horizontal flip.
		"""

		np_img = img.numpy()  # C, H, W
		init_shape = np_img.shape
		new_shape = [init_shape[0],
		             init_shape[1] + self.pad * 2,
		             init_shape[2] + self.pad * 2]
		zeros_padded = np.zeros(new_shape)
		zeros_padded[:, self.pad:init_shape[1] + self.pad, self.pad:init_shape[2] + self.pad] = np_img

		# randomly crop to original size
		init_x = np.random.randint(0, self.pad * 2)
		init_y = np.random.randint(0, self.pad * 2)
		cropped = zeros_padded[:,
		          init_x: init_x + init_shape[1],
		          init_y: init_y + init_shape[2]]
		img = torch.from_numpy(cropped).float()
		return img

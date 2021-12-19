import numpy as np
from imgaug import augmenters as iaa
from scipy.ndimage import interpolation, rotate
import torch

class ImgAugTranslation(object):
	"""Translation
		Arg:
		  Pixels: number of pixels to apply translation to the image"""
	def __init__(self, pixels):
		n_pixels = int(pixels)
		self.aug = iaa.Affine(translate_px=(-n_pixels, n_pixels))

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)

class ImgAugRotation(object):
	"""Rotation
	   Arg:
	      Degrees: number of degrees to rotate the image"""
	def __init__(self, degrees):
		n_degrees = float(degrees)
		self.aug = iaa.Affine(rotate=(-n_degrees, n_degrees), mode='symmetric')

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)


class Translation(object):
	"""Translation"""
	def __init__(self, offset, order=0, isseg=False, mode='nearest'):
		self.order = order if isseg else 5
		self.offset = offset
		self.mode = 'nearest' if isseg else 'mirror'

	def __call__(self, img):
		return interpolation.shift(img, self.offset , order=self.order, mode=self.mode) 

class Rotation(object):
	"""Rotation"""
	def __init__(self, theta, order=0, isseg=False, mode='nearest'):
		self.order = order if isseg else 5
		self.theta = float(theta)
		self.mode = 'nearest' if isseg else 'mirror'

	def __call__(self, img):
		return rotate(img, self.theta, reshape=False, order=self.order, mode=self.mode)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(np.asarray(sample).astype(np.float32))		
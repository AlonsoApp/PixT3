import PIL
import math
import torch
import numpy as np
from enum import Enum
from PIL import Image
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.image_transforms import to_channel_dimension_format, convert_to_rgb, normalize
from typing import Optional, Union, Tuple


class ExplicitEnum(str, Enum):
	"""
	Enum with more explicit error message for missing values.
	"""

	@classmethod
	def _missing_(cls, value):
		raise ValueError(
			f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
		)

class ChannelDimension(ExplicitEnum):
	FIRST = "channels_first"
	LAST = "channels_last"

class Scale:
	DYNAMIC_ALL = -1.
	DYNAMIC_REDUCE_ONLY = -2.
	NO_RESCALE = 1.

def _normalize(image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs) -> np.ndarray:
	"""
	Normalize an image. image = (image - image_mean) / image_std.

	The image std is to mimic the tensorflow implementation of the `per_image_standardization`:
	https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

	Args:
		image (`np.ndarray`):
			Image to normalize.
	"""
	if image.dtype == np.uint8:
		image = image.astype(np.float32)

	# take mean across the whole `image`
	mean = np.mean(image)
	std = np.std(image)
	adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

	return normalize(image, mean=mean, std=adjusted_stddev, **kwargs)

def does_fit(image_height, image_width, patch_height, patch_width, max_patches):
	#return max_patches >= math.ceil(image_height / patch_height) * math.ceil(image_width / patch_width)
	return max_patches >= math.ceil((image_height*image_width) / (patch_width*patch_height))

def compute_scale_numerically(table_img_dim:Tuple, patch_height: int = 16, patch_width: int = 16,
							  max_patches: int = 2048, title_img:PIL.Image=None, max_downscale: float = float('-inf')):
	# Will ti fit right of the batch
	title_height, title_width = title_img.height, title_img.width
	table_height, table_width = table_img_dim
	image_height, image_width = title_height+table_height, max(title_width, table_width)
	if does_fit(image_height, image_width, patch_height, patch_width, max_patches):
		# It fits without downscaling
		return 1.0

	image_height, image_width = math.ceil(title_height + table_height*max_downscale), max(title_width, math.ceil(table_width*max_downscale))
	if not does_fit(image_height, image_width, patch_height, patch_width, max_patches):
		# It doesn't fit even downscaling it to the max_downscale
		return max_downscale

	# Now we know it would fit at some point of the downscale, let's find a better downscaling factor
	scale_no = 1.0 # scale that doesn't fit
	scale_yes = max_downscale # scale that fits
	for _ in range(5):
		new_scale = scale_yes + (scale_no - scale_yes)/2
		image_height, image_width = math.ceil(title_height + table_height * new_scale), max(title_width,
																					 math.ceil(table_width * new_scale))
		if does_fit(image_height, image_width, patch_height, patch_width, max_patches):
			scale_yes = new_scale
		else:
			scale_no = new_scale

	return scale_yes

def img_patch_size(table_img:Image, patch_height: int = 16, patch_width: int = 16, scale_mode: float = Scale.NO_RESCALE,
				   max_patches: int = 2048, max_downscale: float = 0.0, title_img:Image=None):

	table_height, table_width = table_img.height, table_img.width
	if title_img is None:
		# pix2struct original scaling function
		calc_scale = math.sqrt(max_patches * (patch_height / table_height) * (patch_width / table_width))
		calc_scale = max(calc_scale, max_downscale) if scale_mode != Scale.NO_RESCALE else 1.0
		num_feasible_rows = max(math.floor(calc_scale * table_height / patch_height), 1)
		num_feasible_cols = max(math.floor(calc_scale * table_width / patch_width), 1)
		resized_height = max(num_feasible_rows * patch_height, 1)
		resized_width = max(num_feasible_cols * patch_width, 1)
		return num_feasible_rows, num_feasible_cols, resized_height, resized_width
	elif scale_mode == Scale.DYNAMIC_REDUCE_ONLY:
		# numerically compute the scale down without downscaling the title
		calc_scale = compute_scale_numerically(table_img_dim=(table_height, table_width), title_img=title_img,
								 patch_height=patch_height, patch_width=patch_width, max_downscale=max_downscale)
		#title_height, title_width = title_img.height, title_img.width
		#image_height, image_width = title_height + table_height * calc_scale, max(title_width, table_width * calc_scale)
		num_feasible_rows = max(math.ceil(table_height * calc_scale / patch_height), 1)
		num_feasible_cols = max(math.ceil(table_width * calc_scale / patch_width), 1)
		resized_height = max(math.ceil(table_height * calc_scale), 1)
		resized_width = max(math.ceil(table_width * calc_scale), 1)
		return num_feasible_rows, num_feasible_cols, resized_height, resized_width

	elif Scale.DYNAMIC_ALL:
		raise NotImplemented("Dynamic all scaling not yet implemented.")
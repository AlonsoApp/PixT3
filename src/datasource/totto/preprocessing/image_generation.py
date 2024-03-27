import os
import sys
from typing import Dict, List

import numpy as np
from datasource.totto.utils import get_table_html, load_dataset_raw, get_highlighted_cells
from tools.img_utils import Scale, img_patch_size, does_fit
import imgkit
from PIL import Image
import math
import multiprocessing
from itertools import repeat

from tools.multithreading import starmap_with_kwargs

RENDERING_OPTIONS = {
	'quiet': '',
	'format': 'png',
	'width': 50,# 1024 or 512 could be interesting to analyse the amount of tables cut and how more horizontal text improves
	'encoding': "UTF-8",
	'quality':1
}


class Highlights:
	HIGHLIGHTED = "highlighted"
	NO_HIGHLIGHTED = "no_highlighted"
	MASKED = "masked"
	MASKED_INVERTED = "masked_inverted"
	GEN = "gen"
	folders = {HIGHLIGHTED: "highlighted", NO_HIGHLIGHTED: "no_highlighted", MASKED: "masked",
			   MASKED_INVERTED: "masked_inverted", GEN:"gen"}

class HeaderContent:
	PAGE_SECTION_TITLE = "page_section_title"
	HIGHLIGHTS = "highlights"

def highlight_config(highlight_mode: str, example: Dict):
	highlighted_css_path = 'resources/table_style.css'
	masked_css_path = 'resources/table_style_masked.css'
	masked_inverted_css_path = 'resources/table_style_masked_inverted.css'
	highlighted_cells, css_path = example["highlighted_cells"], None
	match highlight_mode:
		case Highlights.HIGHLIGHTED | Highlights.GEN:
			css_path = highlighted_css_path
		case Highlights.NO_HIGHLIGHTED:
			highlighted_cells = []
			css_path = highlighted_css_path
		case Highlights.MASKED:
			css_path = masked_css_path
		case Highlights.MASKED_INVERTED:
			css_path = masked_inverted_css_path
	return highlighted_cells, css_path

def crop_whitespace(img):
	"""
	Crops all white pixels vertically form right to left. Usually made to crop title headers
	:param img:
	:return:
	"""
	pix = np.asarray(img)
	pix = pix[:, :, 0:3]  # Drop the alpha channel
	for i in range(len(pix[0])-1, 0, -1):
		if not (pix[:,i,:]==255).all():
			return img.crop((0, 0, i+1, img.height))

def render_header(example, css_path, table_img_width, header_content: List):
	if not header_content:
		# Header empty
		return Image.new("RGB", (0, 0), "white")
	file_name = os.path.join(f'/tmp/h_{example["example_id"]}.png')
	html_title = ""
	if HeaderContent.PAGE_SECTION_TITLE in header_content:
		html_title += rf"<strong>Title: </strong>{example['table_page_title']}<br>"
		if example['table_section_title'] != '':
			html_title += rf"<strong>Section: </strong>{example['table_section_title']}<br>"
	if HeaderContent.HIGHLIGHTS in header_content:
		highlighted_cells = get_highlighted_cells(example)
		html_title += rf"<strong>Highlights: </strong>{' // '.join(highlighted_cells)}<br>"
	options = {'quiet': '',
	'format': 'png',
	'width': min(512, table_img_width),
	'encoding': "UTF-8",
	'quality':1}
	imgkit.from_string(html_title, file_name, options=options, css=css_path)
	img = Image.open(file_name)
	img = crop_whitespace(img)
	#img.save(file_name, format='PNG', quality=1)
	return img

def render_table(raw_table_path: str, example, highlight_mode):
	"""
	Renders a highlight_mode version of the table and returns a PIL Image of it. The table is rendered in raw_table_path
	so it can be reused for other dataset versions
	:param raw_table_path:
	:param example:
	:param highlight_mode:
	:return:
	"""
	if not os.path.isfile(raw_table_path):
		highlighted_cells, css_path = highlight_config(highlight_mode, example)
		html_table = get_table_html(example['table'], highlighted_cells)
		imgkit.from_string(html_table, raw_table_path, options=RENDERING_OPTIONS, css=css_path)
		img = Image.open(raw_table_path)
		return crop_whitespace(img)
	return Image.open(raw_table_path)

def render_image(example: Dict, img_dir: str, img_table_dir: str, force_gen: bool = False, scale_mode: float = Scale.NO_RESCALE,
				 header_content: List = None, highlight_mode: str = Highlights.HIGHLIGHTED,
				 max_downscale: float = 0.0, patch_height: int = 16, patch_width: int = 16,
				 max_patches: int = 2048, add_padding: bool = False, include_table: bool = True):
	header_content = [] if header_content is None else header_content
	raw_table_path = os.path.join(img_table_dir, str(example['example_id']) + '.png')
	out_file_path = os.path.join(img_dir, str(example['example_id']) + '.png')
	table_img = render_table(raw_table_path, example, highlight_mode)
	if not header_content and scale_mode == Scale.NO_RESCALE:
		# We only needed to render the raw table.
		return
	if not force_gen and os.path.isfile(out_file_path):
		# The image we needed to render is already at the destination folder
		return
	_, css_path = highlight_config(highlight_mode, example)
	max_header_width = table_img.width if include_table else 512
	header_img = render_header(example, css_path, max_header_width, header_content)
	# Now let's rescale it
	_, _, resized_height, resized_width = img_patch_size(table_img, scale_mode= scale_mode, max_downscale=max_downscale,
														 patch_height=patch_height, patch_width=patch_width,
														 max_patches=max_patches, title_img=header_img)

	table_img = table_img.resize((resized_width, resized_height), Image.HAMMING)
	# If after scaling still exceeds the input size we truncate
	crop_width, crop_height = crop_dimensions(table_img, patch_height, patch_width, max_patches,
											  crop_width=resized_width<resized_height, min_width=header_img.width,
											  title_img=header_img)
	if crop_width != resized_width or crop_height != resized_height:
		#table_img = table_img.crop((0, 0, crop_width, crop_height))
		cropped_table_img = Image.new("RGB", (crop_width, crop_height), "white")
		cropped_table_img.paste(table_img, (0, 0))
		table_img = cropped_table_img

	table_img = table_img if include_table else Image.new("RGB", (0, 0), "white")
	img = build_final_image(header_img, table_img, add_padding, patch_height=patch_height, patch_width=patch_width,
														 max_patches=max_patches)

	img.save(out_file_path, format='PNG', quality=1)


def build_final_image(title_img, table_img, add_padding, patch_height: int = 16, patch_width: int = 16,
				 max_patches: int = 2048):
	new_width = max(title_img.width, table_img.width)
	new_height = title_img.height + table_img.height
	if add_padding:
		# Largest image in this aspect ratio
		#ratio = math.ceil(new_width/patch_width) / math.ceil(new_height/patch_height)
		ratio = new_height / new_width
		final_width = max(math.floor(math.sqrt(max_patches/ratio)), math.ceil(new_width/patch_width))
		final_height = math.floor(max_patches/final_width)
		assert math.ceil(final_height)*math.ceil(final_width) <= max_patches
		new_image = Image.new("RGB", (final_width*patch_width, final_height*patch_height), "white")
	else:
		new_image = Image.new("RGB", (new_width, new_height), "white")
	new_image.paste(title_img, (0, 0))
	new_image.paste(table_img, (0, title_img.height))

	return new_image

def crop_dimensions(table_img, patch_height, patch_width, max_patches, limit_factor:float = 1.0,
					crop_width:bool = True, min_width:int = 0, title_img=None):
	"""
	We first crop table_width (if crop_width = True) up to limit_factor or min_width, then table_height up to limit_factor,
	then table_width again, and so on until img fits
	:param table_img:
	:param patch_height:
	:param patch_width:
	:param max_patches:
	:param limit_factor: don't crop more than a 25%
	:param crop_width:
	:param min_width: minimum table_width to crop
	:param title_img
	:return:
	"""
	if title_img is None:
		title_img = Image.new("RGB", (0, 0))
	table_width, table_height, title_width, title_height = table_img.width, table_img.height, title_img.width, title_img.height
	if not does_fit(title_height, title_width, patch_height, patch_width, max_patches):
		# Not even the header fits, there isn't much we can do. Table = 1x1
		return 1, 1
	image_height, image_width = title_height + table_height, max(title_width, table_width)
	#pixels_to_crop = (table_width * table_height) - (patch_height * patch_width * max_patches)
	while not does_fit(image_height, image_width, patch_height, patch_width, max_patches):
		pixels_to_crop = math.ceil((image_height*image_width) / (patch_width*patch_height) - max_patches) * patch_height * patch_width
		crop_dim, op_dim = (table_width, table_height) if crop_width else (table_height, table_width) # dimension to crop
		new_dim = crop_dim - min(math.ceil(pixels_to_crop/op_dim), math.ceil(crop_dim*limit_factor))
		new_dim = max(new_dim, min_width) if crop_width else new_dim
		table_width, table_height = (new_dim, table_height) if crop_width else (table_width, new_dim)  # if we crop table_width the cropped dim is the new table_width
		crop_width = not crop_width and table_width>min_width

		image_height, image_width = title_height + table_height, max(title_width, table_width)
		#return crop_dimensions(table_width, table_height, patch_height, patch_width, max_patches, limit_factor, not crop_width and table_width>min_width)
	else:
		return table_width, table_height

def generate_images(dataset_dir: str, mode: str, header_content: List = None, dataset_variant: str = "totto_data",
					scale_mode: float = Scale.NO_RESCALE, highlight_mode: str = Highlights.HIGHLIGHTED,
					max_downscale: float = 0.0, add_padding: bool = False, include_table: bool = True,
					dataset_file_names: Dict[str, str] = None):
	"""
	Gets a list of the lengths in tokens for each table of the 'mode' set linearized in 'linearization' way
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param mode: which dataset ['train','dev','test']
	:param dataset_variant:
	:param scale_mode: scale of the rendered image (0.5 would result in an image x4 times smaller resolution)
	:param header_content: list of HeaderContent to add into the rendered header
	:param highlight_mode: whether to highlight the highlighted cells in the table
	:param max_downscale: maximum scaling factor. According to our research, 0.58 is a good factor to avoid performance drops
	:param add_padding: if True, images smaller than 16x16x2048 pixels will be padded
	:param include_table: if False, only the header will be rendered
	:param dataset_file_names: names of the dataset files
	:return:
	"""
	dataset_path = os.path.join(dataset_dir, dataset_variant)
	# Build folder name
	highlighted_folder = Highlights.folders[highlight_mode]
	folder_name = f"{highlighted_folder}" if include_table else "notab" # Table mode
	folder_name += '_high' if header_content is not None and HeaderContent.HIGHLIGHTS in header_content else '' # Highlights in header
	folder_name += f"_0{int(max_downscale * 100)}" # Scale

	img_out_dir = os.path.join(dataset_dir, 'img', folder_name, mode)
	img_table_dir = os.path.join(dataset_dir, 'img', 'raw_tables', highlighted_folder, mode)
	os.makedirs(img_out_dir, exist_ok=True)
	os.makedirs(img_table_dir, exist_ok=True)
	failed = []
	examples = load_dataset_raw(dataset_path, mode, indexed=False, file_names=dataset_file_names)
	num_threads = multiprocessing.cpu_count() - 2
	with multiprocessing.Pool(num_threads) as pool:
		args_iter = zip(examples, repeat(img_out_dir), repeat(img_table_dir))
		kwargs_iter = repeat(dict(scale_mode=scale_mode, header_content=header_content, highlight_mode=highlight_mode,
								  max_downscale=max_downscale, add_padding=add_padding, include_table=include_table))
		starmap_with_kwargs(pool, render_image, args_iter, kwargs_iter, len(examples))
	print(failed)
	print(rf"Total failed: {failed}")


def run():
	dataset_dir = "./data/ToTTo/"
	config1 = {"header_content":[HeaderContent.PAGE_SECTION_TITLE, HeaderContent.HIGHLIGHTS], "include_table":False, "highlight_mode":Highlights.HIGHLIGHTED}
	config2 = {"header_content":[HeaderContent.PAGE_SECTION_TITLE], "include_table":True, "highlight_mode":Highlights.NO_HIGHLIGHTED}
	config3 = {"header_content":[HeaderContent.PAGE_SECTION_TITLE], "include_table":True, "highlight_mode":Highlights.MASKED_INVERTED}
	for config in [config1, config2, config3]:
		for max_down in [0.39]:
			for mode in ["dev", "test", "train"]:
				generate_images(dataset_dir, mode, scale_mode=Scale.DYNAMIC_REDUCE_ONLY,
								header_content=config["header_content"],
								highlight_mode=config["highlight_mode"], max_downscale=max_down,
								include_table=config["include_table"])

def run_for_gen():
	dataset_dir = "./data/ToTTo/"
	dataset_file_names = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
	for mode in ["dev", "test", "train"]:
		generate_images(dataset_dir, mode, scale_mode=Scale.NO_RESCALE, dataset_variant="warmup_ssl3",
						highlight_mode=Highlights.GEN, include_table=True, dataset_file_names=dataset_file_names)

def run_for_l2t():
	dataset_dir = "./data/Logic2Text/"
	dataset_file_names = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
	config1 = {"header_content": [HeaderContent.PAGE_SECTION_TITLE], "include_table": True,
			   "highlight_mode": Highlights.HIGHLIGHTED}
	config2 = {"header_content": [HeaderContent.PAGE_SECTION_TITLE], "include_table": True,
			   "highlight_mode": Highlights.NO_HIGHLIGHTED}
	config3 = {"header_content": [HeaderContent.PAGE_SECTION_TITLE, HeaderContent.HIGHLIGHTS], "include_table": False,
			   "highlight_mode": Highlights.NO_HIGHLIGHTED}
	for config in [config1, config2, config3]:
		for max_down in [0.39]:
			for mode in ["dev", "test", "train"]:
				generate_images(dataset_dir, mode, scale_mode=Scale.DYNAMIC_REDUCE_ONLY,
								dataset_variant="l2t_totto_data",
								header_content=config["header_content"],
								highlight_mode=config["highlight_mode"], max_downscale=max_down,
								include_table=config["include_table"], dataset_file_names=dataset_file_names)

def run_for_wg():
	dataset_dir = "./data/WeatherGov/"
	dataset_file_names = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
	config2 = {"header_content": [HeaderContent.PAGE_SECTION_TITLE], "include_table": True,
			   "highlight_mode": Highlights.NO_HIGHLIGHTED}
	for config in [config2]:
		for max_down in [0.39]:
			for mode in ["dev", "test", "train"]:
				generate_images(dataset_dir, mode, scale_mode=Scale.DYNAMIC_REDUCE_ONLY,
								dataset_variant="wg_totto_data",
								header_content=config["header_content"],
								highlight_mode=config["highlight_mode"], max_downscale=max_down,
								include_table=config["include_table"], dataset_file_names=dataset_file_names)

if __name__ == '__main__':
	flag = sys.argv[1]
	if flag == "totto":
		run()
	elif flag == "l2t":
		run_for_l2t()
	elif flag == "slc":
		run_for_gen()
	else:
		print(f"Flag {flag} not supported.")

"""
if __name__ == '__main__':
	css_path = 'resources/table_style.css'
	totto_dir = "data/ToTTo/"
	data_dir = os.path.join(totto_dir, 'totto_data')
	img_dir = os.path.join(totto_dir, 'img')
	os.makedirs(img_dir, exist_ok=True)
	dataset = load_dataset_raw(data_dir, 'dev', indexed=True)
	for id_example, example in tqdm(dataset.items()):
		html_table = get_table_html(example['table'], example["highlighted_cells"])
		# html_table = add_table_stype(html_table)
		file_name = os.path.join(img_dir, str(id_example) + '.jpg')
		imgkit.from_string(html_table, file_name, options=options, css=css_path)
"""

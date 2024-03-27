import os
from typing import List

from transformers import AutoTokenizer, T5Tokenizer

from datasource.totto.baseline_preprocessing import preprocess_utils
from datasource.totto.utils import linearize_table, FILE_NAMES, linearize_table_tripplets, load_dataset_raw, \
	DATASET_EXAMPLES
import json
import six
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from tools.img_utils import img_patch_size
import multiprocessing
from tqdm import tqdm
from pytictoc import TicToc
import math
import numpy as np


# Linearization types
class Linearization:
	BASELINE = 'baseline'
	TABLE2LOGIC = 'table2logic'
	TRIPLETS = 'triplets'
	ALL = [BASELINE, TABLE2LOGIC]


SPECIAL_TOKENS = {
	Linearization.BASELINE: {'additional_special_tokens': ["<row>", "</row>", "<cell>", "</cell>", "<row_header>",
														   "</row_header>", "<col_header>", "</col_header>",
														   "<highlighted_cell>", "</highlighted_cell>", "<table>",
														   "</table>"]}}


def lineraize_table(table, highlighted_cells: List, linearization: str, tokenizer=None) -> str:
	match linearization:
		case Linearization.BASELINE:
			return preprocess_utils.linearize_full_table(table, highlighted_cells, None, None)
		case Linearization.TABLE2LOGIC:
			return linearize_table(table, highlighted_cells, tokenizer, add_row_idx=True)
		case Linearization.TRIPLETS:
			return linearize_table_tripplets(table, tokenizer)
	raise Exception("Linearization '{}' not implemented".format(linearization))


def get_token_lengths(dataset_dir: str, mode: str, linearization: str, model_name: str,
					  dataset_variant: str = "totto_data", highlight_cells: bool = False, force_run: bool = False,
					  use_special_tokens: bool = False):
	"""
	Gets a list of the lengths in tokens for each table of the 'mode' set linearized in 'linearization' way
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param mode: which dataset ['train','dev','test']
	:param linearization: table linearization mode
	:param model_name: name of the huggingface model for the tokenizer
	:param dataset_variant:
	:param highlight_cells: whether to include highlight information on the linearized table
	:param force_run: skip cache and redo the counting process
	:param use_special_tokens:
	:return:
	"""
	token_lengths = []
	dataset_path = os.path.join(dataset_dir, dataset_variant, FILE_NAMES[dataset_variant][mode])
	lin_name_file = linearization+"_st" if use_special_tokens else linearization
	cache_path = os.path.join(dataset_dir, 'cache', model_name, "{}_{}.pkl".format(lin_name_file, mode))
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path.format(mode), 'rb') as cache_file:
			return pickle.load(cache_file)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	if use_special_tokens:
		tokenizer.add_special_tokens(SPECIAL_TOKENS[linearization])
	failed = 0
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			try:
				if len(token_lengths) % 100 == 0:
					print("Num examples processed: %d" % len(token_lengths))

				line = six.ensure_text(line, "utf-8")
				json_example = json.loads(line)
				table = json_example["table"]
				highlight_cells = json_example["highlighted_cells"] if highlight_cells else []
				full_table_str = lineraize_table(table, highlight_cells, linearization, tokenizer)
				token_lengths.append(len(tokenizer(full_table_str)[0]))
			except:
				failed += 1
	print("Failed: {}".format(failed))
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache', model_name), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(token_lengths, f)
	return token_lengths

_tokenizer: T5Tokenizer
def get_text_length(example):
	try:
		lengths = []
		for sentence in example['sentence_annotations']:
			text = sentence['final_sentence']
			lengths.append(len(_tokenizer(text)[0]))
		return example['example_id'], max(lengths)
	except:
		return example['example_id'], None

def get_text_token_lengths(dataset_dir: str, mode: str, model_name: str, dataset_variant: str = "totto_data",
						   force_run: bool = False):
	"""
	Gets a list of the lengths in tokens for each table of the 'mode' set linearized in 'linearization' way
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param mode: which dataset ['train','dev','test']
	:param model_name: name of the huggingface model for the tokenizer
	:param dataset_variant:
	:param force_run: skip cache and redo the counting process
	:return:
	"""
	global _tokenizer
	text_lengths = []
	cache_path = os.path.join(dataset_dir, 'cache', 'text', model_name, rf"{mode}.pkl")
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path, 'rb') as cache_file:
			return pickle.load(cache_file)
	_tokenizer = AutoTokenizer.from_pretrained(model_name)
	failed = []
	num_threads = multiprocessing.cpu_count() - 2
	examples = load_dataset_raw(os.path.join(dataset_dir, dataset_variant), mode, indexed=False)
	with multiprocessing.Pool(num_threads) as pool:
		for example_id, text_length in tqdm(pool.imap_unordered(get_text_length, examples, chunksize=32),
											 total=len(examples),
											 desc=f"Calculating text lengths"):
			if text_length is None:
				failed.append(example_id)
			else:
				text_lengths.append(text_length)
	print(failed)
	print(rf"Total failed: {len(failed)}")
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache', "text", model_name), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(text_lengths, f)
	return text_lengths

image_dir: str
def get_patch_length(example):
	try:
		image_path = os.path.join(image_dir, str(example['example_id']) + '.png')
		image = Image.open(image_path)  # No need to convert to RGB it will be done by Pix2StructImageProcessor
		num_rows, num_cols, _, _ = img_patch_size(image)
		patch_length = num_rows * num_cols
		return example['example_id'], patch_length
	except:
		return example['example_id'], None

def get_patch_lengths_multi(dataset_dir: str, mode: str, dataset_variant: str = "totto_data", force_run: bool = False):
	r"""
	Gets a list of the lengths in patches for each table image of the 'mode'
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param image_dir: path to image directory
	:param mode: which dataset ['train','dev','test']
	:param dataset_variant:
	:param force_run: skip cache and redo the counting process
	:return:
	"""
	global image_dir
	img_lengths = []
	failed = []
	cache_path = os.path.join(dataset_dir, 'cache', "img", rf"{mode}.pkl")
	image_dir = os.path.join(dataset_dir, 'img/raw_tables/no_highlighted', mode)
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path.format(mode), 'rb') as cache_file:
			return pickle.load(cache_file)
	num_threads = multiprocessing.cpu_count() - 2
	dataset = load_dataset_raw(os.path.join(dataset_dir, dataset_variant), mode, indexed=True)
	examples = dataset.values()
	with multiprocessing.Pool(num_threads) as pool:
		for example_id, patch_length in tqdm(pool.imap_unordered(get_patch_length, examples, chunksize=32), total=len(examples),
						   desc=f"Calculating patch length"):
			if patch_length is None:
				failed.append(example_id)
			else:
				img_lengths.append(patch_length)
	print(failed)
	print(rf"Total failed: {len(failed)}")
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache', "img"), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(img_lengths, f)
	return img_lengths


def get_patch_lengths(dataset_dir: str, mode: str, dataset_variant: str = "totto_data", force_run: bool = False):
	r"""
	Gets a list of the lengths in patchs for each table image of the 'mode'
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param image_dir: path to image directory
	:param mode: which dataset ['train','dev','test']
	:param dataset_variant:
	:param force_run: skip cache and redo the counting process
	:return:
	"""
	img_lengths = []
	dataset_path = os.path.join(dataset_dir, dataset_variant, FILE_NAMES[dataset_variant][mode])
	cache_path = os.path.join(dataset_dir, 'cache', "img", rf"{mode}.pkl")
	image_dir = os.path.join(dataset_dir, 'img', mode)
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path.format(mode), 'rb') as cache_file:
			return pickle.load(cache_file)
	failed = []
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			try:
				if len(img_lengths)>0 and len(img_lengths) % 100 == 0:
					print(rf"Num examples processed: {len(img_lengths)}")
				line = six.ensure_text(line, "utf-8")
				json_example = json.loads(line)

				image_path = os.path.join(image_dir, str(json_example['example_id']) + '.png')
				image = Image.open(image_path)  # No need to convert to RGB it will be done by Pix2StructImageProcessor
				num_rows, num_cols, _, _ = img_patch_size(image)
				img_lengths.append(num_rows*num_cols)
			except:
				failed.append(json_example['example_id'])
	print(failed)
	print(rf"Total failed: {len(failed)}")
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache', "img"), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(img_lengths, f)
	return img_lengths

def _build_geometric_bins(max_num, base: int = 512, max_exp: int = 10):
	"""
	Geometric series of numbers from base until min(base * 2 ** max_exp, max_num)
	:param max_num:
	:param base:
	:return:
	"""
	bins = [0]
	for i in range(max_exp):
		num = base * 2 ** i
		bins.append(num)
		if num > max_num:
			break
	return bins if max_num < bins[-1] else bins + [max_num]

def build_log_bins(min_val:int, max_val:int, num_bins:int):
	"""
	Logarithmic series of numbers to fit min_val and max_val into num_bins bins
	:param min_val:
	:param max_val:
	:param num_bins:
	:return:
	"""
	log_min = math.log(min_val)
	log_max = math.log(max_val)
	log_range = log_max - log_min
	log_step = log_range / num_bins
	bins = [math.exp(log_min + n * log_step) for n in range(num_bins)]
	bins.append(max_val)
	return bins


def _build_hist(lengths):
	labels = []
	hist = []
	bins = _build_geometric_bins(max(lengths)) #range(64)
	for i, step in enumerate(bins[1:]):
		labels.append("{} ({})".format(str(step), int(round(step / 512, 0))))
		hist.append(round(sum((bins[i] < length <= step) for length in lengths) / len(lengths) * 100, 2))

	return hist, labels

def plot_distribution(lengths):
	fig, ax = plt.subplots()

	hist, labels = _build_hist(lengths)

	bars = ax.bar(labels, hist, label=labels)
	ax.bar_label(bars)

	ax.set_ylabel('% tables')
	# xticks_pos = [0.5 * patch.get_width() + patch.get_xy()[0] for patch in bars]
	plt.xticks(rotation=30, ha='right')
	plt.subplots_adjust(bottom=0.18)
	plt.show()


"""
def analyze_hist(lengths):
	# Geometric progression
	bins = _build_bins(max(lengths))
	# plt.hist(lengths, bins=bins)
	# plt.show()

	# histogram on linear scale
	hist, bins, _ = plt.hist(lengths, bins=100, weights=np.ones(len(lengths)) / len(lengths))
	# plt.yscale('log')
	plt.xlabel('sequence length in tokens', fontsize=16)
	plt.ylabel('log tables', fontsize=16)
	# plt.xscale('function', functions=(forward, inverse))
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	plt.show()
"""


def build_all():
	dataset_dir = "./data/ToTTo/"
	for model_name in ["bert-base-uncased"]:  # , "t5-base"]:
		for linear in [Linearization.BASELINE, Linearization.TABLE2LOGIC]:
			for mode in ["train", "dev", "test"]:
				get_token_lengths(dataset_dir, mode, linear, model_name, force_run=False)


def last_highlight_loc_distribution(dataset_dir: str, mode: str, linearization: str, model_name: str,
									dataset_variant: str = "totto_data", force_run: bool = False):
	"""
	Produces a list with teh token index of the last highlighted value of each table
	:param dataset_dir:
	:param mode:
	:param linearization:
	:param model_name:
	:param dataset_variant:
	:param force_run:
	:return:
	"""
	# Index of the token referring to each highlighted value
	dataset_path = os.path.join(dataset_dir, dataset_variant, FILE_NAMES[dataset_variant][mode])
	cache_path = os.path.join(dataset_dir, 'cache', model_name, "tk_{}_{}.pkl".format(linearization, mode))
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path.format(mode), 'rb') as cache_file:
			return pickle.load(cache_file)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	failed = 0
	masks_in_tables = 0
	token_idx = []
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			try:
				if masks_in_tables % 100 == 0:
					print("Num examples processed: %d" % masks_in_tables)

				line = six.ensure_text(line, "utf-8")
				json_example = json.loads(line)
				table = json_example["table"]
				highlight_cells = json_example["highlighted_cells"]
				row_index = highlight_cells[-1][0]
				col_index = highlight_cells[-1][1]
				table[row_index][col_index]['value'] += " " + tokenizer.mask_token + " "
				full_table_str = lineraize_table(table, highlight_cells, linearization, tokenizer)
				tokens = tokenizer(full_table_str)
				masks_in_tables += 1 if tokenizer.mask_token_id in tokens['input_ids'] else 0
				if tokenizer.mask_token_id in tokens['input_ids']:
					tok_idx = len(tokens['input_ids']) - tokens['input_ids'][::-1].index(tokenizer.mask_token_id) - 1
					token_idx.append(tok_idx)
			except:
				failed += 1
	print("Failed: {}".format(failed))
	print("Masks found: {}".format(masks_in_tables))
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache', model_name), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(token_idx, f)
	return token_idx

def highlight_loc_distribution(dataset_dir: str, mode: str, dataset_variant: str = "totto_data",
							   force_run: bool = False):
	"""
	Produces a 3x3 heatmap (discretized to 0, 1, 2) of the location of highlighted cells
	:param dataset_dir:
	:param mode:
	:param linearization:
	:param model_name:
	:param dataset_variant:
	:param force_run:
	:return:
	"""
	# Index of the token referring to each highlighted value
	dataset_path = os.path.join(dataset_dir, dataset_variant, FILE_NAMES[dataset_variant][mode])
	cache_path = os.path.join(dataset_dir, 'cache', 'loc_h_cells', f"loc_{mode}.pkl")
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path.format(mode), 'rb') as cache_file:
			return pickle.load(cache_file)
	failed = 0
	h_cell_loc_row = [0]*3
	h_cell_loc_col = [0]*3
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		for line in tqdm(input_file, total=DATASET_EXAMPLES[mode]):
			try:
				line = six.ensure_text(line, "utf-8")
				json_example = json.loads(line)
				table = json_example["table"]
				highlight_cells = json_example["highlighted_cells"]
				num_cols = max([len(row) for row in table])
				num_rows = len(table)
				for h_cell in highlight_cells:
					table_cell = table[h_cell[0]][h_cell[1]]
					increase_counter(h_cell[0], table_cell, num_rows, h_cell_loc_row)
					increase_counter(h_cell[1], table_cell, num_cols, h_cell_loc_col)
			except:
				failed += 1
	print("Failed: {}".format(failed))
	heatmap = [h_cell_loc_row, h_cell_loc_col]
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache', 'loc_h_cells'), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(heatmap, f)
	return heatmap

def select_heatmap_loc(idx, length):
	loc = (idx+1)/length
	if loc <= 1/3:
		return 0 # top / left
	elif loc <= 2/3:
		return 1 # mid
	else:
		return 2 # bot / right

def increase_counter(h_cell_idx, table_cell, total, heatmap):
	if total >= 3:
		heatmap_loc = select_heatmap_loc(h_cell_idx, total)  # table_cell['column_span']
		heatmap[heatmap_loc] += 1
		for span in range(1, table_cell['column_span']):
			hm_loc_span = select_heatmap_loc(h_cell_idx + span-1, total)
			if hm_loc_span != heatmap_loc:
				heatmap[hm_loc_span] += 1

def table_length_distribution():
	lengths = []
	for mode in ["train"]:#, "dev", "test"]:
		lengths += get_token_lengths("./data/ToTTo/", mode, Linearization.BASELINE, 'bert-base-uncased', use_special_tokens=False, force_run=True)
	plot_distribution(lengths)

def text_length_distribution():
	lengths = []
	for mode in ["dev", "train"]:
		lengths += get_text_token_lengths("./data/ToTTo/", mode, 't5-base')
	plot_distribution(lengths)

def last_highlight_distribution(models=None, modes=None, linearizations=None):
	if models is None:
		models = ["bert-base-uncased"]
	if modes is None:
		modes = ["dev", "test", "train"]
	if linearizations is None:
		linearizations = Linearization.ALL
	for model_name in models:
		for lin in linearizations:
			token_idx = []
			for mode in modes:
				token_idx += last_highlight_loc_distribution("./data/ToTTo/", mode, lin, model_name)
			plot_distribution(token_idx)

def print_heatmap(heatmaps):
	row_heatmap = [0]*3
	col_heatmap = [0]*3
	for heatmap in heatmaps:
		for i, e in enumerate(heatmap[0]):
			row_heatmap[i] += e
		for i, e in enumerate(heatmap[1]):
			col_heatmap[i] += e
	print(f"row: {[e/sum(row_heatmap) for e in row_heatmap]}")
	print(f"col: {[e/sum(col_heatmap) for e in col_heatmap]}")


def highlight_loc_heatmap(modes=None):
	if modes is None:
		modes = ["dev", "test", "train"]
	heatmaps = []
	for mode in modes:
		heatmaps.append(highlight_loc_distribution("./data/ToTTo/", mode))
	print_heatmap(heatmaps)


def img_patch_distribution():
	lengths = []
	t = TicToc()
	for mode in ["train"]:
		t.tic()
		lengths += get_patch_lengths_multi("./data/ToTTo/", mode, force_run=False) #get_patch_lengths("./data/ToTTo/", mode)
		t.toc()
	plot_distribution(lengths)

if __name__ == '__main__':
	#table_length_distribution()
	img_patch_distribution()
	#text_length_distribution()
	# build_all()
	# last_highlight_distribution(linearizations=[Linearization.TABLE2LOGIC])
	#highlight_loc_heatmap()

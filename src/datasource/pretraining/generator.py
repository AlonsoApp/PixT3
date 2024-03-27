import itertools
import os
import pickle
from typing import Dict, List, Tuple, Optional, Set

from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from collections import Counter
import colorcet as cc
import random
import uuid
import json
import string

from transformers import AutoTokenizer

from datasource.totto.baseline_preprocessing.table_to_text_html_utils import get_table_html
from datasource.totto.utils import load_dataset_raw, FILE_NAMES


def get_size_span(html_content):
	soup = BeautifulSoup(html_content, 'html.parser')
	table = soup.find('table')  # Assuming there's only one table in the HTML

	if table:
		rows = table.find_all('tr')
		num_rows = 0
		num_columns = 0
		span_pairs = []
		for row in rows:
			cells = row.find_all(['th', 'td'])
			sum_colspan = 0
			for cell in cells:
				colspan = int(cell.get('colspan', 1))
				rowspan = int(cell.get('rowspan', 1))
				span_pairs.append((rowspan, colspan))
				sum_colspan += colspan
			num_columns = max(num_columns, sum_colspan)
			num_rows += max(int(row.get('rowspan', 1)), 1)

		return (num_rows, num_columns), span_pairs
	else:
		return None  # No table found in the HTML

def add_span_pairs_to_dist(span_dist:Dict[str, Dict[str, int]], span_pairs:List[Tuple]):
	"""

	:param span_dist: {rowspan: {colspan, colspan}}
	:param span_pairs: [(rowspan, colspan), (rowspan, colspan), ...]
	:return:
	"""
	for rowspan, colspan in span_pairs:
		rowspan = str(rowspan)
		colspan = str(colspan)
		if rowspan not in span_dist:
			span_dist[rowspan] = {colspan:1}
		elif colspan not in span_dist[rowspan]:
			span_dist[rowspan][colspan] = 1
		else:
			span_dist[rowspan][colspan] += 1
	return span_dist

def convert_to_matrix(span_dist:Dict[str, Dict[str, int]]):
	"""

	:param span_dist: {rowspan: {colspan, colspan}}
	:return:
	"""
	max_rowspan = max([int(key) for key in span_dist.keys()])
	max_colspan = max([max([int(key) for key in colspan_dist.keys()]) for colspan_dist in span_dist.values()])
	dist_matrix = np.zeros((max_rowspan, max_colspan))
	for rowspan in range(max_rowspan):
		for colspan in range(max_colspan):
			if str(rowspan+1) in span_dist and str(colspan+1) in span_dist[str(rowspan+1)]:
				dist_matrix[rowspan, colspan] = span_dist[str(rowspan+1)][str(colspan+1)]
	return dist_matrix

def plot_matrix(dist_matrix:np.matrix, hide_zero_zero:bool = True, subview:Tuple=(10,10)):
	tmp = dist_matrix
	tmp[0,0] = 0 if hide_zero_zero else tmp[0,0]
	tmp = tmp[:subview[0], :subview[1]]
	sns.heatmap(tmp, linewidth=0.5, cmap="crest")
	plt.show()

def plot_distribution(values:List[int], max_value:int = None):
	values = [x for x in values if x <= max_value] if max_value is not None else values
	sns.histplot(data=values, stat='percent')
	plt.show()

def plot_sample_multi_distribution(col_row_dist:Dict[str, List[int]], max_col:int = 40, max_row:int = None):
	"""
	This is the function that generates that nice multy col row distribution figure.
	To get col_row_dist use compute_distributions_old
	:param col_row_dist:
	:param max_col:
	:param max_row:
	:return:
	"""
	row_dists = []
	# First we get the first max_col columns
	for col in range(1, max_col+1):
		if str(col) not in col_row_dist:
			row_dists.append([])
		else:
			row_dist = [x for x in col_row_dist[str(col)] if x <= max_row] if max_row is not None else col_row_dist[str(col)]
			row_dists.append(row_dist)
	# We group the rest
	fig, ax = plt.subplots()
	for row_dist in row_dists:
		sns.histplot(data=row_dist, ax=ax, stat='probability')
	plt.show()

def plot_prob_distribution(row_prob_matrix:np.array):
	sns.barplot(x=np.arange(len(row_prob_matrix)), y=row_prob_matrix)
	plt.show()

def plot_prob_multi_distribution(row_prob_matrix:np.array):
	total_cols = row_prob_matrix.shape[1]
	palette = sns.color_palette(cc.glasbey, n_colors=row_prob_matrix.shape[1])
	ax = plt.subplots()
	for col in range(total_cols):
		ax=sns.barplot(x=np.arange(len(row_prob_matrix[:,col])), y=row_prob_matrix[:,col], color = palette[col], alpha=0.6)#, ax=ax)
	ax.set(xlabel="Columns", ylabel="Probability")
	#ax.set_xticklabels(labels=[x for x in range(row_prob_matrix.shape[0])], rotation=90)
	plt.show()

def compute_distributions_old(dataset_dir: str = "./data/ToTTo/", dataset_variant: str = "totto_data", force_run: bool = False):
	"""
	Gets three different distributions
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param dataset_variant:
	:param force_run: skip cache and redo the counting process
	:return:col_dist:List[int], row_dist:Dict[str, List[int]], span_dist:np.matrix
	"""
	cache_dir = os.path.join(dataset_dir, 'cache', 'stats')
	cache_path = os.path.join(cache_dir, f"dist.pkl")
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path, 'rb') as cache_file:
			results = pickle.load(cache_file)
			return results["all_columns"], results["all_rows"], results["dist_matrix"]
	all_columns = []
	span_dist = {}
	all_rows = {}
	all_rows = defaultdict(lambda: [], all_rows)
	for mode in ["train"]:#["dev", "test", "train"]:
		examples = load_dataset_raw(os.path.join(dataset_dir, dataset_variant), mode, indexed=False)
		for example in tqdm(examples):
			table_html = get_table_html(example['table'], [])
			size_pair, span_pairs = get_size_span(table_html)
			num_rows, num_columns = size_pair
			all_columns.append(num_columns)
			all_rows[str(num_columns)].append(num_rows)
			span_dist = add_span_pairs_to_dist(span_dist, span_pairs)
		# Save cache
		os.makedirs(cache_dir, exist_ok=True)
		dist_matrix = convert_to_matrix(span_dist)
		results = {"all_columns": all_columns, "all_rows": dict(all_rows), "dist_matrix":dist_matrix}
		with open(cache_path, 'wb') as f:
			pickle.dump(results, f)
	return all_columns, dict(all_rows), dist_matrix


def compute_distributions(dataset_dir: str, dataset_variant: str = "totto_data", force_run: bool = False):
	"""
	Gets out of the given ToTTo dataset, row v column and rowspan v colspan count matrices
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param dataset_variant:
	:param force_run: skip cache and redo the counting process
	:return:col_dist:List[int], row_dist:Dict[str, List[int]], span_dist:np.matrix
	"""
	cache_dir = os.path.join(dataset_dir, 'cache', 'stats')
	cache_path = os.path.join(cache_dir, f"dist.pkl")
	# Load form cache
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path, 'rb') as cache_file:
			results = pickle.load(cache_file)
			return results["size_dist_matrix"], results["span_dist_matrix"]
	size_dict = {}
	span_dict = {}
	for mode in ["train"]:#["dev", "test", "train"]:
		examples = load_dataset_raw(os.path.join(dataset_dir, dataset_variant), mode, indexed=False)
		for example in tqdm(examples):
			table_html = get_table_html(example['table'], [])
			size_pair, span_pairs = get_size_span(table_html)
			span_dict = add_span_pairs_to_dist(span_dict, span_pairs)
			size_dict = add_span_pairs_to_dist(size_dict, [size_pair])
		# Save cache
		os.makedirs(cache_dir, exist_ok=True)
		size_dist_matrix = convert_to_matrix(size_dict)
		span_dist_matrix = convert_to_matrix(span_dict)
		results = {"size_dist_matrix": size_dist_matrix, "span_dist_matrix": span_dist_matrix}
		with open(cache_path, 'wb') as f:
			pickle.dump(results, f)
	return size_dist_matrix, span_dist_matrix

def samples_to_probs(samples:List[int], max_cap:int=None) -> List[float]:
	"""
	We consider a discrete distribution
	:param samples:
	:param max_cap:
	:return:
	"""
	max_cap = max(samples) if max_cap is None else max_cap
	samples = [x for x in samples if x <= max_cap]
	counts = Counter(samples)
	total = sum(counts.values())
	probability_mass = {k: v / total for k, v in counts.items()}
	probability_mass = defaultdict(lambda: 0.0, probability_mass)
	return [probability_mass[i+1] for i in range(max_cap)]

def count_to_probs(counts:np.array, max_cap:int=None) -> List[float]:
	"""
	We consider a discrete distribution
	:param counts:
	:param max_cap:
	:return:
	"""
	counts = counts[0:max_cap] if max_cap is not None else counts
	return [c / sum(counts) for c in counts] if sum(counts) > 0 else [0.0]*len(counts)


def compute_probabilities(size_dist_matrix:np.ndarray[np.ndarray[int]], span_dist_matrix:np.ndarray[np.ndarray[int]], max_cols:int, max_rows:int):
	"""
	Transforms the count matrices to probabilities
	:param size_dist_matrix:
	:param span_dist_matrix:
	:param max_cols:
	:param max_rows:
	:return:
		col_probs: a set of probabilities from 1 to max_cols
		row_probs: a set of probabilities from 1 to max_rows
		col_span_probs: a set of probabilities from 1 to max_cols for general probability of a colspan
		row_span_probs_matrix: a set of probabilities from 1 to max_rows for general probability of a rowspan for every colspan from 1 to max_cols
	"""
	col_probs = count_to_probs(size_dist_matrix.sum(axis=0), max_cols)

	# The following code would compute the prob for each col number.
	# But as we discovered that they all follow the same distribution, we will aggregate all the counts and compute a
	# single distribution for rows
	#row_probs = np.zeros((max_rows, max_cols))
	#for col in range(max_cols):
	#	probs = count_to_probs(size_dist_matrix[:, col], max_rows)
	#	row_probs[:, col] = probs
	row_probs = count_to_probs(size_dist_matrix.sum(axis=1), max_rows)

	col_span_probs = count_to_probs(span_dist_matrix.sum(axis=0), max_cols)

	row_span_probs_matrix = np.zeros((max_rows, max_cols))
	for colspan in range(max_cols):
		probs = count_to_probs(span_dist_matrix[:,colspan], max_rows)
		row_span_probs_matrix[:,colspan] = probs

	return col_probs, row_probs, col_span_probs, row_span_probs_matrix

class Cell:
	def __init__(self, row:int, column:int, colspan:int, rowspan:int, text:str=None, is_header:bool = False,
				 is_masked:bool = False):
		self.row = row
		self.column = column
		self.colspan = colspan
		self.rowspan = rowspan
		self.text = text
		self.is_header = is_header
		self.is_masked = is_masked
		if text is None:
			text_len = random.randint(1, 5)
			self.text = ''.join(random.choices(string.ascii_letters + string.digits, k=text_len))

	def to_totto_dict(self):
		result = {
			"column_span": self.colspan,
			"is_header": self.is_header,
			"row_span": self.rowspan,
			"value": "" if self.is_masked else self.text
		}
		return result

	def row_range(self) -> List[int]:
		return list(range(self.row, self.row + self.rowspan))

	def column_range(self) -> List[int]:
		return list(range(self.column, self.column + self.colspan))

	def __str__(self):
		return f"{self.rowspan}:{self.colspan}"

class Table:
	def __init__(self, raw_list:List[List[Cell]], shape:Tuple[int,int]):
		self.raw_list = raw_list
		self.shape = shape
		self.highlighted_cells:Optional[List[Cell]] = None
		# Assign row and column names for a possible ssl3 cell naming
		self.row_idx_names = self.gen_alph_seq(shape[0]) if bool(random.getrandbits(1)) else [f"{i:02d}" for i in range(shape[0])]
		self.column_idx_names = self.gen_alph_seq(shape[1]) if bool(random.getrandbits(1)) else [f"{i:02d}" for i in range(shape[1])]

	def get_highlighted_cells(self) -> List[Cell]:
		if self.highlighted_cells is not None:
			return self.highlighted_cells
		h_row = random.randint(0, self.shape[0] - 1)
		h_col = random.randint(0, self.shape[1] - 1)
		# We need the origin coordinate of the highlighted cell for the image generation to highlight properly
		h_cell = self.get(h_row, h_col)
		self.highlighted_cells = [h_cell]
		return self.highlighted_cells

	def to_totto_example(self, ssl_target:str = "ssl1"):
		if ssl_target == "ssl3":
			self.fill_text_ssl3()
			self.mask_random_cells()
			self.highlighted_cells = []
		table = self.get_totto_table()
		return {
			"example_id": str(uuid.uuid4()),
			"table": table,
			"highlighted_cells": [self.raw_list_index(cell) for cell in self.get_highlighted_cells()],
			"sentence_annotations": [{"final_sentence":self.get_target_text(ssl_target)}]
		}

	def fill_text_ssl3(self):
		self.clean_all_text()
		for r_index, row in enumerate(self.raw_list):
			for c_index, cell in enumerate(row):
				if cell.text is None:
					cell.text = self.ssl3_cell_text(r_index, c_index, cell.rowspan, cell.colspan)

	def mask_random_cells(self):
		num_masked_cells = random.randint(2,5)
		for i in range(num_masked_cells):
			h_row = random.randint(0, self.shape[0] - 1)
			h_col = random.randint(0, self.shape[1] - 1)
			self.get(h_row, h_col).is_masked = True

	def ssl3_cell_text(self, row, column, rowspan, colspan):
		return f"{','.join(self.row_idx_names[row:row+rowspan])}-{','.join(self.column_idx_names[column:column+colspan])}"

	@staticmethod
	def gen_alph_seq(n, lowercase:bool=None):
		lowercase = bool(random.getrandbits(1)) if lowercase is None else lowercase
		base = 'a' if lowercase else 'A'
		if n <= 0:
			return []
		result = []
		for i in range(n):
			first_char = chr(ord(base) + (i // 26))
			second_char = chr(ord(base) + (i % 26))
			result.append(first_char + second_char)
		return result

	def clean_all_text(self):
		for r_index, row in enumerate(self.raw_list):
			for c_index, cell in enumerate(row):
				cell.text = None

	def raw_list_index(self, search_cell:Cell) -> List[int]:
		for r_index, row in enumerate(self.raw_list):
			for c_index, cell in enumerate(row):
				if cell == search_cell:
					return [r_index, c_index]

	def get_totto_table(self):
		return [[cell.to_totto_dict() for cell in row] for row in self.raw_list]

	def get_target_text(self, ssl_target:str = "ssl1"):
		match ssl_target:
			case 'ssl1':
				return self.ssl_01_target()
			case 'ssl3':
				return self.ssl_03_target()
	def ssl_03_target(self):
		masked_cells = []
		for r_index, row in enumerate(self.raw_list):
			for c_index, cell in enumerate(row):
				if cell.is_masked and cell not in masked_cells:
					masked_cells.append(cell)
		return f"<{'><'.join([cell.text for cell in masked_cells])}>"
	def ssl_01_target(self):
		"""
		All cells related to the highlighted cell. All the cells of the same column and row as the highlighted cell
		:return:
		"""
		result = "<"
		for h_cell in self.get_highlighted_cells():
			rows_str = self.build_values_str(self.row_values(h_cell.row_range(), self.column_range(), h_cell), "row")
			columns_str = self.build_values_str(self.column_values(self.row_range(), h_cell.column_range(), h_cell), "column")
			result += f"<<{h_cell.text}>{rows_str}{columns_str}>"
		result += ">"
		return result

	def row_range(self) -> List[int]:
		return list(range(self.shape[0]))

	def column_range(self) -> List[int]:
		return list(range(self.shape[1]))
	@staticmethod
	def build_values_str(values:List[List[str]], tag:str) -> str:
		result = "<"
		for sub_values in values:
			result+=f"<<{'><'.join(sub_values)}>>"
		result += ">"
		return result

	def row_values(self, rows:List[int], cols:List[int], skip:Cell):
		cells:List[List[Cell]] = []
		for i, row in enumerate(rows):
			for col in cols:
				cell = self.get(row, col)
				no_skip = not (row in skip.row_range() and col in skip.column_range())
				not_repeated = len(cells) == 0 or len(cells[-1]) == 0 or cells[-1][-1] != cell
				if no_skip and not_repeated:
					if i >= len(cells):
						cells.append([])
					cells[-1].append(cell)
		return [[c.text for c in sub_cells] for sub_cells in cells]

	def column_values(self, rows:List[int], cols:List[int], skip:Cell):
		cells:List[List[Cell]] = []
		for i, col in enumerate(cols):
			for row in rows:
				cell = self.get(row, col)
				no_skip = not (row in skip.row_range() and col in skip.column_range())
				not_repeated = len(cells) == 0 or len(cells[-1]) == 0 or cells[-1][-1] != cell
				if no_skip and not_repeated:
					if i >= len(cells):
						cells.append([])
					cells[-1].append(cell)
		return [[c.text for c in sub_cells] for sub_cells in cells]


	def to_html(self):
		"""
		Table linearization HTML
		:return:
		"""
		return get_table_html(self.get_totto_table(), [])

	def get(self, i_row:int, i_col:int) -> Optional[Cell]:
		rows, cols =  self.shape
		row_overload = [(0, None)] * cols
		row = 0
		while row < rows:
			col = 0
			col_offset = 0
			while col < cols:
				if row_overload[col][0] > 0:
					if row == i_row and col == i_col:
						return row_overload[col][1]
					row_overload[col] = (0, None) if row_overload[col][0] == 1 else (row_overload[col][0]-1, row_overload[col][1])
					col += 1
					col_offset += 1
				else:
					cell = self.raw_list[row][col-col_offset]
					if i_row == row  and i_col in range(col, col+cell.colspan, 1):
						return cell
					row_overload[col:col+cell.colspan] = itertools.repeat((cell.rowspan-1, cell), cell.colspan)
					col += cell.colspan
					col_offset += cell.colspan-1
			row += 1
		return None

def col_limit(cur_col, total_cols, row_overload):
	if sum(row_overload[cur_col+1:]) > 0:
		return next(i for i, overload in enumerate(row_overload[cur_col:]) if overload > 0)
	else:
		return total_cols - cur_col


def generate_tables(sizes:zip, col_span_probs:np.array, row_span_prob_matrix:np.array, total):
	all_colspan = [x for x in range(1, len(col_span_probs) + 1)]
	all_rowspan = [x for x in range(1, row_span_prob_matrix.shape[0] + 1)]
	gen_tables = []
	for cols, rows in tqdm(sizes, f"Generating tables...", total=total):
		table = [[]]
		row_overload = [0]*cols
		row = 0
		while row < rows:
			rows_left = rows - row
			col = 0
			while col < cols:
				if row_overload[col] > 0:
					row_overload[col] -= 1
					col += 1
				else:
					cols_left = col_limit(col, cols, row_overload)
					colspan = random.choices(population=all_colspan[:cols_left], k=1, weights=col_span_probs[:cols_left])[0]
					rowspan = random.choices(population=all_rowspan[:rows_left], k=1, weights=row_span_prob_matrix[:rows_left,colspan-1])[0]
					table[row].append(Cell(row, col, colspan, rowspan))
					row_overload[col:col+colspan] = itertools.repeat(rowspan-1, colspan)
					col += colspan
			row+=1
			if row < rows:
				table.append([])
		gen_tables.append(Table(table, shape=(rows, cols)))
	return gen_tables

def save_dataset(gen_tables:List[Table], dataset_dir:str, generations:dict, ssl_target:str):
	gen_dataset_dir = os.path.join(dataset_dir, f"warmup_{ssl_target}")
	os.makedirs(os.path.join(gen_dataset_dir), exist_ok=True)

	sizes = list(generations.values())
	split_gen_tables = [gen_tables[sum(sizes[:i]):sum(sizes[:i])+size] for i,size in enumerate(sizes)]
	for i, dataset in enumerate(generations.keys()):
		gen_dataset_path = os.path.join(gen_dataset_dir, f"{dataset}.jsonl")
		gen_tables = split_gen_tables[i]
		with open(gen_dataset_path, 'w') as outfile:
			for table in tqdm(gen_tables, f"Saving {dataset}..."):
				json.dump(table.to_totto_example(ssl_target), outfile)
				outfile.write('\n')


def run_generator(max_cols:int = 20, max_rows:int = 75, total:int=1):
	"""

	:param max_cols: 30 based on general distribution and the table size we want to produce
	:param max_rows: 125 based on general distribution and the table size we want to produce
	:param total: amount of tables to generate
	:return:
	"""
	size_dist_matrix, span_dist_matrix = compute_distributions("./data/ToTTo/")

	col_probs, row_probs, col_span_probs, row_span_prob_matrix = compute_probabilities(size_dist_matrix, span_dist_matrix, max_cols, max_rows)
	random.seed(117)
	gen_cols = random.choices(population=[x for x in range(1, max_cols+1)], k=total, weights=col_probs)
	gen_rows = random.choices(population=[x for x in range(1, max_rows+1)], k=total, weights=row_probs)
	return generate_tables(zip(gen_cols, gen_rows), col_span_probs, row_span_prob_matrix, total)


def analytics():
	tokenizer = AutoTokenizer.from_pretrained('t5-base')
	tables = run_generator(max_cols=20, max_rows=75, total=1000)
	ssl_01_tokens, ssl_02_tokens = [], []
	for table in tqdm(tables):
		ssl_01_tokens.append(len(tokenizer(table.ssl_01_target())[0]))
		ssl_02_tokens.append(len(tokenizer(table.to_html())[0]))
	print(f"SSL_01 - avg: {np.mean(ssl_01_tokens):.2f}, max: {max(ssl_01_tokens)}")
	print(f"SSL_02 - avg: {np.mean(ssl_02_tokens):.2f}, max: {max(ssl_02_tokens)}")
	plot_distribution(ssl_01_tokens)
	plot_distribution(ssl_02_tokens)


def run():
	dataset_dir = "./data/ToTTo/"
	generations = {"train": 120000, "dev": 7700, "test": 7700}
	#generations = {"dev": 50}
	gen_tables = run_generator(max_cols=20, max_rows= 75, total=sum(generations.values()))
	save_dataset(gen_tables, dataset_dir, generations, ssl_target="ssl3")

def update_old_format():
	dataset_dir = "./data/ToTTo/warmup_ssl1.old/"
	for mode in ["dev","test","train"]:
		dataset = load_dataset_raw(dataset_dir, mode, file_names=FILE_NAMES["warmup_ssl1"])

		out_dataset_path = os.path.join("./data/ToTTo/warmup_ssl1/", FILE_NAMES["warmup_ssl1"][mode])
		with open(out_dataset_path, 'w') as outfile:
			for example in tqdm(dataset):
				example["sentence_annotations"] = [{"final_sentence":example.pop("ssl_01")}]
				json.dump(example, outfile)
				outfile.write('\n')

if __name__ == '__main__':
	run()
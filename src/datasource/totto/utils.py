from typing import List, Dict

import os

import pandas as pd
import six
import json
from tqdm import tqdm

from datasource.totto.baseline_preprocessing.table_to_text_html_utils import get_table_html
from datasource.totto.baseline_preprocessing.preprocess_utils import linearize_full_table as baseline_linearization
from datasource.utils import build_table_from_df, table2logic_table_linearization, tokenize_column_names

import xml.etree.ElementTree as et

DATASET_EXAMPLES = {"train": 120761, "dev": 7700, "test": 7700}

FILE_NAMES = {
	"totto_data":{"train": "totto_train_data.jsonl", "dev": "totto_dev_data.jsonl",
				  "test": "unlabeled_totto_test_data.jsonl"},
	"totto_toy":{"train": "totto_train_data_toy.jsonl", "dev": "totto_dev_data_toy.jsonl",
				 "test": "unlabeled_totto_test_data.jsonl"},
	"warmup_ssl1":{"train": "train.jsonl", "dev": "dev.jsonl",
				 "test": "test.jsonl"},
	"warmup_ssl3":{"train": "train.jsonl", "dev": "dev.jsonl",
				 "test": "test.jsonl"},
	"l2t_totto_data": {"train": "train.jsonl", "dev": "dev.jsonl",
					"test": "test.jsonl"},
	"wg_totto_data": {"train": "train.jsonl", "dev": "dev.jsonl",
					"test": "test.jsonl"},
	"cont": {"train": "train.jsonl", "dev": "val.jsonl",
					"test": "test.jsonl"},
	"t5": {"train": "train.jsonl", "dev": "val.jsonl",
					"test": "test.jsonl"},
	"lattice": {"train": "train_linearized.jsonl", "dev": "dev_linearized.jsonl",
					"test": "test_linearized.jsonl"}}

NO_TABLE = [['No table']]
NO_COLUMNS = ['No columns']


def load_dataset_raw(dataset_dir, mode, indexed=False, allow_duplicates: bool = False, file_names=None):
	"""
	Loads the ToTTo dataset without any preprocessing
	:param dataset_dir:
	:param mode:
	:param indexed: if True it will return a dictionary with {"example_id": example}
	:param allow_duplicates: values are now List and can contain multiple examples associated to one id
	:param file_names: names of the dataset files, usually FILE_NAMES
	:return: list of json objects or dict {"example_id": example}
	"""
	if file_names is None:
		file_names = FILE_NAMES["totto_data"]
	dataset_path = os.path.join(dataset_dir, file_names[mode])
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		data = []
		data_dict = {}
		for line in tqdm(input_file, desc="Loading ToTTo ({})".format(mode)):
			line = six.ensure_text(line, "utf-8")
			example = json.loads(line)
			if indexed:
				if example['example_id'] in data_dict and allow_duplicates:
					data_dict[example['example_id']].append(example)
				else:
					data_dict[example['example_id']] = [example] if allow_duplicates else example
			else:
				data.append(example)
	return data_dict if indexed else data


def is_first_row_title(example):
	"""
	Some tables have a first row that covers all column span and is usually mistaken as a column header
	:param example:
	:return:
	"""
	table = example['table']
	max_cols = max([len(row) for row in table])
	is_row_max_col_span = len(table[0]) == 1 and 1 < max_cols == table[0][0]['column_span']
	# If any value in the second row is header
	has_row_1_header = len(table) > 1 and any([cell['is_header'] for cell in table[1]])
	# Alternatively if first row is max col span and table has > 3 rows with more than 2 columns its highly probable that it is a row title
	is_large_table = len(table) > 3 and max_cols > 2
	# TBD
	return is_row_max_col_span and (has_row_1_header or is_large_table)

def linearize_table_tripplets(table, tokenizer):
	"""
	0 : col , col : value [SEP] 0 : col , col : value
	:param table:
	:param tokenizer:
	:return:
	"""
	table_xml = baseline_linearization(table, [], None, None)
	root = et.fromstring(table_xml)
	for row in root:
		for cell in row:
			print(cell.tag, cell.attrib)
	None

def linearize_table(table, highlighted_cells, tokenizer, add_row_idx: bool = True):
	"""
	Given a data example form totto, returns a data_sample dict to build a Table2Logic Example
	:param table:
	:param tokenizer:
	:param highlighted_cells:
	:param add_row_idx:
	:return:
	"""
	pd_table = build_df_through_html(table, highlighted_cells)
	pd_table = pd_table.astype(str)
	# Simplify MultiIndex columns
	if isinstance(pd_table.columns, pd.MultiIndex):
		pd_table.columns = list(pd_table.columns.to_series().apply(lambda x: '/'.join([*x])).values.astype(str))
	table = build_table_from_df(pd_table, add_row_idx)
	str_table = table2logic_table_linearization(table, tokenizer, add_row_idx)
	str_columns = tokenize_column_names(list(pd_table.columns.values.astype(str)), tokenizer)
	return "<columns>" + str_columns + "</columns><table>" + str_table + "</table>"


def build_df_through_html(table, highlighted_cells) -> pd.DataFrame:
	"""
	More reliable but less faithful to the source table. Uses ToTTo's supplementary code implementation to: table -> html -> DataFrame
	:param example:
	:param highlighted_cells: mark highlighted cells as highlighted in the html
	:return:
	"""
	html_table = get_table_html(table, highlighted_cells)
	return pd.read_html(html_table)[0]


def get_highlighted_cells(example):
	"""
	Gets the value of the cells highlighted
	:param example:
	:return:
	"""
	values = []
	highlighted_cells = example['highlighted_cells']
	table = example['table']
	for h_cell in highlighted_cells:
		values.append(table[h_cell[0]][h_cell[1]]['value'])
	return values


def build_topic(example):
	"""
	Builds the text that will be represented as 'topic'
	:param example:
	:return:
	"""
	return example['table_page_title'] + " - " + example[
		'table_section_title']  # + " - " + example['table_section_text']


def preprocess_table(table, add_row_idx: bool = True, skip_first_row=False):
	"""
	Converts the table to a [['row 0', 'cell11', 'cell12'],['row 1', 'cell21', 'cell22'], etc] format
	:param table: table in the way it comes in the totto dataset
	:param add_row_idx: whether to include or not the index of the row at the beginning of each row
	:param skip_first_row: used for tables where first row is a title and the actual column is in the second row
	:return:
	"""
	res = []
	offset = 2 if skip_first_row else 1
	num_cols = max([len(row) for row in table])
	row_span_cache: List[List] = [None] * num_cols
	for ind, row in enumerate(table):
		next_row_span_cache: List[List] = row_span_cache.copy()
		res_row = ["row " + str(ind - offset)] if add_row_idx else []
		col_span_cache: List[List, int] = None
		col_span_offset = 0
		for cell_i in range(num_cols):
			if row_span_cache[cell_i] is not None:
				# row_span policy: duplicate value
				res_row.append(row_span_cache[cell_i][0]['value'])
				next_row_span_cache[cell_i][1] -= 1
				if next_row_span_cache[cell_i][1] == 0:
					next_row_span_cache[cell_i] = None
			elif col_span_cache is not None:
				# col_span policy: empty cell
				res_row.append('')
				col_span_cache[1] -= 1
				if col_span_cache[1] == 0:
					col_span_cache = None
			else:
				row_i = cell_i - sum(x is not None for x in row_span_cache[0:cell_i]) - col_span_offset
				cell = row[row_i]
				res_row.append(cell['value'])
				if cell['row_span'] > 1:
					next_row_span_cache[cell_i] = [cell, cell['row_span'] - 1]
				if cell['column_span'] > 1:
					col_span_cache = [cell, cell['column_span'] - 1]
					col_span_offset += cell['column_span'] - 1
		row_span_cache = next_row_span_cache
		res.append(res_row)
	return res[offset:]


def get_column_list(table, skip_first_row=False):
	"""
	Gets a list of columns inna ToTTo table
	:param table:
	:param skip_first_row
	:return:
	"""
	columns = []
	col_idx = 1 if skip_first_row else 0
	for cell in table[col_idx]:
		columns.append(cell['value'])
	return columns


def get_sent(sentence_annotations) -> str:
	for sentence_annotation in sentence_annotations:
		if "final_sentence" in sentence_annotation:
			return sentence_annotation['final_sentence']


def is_norm_table(example):
	table = example['table']
	# All cells in row 0 must be headers
	if not is_first_row_header(table, condition='any'):
		return False
	# All the rest of cells must not be headers
	for row in table[1:]:
		for cell in row:
			if cell['is_header']:
				return False
	# Let's see if there is any cell with a row span > 1
	for row in table[1:]:
		for cell in row:
			if cell['column_span'] > 1:
				return False

	return True


# Search tables with header outside row 0
def is_table_valid(example, strictness='medium') -> (bool, int):
	"""
	# Criteria (high)
	1. Columns must be unique and not ''
	2. Must have a header (first row must be the largest all column names must be  and all cells must be header)
	3. It must pass the matrix span check
	4. No 'column_span' > 1
	5. The rest of cells must not be headers
	# Criteria (medium)
	1. Columns must be unique and not ''
	2. First row must contain at least one cell marked as header
	3. First row must be the row with the most columns
	4. It must not have a cell that covers all columns (which its usually considered as a new header)
	5. It must pass the span check (which enables it to be converted to pandas DataFrame)
	:param example:
	:param strictness: 'high'|'medium'|'low'
	:return:
	"""
	table = example['table']
	columns = [col['value'] for col in table[0]]
	# 1. Columns must be unique and not ''
	if '' in columns or len(set(columns)) != len(columns):
		return False, 1
	is_first_row_header_condition = 'any' if strictness in ['medium', 'low'] else 'all'
	# 2. First row must contain at least one cell marked as header
	if not is_first_row_header(table, condition=is_first_row_header_condition):
		return False, 2
	# 3. First row must be the row with the most columns
	if len(columns) != max([len(row) for row in table]):
		return False, 3
	# 4. Shouldn't be a row that covers all columns
	for i, row in enumerate(table[1:]):
		for j, cell in enumerate(row):
			if strictness == 'high' and (cell['is_header'] or cell['column_span'] > 1):
				return False, 4
			if 1 < len(columns) == cell['column_span']:
				return False, 4
	# 5. It must pass the span check
	if not indepth_span_check(example):
		return False, 5
	return True, -1


def adapt_table(example):
	"""
	# Based on Criteria from is_table_valid
	0. If first row col_span = max -> remove
	1. Duplicated columns will be added a #1, #2 at the end ‘’ columns will be called ‘Column’
	2. if rows>1 -> second row = header else add col placeholder names
	3. Adapted on 2
	4. Remove row that covers all cols
	5. -
	6. Truncate, remove rows until < 512
	:param example:
	:param strictness: 'high'|'medium'|'low'
	:return:
	"""
	table = example['table']
	max_col_span = max([len(row) for row in table])
	# 0. If first row col_span = max -> remove
	if max_col_span != 1 and len(table[0]) == 1:
		table.remove(0)
	columns = [col['value'] for col in table[0]]
	# 1. Columns must be unique and not ''
	if '' in columns or len(set(columns)) != len(columns):
		return False
	is_first_row_header_condition = 'any'
	# 2. First row must contain at least one cell marked as header
	if not is_first_row_header(table, condition=is_first_row_header_condition):
		return False
	# 3. First row must be the row with the most columns
	if len(columns) != max([len(row) for row in table]):
		return False
	# 4. Shouldn't be a row that covers all columns
	for i, row in enumerate(table[1:]):
		for j, cell in enumerate(row):
			if 1 < len(columns) == cell['column_span']:
				return False
	# 5. It must pass the span check
	if not indepth_span_check(example):
		return False
	return True


def matrix_span_check(example):
	table = example['table']
	col_span = 0
	row_span = 0
	for row in table:
		for cell in row:
			col_span += cell['column_span']
			row_span += cell['row_span']
			if cell['column_span'] > 1:
				row_span += cell['column_span'] - 1
			if cell['row_span'] > 1:
				col_span += cell['row_span'] - 1
	max_cols = max([len(row) for row in table])
	result = col_span == row_span and col_span == len(table) * max_cols
	return result


def indepth_span_check(example):
	"""
	Checks all row and column spans are correct in the table
	:param example:
	:return:
	"""
	table = example['table']
	num_cols = max([len(row) for row in table])
	row_span_cache: List[int] = [0] * num_cols
	for ind, row in enumerate(table):
		next_row_span_cache: List[int] = row_span_cache.copy()
		col_span_cache = 0
		col_span_offset = 0
		for cell_i in range(num_cols):
			if row_span_cache[cell_i] > 0:
				next_row_span_cache[cell_i] -= 1
			elif col_span_cache > 0:
				col_span_cache -= 1
			else:
				row_i = cell_i - sum(x != 0 for x in row_span_cache[0:cell_i]) - col_span_offset
				if row_i < 0 or row_i >= len(row):
					return False
				cell = row[row_i]
				if cell['row_span'] > 1:
					next_row_span_cache[cell_i] = cell['row_span'] - 1
				if cell['column_span'] > 1:
					col_span_cache = cell['column_span'] - 1
					col_span_offset += cell['column_span'] - 1
		row_span_cache = next_row_span_cache
	return True


def is_first_row_header(table, condition='all'):
	for cell in table[0]:
		if not cell['is_header'] and condition == 'all':
			return False
		elif cell['is_header'] and condition == 'any':
			return True
	return condition == 'all'


def is_header_ok(example):
	"""
	1. Must have a header (first row must be the largest and must have at least one header value)
	:param example:
	:return:
	"""
	table = example['table']
	return is_first_row_header(table, condition='any') and len(table[0]) == max([len(row) for row in table])


def count_accepted():
	input_path = "data/ToTTo/totto_data/totto_train_data.jsonl"
	with open(input_path, "r", encoding="utf-8") as input_file:
		total = 0
		criteria_count = {"high": 0, "medium": 0}
		for line in tqdm(input_file):
			line = six.ensure_text(line, "utf-8")
			json_example = json.loads(line)
			for criteria, count in criteria_count.items():
				criteria_count[criteria] += 1 if is_table_valid(json_example, strictness=criteria)[0] else 0
			total += 1

	for criteria, count in criteria_count.items():
		print("{}: {}/{} ({:.2f}%)".format(criteria, count, total, (count / total) * 100))


def count_span_correct():
	input_path = "data/ToTTo/totto_data/totto_train_data.jsonl"
	with open(input_path, "r", encoding="utf-8") as input_file:
		total = 0
		criteria_count = {"matrix": 0, "indepth": 0}
		diff = []
		for line in tqdm(input_file):
			line = six.ensure_text(line, "utf-8")
			example = json.loads(line)
			result_matrix = matrix_span_check(example)
			result_indepth = indepth_span_check(example)
			criteria_count['matrix'] += 1 if result_matrix else 0
			criteria_count['indepth'] += 1 if result_indepth else 0
			if not result_matrix and result_indepth:
				diff.append(example)
			total += 1

	for criteria, count in criteria_count.items():
		print("{}: {}/{} ({:.2f}%)".format(criteria, count, total, (count / total) * 100))
	print("Diff: {}".format(len(diff)))


def next_example(input_file):
	line = input_file.readline()
	line = six.ensure_text(line, "utf-8")
	return json.loads(line)


def create_dataset_with_predictions_totto(dataset, output_dir, predictions):
	indexed_dataset = {}
	for example in dataset:
		indexed_dataset[example['example_id']] = example
	data_out = []
	for example_id, pred_tree in predictions:
		example = indexed_dataset[example_id]
		# Remove 'Row 0'
		table = [row[1:] for row in example['table']] if example['table'] != NO_TABLE else example['table']

		data = {"example_id": example_id,
				"sent": "No text. LF generated with T2L model.",
				"annotation": None,
				"logic": {},
				"logic_str": pred_tree.to_logic_str(add_dataset_formatting=True, resolve_substring=True),
				"interpret": "No interpret",
				"topic": example['topic'],
				"wiki": "",
				"url": "",
				"action": "",
				"num_func": 0,
				"nid": 0,
				"g_ids": None,
				"g_ids_features": None,
				"g_adj": None,
				"table_header": example['columns'],
				"table_cont": table}
		data_out.append(data)

	with open(os.path.join(output_dir, "out.json"), 'w', encoding='utf-8') as f:
		json.dump(data_out, f, ensure_ascii=False, indent=4)

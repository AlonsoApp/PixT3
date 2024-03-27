import os
from typing import List
from tqdm import tqdm
import json

from datasource.logic2text.logical_form.lf_grammar import V, C
from datasource.logic2text.logical_form.lf_parser import ASTTree

from datasource.logic2text.utils import FILE_NAMES, load_dataset
from datasource.totto.utils import FILE_NAMES as TOTTO_FILE_NAMES
from datasource.logic2text.preprocessing.fix_all import run as generate_fixed_dataset


def to_totto_dict(cell, is_header):
	result = {
		"column_span": 1,
		"is_header": is_header,
		"row_span": 1,
		"value": cell
	}
	return result

def table_to_totto(table):
	return [[to_totto_dict(cell, False) for cell in row] for row in table]

def get_highlighted_cells(table, h_values:List[str], h_columns:List[str]):
	h_cells = []
	for i_row, row in enumerate(table):
		for i_col, cell in enumerate(row):
			if not cell["is_header"] and cell["value"] in h_values:
				h_cells.append([i_row, i_col])
			elif cell["is_header"] and cell["value"] in h_columns:
				h_cells.append([i_row, i_col])
	return h_cells

def convert_to_totto(example):
	"""
	{"example_id": sample["example_id"], "topic": topic, "table": table, "columns": columns, "logic_str": logic_str,
	 "sent": sent, "pd_table": pd_table, "cased_values": cased_values}
	"""
	table = example["table"]
	# Remove the 'Row n' cells added after loading, we want to keep the tlt code intact
	table = [row[1:] for row in table]
	totto_table = table_to_totto(table)
	totto_headers = [to_totto_dict(column, True) for column in example["columns"]]
	totto_table.insert(0, totto_headers)
	cased_values = example["cased_values"]
	tree = ASTTree.from_logic_str(example["logic_str"])
	cols_in_lf = list(set(tree.get_columns()))

	return {
		"example_id": example["example_id"],
		"overlap_subset": True,
		"table_section_text": "",
		"table_page_title": example["topic"],
		"table_section_title": "",
		"table": totto_table,
		"highlighted_cells": get_highlighted_cells(totto_table, cased_values[V.TAB], cols_in_lf),
		"sentence_annotations": [{"final_sentence": example["sent"]}]
	}

def run():
	dataset_dir = "./data/Logic2Text/original_data_fix"
	out_dataset_dir = "./data/Logic2Text/totto_like_data"
	os.makedirs(out_dataset_dir, exist_ok=True)
	for mode in ["train", "dev", "test"]:
		in_dataset_path = os.path.join(dataset_dir, FILE_NAMES[mode])
		out_dataset_path = os.path.join(out_dataset_dir, TOTTO_FILE_NAMES["logic2text"][mode])
		if not os.path.isfile(in_dataset_path):
			generate_fixed_dataset()
		print("Loading Logic2Text fixed dataset...")
		l2t_dataset = load_dataset(dataset_dir, mode)
		with open(out_dataset_path, 'w') as outfile:
			for l2t_example in tqdm(l2t_dataset, f"Saving {mode}..."):
				json.dump(convert_to_totto(l2t_example), outfile)
				outfile.write('\n')


if __name__ == '__main__':
	run()
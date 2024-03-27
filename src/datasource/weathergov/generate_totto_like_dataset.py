from typing import List
import os
from collections import defaultdict
from hashlib import sha1
from tqdm import tqdm
import json
from datasource.totto.utils import FILE_NAMES as TOTTO_FILE_NAMES


def read_lines(path:str) -> List[str]:
	with open(path) as file:
		lines = [line.rstrip() for line in file]# if line.rstrip() != ""]
	return lines

def write_lines(path:str, lines:List[str]):
	with open(path, 'w') as f:
		for line in lines:
			f.write(f"{line}\n")

def split_original_data():
	main_dir = "./data/WeatherGov"
	out_dir = os.path.join(main_dir, "original_data")
	mapping_dir = os.path.join(main_dir, "split_mapping")
	events_path = os.path.join(main_dir, "all.events")
	#texts_path = os.path.join(main_dir, "all.text")
	events_str = read_lines(events_path)
	#texts_str = read_lines(texts_path)
	for mode in ["test", "train", "valid"]:
		mapping_path = os.path.join(mapping_dir, f"{mode}tgt_indices2orig.txt")
		mapping = read_lines(mapping_path)
		tgt_texts_str = read_lines(os.path.join(main_dir, f"{mode}.tgt"))
		mode_events = [events_str[int(i)-1] for i in mapping]#[event for i, event in enumerate(events_str) if str(i+1) in mapping]
		#mode_texts = [texts_str[int(i)-1] for i in mapping]
		write_lines(os.path.join(out_dir, f"{mode}.events"), mode_events)
		write_lines(os.path.join(out_dir, f"{mode}.text"), tgt_texts_str)

def table_to_totto(src_table:str):
	rows = []
	cur_row = defaultdict(lambda: "--")
	for cell in src_table.split(" "):
		c_type, rest = cell.split(".")
		col, value = rest.split(":")
		if col in cur_row or ("type" in cur_row and cur_row["type"] != c_type):
			rows.append(cur_row)
			cur_row = defaultdict(lambda: "--")
			cur_row["type"] = c_type
		if "type" not in cur_row:
			cur_row["type"] = c_type
		cur_row[col] = value
	rows.append(cur_row)
	# Extract columns
	columns = []
	for row in rows:
		columns.extend([col for col in row.keys() if col not in columns])
	totto_table = [[{"column_span": 1, "is_header": True, "row_span": 1, "value": column} for column in columns]]
	for row in rows:
		totto_row = []
		for column in columns:
			totto_row.append({"column_span": 1, "is_header": column == "type", "row_span": 1, "value": row[column]})
		totto_table.append(totto_row)
	return totto_table


def convert_to_totto(mode:str, index:int, src_table:str, tgt_text:str):
	totto_table = table_to_totto(src_table)

	return {
		"example_id": sha1(f"{mode}-{index}".encode('utf-8')).hexdigest(),
		"overlap_subset": True,
		"table_section_text": "",
		"table_page_title": "Weather Forecast Data",
		"table_section_title": "",
		"table": totto_table,
		"highlighted_cells": [],
		"sentence_annotations": [{"final_sentence": tgt_text}]
	}

def run():
	main_dir = "./data/WeatherGov"
	original_data_dir = os.path.join(main_dir, "original_data")
	out_dataset_dir = os.path.join(main_dir, "wg_totto_data")
	os.makedirs(out_dataset_dir, exist_ok=True)
	for mode in ["test", "train", "valid"]:
		out_dataset_path = os.path.join(out_dataset_dir, TOTTO_FILE_NAMES["wg_totto_data"]["dev" if mode == "valid" else mode])
		with open(out_dataset_path, 'w') as outfile:
			src_tables = read_lines(os.path.join(original_data_dir, f"{mode}.events"))
			tgt_texts = read_lines(os.path.join(original_data_dir, f"{mode}.text"))
			for i, (src_table, tgt_text) in tqdm(enumerate(zip(src_tables, tgt_texts)), f"Saving {mode}..."):
				if src_table == '':
					# We skip empty samples
					continue
				json.dump(convert_to_totto(mode, i, src_table, tgt_text), outfile)
				outfile.write('\n')



if __name__ == '__main__':
	#split_original_data()
	run()
"""
Replicate the preprocessing used by CoNT: Contrastive Neural Text Generation https://github.com/Shark-NLP/CoNT
With this we can generate a similar dataset to the one provided but instead of having just the highlighted cells, will
all the cells in the table. This way we can compare our model with CoNT (SOTA) in the Controlled Table-to-text and
Table-to-Text settings. We mimic the format of CoNT dataset to create the new one
"""
import json
import os
import sys
from typing import List

from tqdm import tqdm
import shutil

from transformers import PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding

from datasource.totto.baseline_preprocessing import preprocess_utils
from datasource.totto.utils import FILE_NAMES, load_dataset_raw
from model.t5.dataset_t5 import prepare_data


def convert_example(example, highlight_cells: bool = False, full_table: bool = True, append_targets: bool = True):
	"""
	Generates a CoNT dataset sample with a full table linearized using ToTTo repo preprocessing
	:param example:
	:param highlight_cells: True to include a highlighted marker on highlighted cells
	:param full_table: True to linearize the entire table or False for only the highlighted cells
	:param append_targets:
	:return:
	"""
	table = example["table"]
	table_page_title = example["table_page_title"]
	table_section_title = example["table_section_title"]
	highlighted_cells = example["highlighted_cells"] if highlight_cells or not full_table else []
	if full_table:
		source = preprocess_utils.linearize_full_table(table, highlighted_cells, table_page_title, table_section_title)
	else:
		subtable = (preprocess_utils.get_highlighted_subtable(table=table, cell_indices=highlighted_cells, with_heuristic_headers=True))
		source = preprocess_utils.linearize_subtable(subtable, table_page_title, table_section_title)
	if 'sentence_annotations' in example:
		if append_targets:
			target = " ".join([ann['final_sentence'] for ann in example['sentence_annotations']])
		else:
			target = example['sentence_annotations'][0]['final_sentence']
	else:
		target = "none"
	return {"example_id": str(example["example_id"]),
			"source": source,
			"target": target}

def generate_dataset(dataset_dir: str, mode: str, highlight_cells: bool = False, full_table: bool = True,
					 dataset_variant:str = "totto_data", append_targets: bool = True, model:str="cont",
					 tokenizer:PreTrainedTokenizerBase=None):
	original_data_dir = os.path.join(dataset_dir, dataset_variant)
	out_data_dir = os.path.join(dataset_dir, dataset_variant.replace('data', model))
	out_data_path = os.path.join(out_data_dir, FILE_NAMES[model][mode])
	examples = load_dataset_raw(original_data_dir, mode, indexed=False, file_names=FILE_NAMES[dataset_variant])
	out_examples = []
	for example in tqdm(examples):
		conv_example = convert_example(example, highlight_cells=highlight_cells, full_table=full_table, append_targets=append_targets)
		if tokenizer is not None:
			tok_example = prepare_data(conv_example, tokenizer)[0].data
			conv_example["example_id"] = example["example_id"]
			conv_example["input_ids"] = tok_example["input_ids"]
			conv_example["labels"] = tok_example["labels"]
		out_examples.append(conv_example)

	# Save  new dataset
	os.makedirs(os.path.join(out_data_dir), exist_ok=True)
	with open(out_data_path, 'w') as outfile:
		for entry in out_examples:
			json.dump(entry, outfile)
			outfile.write('\n')


def run(dataset_dir = "./data/ToTTo/", dataset_variant = "totto_data", tokenizer = None):
	copy_original_dev = False
	#dataset_dir = "./data/ToTTo/"
	#dataset_variant = "totto_data"
	#tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("t5-base", add_eos_token=True)
	for mode in ["train", "dev", "test"]:
		generate_dataset(dataset_dir, mode, dataset_variant=dataset_variant, full_table=True, highlight_cells=False,
						 append_targets=False, model="t5", tokenizer=tokenizer)
	if copy_original_dev:
		# Copy source dev because for some reason CoNT generation needs it
		src_dev_path = os.path.join(dataset_dir, dataset_variant, FILE_NAMES[dataset_variant]['dev'])
		dst_dev_path = os.path.join(dataset_dir, dataset_variant.replace('data','cont'), FILE_NAMES["totto_data"]['dev'])
		shutil.copyfile(src_dev_path, dst_dev_path)


if __name__ == '__main__':
	#run()
	flag = sys.argv[1]
	if flag == "totto":
		run(tokenizer=AutoTokenizer.from_pretrained("t5-base", add_eos_token=True))
	elif flag == "l2t":
		run(dataset_dir = "./data/Logic2Text/", dataset_variant = "original_data", tokenizer=None)
	else:
		print(f"Flag {flag} not supported.")
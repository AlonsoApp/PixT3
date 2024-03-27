import os
import json
from typing import List
import torch

from datasource.logic2text.logical_form.lf_parser import ASTTree
from datasource.utils import build_table
from datasource.logic2text.logical_form.lf_grammar import V

FILE_NAMES = {"train":"train.json", "dev":"valid.json", "test":"test.json"}

def load_dataset(dataset_dir, mode):
	dataset_path = os.path.join(dataset_dir, FILE_NAMES[mode])
	with open(dataset_path) as f:
		dataset = json.load(f)
	data = []
	for sample in dataset:
		logic_str = sample["logic_str"]
		topic = sample["topic"]
		table = preprocess_table(sample["table_cont"])
		columns = sample["table_header"]
		cased_values = ASTTree.from_logic_str(logic_str).get_cased_values()

		# we use this to build the pandas datatable later to execute the LFs
		pd_table = build_table_from_data_sample(sample)
		sent = sample["sent"]

		data_sample = {"example_id":sample["example_id"], "topic": topic, "table": table, "columns": columns, "logic_str": logic_str,
					   "sent": sent, "pd_table": pd_table, "cased_values": cased_values}
		data.append(data_sample)
	return data

def build_tree(data, cs_values_to_include:List[str]=None):
	"""
	Builds an ASTTree out of this dataset data sample.
	:param data:
	:param cs_values_to_include: ['AUX'] list of cs values that would be added to the model input (to assign its corresponding id_c)
	:return:
	"""
	table = build_table_from_data_sample(data)
	logic_str = data["logic_str"]

	columns = list(table.columns.values)
	ordinals = [str(x) for x in range(1, 20 + 1)]
	cs_values = {}
	if cs_values_to_include is not None:
		cased_values = ASTTree.from_logic_str(logic_str).get_cased_values()
		for case in cs_values_to_include:
			if case in cased_values:
				cs_values[case] = cased_values[case]
	all_table_vals = flatten(preprocess_table(data["table_cont"]))
	return ASTTree.from_logic_str(logic_str=logic_str, columns=columns, ordinals=ordinals, flat_table=all_table_vals, cs_values=cs_values)

def build_table_from_data_sample(data):
	"""
	Given an etire Logi2Text data sample build a pandas DataFrame of its table
	:param data: Logi2Text data sample
	:return:
	"""
	table_header = data["table_header"]
	table_cont = data["table_cont"]
	return build_table(table_header, table_cont)


def preprocess_table(table):
	"""
	Converts Logic2Text table to a [['row 0', 'cell11', 'cell12'],['row 1', 'cell21', 'cell22'], etc] format
	:param table: table in the way it comes in the Logic2Text dataset
	:return:
	"""
	res = []
	for ind, row in enumerate(table):
		res.append(["row " + str(ind)] + row)
	return res


def flatten(table):
	"""
	Flattens list of lists table into a 1D list of its values.
	:param table: table in list of lists format [['row 0', 'cell11', 'cell12'],['row 1', 'cell21', 'cell22'], etc]
	:return:
	"""
	return [item for sublist in table for item in sublist]

def create_dataset_with_predictions_l2t(dataset, output_dir, predictions):
	assert len(predictions) == len(dataset)
	for i, data in enumerate(dataset):
		# in l2t the sample parity is 1:1 so the example_id will be the one in this sample from the dataset
		data["sent"] = "No text. LF generated with T2L model."
		data["annotation"] = None
		data["logic"] = {}  # no time to implement tree.to_logic_dic and isn't used to train Logic2Text
		data["logic_str"] = predictions[i][1].to_logic_str(add_dataset_formatting=True, resolve_substring=True)
		data["interpret"] = "No interpret"

	with open(os.path.join(output_dir, "out.json"), 'w', encoding='utf-8') as f:
		json.dump(dataset, f, ensure_ascii=False, indent=4)


def save_model(model, model_save_path, model_name="best_model.pt"):
	torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
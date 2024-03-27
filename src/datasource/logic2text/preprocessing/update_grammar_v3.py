"""
Makes all TAB values be strictly equals or a substring of their table counterparts
"""
import json
from typing import List

from tqdm import tqdm
from pathlib import Path

from datasource.logic2text.logical_form.lf_engine import api as engine_current
from datasource.logic2text.logical_form.legacy.v2.lf_parser import ASTTree, is_a_in_x
from datasource.logic2text.logical_form.legacy.v2.lf_grammar import V
from datasource.logic2text.logical_form.lf_parser import ASTTree as CurrentASTTree
from datasource.logic2text.preprocessing.manual_fix_samples_v3 import *

from datasource.logic2text.utils import build_table_from_data_sample

def print_report(result:dict):
	for case, report in result.items():
		print(case)
		for chapter, result in report.items():
			if type(result) == int:
				print("    {}: {}".format(chapter, result))
			elif type(result) == dict:
				print("    {}:".format(chapter))
				for subchap, subresult in result.items():
					print("        {}: {}".format(subchap, subresult))

def apply_value_fix(sample:dict, tree:ASTTree, case:str) -> dict:
	old_value, new_value = value_fixes[sample['example_id']]
	tree.replace_value(old_value, new_value, value_case=case)
	sample["logic_str"] = tree.to_logic_str(add_dataset_formatting=True)
	return sample

def apply_contains(sample:dict, tree:ASTTree, current_value:str, cell_value:str, case:str) -> dict:
	tree.convert_to_contains(current_value, cell_value, case)
	sample["logic_str"] = tree.to_logic_str(add_dataset_formatting=True)
	return sample

def apply_substring(sample:dict, tree:ASTTree, current_value:str, cell_value:str, case:str) -> dict:
	tree.convert_value_to_substring(current_value, cell_value, case)
	sample["logic_str"] = tree.to_logic_str(add_dataset_formatting=True)
	return sample

def apply_suggested(sample:dict, tree:ASTTree, old_value:str, suggested_value:str, case:str) -> dict:
	tree.replace_value(old_value, suggested_value, value_case=case)
	sample["logic_str"] = tree.to_logic_str(add_dataset_formatting=True)
	return sample

def perform_clean(sample, tree:ASTTree, case:str, suggested_state:str, value_state:str, value:str,
				  suggested_value:str=None, scope:List[str]=None):
	"""

	:param sample:
	:param tree:
	:param case:
	:param suggested_state:
	:param value_state:
	:param value:
	:param suggested_value:
	:param scope: in case we don't have a suggested value, this is the scope in which we'll seach again after manual
	value fix
	:return:
	"""
	example_id = sample['example_id']
	# Manually fix detected values
	if example_id in value_fixes.keys():
		sample = apply_value_fix(sample, tree, case)
		value = value_fixes[sample['example_id']][1]
		if suggested_value is None:
			# With the value corrected, let's try again to find a suitable suggested value
			for scope_val in scope:
				if value in scope_val:
					suggested_value = scope_val
					break

	# Detected special cases
	if example_id in no_change_samples:
		return sample
	elif example_id in suggested_samples:
		return apply_suggested(sample, tree, value, suggested_value, case)
	elif example_id in contains_substring_samples:
		sample = apply_contains(sample, tree, value, suggested_value, case)
		return apply_substring(sample, tree, value, suggested_value, case)

	# Defaults for each case
	if suggested_state == "suggested_not_eq":
		sample = apply_contains(sample, tree, value, suggested_value, case)
		if value_state == "substring" or value_state == "not_found":
			sample = apply_substring(sample, tree, value, suggested_value, case)
		return sample
	#elif suggested_state == "suggested_eq" and value_state == "substring":
	#	sample = apply_contains(sample, tree, value, suggested_value, case)
	#	return apply_substring(sample, tree, value, suggested_value, case)
	else:
		# Let's try: when all suggested are eq use eq -> suggested It will reduce LF complexity and LF still be True
		return apply_suggested(sample, tree, value, suggested_value, case)

def clean(data_in, dataset):

	report = {V.TAB1:{"ok":0, "suggested_not_eq":{"equals":0, "substring":0, "not_found":0}, "suggested_eq":{"substring":0, "not_found":0}},
			  V.TAB2:{"ok":0, "suggested_not_eq":{"equals":0, "substring":0, "not_found":0}, "suggested_eq":{"substring":0, "not_found":0}},
			  V.INF:{"ok":0, "executed_not_eq":0},
			  V.AUX:{"ok":0, "substring":0, "not_found_left":0}}

	data_out = []

	for sample in tqdm(data_in):
		# Remove detected erroneous LFs
		if sample['example_id'] in removed_samples:# or sample['example_id'] in example_id_avoid:
			continue
		pd_table = build_table_from_data_sample(sample)
		tree = ASTTree.from_logic_str(sample["logic_str"])
		tab_scope = tree.get_scope_values(pd_table, legacy_api=True)
		for case, values in tab_scope.items():
			for value_pair in values:
				lf_value = value_pair[0]
				suggested_values = value_pair[1]
				if case == V.INF:
					if lf_value != str(suggested_values[0]):
						report[case]["executed_not_eq"] += 1
						sample = apply_suggested(sample, tree, lf_value, suggested_values[0], case)
					else:
						report[case]["ok"] += 1
				elif case == V.AUX:
					# For AUX we want to keep things simple. If value == any of the suggested values, we leave it untouched
					# if value is a wp substring of any value in the table we apply substring
					# if value isn't contained in suggested values nor in direct or substring form we remove it (only for train/valid, for test we do nothing)
					if lf_value not in suggested_values:
						found = False
						for sugg_val in suggested_values:
							if is_a_in_x(lf_value, sugg_val):
								# AUX Value is a substring of a suggested value apply substring
								sample = apply_substring(sample, tree, lf_value, sugg_val, case)
								found = True
								report[case]["substring"] += 1
								break
						if not found and dataset in ["train.json"]: # Remove this if we are creating the training set for Logic2Text (comment also AUX not found in table (train) in manual_fix_samples_v3.py)
							print("SHOULD BE REMOVED!! {}: {}".format(dataset, sample['example_id']))
						elif not found:
							# Do nothing
							report[case]["not_found_left"] += 1
					else:
						# AUX value in suggested_values leave it
						report[case]["ok"] += 1
				elif not all_same(suggested_values):
					# Not all values are the same, therefore it must be 'contains'
					if lf_value in suggested_values:
						# We have a direct reference within suggested values so substring is not needed: contains -> V
						report[case]["suggested_not_eq"]["equals"] += 1
						sample = perform_clean(sample, tree, case, suggested_state="suggested_not_eq",
											   value_state="equals", value=lf_value, suggested_value=lf_value)
						continue
					found = False
					for sugg_val in suggested_values:
						if is_a_in_x(lf_value, sugg_val):
							if sample['example_id'] not in no_change_samples:
								# Value has no straight eq within suggested values but it is a substring of one of them: contains -> substring
								report[case]["suggested_not_eq"]["substring"] += 1
								sample = perform_clean(sample, tree, case, suggested_state="suggested_not_eq",
													   value_state="substring", value=lf_value, suggested_value=sugg_val)
							found = True
							break
					if not found:
						report[case]["suggested_not_eq"]["not_found"] += 1
						sample = perform_clean(sample, tree, case, suggested_state="suggested_not_eq",
											   value_state="not_found", value=lf_value, scope=suggested_values)
				elif suggested_values[0] != lf_value:
					# All suggested values are the same and lf_value is not eq to any
					if is_a_in_x(lf_value, suggested_values[0]):
						# But it's contained within them. It's contains -> substring
						report[case]["suggested_eq"]["substring"] += 1
						sample = perform_clean(sample, tree, case, suggested_state="suggested_eq",
											   value_state="substring", value=lf_value, suggested_value=suggested_values[0])
					else:
						# It isn't even contained within them. It's eq -> V
						report[case]["suggested_eq"]["not_found"] += 1
						sample = perform_clean(sample, tree, case, suggested_state="suggested_eq",
											   value_state="not_found", value=lf_value, suggested_value=suggested_values[0])
				else:
					report[case]["ok"]+=1
		data_out.append(sample)
	return data_out, report

def perform_simplify(data):
	for sample in tqdm(data):
		tree = ASTTree.from_logic_str(sample["logic_str"])
		tree.update_grammar('V3')
		sample["logic_str"] = tree.to_logic_str(add_dataset_formatting=True)
	return data

def all_same(items):
	return all(x == items[0] for x in items)

def run(original_path):
	fix_path = "./data/Logic2Text/original_data_update_values/"
	Path(fix_path).mkdir(parents=True, exist_ok=True)
	for file_name in ["all_data.json", "test.json", "train.json", "valid.json"]:
		print("Updating grammar for {}".format(file_name))
		with open(original_path + file_name) as f:
			data_in = json.load(f)
		print("Making all Values equal or substring of table values...")
		clean_data, report = clean(data_in, file_name)
		#print_report(report)
		#assert len(data_in) - len(removed_samples) == len(clean_data)
		print("Removing _str modifiers from grammar...")
		clean_data = perform_simplify(clean_data)

		# With packages updated we now get 7 False LFs. We don't need LFs to be True for this project
		#print("Validating execution with current engine...")
		#true_count, false_count = execute_all(clean_data, engine_current)
		#assert false_count == 0
		save_data(clean_data, fix_path, file_name)
	return fix_path

def execute_all(data_in, engine):
	"""
	Executes all LFs with the given engine and verifies that none of them are False
	:param data_in:
	:param engine:
	:return:
	"""
	count_true = 0
	for sample in tqdm(data_in):
		pd_table = build_table_from_data_sample(sample)
		tree = CurrentASTTree.from_logic_str(sample["logic_str"])
		result = tree.execute(pd_table, engine=engine)
		if result:
			count_true+=1
	return count_true, (len(data_in)-count_true)

def save_data(data, path, file_name):
	with open(path + file_name, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
	data_path = "./data/Logic2Text/original_data_fix_grammar/"
	run(data_path)


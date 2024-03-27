import json
from tqdm import tqdm
from pathlib import Path

from datasource.logic2text.logical_form.legacy.v2.lf_parser import add_sep_spaces

'''
Change the logic_str of the dataset for the str built out of the "logic" in json format provided with every sample. We 
do this to solve the ambiguity introduced by not using actions such as filter_eq instead of filter_str_eq. Originally, 
logic_str featured only filter_eq but considering that "" and "filter_str_eq" have different functionalities we need to 
reflect this disambiguation in the logic_str that we will feed to the model. So we can make the model produce 
unambiguous LFs that can be correctly executed
'''

def fix_dataset(original_path, fix_path, file_name):
	'''
	execute all logic forms
	'''

	with open(original_path + file_name) as f:
		data_in = json.load(f)

	num_all = 0
	num_correct = 0

	for data in tqdm(data_in):
		num_all += 1
		# print("Processing: {}".format(num_all))
		logic_str = data["logic_str"]
		logic_json = data["logic"]

		fixed_str = logic_to_str(logic_json)
		fixed_str = add_sep_spaces(fixed_str)
		fixed_str = fixed_str + "= true"
		data["logic_str"] = fixed_str

	with open(fix_path + file_name, 'w', encoding='utf-8') as f:
		json.dump(data_in, f, ensure_ascii=False, indent=4)

	return num_all, num_correct

def logic_to_str(logic):
	if type(logic) is not dict:
		return logic

	str_args = [logic_to_str(x) for x in logic['args']]
	return "{func}{{{args}}}".format(func=logic["func"], args=";".join(str_args))

def run(original_path):
	fix_path = "./data/Logic2Text/original_data_fix_ambiguous/"
	Path(fix_path).mkdir(parents=True, exist_ok=True)
	for file_name in ["all_data.json", "test.json", "train.json", "valid.json"]:
		fix_dataset(original_path, fix_path, file_name)
	return fix_path


if __name__ == '__main__':
	run("./data/Logic2Text/original_data_sha1/")

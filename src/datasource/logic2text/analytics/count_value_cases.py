import json
from tqdm import tqdm

from datasource.logic2text.logical_form.legacy.v2.lf_grammar import V
from datasource.logic2text.logical_form.legacy.v2.lf_parser import ASTTree

def execute_all(json_in):
	'''
	execute all logic forms
	'''

	with open(json_in) as f:
		data_in = json.load(f)

	num_actions = []

	report = {"values":{V.TAB1:0, V.TAB2:0, V.INF:0, V.AUX:0}, "lf_with":{V.TAB1:0, V.TAB2:0, V.INF:0, V.AUX:0, "TAB":0}}

	for data in tqdm(data_in):
		# print("Processing: {}".format(num_all))
		logic_str = data["logic_str"]

		cased_values = ASTTree.from_logic_str(logic_str).get_cased_values()
		for case, values in cased_values.items():
			report["values"][case]+=len(values)
			report["lf_with"][case]+= 1 if len(values)>0 else 0
		if len(cased_values[V.TAB1]) > 0 or len(cased_values[V.TAB2]) > 0:
			report["lf_with"]["TAB"] += 1

	total_values = sum(report["values"].values())
	print("Values:")
	for case, value in report["values"].items():
		print("    {}: {} ({:.2f}%)".format(case, value, (value/total_values)*100))
	print("LFs with at least one:")
	for case, value in report["lf_with"].items():
		print("    {}: {} ({:.2f}%)".format(case, value, (value/len(data_in))*100))


if __name__ == '__main__':

	data_path = "./data/Logic2Text/original_data_fix_grammar/"

	for file_name in ["all_data.json"]:#, "test.json", "train.json", "valid.json"]:
		execute_all(data_path + file_name)

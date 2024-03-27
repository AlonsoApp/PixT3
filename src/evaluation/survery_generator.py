import os
import random
import json
import string
from typing import Dict, List
import re

from statsmodels.stats import inter_rater as irr

import math
import pandas as pd
import numpy as np

from datasource.totto.utils import load_dataset_raw, FILE_NAMES
from evaluation.utils import DATASET_PATHS, generate_reference_inference_files, generate_dataset_inference_files
"""
EVAL_EXPERIMENTS = {
	"no_highlighted":["reference","cont_no_highlighted","f3__20230905_114832_no_highlighted_039_bs", "i3__20230905_134649_no_highlighted_039_bs"],
	"highlighted":["reference","cont_highlighted","f2__20230809_172951_highlighted_039_bs", "i1__20230905_134725_highlighted_039_bs"],
	"notab_high":["reference","cont_notab_high","f4__20230918_201309_notab_high_00_bs"],
	"highlighted_test":["reference","cont_highlighted","f2__20230809_172951_highlighted_039_bs", "i1__20230905_134725_highlighted_039_bs"]
}
"""
EVAL_EXPERIMENTS = {
	"no_highlighted":["reference","lat_base_no_high"],
	"highlighted":["reference","lat_base_high_full"],
	"notab_high":["reference","lat_base_high_only"]
}

# List of IDs used as examples. Removed from evaluation samples to avoid bias
ID_BLACK_LIST = []

def form_plan(num_experiments:int, examples_to_eval:int, evaluators:int, duplicates:int):
	num_unique_forms = int(evaluators / duplicates)
	questions_per_form = math.ceil(examples_to_eval * num_experiments / num_unique_forms)
	return num_unique_forms, questions_per_form

def img_size(example) -> int:
	# WATCH OUT!!! This is the only function that must be tweaked when changing between Logic2Text and ToTTo and dev and test
	#img_path = os.path.join("data/ToTTo/img/raw_tables/no_highlighted/dev",f"{example['example_id']}.png")
	img_path = os.path.join("data/Logic2Text/img/raw_tables/no_highlighted/test",f"{example['example_id']}.png")
	return os.stat(img_path).st_size

def generate_evaluation_file(dataset, evaluation:str, evaluators:int, examples_to_eval:int, duplicates:int,
							 ref_example_ids:List[str] = None, skip_ref:bool=True, form_name_offset:int = 0, mode:str = "test",
							 mayor_evaluation:str="totto"):
	all_examples = {}
	for experiment in EVAL_EXPERIMENTS[evaluation]:
		experiment_dir_name = experiment if mode == "dev" else experiment+f"_{mode}"
		path = f"./out/inferences/{mayor_evaluation}/{experiment_dir_name}/hashed_inferred_texts.csv"
		if not os.path.isfile(path):
			generate_dataset_inference_files([experiment], mayor_evaluation, mode=mode)
		df = pd.read_csv(path)
		all_examples[experiment] = {}
		for index, row in df.iterrows():
			# example_id,prediction,reference
			example_id = row['example_id']
			prediction = row['prediction']
			title = dataset[example_id]["table_page_title"]
			section = dataset[example_id]['table_section_title']
			if example_id not in ID_BLACK_LIST:
				all_examples[experiment][example_id] = {"experiment":experiment,"example_id":str(example_id), "sent":prediction, "title":title, "section":section}
	example_ids = list(list(all_examples.values())[0].keys())

	# Filter ids we (don't) want to include
	if ref_example_ids and len(ref_example_ids) > 0:
		example_ids = [x for x in example_ids if x not in ref_example_ids] if skip_ref else ref_example_ids
	random.seed(117)
	random.shuffle(example_ids)
	selected_ids = example_ids[:examples_to_eval]
	selected_examples = []
	for selected_id in selected_ids:
		example_group = []
		for experiment, examples in all_examples.items():
			example_group.append(examples[selected_id])
		# we do this to break the model order within the same group
		random.shuffle(example_group)
		selected_examples.extend(example_group)

	num_unique_forms, questions_per_form = form_plan(len(EVAL_EXPERIMENTS[evaluation]), examples_to_eval, evaluators, duplicates)
	forms = []
	for _ in range(num_unique_forms):
		forms.append(selected_examples[:min(len(selected_examples), questions_per_form)])
		del selected_examples[:min(len(selected_examples), questions_per_form)]

	# Sort examples based on table size
	for form in forms:
		form.sort(key=img_size)

	# Group forms in dict
	forms_dict = {}
	for i, form in enumerate(forms):
		forms_dict[string.ascii_lowercase[i+form_name_offset]] = form
	return forms_dict

def get_eval_example_ids(eval_form_dir:str, eval_type:str) -> List[str]:
	example_ids = []
	eval_form_path = os.path.join(eval_form_dir, f"form_{eval_type}.json")
	with open(eval_form_path, "r", encoding="utf-8") as f:
		forms = json.load(f)
	for form_name, form in forms.items():
		for question in form:
			example_ids.append(str(question["example_id"]))
	return list(set(example_ids))

def generate_evaluation_files(mayor_evaluation:str="totto", mode="test", ref_eval_form_dirs:List[str]=None, skip_ref:bool=True):
	out_forms_dir = f"out/evaluation/human/{mayor_evaluation}/forms"
	ref_eval_form_dirs = [] if ref_eval_form_dirs is None else ref_eval_form_dirs
	file_names = FILE_NAMES["l2t_totto_data"] if mayor_evaluation == "l2t" else None
	dev_dataset = load_dataset_raw(DATASET_PATHS[mayor_evaluation], mode, indexed=True, file_names=file_names)
	for eval_type in ["no_highlighted", "highlighted", "notab_high"]:
		ref_example_ids = []
		for ref_eval_form_dir in ref_eval_form_dirs:
			ref_example_ids.extend(get_eval_example_ids(ref_eval_form_dir, eval_type))
		eval_forms = generate_evaluation_file(dev_dataset, eval_type, evaluators=15, examples_to_eval=100, duplicates=3, ref_example_ids=ref_example_ids, skip_ref=skip_ref, form_name_offset=10, mayor_evaluation=mayor_evaluation, mode=mode)
		with open(os.path.join(out_forms_dir, f"{mayor_evaluation}_form_{eval_type}.json"), 'w') as f:
			json.dump(eval_forms, f)

def extract_answers(result_df):
	"""
	:param result_df:
	:return: [{"sent":"Sentence x.", "evaluations":[True, False, True]}, {"sent"...]
	"""
	cols_to_drop = 5
	answers:List[Dict] = []
	for i, (series_name, series) in enumerate(result_df.items()):
		if i < cols_to_drop:
			continue
		sent = series_name.replace("𝗗𝗲𝘀𝗰𝗿𝗶𝗽𝘁𝗶𝗼𝗻: ", "")
		# When sents are duplicates, Google Forms adds .1, .2, etc to avoid id collision
		sent = re.sub(r"\.\d$", '', sent)
		answers.append({"sent": sent, "evaluations":series.tolist()})
	return answers

def agg(row):
	# True, False
	return row.value_counts()

def load_forms(mayor_evaluation:str="totto", evaluation="highlighted", evaluations:int = 2):
	forms_dir = f"out/evaluation/human/{mayor_evaluation}/forms"
	forms = {}
	for i in range(evaluations):
		with open(os.path.join(forms_dir, f"eval_0{i + 1}", f"form_{evaluation}.json")) as f:
			forms = forms | json.load(f)
	return forms

def generate_significance_file(mayor_evaluation:str="totto", evaluation="highlighted_test", evaluations:int = 2):
	# item-id         human-id         model-id    score (e.g., yes/no in your case)
	columns = ["item-id", "human-id", "model-id", "score"]
	results_dir = os.path.join("out/evaluation/human/",mayor_evaluation,"results", evaluation)
	forms = load_forms(mayor_evaluation, evaluation, evaluations)
	rows = []
	num_evaluators = None
	for form_name, form in forms.items():
		result_df = pd.read_csv(os.path.join(results_dir, f"{form_name}.csv"))
		answers = extract_answers(result_df)
		assert len(answers) == len(form)
		for question, answer in zip(form, answers):
			# The amount of evaluators must remain constant across documents
			assert question["sent"] == answer["sent"]
			evals = answer["evaluations"]
			if num_evaluators is None:
				num_evaluators = len(evals)
			else:
				# The amount of evaluators must remain constant across documents
				assert num_evaluators == len(evals)
			for evaluator in range(num_evaluators):
				row = [question["example_id"], result_df["Please enter your Prolific ID"][evaluator], question["experiment"], "yes" if evals[evaluator] else "no"]
				rows.append(row)
	df = pd.DataFrame(rows, columns=columns)
	df.to_csv(os.path.join("out/evaluation/human/results", f"{evaluation}_sig.csv"), index=False)

def print_faithfulness_vote(df):
	faithfulness_df = df.groupby(["experiment", "final_eval"]).size().reset_index(name='counts')
	experiments = faithfulness_df["experiment"].unique().tolist()
	for experiment in experiments:
		exp_df = faithfulness_df[faithfulness_df["experiment"] == experiment]
		faithfulness = exp_df[exp_df["final_eval"]].iloc[0]['counts'] / sum(exp_df['counts'])
		print(f"{experiment}: {faithfulness:.2f}")

def print_faithfulness_count(df, num_evaluators):
	models = df["experiment"].unique()
	for model in models:
		count_trues = 0
		for evaluator in range(num_evaluators):
			count_trues += df[df['experiment'] == model][f"eval_{evaluator+1}"].sum()
		faithfulness = int((count_trues / (len(df[df['experiment'] == model]) * num_evaluators))*100 + 0.5)

		print(f"{model}: {faithfulness:.2f}")


def analyze_results(mayor_evaluation="totto", evaluation="highlighted_test", evaluations:int = 2, count_func:str="vote", num_evaluators=3):
	results_dir = os.path.join(f"out/evaluation/human/{mayor_evaluation}/results", evaluation)
	forms = load_forms(mayor_evaluation, evaluation, evaluations)
	rows = []
	for form_name, form in forms.items():
		#if form_name not in ["a", "b", "c", "d", "e"]:
		#	continue
		result_df = pd.read_csv(os.path.join(results_dir, f"{form_name}.csv"))
		answers = extract_answers(result_df)
		assert len(answers) == len(form)
		for question, answer in zip(form, answers):
			# The amount of evaluators must remain constant across documents
			assert question["sent"] == answer["sent"]
			evals = answer["evaluations"]
			row = [question["example_id"], question["experiment"]] + evals
			if count_func == "vote":
				row.append(max(set(evals), key=evals.count))
			rows.append(row)
	eval_columns = [f"eval_{n+1}" for n in range(num_evaluators)]
	result_columns = ["example_id", "experiment"] + eval_columns
	if count_func == "vote":
		result_columns.append("final_eval")
	df = pd.DataFrame(rows, columns=result_columns)
	# Now with the results df we calculate results and metrics ["example_id", "experiment", "eval_1", "eval_2", "eval_3", "final_eval"]
	match count_func:
		case "vote":
			print_faithfulness_vote(df)
		case "count":
			print_faithfulness_count(df, num_evaluators)


	# Agreement
	eval_df = df[eval_columns]
	tmp = eval_df.apply(agg, axis=1)
	tmp = tmp.fillna(0).astype(int)
	fleiss_kappa = irr.fleiss_kappa(tmp, method='fleiss')
	print(f"Fleiss' kappa: {fleiss_kappa:.2f}")
	# .values.tolist()
	return df

def extract_evaluator_vector(df:pd.DataFrame):
	"""
	For a answer document df returns
	{"evaluator_id": [1, 0, 0, 1, ...], "evaluator_id": [0, 1, ...]}
	:param df:
	:return:
	"""
	eval_vecs = {}
	for index, row in df.iterrows():
		evaluator = row[2]
		ans_vector = np.array(row.iloc[5:].to_list(), dtype=bool).astype(int)
		eval_vecs[evaluator] = ans_vector
	return eval_vecs

def print_similarity_matrix(answers:dict):
	mat = []
	for _, a in answers.items():
		eval_sim = []
		for _, b in answers.items():
			similarity = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
			eval_sim.append(similarity)
		mat.append(eval_sim)
	print('\n'.join(['\t'.join([f"{cell:.2f}" for cell in row]) for row in mat]))

def print_true_ref(result_df, form):
	ref_idx = []
	for i, question in enumerate(form):
		if "reference" in question["experiment"]:
			ref_idx.append(i)
	eval_ratio = {}
	for index in ref_idx:
		for _, row in result_df.iterrows():
			evaluator = row[2]
			if evaluator not in eval_ratio:
				eval_ratio[evaluator] = 1 if row[5+index] else 0
			else:
				eval_ratio[evaluator] += 1 if row[5+index] else 0
	for evaluator, total in eval_ratio.items():
		print(f"{evaluator}: {int((total/len(ref_idx))*100)}%")

def find_divergent_evaluators(mayor_evaluation="totto", evaluation="highlighted", print_ids:bool=False, evaluations:int=1):
	results_dir = os.path.join(f"out/evaluation/human/{mayor_evaluation}/results", evaluation)
	directory = os.fsencode(results_dir)
	#for file in os.listdir(directory):
		#filename = os.fsdecode(file)
		#if not filename.endswith(".csv"):
			#continue
	forms = load_forms(mayor_evaluation, evaluation, evaluations)
	rows = []
	num_evaluators = None
	for form_name, form in forms.items():
		print(f"--------- {form_name} ---------")
		result_df = pd.read_csv(os.path.join(results_dir, f"{form_name}.csv"))
		answers = extract_evaluator_vector(result_df)
		if print_ids:
			print_true_ref(result_df, form)
		print_similarity_matrix(answers)

if __name__ == '__main__':
	#generate_cont_inference_files(["cont_highlighted", "cont_no_highlighted", "cont_notab_high"])
	#generate_ref_inference_files()
	#generate_evaluation_files(mayor_evaluation="l2t", mode="test", skip_examples_dir="out/evaluation/human/l2t/forms/eval_01")
	#generate_evaluation_files(mayor_evaluation="l2t", mode="test", ref_eval_form_dirs=["out/evaluation/human/l2t/forms/eval_01", "out/evaluation/human/l2t/forms/eval_02"], skip_ref=False)
	#generate_evaluation_files(skip_examples_dir="out/evaluation/human/totto/forms/eval_01")
	#analyze_results("totto", "highlighted", count_func="vote", evaluations=3)
	#analyze_results("l2t", "notab_high", count_func="vote", evaluations=3)
	for evalu in ["notab_high"]:#["no_highlighted", "highlighted", "notab_high"]:
		generate_significance_file(mayor_evaluation="totto",evaluation=evalu,evaluations=3)
	#find_divergent_evaluators(mayor_evaluation="l2t", evaluation="notab_high", print_ids=True, evaluations=2)
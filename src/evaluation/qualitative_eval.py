import sys
import os
import pandas as pd

from datasource.totto.utils import load_dataset_raw


def generate_evaluation_file():
	args = sys.argv[1:]
	infer_dir = args[0]
	infer_path = os.path.join(infer_dir, "hashed_inferred_texts.csv")
	exp_name = os.path.basename(os.path.normpath(infer_dir))
	eval_path = os.path.join("./out/evaluation/qualitative/", exp_name)
	overlap_path = os.path.join(eval_path, "overlap.txt")
	no_overlap_path = os.path.join(eval_path, "no_overlap.txt")
	df_infer = pd.read_csv(infer_path)
	totto_dataset = load_dataset_raw(infer_dir, "dev", indexed=True)

	os.makedirs(eval_path, exist_ok=True)
	with open(overlap_path, 'w') as overlap_file:
		with open(no_overlap_path, 'w') as no_overlap_file:
			for index, row in df_infer.iterrows():
				example_id, prediction, reference = row['example_id'], row['prediction'], row['reference']
				file = overlap_file if totto_dataset[example_id]['overlap_subset'] else no_overlap_file
				file.write(str(example_id)+"\n")
				file.write(str(prediction)+"\n")
				file.write(str(reference)+"\n")
				file.write("\n")

def print_stats(exp_type: str = "highlighted"):
	if exp_type == "highlighted":
		df = pd.read_csv("out/evaluation/qualitative/exp__20230606_123420_highlighted/qualitative_highlighted_no_overlap.csv")
	elif exp_type == "no_highlighted":
		df = pd.read_csv("out/evaluation/qualitative/exp__20230606_155215_no_highlighted/qualitative_no_highlighted_no_overlap.csv")
	elif exp_type == "masked_inverted":
		df = pd.read_csv("out/evaluation/qualitative/exp__20230606_160100_masked_inverted/qualitative_masked_inverted_no_overlap.csv")
	else:
		df = pd.read_csv("out/evaluation/qualitative/i8__20230918_120129_no_highlighted_039_totto/qualitative_no_highlighted_no_overlap_i8.csv")
	print(rf"Semantic: {df['semantic'].value_counts()}")
	print(rf"Faithfulness errors: {df['faithfulness_error'].value_counts()}")
	if exp_type == "highlighted":
		print(rf"Highlighted not mentioned (avg): {df.loc[df['faithfulness_error'] != 'co'].loc[:, 'highlighted_not_mentioned'].mean():.2f}")
		print(rf"Not-Highlighted mentioned (avg): {df.loc[df['faithfulness_error'] != 'co'].loc[:, 'non_highlighted_mentioned'].mean():.2f}")
	elif exp_type == "no_highlighted":
		print(rf"Cells mentioned (avg): {df.loc[df['faithfulness_error'] != 'co'].loc[:, 'cells_mentioned'].mean():.2f}")
		print(rf"Cell loc row: {df['cell_loc_row'].value_counts()}")
		print(rf"Cell loc col: {df['cell_loc_col'].value_counts()}")
	elif exp_type == "masked_inverted":
		print(df['values_not_mentioned'].value_counts(normalize=True))
		#print(rf"Cells mentioned (avg): {df.loc[df['values_not_mentioned'] != 'co'].loc[:, 'cells_mentioned'].mean():.2f}")
	return None


if __name__ == '__main__':
	print_stats("i8")
	#generate_evaluation_file()
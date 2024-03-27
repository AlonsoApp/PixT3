import os

import math
from tqdm import tqdm
from typing import List

from datasource.totto.analytics.histogram import build_log_bins
from datasource.totto.preprocessing.image_generation import Highlights
from os import listdir
from os.path import isfile, join
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from operator import itemgetter
import json

from datasource.totto.utils import load_dataset_raw
from evaluation.utils import generate_dataset_inference_files, DATASET_PATHS


def get_sizes(force_run: bool = False, mode:str= 'dev', highlight_mode:str = Highlights.HIGHLIGHTED):
	img_dir = os.path.join("data/ToTTo/img/raw_tables", Highlights.folders[highlight_mode], mode)
	csv_dir = os.path.join('./data/ToTTo/', 'cache', 'img_sizes')
	csv_path = os.path.join(csv_dir, rf"{highlight_mode}_{mode}.csv")
	# Load form cache
	if not force_run and os.path.isfile(csv_path):
		return pd.read_csv(csv_path)

	sizes = {'example_id': [], 'width': [], 'height': [], 'total_resolution': [], 'file_size': []}
	filenames:List[str] = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
	for file_name in tqdm(filenames):
		img_path = join(img_dir, file_name)
		im = Image.open(img_path)
		width, height = im.size
		example_id = file_name.replace('.png','')
		sizes['example_id'].append(example_id)
		sizes['width'].append(width)
		sizes['height'].append(height)
		sizes['total_resolution'].append(width*height)
		sizes['file_size'].append(os.stat(img_path).st_size)

	# Save csv
	os.makedirs(csv_dir, exist_ok=True)
	df = pd.DataFrame.from_dict(sizes)
	df.to_csv(csv_path, index=False, encoding='utf-8')
	return df

def plot_bin_size_dist(sort_img_size_by:str='total_resolution', highlight_mode:str=Highlights.NO_HIGHLIGHTED, num_bins:int=20):
	df = get_sizes(highlight_mode=highlight_mode)
	bins = build_log_bins(min(df[sort_img_size_by]), max(df[sort_img_size_by]), num_bins)
	plot_size_dist(df, bins, sort_img_size_by)

def plot_size_dist(df:pd.DataFrame, bins:List[float], sort_img_size_by:str = 'total_resolution'):
	# sns.displot(size_df.loc[size_df['total'] < 1e6], x="total", bins=20)
	fg = sns.displot(df, x=sort_img_size_by, stat='percent', bins=bins)
	plt.xscale('log')
	plt.xlabel("Image size in pixels", fontsize=14)
	plt.ylabel("Percent", fontsize=14)
	for ax in fg.axes.ravel():

		# add annotations
		for c in ax.containers:
			# custom label calculates percent and add an empty string so 0 value bars don't have a number
			labels = [rf'{i+1}: {w:0.1f}%' if (w := v.get_height()) > 0 else '' for i,v in enumerate(c)]

			ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=90, padding=2)

		ax.margins(y=0.2)
	plt.show()

def save(dataset_subset, df_inferences, output_dir, mode, subset:int):
	with open(os.path.join(output_dir, rf"inferred_texts_{subset+1:02d}.txt"), 'w') as texts_file:
		with open(os.path.join(output_dir, rf"totto_{mode}_{subset+1:02d}.jsonl"), 'w') as dataset_file:
			for example in dataset_subset:
				prediction = df_inferences[df_inferences['example_id'] == example['example_id']]['prediction'].iloc[0]
				# Write text line
				texts_file.write(f"{prediction}\n")
				# Write json
				json.dump(example, dataset_file)
				dataset_file.write('\n')

def split_inferences(inference_folder:str, num_bins:int, mode:str= 'dev', highlight_mode:str = Highlights.HIGHLIGHTED,
					 sort_img_size_by:str = "total_resolution", dataset_name:str="totto"):
	"""
	Splits inference files into num_bins splits based on the size of the table images
	:param inference_folder: folder name where inferred_texts.txt file is stored "f2__20230809_172951_highlighted_039_bs"
	:param num_bins: 20 (we use 20 in the paper)
	:param mode: train | dev
	:param highlight_mode:
	:param sort_img_size_by: "total_resolution" | "file_size"
	:param dataset_name: "totto" | "l2t"
	:return:
	"""
	dataset_dir = DATASET_PATHS[dataset_name]
	inference_dir = f"./out/inferences/{dataset_name}/{inference_folder}"
	inference_path = os.path.join(inference_dir, "hashed_inferred_texts.csv")
	output_dir = os.path.join(inference_dir, "table_size_split")
	os.makedirs(output_dir, exist_ok=True)
	if not os.path.isfile(inference_path):
		generate_dataset_inference_files([inference_folder], "totto", mode=mode)

	df_size = get_sizes(highlight_mode=highlight_mode, mode=mode)
	bins = build_log_bins(min(df_size[sort_img_size_by]), max(df_size[sort_img_size_by]), num_bins)
	#bins = build_log_bins(400428, 900551, num_bins)
	#plot_size_dist(df_size, bins)
	#return
	hashed_dataset = load_dataset_raw(dataset_dir, mode, indexed=True)
	df_inferences = pd.read_csv(inference_path)
	# Get example_ids per bin
	for i in tqdm(range(num_bins)):
		lower_bin = bins[i]
		upper_bin = bins[i+1]
		sel_example_ids = list(df_size.loc[df_size[sort_img_size_by].between(lower_bin, upper_bin, inclusive="right")]['example_id'])
		sel_example_ids = list(map(int, sel_example_ids))
		sel_dataset = list(itemgetter(*sel_example_ids)(hashed_dataset))
		sel_df_inferences = df_inferences[df_inferences['example_id'].isin(sel_example_ids)]
		save(sel_dataset, sel_df_inferences, output_dir, mode, i)

def old_add_scale_line(scale, min_val):
	colors = {1:'#3BAEA3',2:"#F7D55C",10:"#EC563B",40:"#DD3F3F"}
	x_val = math.log(16 * 16 * 2048*scale) - math.log(min_val)
	plt.axvline(x_val, 0, 1.0, color=colors[scale])
	# scale = (math.e**(6+math.log(min_val)))/(16 * 16 * 2048) # calc scale factor at chunk 6
	plt.text(x_val, 67, rf'x{scale}', rotation=45, color="black")

def find_bounds(bins, size_to_search):
	for i in tqdm(range(len(bins))):
		lower_bound = bins[i]
		upper_bound = bins[i + 1]
		if lower_bound < size_to_search < upper_bound:
			return i+1, lower_bound, upper_bound
	return 1, bins[0], bins[-1]

def add_scale_line(scale, bins):
	colors = {1: '#3BAEA3', 2: "#F7D55C", 10: "#EC563B", 40: "#DD3F3F"}
	size_to_search = (16 * 16 * 2048 * scale)
	chunk, lower_bound, upper_bound = find_bounds(bins, size_to_search)
	#(size_to_search-lower_bound) / (upper_bound-lower_bound)
	x_val = chunk + (math.log(size_to_search) - math.log(lower_bound)) / (math.log(upper_bound) - math.log(lower_bound))
	#x_val = math.log(16 * 16 * 2048*scale) - math.log(min_val)
	plt.axvline(x_val, 0, 1.0, color=colors[scale])
	# scale = (math.e**(5+math.log(min_val)))/(16 * 16 * 2048) # calc scale factor at chunk 6
	plt.text(x_val, 67, rf'x{scale}', rotation=45, color="black")

def add_scale_line_at_chunk(chunk, bins):
	lower = bins[chunk-1]
	upper = bins[chunk]
	lower_scale = (16 * 16 * 2048)/lower
	upper_scale = (16 * 16 * 2048)/upper
	plt.axvline(chunk, 0, 1.0, color="grey")
	label_height = 67 if len(bins) == 10 else 70
	plt.text(chunk, label_height, rf'{lower_scale:.2f} - {upper_scale:.2f}', rotation=25, color="black")

def plot_performance_v_size(chunks:int, mode:str= 'dev', highlight_mode:str = Highlights.HIGHLIGHTED,
							sort_img_size_by:str = "total_resolution"):
	evaluation_path = f"./out/evaluation/qualitative/exp__20230606_123420_highlighted/split_{chunks}.csv"
	df = pd.read_csv(evaluation_path)
	# descaling starts at 16*16*2048
	# x2
	df_size = get_sizes(highlight_mode=highlight_mode, mode=mode)
	#bins = build_log_bins(min(df_size['total']), max(df_size['total']), num_bins)


	df.set_index('chunk', inplace=True)
	sns.lineplot(data=df)
	#for scale in [1, 2, 10]:#, 40]:
	#	bins = build_log_bins(min(df_size['total']), max(df_size['total']), df.index.max())
	#	add_scale_line(scale, bins)
	scale_bar_chunks = [6, 7, 8] if chunks == 10 else [11, 13, 15]
	for chunk in scale_bar_chunks:
		bins = build_log_bins(min(df_size[sort_img_size_by]), max(df_size[sort_img_size_by]), df.index.max())
		add_scale_line_at_chunk(chunk, bins)
	plt.show()

EVAL_TO_MODEL = {
	"l5__20231201_000000": "T5",
	"cont_highlighted": "CoNT",
	"i1__20230905_134725_highlighted_039_bs": "PixT3",
}

def old_plot_multi_performance_v_size(evaluations:List[str], chunks:int, metric:str="bleu"):
	"""

	:param evaluations:
	:param chunks:
	:param metric: bleu | parent
	:return:
	"""
	df = None
	for evaluation in evaluations:
		evaluation_path = f"./out/evaluation/qualitative/{evaluation}/split_{chunks}.csv"
		cur_df = pd.read_csv(evaluation_path)
		cur_df.set_index('chunk', inplace=True)
		for col in cur_df.columns:
			if f"{metric}_overall" == col:
			#if metric in col:
				new_col = col.replace(metric, EVAL_TO_MODEL[evaluation])
				cur_df = cur_df.rename(columns={col: new_col})
			else:
				cur_df = cur_df.drop(col, axis=1)
		if df is None:
			df = cur_df
		else:
			df = pd.concat([df, cur_df], axis=1)

	sns.lineplot(data=df)
	plt.show()

def plot_multi_performance_v_size(evaluations:List[str], chunks:int, metric:str="bleu", chunk_range:List[int] = None):
	"""

	:param evaluations:
	:param chunks:
	:param metric: bleu | parent
	:param chunk_range:
	:return:
	"""
	df = pd.read_csv("./out/evaluation/qualitative/multi_eval_split_20.csv")
	#sns.set(font_scale=1)
	if chunk_range:
		df = df[df["chunk"].isin(chunk_range)]

	a = sns.lineplot(
		data=df,
		x="chunk", y=metric, hue="model", style="model",
		markers=True, dashes=False
	)

	plt.xlabel("Table size group", fontsize=14)
	plt.ylabel("PARENT", fontsize=14)
	plt.legend(title='', fontsize=11)
	a.tick_params(labelsize=12)
	plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
	plt.show()

if __name__ == '__main__':
	#split_inferences("lat_base_high_full", 20,
	#				 sort_img_size_by='total_resolution', highlight_mode=Highlights.NO_HIGHLIGHTED)
	#plot_performance_v_size(20, sort_img_size_by='total_resolution')
	plot_multi_performance_v_size(list(EVAL_TO_MODEL.keys()), 20, metric= "parent", chunk_range=list(range(1, 19)))
	#plot_bin_size_dist()

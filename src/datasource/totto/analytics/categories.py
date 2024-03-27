from typing import List, Dict

from datasource.totto.utils import load_dataset_raw
import os
from tqdm import tqdm
import pickle
import random

def add_example(category_clusters: dict, example: dict):
	for category in example['categories']:
		if category not in category_clusters:
			category_clusters[category] = []
		category_clusters[category].append(example['example_id'])

def gen_category_cluster(dataset_dir: str, mode: str, category_clusters: dict, dataset_variant: str = "totto_cat"):
	dataset_path = os.path.join(dataset_dir, dataset_variant)
	dataset = load_dataset_raw(dataset_path, mode, indexed=True)
	for _, example in tqdm(dataset.items()):
		add_example(category_clusters, example)

def load_category_cluster(dataset_dir: str, force_run: bool = False) -> Dict[int, List]:
	cache_path = os.path.join(dataset_dir, 'cache', 'category_clusters.pkl')
	category_clusters: Dict[int, List] = {}
	if not force_run and os.path.isfile(cache_path):
		with open(cache_path, 'rb') as cache_file:
			return pickle.load(cache_file)
	else:
		for mode in ["dev", "train"]:
			gen_category_cluster(dataset_dir, mode, category_clusters)
	# Save cache
	os.makedirs(os.path.join(dataset_dir, 'cache'), exist_ok=True)
	with open(cache_path, 'wb') as f:
		pickle.dump(category_clusters, f)
	return category_clusters

def load_dataset_full(dataset_dir: str, dataset_variant: str = "totto_cat"):
	dataset_full = {}
	dataset_path = os.path.join(dataset_dir, dataset_variant)
	for mode in ["dev", "train"]:
		dataset_full = dataset_full | load_dataset_raw(dataset_path, mode, indexed=True)
	return dataset_full

def run(force_run: bool = False):
	dataset_dir = "./data/ToTTo/"
	report_dir = "./out/clustering/"
	report_path = os.path.join(report_dir, "report.txt")
	category_clusters = load_category_cluster(dataset_dir, force_run=force_run)
	# Sort by freq
	category_clusters = {k: v for k, v in sorted(category_clusters.items(), key=lambda item: -len(item[1]))}

	print(rf"Unique categories: {len(category_clusters)}")
	unique_ids = set()
	for value in category_clusters.values():
		unique_ids |= set(value)
	print(rf"Unique ids: {len(unique_ids)}")

	dataset_full = load_dataset_full(dataset_dir)
	# Generating report
	random.seed(42)
	os.makedirs(os.path.join(report_dir), exist_ok=True)
	with open(report_path, 'w') as outfile:
		categories_in_report = 0
		for category, example_ids in category_clusters.items():
			outfile.write(rf"{category} ({len(example_ids)}):")
			outfile.write('\n')
			for example_id in random.sample(example_ids, 10):
				outfile.write('\t')
				outfile.write(rf"- {dataset_full[example_id]['sentence_annotations'][0]['final_sentence']}")
				outfile.write('\n')
			outfile.write('\n\n')
			categories_in_report += 1
			if categories_in_report == 50:
				break


if __name__ == '__main__':
	run()
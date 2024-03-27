import os.path
import json
import wikipedia
import multiprocessing
from datasource.totto.utils import load_dataset_raw, FILE_NAMES
from tqdm import tqdm

def get_categories(example: dict) -> dict:
	try:
		categories = wikipedia.page(example['table_page_title'], auto_suggest=False).categories
		example['categories'] = categories
		return example
	except:
		return example

def debug(example_id: int):
	mode = "dev"
	dataset_dir = "./data/ToTTo/"
	dataset_path = os.path.join(dataset_dir, "totto_data")
	dataset = load_dataset_raw(dataset_path, mode, indexed=True)
	example = dataset[example_id]
	categories = wikipedia.page(example['table_page_title'], auto_suggest=False).categories
	example['categories'] = categories

def create_category_dataset(dataset_dir: str, mode: str, dataset_variant: str = "totto_data", force_run: bool = False):
	dataset_path = os.path.join(dataset_dir, dataset_variant)
	output_dir = os.path.join(dataset_dir, "totto_cat")
	output_path = os.path.join(output_dir, FILE_NAMES[dataset_variant][mode])
	if not force_run and os.path.isfile(output_path):
		return
	dataset = load_dataset_raw(dataset_path, mode, indexed=True)
	results = {}
	failed_ids = []
	examples = dataset.values()
	num_threads = multiprocessing.cpu_count() - 2
	with multiprocessing.Pool(num_threads) as pool:
		for output in tqdm(pool.imap_unordered(get_categories, examples, chunksize=32), total=len(examples),
						 desc=f"Finding categories"):
			if 'categories' not in output:
				# Set default categories but log the failure
				output['categories'] = []
				# print(rf"FAIL: {output['example_id']}")
				failed_ids.append(output['example_id'])
			results[output['example_id']] = output
	"""
	for key, value in dataset.items():
		output = get_categories(key)
		results[output['example_id']] = output
		break
	"""
	print("Failures:")
	print(rf"Total: {len(failed_ids)}")
	print(rf"List: {failed_ids}")
	# Sort
	print("Sorting result")
	index_map = {v: i for i, v in enumerate(dataset.keys())}
	sorted(results.items(), key=lambda pair: index_map[pair[0]])

	# Save  new dataset
	print("Saving dataset")
	os.makedirs(os.path.join(output_dir), exist_ok=True)
	with open(output_path, 'w') as outfile:
		for entry in results.values():
			json.dump(entry, outfile)
			outfile.write('\n')

def run():
	dataset_dir = "./data/ToTTo/"
	for mode in ["dev", "train", "test"]:
		create_category_dataset(dataset_dir, mode)


if __name__ == '__main__':
	run()
	#debug(6258794881679192426)
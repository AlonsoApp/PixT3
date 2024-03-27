import os

from datasource.totto.utils import FILE_NAMES, DATASET_EXAMPLES, load_dataset_raw
import json
import six
from tqdm import tqdm

def align_dataset(dataset_dir: str, mode: str, dataset_variant: str = "totto_data"):
	"""
	Gets a list of the lengths in tokens for each table of the 'mode' set linearized in 'linearization' way
	:param dataset_dir: path to the main directory to the dataset (usually 'ToTTo')
	:param mode: which dataset ['train','dev','test']
	:param dataset_variant:
	:return:
	"""
	dataset_path = os.path.join(dataset_dir, dataset_variant, FILE_NAMES[dataset_variant][mode])
	img_dir = os.path.join(dataset_dir, 'img', mode)
	output_dir = os.path.join(dataset_dir, "totto_align")
	output_path = os.path.join(output_dir, FILE_NAMES[dataset_variant][mode])
	removed = []
	results = []
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		for line in tqdm(input_file, total=DATASET_EXAMPLES[mode]):
			line = six.ensure_text(line, "utf-8")
			example = json.loads(line)
			img_path = os.path.join(img_dir, str(example['example_id']) + '.png')
			if os.path.isfile(img_path):
				results.append(example)
			else:
				removed.append(example['example_id'])
	print("Removed: {}".format(removed))
	print(rf"Removed total: {len(removed)}")
	return
	# Save  new dataset
	print("Saving dataset")
	os.makedirs(os.path.join(output_dir), exist_ok=True)
	with open(output_path, 'w') as outfile:
		for entry in results:
			json.dump(entry, outfile)
			outfile.write('\n')

def run_aligment():
	for mode in ["dev", "test", "train"]:
		align_dataset("./data/ToTTo/", mode)  # get_patch_lengths("./data/ToTTo/", mode)

def run_duplicates():
	for mode in ["train", "dev", "test"]:
		dataset, duplicates = get_duplicates("./data/ToTTo/", mode=mode)

def get_duplicates(dataset_dir, mode, dataset_variant: str = "totto_data"):
	dataset_dir = os.path.join(dataset_dir, dataset_variant)
	dataset = load_dataset_raw(dataset_dir, mode, indexed=True, allow_duplicates=True)
	duplicate_ids = []
	for example_id, examples in dataset.items():
		if len(examples) > 1:
			duplicate_ids.append(example_id)
			print(rf"DUPLICATED: {len(examples)}")
	return dataset, duplicate_ids

if __name__ == '__main__':
	"""
	NO ALIGNMENT IS NEEDED. THIS MODULE IS USELESS
	Duplicate ids have the same table so there's no problem on having just one table for the same id. There are a total 
	of 108 duplicated ids. Many with 2 samples but some with 3 a total of 212 examples with duplicated ids.
	
	The difference between renderized tables and amount of examples in train (-108) corresponds uniquely to these 
	duplicates. There's no need to clean the dataset for aligment
	"""
	run_aligment()
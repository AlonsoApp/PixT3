import json
import hashlib
from tqdm import tqdm
from pathlib import Path

def test_unique():
	dataset_path = "./data/Logic2Text/original_data/all_data.json"
	with open(dataset_path) as f:
		data_in = json.load(f)
	hashed_dataset = {}
	for sample in data_in:
		sha1 = hashlib.sha1(bytes(sample["url"] + sample["action"], "utf-8")).hexdigest()
		if sha1 not in hashed_dataset:
			hashed_dataset[sha1] = []
		hashed_dataset[sha1].append(sample)
	for key in hashed_dataset.keys():
		if len(hashed_dataset[key]) > 1:
			print(":)")

def build_hashed_dataset(original_path, file_name):
	with open(original_path + file_name) as f:
		data_in = json.load(f)
	hashed_dataset = {}
	for data in tqdm(data_in):
		sha1 = hashlib.sha1(bytes(data["url"] + data["action"], "utf-8")).hexdigest()
		hashed_dataset[sha1] = data
	return hashed_dataset

def test_no_collision():
	original_path = "./data/Logic2Text/original_data/"
	all_data = build_hashed_dataset(original_path, "all_data.json")
	test = build_hashed_dataset(original_path, "test.json")
	train = build_hashed_dataset(original_path, "train.json")
	valid = build_hashed_dataset(original_path, "valid.json")
	original_len = len(all_data)
	for key in test.keys():
		all_data.pop(key)
	assert len(all_data) == original_len - len(test)
	for key in train.keys():
		all_data.pop(key)
	assert len(all_data) == original_len - len(test) - len(train)
	for key in valid.keys():
		all_data.pop(key)
	assert len(all_data) == 0


def modify_dataset(original_path, fix_path, file_name):
	'''
	execute all logic forms
	'''

	with open(original_path + file_name) as f:
		data_in = json.load(f)

	for data in tqdm(data_in):
		data["example_id"] = hashlib.sha1(bytes(data["url"] + data["action"], "utf-8")).hexdigest()

	with open(fix_path + file_name, 'w', encoding='utf-8') as f:
		json.dump(data_in, f, ensure_ascii=False, indent=4)

def run(original_path):
	fix_path = "./data/Logic2Text/original_data_sha1/"
	Path(fix_path).mkdir(parents=True, exist_ok=True)
	for file_name in ["all_data.json", "test.json", "train.json", "valid.json"]:
		modify_dataset(original_path, fix_path, file_name)
	return fix_path


if __name__ == '__main__':
	run("./data/Logic2Text/original_data/")

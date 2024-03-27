import json
from tqdm import tqdm
import os
import sys

"""
Generates the .text files corresponding to the .json dataset files in the dataset_path path
"""

def generate_text_file(dataset_path, file_name):

	with open(os.path.join(dataset_path, file_name)) as f:
		data_in = json.load(f)

	num_all = 0
	num_correct = 0

	texts = []

	for data in tqdm(data_in):
		num_all += 1
		# print("Processing: {}".format(num_all))
		texts.append(data["sent"])

	text_file_name = file_name.replace(".json", ".text")
	with open(os.path.join(dataset_path, text_file_name), 'w') as f:
		for text in texts:
			f.write("%s\n" % text)

	return num_all, num_correct

def run(dataset_path):
	json_files = [pos_json for pos_json in os.listdir(dataset_path) if pos_json.endswith('.json')]
	if 'args.json' in json_files:
		json_files.remove('args.json')
	for file_name in json_files:
		generate_text_file(dataset_path, file_name)
	return dataset_path


if __name__ == '__main__':
	run(sys.argv[1])

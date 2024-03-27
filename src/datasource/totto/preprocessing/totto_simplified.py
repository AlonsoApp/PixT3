import os
from datasource.totto.utils import FILE_NAMES, is_table_valid
import six
import json
from collections import defaultdict

def create_totto_simplified(mode: str, dataset_dir: str = "./data/ToTTo/"):
	original_data_path = os.path.join(dataset_dir, 'totto_data', FILE_NAMES["totto_data"][mode])
	simplified_data_dir = os.path.join(dataset_dir, 'totto_simplified')
	simplified_data_path = os.path.join(simplified_data_dir, FILE_NAMES["totto_data"][mode])
	failed, processed = 0, 0
	report = defaultdict(lambda: 0)
	simplified_examples = []
	with open(original_data_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			try:
				if processed % 100 == 0:
					print("Num examples processed: %d" % processed)

				line = six.ensure_text(line, "utf-8")
				json_example = json.loads(line)
				is_valid, reject_reason = is_table_valid(json_example)
				if is_valid:
					simplified_examples.append(json_example)
				else:
					report[reject_reason] += 1
			except:
				failed += 1
			processed += 1
	print("Report ----------------------------")
	remaining = processed
	percentage = 100.0
	for key in sorted(report):
		remaining -= report[key]
		new_percentage = round((remaining/processed)*100, 1)
		print("{}: {}/{} {}% ({}%)".format(key, remaining, processed, new_percentage, round(new_percentage-percentage, 1)))
		percentage = new_percentage
	print("Failed: {}/{}".format(failed, processed))
	# Save  new dataset
	os.makedirs(os.path.join(simplified_data_dir), exist_ok=True)
	with open(simplified_data_path, 'w') as outfile:
		for entry in simplified_examples:
			json.dump(entry, outfile)
			outfile.write('\n')

def run():
	for mode in list(FILE_NAMES["totto_data"].keys()):
		create_totto_simplified(mode)


if __name__ == '__main__':
	run()

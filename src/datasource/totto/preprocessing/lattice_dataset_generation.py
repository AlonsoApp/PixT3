"""
Adapted code from Lattice preprocessing: https://github.com/luka-group/Lattice/tree/main
With this we can generate a similar dataset to the one provided but instead of having just the highlighted cells, will
all the cells in the table. This way we can compare our model with Lattice in the Controlled Table-to-text and
Table-to-Text settings. We mimic the format of Lattice dataset to create the new one
"""
import os

from datasource.totto.utils import FILE_NAMES
import dataset.totto.preprocessing.lattice.preprocess_data as preprocess_data
import dataset.totto.preprocessing.lattice.json_to_csv as json_to_csv


def run():
	#dataset_dir = "./data/ToTTo/"
	dataset_dir = "./data/Logic2Text/"
	settings = [{"highlight_cells": True, "full_table": True, "name":"highlighted_full"},
				{"highlight_cells": True, "full_table": False, "name":"highlighted_only"},
				{"highlight_cells": False, "full_table": True, "name":"no_highlighted"}]
	for setting in settings:
		print(f"Setting: {setting['name']}")
		for mode in ["train", "dev", "test"]:
			print(f"Mode: {mode}")
			#input_path = os.path.join(dataset_dir, "totto_data", FILE_NAMES["totto_data"][mode])
			input_path = os.path.join(dataset_dir, "l2t_totto_data", FILE_NAMES["l2t_totto_data"][mode])
			#output_dir = os.path.join(dataset_dir, "totto_lattice", setting["name"])
			output_dir = os.path.join(dataset_dir, "l2t_totto_lattice", setting["name"])
			os.makedirs(output_dir, exist_ok=True)
			output_json_path = os.path.join(output_dir, FILE_NAMES["lattice"][mode])
			output_csv_path = os.path.join(output_dir, f"{mode}.csv")
			preprocess_data.run(input_path, output_json_path, setting["highlight_cells"], setting["full_table"])
			json_to_csv.run(output_json_path, output_csv_path)


if __name__ == '__main__':
	run()
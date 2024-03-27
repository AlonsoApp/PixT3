import json

import six
from tqdm import tqdm


def run(file_path:str):
	out_path = file_path.replace(".sys", ".txt")
	with open(file_path, "r", encoding="utf-8") as input_file:
		with open(out_path, "w", encoding="utf-8") as output_file:
			for line in tqdm(input_file, desc="Loading..."):
				line = six.ensure_text(line, "utf-8")
				example = json.loads(line)
				text = example["sys_out"]
				output_file.write(f"{text}\n")

if __name__ == '__main__':
	for path in ["out/inferences/totto/cont_notab_high_test/epoch-1_step-26500.test.sys"]:
		run(path)
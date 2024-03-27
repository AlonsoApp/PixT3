import json

import six
from tqdm import tqdm


def run(file_path:str):
	out_path = file_path.replace(".jsonl", ".txt")
	with open(file_path, "r", encoding="utf-8") as input_file:
		with open(out_path, "w", encoding="utf-8") as output_file:
			for line in tqdm(input_file, desc="Loading..."):
				line = six.ensure_text(line, "utf-8")
				example = json.loads(line)
				text = example["output"]
				if text == "":
					text = " "
				output_file.write(f"{text.encode('unicode_escape').decode('utf-8')}\n")

if __name__ == '__main__':
	for path in ["out/inferences/llava/dev/highlighted.jsonl", "out/inferences/llava/dev/no_highlighted.jsonl", "out/inferences/llava/dev/notab_high.jsonl"]:
		run(path)
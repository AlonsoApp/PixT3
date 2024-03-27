import json

import evaluate
import six
import itertools

def read_file(file_path):
	lines = []
	with open(file_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			line = line.strip()
			lines.append(line.lower())
	return lines

def get_references(json_example, mode="dev"):
	"""Get references from json example."""
	multi_reference = []
	for annotation in json_example["sentence_annotations"]:
		final_sentence = annotation["final_sentence"]
		multi_reference.append(final_sentence.lower())
	if mode == "dev" or mode == "test":
		while len(multi_reference) < 3:
			multi_reference.append("<null>")

	return multi_reference

def read_references(ref_path):
	all_references = []
	with open(ref_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			line = six.ensure_text(line, "utf-8")
			json_example = json.loads(line)
			all_references.append(get_references(json_example))

	return all_references

def count_nulls(references):
	total = 0
	for refs in references:
		for ref in refs:
			total += 1 if ref == '<null>' else 0
	print(total)

def flatten(l, dim:int=0):
	return [item for sublist in l for item in sublist]

def repeat(l, times=3):
	all_items = []
	for item in l:
		all_items.extend(list(itertools.repeat(item, times)))
	return all_items

def compute_metrics(predictions, labels):
	metric = evaluate.load('sacrebleu')
	return {"sacrebleu":metric.compute(predictions=predictions, references=labels)['score']}

def run():
	preditions = read_file("out/inferences/totto/l1__20231127_000000/inferred_texts.txt")
	references = read_references("data/ToTTo/totto_data/totto_dev_data.jsonl")
	exp_pred = repeat(preditions, 3)
	exp_ref = flatten(references)
	assert len(exp_pred) == len(exp_ref)
	print(compute_metrics(exp_pred, exp_ref))


	pass

if __name__ == '__main__':
	run()
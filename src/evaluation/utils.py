import os
from typing import List

import pandas as pd
from datasource.totto.utils import load_dataset_raw, FILE_NAMES


DATASET_PATHS = {
	"l2t": "data/Logic2Text/l2t_totto_data",
	"totto": "data/ToTTo/totto_data",
}

def generate_dataset_inference_files(inferences:List[str], mayor_evaluation:str="totto", mode="test"):
	"""
	Creates the hashed_inferred_texts for the inferences
	:param inferences: inference names like ["cont_highlighted", "cont_no_highlighted", "cont_notab_high"]
	:param mayor_evaluation:
	:param mode:
	:return:
	"""
	for origin in inferences:
		if mode != "dev":
			origin+=f"_{mode}"
		infer_dir = f"./out/inferences/{mayor_evaluation}/{origin}/"
		file_names = FILE_NAMES["l2t_totto_data"] if mayor_evaluation == "l2t" else None
		val_dataset = load_dataset_raw(DATASET_PATHS[mayor_evaluation], mode, indexed=True, file_names=file_names)
		hashed_inferred_texts = {"example_id": [], "prediction": [], "reference": []}
		with open(os.path.join(infer_dir, "inferred_texts.txt")) as file:
			inferred_texts = [line.rstrip() for line in file]
		for example_id, example, infer_text in zip(val_dataset.keys(), val_dataset.values(), inferred_texts):
			hashed_inferred_texts["example_id"].append(example_id)
			hashed_inferred_texts["prediction"].append(infer_text)
			hashed_inferred_texts["reference"].append(example['sentence_annotations'][0]['final_sentence'])
		# Save hashed inferences as csv
		pd.DataFrame(data=hashed_inferred_texts).to_csv(os.path.join(infer_dir, "hashed_inferred_texts.csv"), index=False)

def generate_reference_inference_files(mayor_evaluation:str="totto", mode="test"):
	origin = "reference"
	if mode != "dev":
		origin+=f"_{mode}"
	infer_dir = f"./out/inferences/{mayor_evaluation}/{origin}/"
	file_names = FILE_NAMES["l2t_totto_data"] if mayor_evaluation == "l2t" else None
	val_dataset = load_dataset_raw(DATASET_PATHS[mayor_evaluation], mode, indexed=True, file_names=file_names)
	hashed_inferred_texts = {"example_id": [], "prediction": [], "reference": []}
	inferred_texts = []
	for example_id, example in zip(val_dataset.keys(), val_dataset.values()):
		hashed_inferred_texts["example_id"].append(example_id)
		hashed_inferred_texts["prediction"].append(example['sentence_annotations'][0]['final_sentence'])
		hashed_inferred_texts["reference"].append(example['sentence_annotations'][0]['final_sentence'])
		inferred_texts.append(example['sentence_annotations'][0]['final_sentence'])
	# Save hashed inferences as csv
	pd.DataFrame(data=hashed_inferred_texts).to_csv(os.path.join(infer_dir, "hashed_inferred_texts.csv"), index=False)
	with open(os.path.join(infer_dir, 'inferred_texts.txt'), 'w') as f:
		for line in inferred_texts:
			f.write(f"{line}\n")
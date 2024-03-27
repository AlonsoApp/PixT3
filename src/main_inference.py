from typing import Dict

from model.pix2struct import config, data_loader
from tools.torch_utils import set_seed
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from tools.torch_utils import get_device
import re
import os
import pandas as pd
from tqdm import tqdm
import time


def run():
	args = config.read_arguments()
	set_seed(args.seed)

	processor = AutoProcessor.from_pretrained(args.hf_model_name)

	model = Pix2StructForConditionalGeneration.from_pretrained(args.model_to_load_path)

	model = model.to(get_device())

	eval_dataset = data_loader.get_dataset(args, args.mode, processor)
	eval_dataloader = data_loader.get_dataloader(args, eval_dataset, processor, batch_size=args.eval_batch_size,
												 shuffle=False)
	hashed_inferred_texts = calculate_metrics(args, model, eval_dataloader, processor)
	save_inferences(hashed_inferred_texts, args)

def save_inferences(hashed_inferred_texts: Dict, args):
	# I don't want to specify another parameter, keeping this until it breaks
	#experiment_name = re.search('/([^/]*)/checkpoints', args.model_to_load_path).group(1)
	experiment_name = str(int(time.time()))
	out_name = rf"{experiment_name}_{str(args.image_dir).split('/')[-2]}_{args.mode}"
	output_dir = os.path.join("./out/inferences/",out_name)
	output_path = os.path.join(output_dir, "hashed_inferred_texts.csv")
	os.makedirs(output_dir, exist_ok=True)
	# Save hashed inferences as csv
	pd.DataFrame(data=hashed_inferred_texts).to_csv(output_path, index=False)

	with open(os.path.join(output_dir, "inferred_texts.txt"), 'w') as texts_file:
		for example_id, prediction in zip(hashed_inferred_texts['example_id'], hashed_inferred_texts['prediction']):
			# Write text line
			texts_file.write("{}\n".format(prediction))

def calculate_metrics(args, model, eval_dataloader, processor) -> Dict:
	"""
	based on calculate_metrics from main_train.py
	:param args: dict that contains all the necessary information passed by user while training
	:param model: finetuned pix2struct model
	:param eval_dataloader: Table2TextDataset object for validation data
	:param processor: dataset processor
	"""
	nb_eval_steps = 0
	model.eval()
	device = get_device()
	hashed_inferred_texts = {"example_id": [], "prediction": [], "reference": []}

	for local_step, batch in enumerate(tqdm(eval_dataloader)):
		flattened_patches = batch.pop("flattened_patches").to(device)
		attention_mask = batch.pop("attention_mask").to(device)
		references = batch.pop("texts")
		example_ids = batch.pop("example_ids")

		with torch.no_grad():
			prediction_tokens = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask,
											   max_new_tokens=args.max_text_length, num_beams=args.num_beams, early_stopping=True)
			predictions = processor.batch_decode(prediction_tokens, skip_special_tokens=True)
			hashed_inferred_texts["example_id"] += example_ids
			hashed_inferred_texts["prediction"] += predictions
			hashed_inferred_texts["reference"] += references
		nb_eval_steps += 1
	return hashed_inferred_texts


if __name__ == '__main__':
	run()
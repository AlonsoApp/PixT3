import itertools
import os

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding, DataCollatorForSeq2Seq
import json
from typing import List
from tqdm.auto import tqdm

def prepare_data(example, tokenizer: PreTrainedTokenizerBase, is_encoder_decoder: bool = True,
				 inference: bool = False, max_length: int = 512, pad_to_max_length: bool = False) -> List[BatchEncoding]:
	text_inputs = []
	text_labels = []

	if is_encoder_decoder:
		text_inputs.append(example["source"])
		text_labels.append(example["target"].strip())
	else:
		text = f"{example['source']} Description:"
		if inference:
			text_inputs.append(text)
			text_labels.append(example["target"].strip())
		else:
			text = f"{text} {example['target'].strip()}"
			text_inputs.append(text)
			text_labels.append(text)

	tokenizer_dataset = []
	for text_input, text_label in zip(text_inputs, text_labels):
		model_inputs = tokenizer(
			text=text_input,
			max_length=max_length,
			truncation=True,
			padding="max_length" if pad_to_max_length else False,
			return_tensors=None,
			add_special_tokens=True,
		)

		if is_encoder_decoder:
			model_inputs["labels"] = tokenizer(
				text_target=text_label,
				max_length=max_length,
				truncation=True,
				padding="max_length" if pad_to_max_length else False,
				return_tensors=None,
				add_special_tokens=True,
			)["input_ids"]
		else:
			model_inputs["labels"] = tokenizer(
				text=text_label,
				max_length=max_length,
				truncation=True,
				padding="max_length" if pad_to_max_length else False,
				return_tensors=None,
				add_special_tokens=True,
			)["input_ids"]

		tokenizer_dataset.append(model_inputs)

	return tokenizer_dataset


class T5ToTToDataset(Dataset):
	def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int,
				 pad_to_max_length: bool = False, is_encoder_decoder: bool = True, inference: bool = False):
		self.data = []
		with (open(jsonl_path, "r") as f):
			lines = f.readlines()
			for line in tqdm(lines, desc="Loading dataset"):
				example_dict = json.loads(line.strip())
				#prepared_example = prepare_data(example_dict, tokenizer, is_encoder_decoder=is_encoder_decoder,
				#								inference=inference, max_length=max_length,
				#								pad_to_max_length=pad_to_max_length,)
				prepared_example = {
					"input_ids": example_dict["input_ids"],
					"labels": example_dict["labels"],
					"attention_mask": list(itertools.repeat(1, len(example_dict["input_ids"]))),
				}
				self.data.append(prepared_example)
		print(f"Loaded {len(self.data)} examples from {jsonl_path}.")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def get_dataloader(jsonl_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int, batch_size: int,
				   pad_to_max_length: bool = False, is_encoder_decoder: bool = True,
				   inference: bool = False) -> DataLoader:
	dataset = T5ToTToDataset(jsonl_path=jsonl_path, tokenizer=tokenizer, max_length=max_length,
					  pad_to_max_length=pad_to_max_length, is_encoder_decoder=is_encoder_decoder, inference=inference)
	collator = DataCollatorForSeq2Seq(tokenizer, padding="max_length" if pad_to_max_length else True,
									  return_tensors="pt", max_length=max_length, label_pad_token_id=-100,
									  pad_to_multiple_of=8)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True,
							num_workers=min(16, os.cpu_count()))

	return dataloader

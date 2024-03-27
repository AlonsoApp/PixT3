import os
import torch
from transformers import AutoProcessor, TensorType
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers.tokenization_utils_base import TruncationStrategy, TextInput, PreTokenizedInput
from transformers.utils import PaddingStrategy

from datasource.totto.utils import FILE_NAMES
from PIL import Image
from typing import List, Optional, Union
import numpy as np

class Table2TextDataset(Dataset):
	def __init__(self, datasets, processor, max_patches, image_dir, mode):
		"""
		During fine-tuning only one dataset is used. During warmup, multiple datasets can be combined for multiple objectives
		:param datasets:
		:param processor:
		:param max_patches:
		:param image_dir:
		"""
		self.datasets = datasets
		self.processor = processor
		self.max_patches = max_patches
		self.image_dir = image_dir
		self.processor.image_processor.is_vqa = False
		self.mode = mode

	def __len__(self):
		return len(self.datasets[0])

	def __getitem__(self, idx):
		items = [dataset[idx] for dataset in self.datasets]
		multi = len(self.datasets)>1 # multi datasets must follow image_dir/0/dev
		image_paths = [os.path.join(self.image_dir, f"{str(dataset_i) if multi else ''}", self.mode, f"{str(item['example_id'])}.png") for dataset_i, item in enumerate(items)]
		images = [Image.open(path) for path in image_paths] # No need to convert to RGB Pix2StructImageProcessor will do it
		encodings = self.processor(images=images, return_tensors="pt", add_special_tokens=True, max_patches=self.max_patches)

		examples = [{k: v[i].squeeze() for k, v in encodings.items()} for i in range(len(items))]
		for example, item in zip(examples, items):
			example["text"] = item['sentence_annotations'][0]['final_sentence'] if 'sentence_annotations' in item else ''
			example["example_id"] = item['example_id']
		return examples

class Table2TextCollator:
	def __init__(self, processor, max_text_length, dataset_variant_weights, batch_dataset=None):
		self.processor = processor
		self.max_text_length = max_text_length
		self.dataset_variant_weights = dataset_variant_weights
		self.batch_dataset = batch_dataset # To force one specific ssl dataset on warmup evaluation

	def __call__(self, *args, **kwargs):
		batch = args[0]
		batch_dataset = np.random.choice(np.arange(len(batch[0])), p=self.dataset_variant_weights) if self.batch_dataset is None else self.batch_dataset
		batch = [examples[batch_dataset] for examples in batch]
		new_batch = {"flattened_patches": [], "attention_mask": []}
		texts = [item["text"] for item in batch]
		example_ids = [item["example_id"] for item in batch]

		if self.max_text_length is not None:
			padding, truncation, max_length = True, True, self.max_text_length
		else:
			padding, truncation, max_length = True, False, None

		if self.processor.image_processor.is_vqa:
			processed_texts = self.process_text_vqa(text=texts, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt", add_special_tokens=True)
		else:
			processed_texts = self.processor(text=texts, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt", add_special_tokens=True)

		new_batch["labels"] = processed_texts.input_ids
		new_batch["texts"] = texts
		new_batch["example_ids"] = example_ids

		for item in batch:
			new_batch["flattened_patches"].append(item["flattened_patches"])
			new_batch["attention_mask"].append(item["attention_mask"])

		new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
		new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
		return new_batch

	def process_text_vqa(self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,):
		"""As VQA Processors currently don't support text only processing we replicate the text only processing of
		pix2struct-base"""
		text_encoding = self.processor.tokenizer(
			text=text,
			add_special_tokens=add_special_tokens,
			padding=padding,
			truncation=truncation,
			max_length=max_length,
			stride=stride,
			pad_to_multiple_of=pad_to_multiple_of,
			return_attention_mask=return_attention_mask,
			return_overflowing_tokens=return_overflowing_tokens,
			return_special_tokens_mask=return_special_tokens_mask,
			return_offsets_mapping=return_offsets_mapping,
			return_token_type_ids=return_token_type_ids,
			return_length=return_length,
			verbose=verbose,
			return_tensors=return_tensors,
			**kwargs,
		)
		return text_encoding

def get_dataset(args, mode: str, processor: AutoProcessor = None) -> Table2TextDataset:
	dataset_paths = [os.path.join(args.dataset_dir, variant, FILE_NAMES[variant][mode]) for variant in args.dataset_variant.split(',')]
	# I know I should not use split="train" but I'm too tired
	dataset_jsons = [load_dataset("json", data_files=path, split="train") for path in dataset_paths]
	return Table2TextDataset(dataset_jsons, processor, args.max_patches, args.image_dir, mode)

def get_dataloader(args, dataset, processor: AutoProcessor, batch_size: int = None, shuffle:bool = True, batch_dataset:int=None):
	max_text_length = args.max_text_length if args.truncate_train_length else None
	dataset_variant_weights = [float(x) for x in args.dataset_variant_weights.split(",")] if args.dataset_variant_weights is not None else None
	cls_collator = Table2TextCollator(processor, max_text_length, dataset_variant_weights, batch_dataset=batch_dataset)
	batch_size = args.batch_size if batch_size is None else batch_size
	return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=cls_collator)
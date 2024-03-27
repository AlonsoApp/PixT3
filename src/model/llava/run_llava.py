import argparse
import torch
import os
import json
from tqdm import tqdm
import six

from llava.constants import (
	IMAGE_TOKEN_INDEX,
	DEFAULT_IMAGE_TOKEN,
	DEFAULT_IM_START_TOKEN,
	DEFAULT_IM_END_TOKEN,
	IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
	process_images,
	tokenizer_image_token,
	get_model_name_from_path,
	KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


FILE_NAMES = {
	"totto_data":{"train": "totto_train_data.jsonl", "dev": "totto_dev_data.jsonl",
				  "test": "unlabeled_totto_test_data.jsonl"},
	"totto_toy":{"train": "totto_train_data_toy.jsonl", "dev": "totto_dev_data_toy.jsonl",
				 "test": "unlabeled_totto_test_data.jsonl"},
	"warmup_ssl1":{"train": "train.jsonl", "dev": "dev.jsonl",
				 "test": "test.jsonl"},
	"warmup_ssl3":{"train": "train.jsonl", "dev": "dev.jsonl",
				 "test": "test.jsonl"},
	"l2t_totto_data": {"train": "train.jsonl", "dev": "dev.jsonl",
					"test": "test.jsonl", "toy": "toy.jsonl"},
	"wg_totto_data": {"train": "train.jsonl", "dev": "dev.jsonl",
					"test": "test.jsonl"},
	"cont": {"train": "train.jsonl", "dev": "val.jsonl",
					"test": "test.jsonl"},
	"t5": {"train": "train.jsonl", "dev": "val.jsonl",
					"test": "test.jsonl"}}

NO_TABLE = [['No table']]
NO_COLUMNS = ['No columns']


def load_dataset_raw(dataset_dir, mode, indexed=False, allow_duplicates: bool = False, file_names=None):
	"""
	Loads the ToTTo dataset without any preprocessing
	:param dataset_dir:
	:param mode:
	:param indexed: if True it will return a dictionary with {"example_id": example}
	:param allow_duplicates: values are now List and can contain multiple examples associated to one id
	:param file_names: names of the dataset files, usually FILE_NAMES
	:return: list of json objects or dict {"example_id": example}
	"""
	if file_names is None:
		file_names = FILE_NAMES["totto_data"]
	dataset_path = os.path.join(dataset_dir, file_names[mode])
	with open(dataset_path, "r", encoding="utf-8") as input_file:
		data = []
		data_dict = {}
		for line in tqdm(input_file, desc="Loading ToTTo ({})".format(mode)):
			line = six.ensure_text(line, "utf-8")
			example = json.loads(line)
			if indexed:
				if example['example_id'] in data_dict and allow_duplicates:
					data_dict[example['example_id']].append(example)
				else:
					data_dict[example['example_id']] = [example] if allow_duplicates else example
			else:
				data.append(example)
	return data_dict if indexed else data

def image_parser(args):
	out = args.image_file.split(args.sep)
	return out


def load_image(image_file):
	if image_file.startswith("http") or image_file.startswith("https"):
		response = requests.get(image_file)
		image = Image.open(BytesIO(response.content)).convert("RGB")
	else:
		image = Image.open(image_file).convert("RGB")
	return image


def load_images(image_files):
	out = []
	for image_file in image_files:
		image = load_image(image_file)
		out.append(image)
	return out

EXAMPLES = [
	"'chilawathurai had the 2nd lowest population density among main towns in the mannar district .'",
	"'zhou mi only played in one bwf super series masters finals tournament .'",
	"'tobey maguire appeared in vanity fair later than mike piazza in 2003 .'"
]

PROMPTS_V2 = {
	"notab_high": f"Here are some descriptions based on other highlights of other tables {', '.join(EXAMPLES)}. Now write a short description based on the following highlighted cells extracted form a table.",
	"highlighted": f"Here are some descriptions based on the highlights of other tables not present in the input: {', '.join(EXAMPLES)}. Now write a short description based on the highlighted cells in this table following the same style as the example descriptions.",
	"no_highlighted": f"Here are some descriptions from other tables not present in the input: {', '.join(EXAMPLES)}. Now write a short description stating something from this table following the same style as the example descriptions.",
}

PROMPTS_V1 = {
	"notab_high": f"Write a short description based on the following highlighted cells extracted form a table.",
	"highlighted": f"Write a short description based on the highlighted cells in this table.",
	"no_highlighted": f"Write a short description stating something from this table.",
}

def eval_model(args):
	# Model
	disable_torch_init()

	model_name = get_model_name_from_path(args.model_path)
	tokenizer, model, image_processor, context_len = load_pretrained_model(
		args.model_path, args.model_base, model_name
	)

	dataset = load_dataset_raw(os.path.join(args.dataset_dir, args.dataset_variant), args.mode, indexed=True,
							   file_names=FILE_NAMES[args.dataset_variant])
	results = []
	for example_id, example in tqdm(dataset.items()):
		qs = PROMPTS_V2[args.setting]
		image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
		if IMAGE_PLACEHOLDER in qs:
			if model.config.mm_use_im_start_end:
				qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
			else:
				qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
		else:
			if model.config.mm_use_im_start_end:
				qs = image_token_se + "\n" + qs
			else:
				qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

		if "llama-2" in model_name.lower():
			conv_mode = "llava_llama_2"
		elif "v1" in model_name.lower():
			conv_mode = "llava_v1"
		elif "mpt" in model_name.lower():
			conv_mode = "mpt"
		else:
			conv_mode = "llava_v0"

		if args.conv_mode is not None and conv_mode != args.conv_mode:
			print(
				"[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
					conv_mode, args.conv_mode, args.conv_mode
				)
			)
		else:
			args.conv_mode = conv_mode

		conv = conv_templates[args.conv_mode].copy()
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()

		#image_files = image_parser(args)
		# ./data/Logic2Text/img/highlighted_039/test/33270b2ef29196dd717bcfc54cf38572743b12f3.png
		# ./data/Logic2Text/img/highlighted_039/test/
		image_files = [os.path.join(args.image_dir, args.mode, f"{example_id}.png")] # TODO change test with args.mode
		images = load_images(image_files)
		images_tensor = process_images(
			images,
			image_processor,
			model.config
		).to(model.device, dtype=torch.float16)

		input_ids = (
			tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
			.unsqueeze(0)
			.cuda()
		)

		stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
		keywords = [stop_str]
		stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

		with torch.inference_mode():
			output_ids = model.generate(
				input_ids,
				images=images_tensor,
				do_sample=True if args.temperature > 0 else False,
				temperature=args.temperature,
				top_p=args.top_p,
				num_beams=args.num_beams,
				max_new_tokens=args.max_new_tokens,
				use_cache=True,
				stopping_criteria=[stopping_criteria],
			)

		input_token_len = input_ids.shape[1]
		n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
		if n_diff_input_output > 0:
			print(
				f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
			)
		outputs = tokenizer.batch_decode(
			output_ids[:, input_token_len:], skip_special_tokens=True
		)[0]
		outputs = outputs.strip()
		if outputs.endswith(stop_str):
			outputs = outputs[: -len(stop_str)]
		outputs = outputs.strip()
		results.append({"example_id": example_id, "prompt": prompt, "output": str(outputs)})
		#print(outputs)

	out_dir = os.path.join("./out/inferences/llava/")
	out_path = os.path.join(out_dir, f"{args.setting}.jsonl")
	os.makedirs(out_dir, exist_ok=True)
	with open(out_path, 'w') as f:
		for line in results:
			# f.write(f"{line}\n")
			f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
	parser.add_argument("--model-base", type=str, default=None)
	#parser.add_argument("--image-file", type=str, required=True)
	#parser.add_argument("--query", type=str, required=True)
	parser.add_argument("--conv-mode", type=str, default=None)
	parser.add_argument("--sep", type=str, default=",")
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--top_p", type=float, default=None)
	parser.add_argument("--num_beams", type=int, default=1)
	parser.add_argument("--max_new_tokens", type=int, default=512)

	parser.add_argument("--image_dir", type=str, default="")
	parser.add_argument("--mode", type=str, default="dev")
	parser.add_argument("--dataset_variant", type=str, default="l2t_totto_data")
	parser.add_argument("--dataset_dir", type=str, default="./data/Logic2Text")
	parser.add_argument("--setting", type=str, default="highlighted")
	args = parser.parse_args()

	eval_model(args)
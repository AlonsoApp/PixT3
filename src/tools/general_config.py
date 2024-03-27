import argparse
import json
import os

def save_config(args, output_path):
	config_path = os.path.join(output_path, "args.json")

	with open(config_path, 'w', encoding='utf-8') as f:
		json.dump(args.__dict__, f, indent=2)

def load_parser():
	parser = argparse.ArgumentParser(description="Running with following arguments")

	parser.add_argument('--mode', default='train', type=str)

	# Paths and general configuration
	parser.add_argument('--exp_name', default='exp', type=str)
	parser.add_argument('--seed', default=42, type=int)
	parser.add_argument('--dataset_dir', default="./data/ToTTo/", type=str)
	parser.add_argument('--dataset_variant', default="totto_data", type=str, help="Also supports multiple variants for multi ssl objective warmup")
	parser.add_argument('--image_dir', default="./data/ToTTo/img/highlighted/", type=str)
	parser.add_argument('--model_output_dir', default="./out/experiments/", type=str)
	parser.add_argument('--inference_output_dir', default="./out/inference/", type=str)
	parser.add_argument('--model_to_load_dir', default=argparse.SUPPRESS, type=str, help="The path of the model to be loaded. E.g: exp__20220425_165912")
	parser.add_argument('--wandb_project', default="MMTable2Text", type=str, help="Project of wandb to report")

	# Model configuration
	parser.add_argument('--hf_model_name', default="google/pix2struct-base", type=str) # ybelkada/pix2struct-base
	parser.add_argument('--max_patches', default=2048, type=int) # 512: 58.93%, 2048: 88.01%, 4096: 94.99%
	parser.add_argument('--max_text_length', default=50, type=int) # 40=92.62%, 45=95.7%, 50=97.49% of samples 55=98.52

	# Training & optimizer configuration
	parser.add_argument('--truncate_train_length', default=False, type=bool) # True train|eval target truncated to max_text_length
	parser.add_argument('--shuffle_dataset', default=True, type=bool)
	parser.add_argument('--gradient_checkpointing', default=False, type=bool)
	parser.add_argument('--mixed_precision', default="fp16", type=str)
	parser.add_argument("--batch_size", default=8, type=int, help="batch_size") # 1 local, 4 GPUs
	parser.add_argument("--eval_batch_size", default=32, type=int, help="batch_size") # 1 local, 4 GPUs
	parser.add_argument("--epochs", default=21, type=int, help="no of epochs of training (11 would match the 10000 steps"
															   " at 128 batch size in the finetuning of the pix2struct "
															   "paper")
	parser.add_argument("--eval_freq_steps", default=250, type=int, help="Evaluate every n global steps") # Equivalent to 12800 samples
	parser.add_argument("--resume_from_state", default=argparse.SUPPRESS, type=str, help="Experiment name to carry the training from. E.g: exp__20230519_181818")

	#parser.add_argument("--training_steps", default=10000, type=int, help="no of epochs of training")
	parser.add_argument("--lr", default=1e-4, type=float, help="training learning rate") # AdamW constant 1e-5 and scheduler 1e-4
	parser.add_argument("--num_warmup_steps", default=1000, type=int, help="Number of warmup steps for the lr scheduler") # 1000 as in the paper
	parser.add_argument("--gradient_accumulation_steps", default=32, type=int, help="gradient_accumulation_steps 32 "
																					 "for a batch size of 8 to match the "
																					 "effective 256 batch_size in the "
																					 "fine-tuning of the pix2struct paper")
	parser.add_argument("--checkpoints_to_save", default='best', type=str, help="Checkpoint saving criteria. 'all' saves one for each epoch, 'best' only saves the best")
	parser.add_argument("--freeze_decoder", default=False, type=bool,
						help="Whether to freeze the decoder parameters during training. Used for structure training curriculum (aka warmup)")
	parser.add_argument('--dataset_variant_weights', default=None, type=str, help="weights separated by "
							"commas to balance the amount of samples of each dataset variant to be used during warmup")  # "0.3,0.7"

	# Inference configuration
	parser.add_argument('--num_beams', default=8, type=int, help='beam size for beam search')
	parser.add_argument('--model_to_load_path', default="", type=str, help='Path to the experiment '
																						  'folder where the model is '
																						  'stored')

	return parser

def finish_args(args):
	if hasattr(args, 'model_to_load_dir'):
		args.model_to_load_path = os.path.join(args.model_to_load_dir, "best_model.pt")
		args.config_to_load_path = os.path.join(args.model_to_load_dir, "model_config.json")

	print("*** parsed configuration from command line and combine with constants ***")

	for argument in vars(args):
		print("argument: {}={}".format(argument, getattr(args, argument)))
	return args
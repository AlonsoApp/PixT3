import os.path
import shutil
from typing import Dict

from math import ceil
from transformers.optimization import get_cosine_schedule_with_warmup

from model.pixt3 import config, data_loader
from tools.torch_utils import set_seed, create_experiment_folder
import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, Adafactor, Pix2StructProcessor
import evaluate as hf_evaluate
from torch.utils.data import DataLoader
import time
import datetime
from accelerate import Accelerator

def run():
	args = config.read_arguments()
	if hasattr(args, 'resume_from_state'):
		experiment_name, output_path = args.resume_from_state, os.path.join(args.model_output_dir, args.resume_from_state)
	else:
		experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
	set_seed(args.seed)

	wandb_resume = hasattr(args, 'resume_from_state')
	accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb", mixed_precision=args.mixed_precision)
	accelerator.init_trackers(project_name=args.wandb_project, config=args, init_kwargs={"wandb": {"id":experiment_name,
																								   "name":experiment_name,
																								   "resume":wandb_resume}})

	metrics = {'bleu': hf_evaluate.load('sacrebleu')}
	best_results = None

	processor = AutoProcessor.from_pretrained(args.hf_model_name)
	model = Pix2StructForConditionalGeneration.from_pretrained(args.hf_model_name)

	train_dataset = data_loader.get_dataset(args, "train", processor)
	dev_dataset = data_loader.get_dataset(args, "dev", processor)
	train_dataloader = data_loader.get_dataloader(args, train_dataset, processor, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
	eval_dataloader = data_loader.get_dataloader(args, dev_dataset, processor, batch_size=args.eval_batch_size, shuffle=False)

	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()

	# Training
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	t_training_steps = args.epochs * ceil(len(train_dataloader)/args.gradient_accumulation_steps)
	#optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)
	accelerator.print(f"Total train steps: {t_training_steps}")
	lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_training_steps)

	train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer, lr_scheduler)
	#train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

	checkpoint = -1
	if hasattr(args, 'resume_from_state'):
		checkpoint_path, checkpoint = get_checkpoint_path(output_path)
		accelerator.load_state(checkpoint_path)
	if args.freeze_decoder:
		freeze_parameters(model)
	model.train()
	global_step = 0
	for epoch in range(checkpoint+1, args.epochs):
		accelerator.print("Epoch:", epoch)
		assert_frozen(model, args.freeze_decoder)
		for local_step, batch in enumerate(train_dataloader):
			with accelerator.accumulate(model):
				flattened_patches = batch.pop("flattened_patches")
				attention_mask = batch.pop("attention_mask")
				labels = batch.pop("labels")

				outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
				loss = outputs.loss
				accelerator.backward(loss)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				global_step = (local_step + (epoch*len(train_dataloader)))/args.gradient_accumulation_steps

				if (local_step+(epoch*len(train_dataloader))) % args.gradient_accumulation_steps == 0:
					# Report on accumulated step
					accelerator.log({"train_loss": accelerator.gather(loss).mean()}, step=int(global_step))
					accelerator.log({"learning_rate": scheduler.get_lr()[0]}, step=int(global_step))
			if (global_step+1) % args.eval_freq_steps == 0:
				eval_loss = evaluate(accelerator, model, eval_dataloader)
				accelerator.log({'eval_loss': eval_loss}, step=int(global_step))
				model.train()
		metric_results = calculate_metrics(accelerator, args, model, eval_dataloader, processor, metrics)
		accelerator.log(metric_results, step=int(global_step))
		if args.checkpoints_to_save == 'all' or (args.checkpoints_to_save == 'best' and is_better_checkpoint(metric_results, best_results)) and accelerator.is_main_process:
			save_checkpoint(model, accelerator, processor, epoch, output_path, args.checkpoints_to_save)
			best_results = metric_results
	accelerator.end_training()

def evaluate(accelerator, model, eval_dataloader: DataLoader) -> float:
	"""
	Returns the loss on validation dataset.
	:param accelerator:
	:param model: finetuned pix2struct model
	:param eval_dataloader: eval data loader
	"""
	eval_loss = 0.0
	nb_eval_steps = 0
	model.eval()
	accelerator.print("Evaluating...")
	t_start = time.time()
	accelerator.wait_for_everyone()
	for local_step, batch in enumerate(eval_dataloader):
		flattened_patches = batch.pop("flattened_patches")
		attention_mask = batch.pop("attention_mask")
		labels = batch.pop("labels")
		with torch.no_grad():
			outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
			loss = outputs.loss
			eval_loss += accelerator.gather(loss).mean()
		nb_eval_steps += 1
	accelerator.print(f"Finished: {str(datetime.timedelta(seconds=time.time()-t_start))}")
	return eval_loss / nb_eval_steps

def calculate_metrics(accelerator, args, model, eval_dataloader, processor, metrics: Dict) -> Dict:
	"""
	Returns the loss on validation dataset.
	:param accelerator:
	:param args: dict that contains all the necessary information passed by user while training
	:param model: finetuned pix2struct model
	:param eval_dataloader: Table2TextDataset object for validation data
	:param processor: dataset processor
	:param metrics: metrics to calculate over the predictions
	"""
	accelerator.print("Calculating metrics...")
	t_start = time.time()
	result_metrics = {k: [] for k in metrics.keys()}
	model.eval()
	accelerator.wait_for_everyone()
	prediction_tokens, label_tokens = [], []
	if accelerator.state.num_processes > 1:
		# We need to unwrap the model when using multi GPU to use generate (DistributedDataParallel)
		model =  accelerator.unwrap_model(model)
	for local_step, batch in enumerate(eval_dataloader):
		flattened_patches = accelerator.gather(batch["flattened_patches"])
		attention_mask = accelerator.gather(batch["attention_mask"])
		labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=processor.tokenizer.pad_token_id)
		label_tokens += accelerator.gather(labels)
		with torch.no_grad():
			prediction_tokens += model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask,
											   max_new_tokens=args.max_text_length)
	references = processor.batch_decode(label_tokens, skip_special_tokens=True)
	predictions = processor.batch_decode(prediction_tokens, skip_special_tokens=True)
	for metric_name, metric in metrics.items():
		result_metrics[metric_name] = metric.compute(predictions=predictions, references=references)['score']
	accelerator.print(f"Finished: {str(datetime.timedelta(seconds=time.time() - t_start))}")
	return result_metrics

def freeze_parameters(model:Pix2StructForConditionalGeneration):
	for param in model.decoder.parameters():
		param.requires_grad = False

def assert_frozen(model:Pix2StructForConditionalGeneration, freeze:bool):
	if not freeze:
		return
	for param in model.decoder.parameters():
		assert param.requires_grad == False

def save_checkpoint(model, accelerator: Accelerator, processor: Pix2StructProcessor, epoch: int, output_path: str, saving_criteria: str):
	checkpoint_path = os.path.join(output_path, "checkpoints")
	path = os.path.join(checkpoint_path, str(epoch))
	os.makedirs(path, exist_ok=True)
	accelerator.save_state(path)
	#torch.save(model.state_dict(), os.path.join(model_save_path, model_name))

	# Good practice: save your training arguments together with the trained model
	if accelerator.state.num_processes > 1:
		accelerator.unwrap_model(model).config.to_json_file(os.path.join(path, "config.json"))
	else:
		model.config.to_json_file(os.path.join(path, "config.json"))
	processor.save_pretrained(path)
	if saving_criteria == 'best':
		# Delete all the other checkpoints, we already know this is the best checkpoint
		for checkpoint in os.listdir(checkpoint_path):
			if checkpoint != str(epoch):
				shutil.rmtree(os.path.join(checkpoint_path, checkpoint))

def get_checkpoint_path(experiment_path:str, checkpoint: int = None):
	"""
	:param experiment_path: path where the experiment is being saved
	:param checkpoint: default = latest
	:return:
	"""
	checkpoint_dir = os.path.join(experiment_path, "checkpoints")
	checkpoints = [int(x) for x in os.listdir(checkpoint_dir)]
	checkpoint = checkpoint if checkpoint is not None else max(checkpoints)
	checkpoint_path = os.path.join(checkpoint_dir, str(checkpoint))
	return checkpoint_path, checkpoint

def is_better_checkpoint(current_results: Dict, best_results: Dict):
	# Easy and simple, if BLEU is better, return True
	return best_results is None or current_results['bleu'] > best_results['bleu']

if __name__ == '__main__':
	run()
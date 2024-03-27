from transformers import (
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments, AutoProcessor, T5TokenizerFast
)
from datasets import DatasetDict
import evaluate
import numpy as np

from model.t5.dataset_t5 import T5ToTToDataset
from model.t5.load_model import load_model_for_training, load_model_for_inference
from model.t5.config_t5 import (
    ModelArguments,
    DataTrainingArguments,
    Seq2SeqTrainingArguments,
)

import sys
import os
import torch.utils.data
import json

def clean_cache():
    import gc

    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""

    print(f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")
tokenizer:T5TokenizerFast = None
def train_seq2seq(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: Seq2SeqTrainingArguments):
    print("Loading model...")
    global tokenizer
    model, tokenizer = load_model_for_training(
        model_weights_name_or_path=model_args.model_name_or_path,
        int8_quantization=model_args.int8_quantization,
        use_lora=model_args.use_lora,
    )

    print("Loading datasets...")
    training_datsets = []
    for train_path in data_args.train_files:
        train_dataset = T5ToTToDataset(
            jsonl_path=train_path,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            pad_to_max_length=False,
            is_encoder_decoder=model.config.is_encoder_decoder,
            inference=False,
        )
        training_datsets.append(train_dataset)

    train_dataset = torch.utils.data.ConcatDataset(training_datsets)

    dev_datasets = DatasetDict()
    for dev_path in data_args.validation_files:
        dev_dataset = T5ToTToDataset(
            jsonl_path=dev_path,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            pad_to_max_length=False,
            is_encoder_decoder=model.config.is_encoder_decoder,
            inference=False,
        )
        dev_datasets[os.path.splitext(os.path.basename(dev_path))[0]] = dev_dataset

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id,
        ),
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

def compute_metrics(eval_preds):
    metric = evaluate.load('sacrebleu')
    prediction_tokens, label_tokens = eval_preds
    predictions, labels = [], []
    for i in range(len(label_tokens)):
        labels.append(tokenizer.decode([x for x in label_tokens[i] if x != -100], skip_special_tokens=True).lower())
    for i in range(len(prediction_tokens)):
        predictions.append(tokenizer.decode([x for x in prediction_tokens[i] if x != -100], skip_special_tokens=True).lower())
    print(f"Example 01:\nRef: {labels[0]}\nPred tokens: {prediction_tokens[0]}\nPred text:{predictions[0]}")
    return {"sacrebleu":metric.compute(predictions=predictions, references=labels)['score']}

def eval_seq2seq(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: Seq2SeqTrainingArguments):
    print("Loading model...")
    model, tokenizer = load_model_for_inference(
        weights_path=training_args.output_dir,
        int8_quantization=model_args.int8_quantization,
        lora_weights_name_or_path=(
            (
                model_args.lora_weights_name_or_path
                if model_args.lora_weights_name_or_path is not None
                else training_args.output_dir
            )
            if model_args.use_lora
            else None
        ),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )

    for val_path in data_args.validation_files:
        print(f"Evaluate {val_path}...")
        test_dataset = T5ToTToDataset(
            jsonl_path=val_path,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            pad_to_max_length=False,
            is_encoder_decoder=model.config.is_encoder_decoder,
            inference=True,
        )
        output_path = os.path.join(
            training_args.output_dir,
            os.path.splitext(os.path.basename(val_path))[0] + ".predictions",
        )

        gen_kwargs = {
            "max_new_tokens": training_args.generation_max_length,
            "num_beams": training_args.generation_num_beams,
            "do_sample": training_args.do_sample,
            "temperature": training_args.temperature,
            "top_k": training_args.top_k,
            "top_p": training_args.top_p,
            "repetition_penalty": training_args.repetition_penalty,
        }

        predictions = trainer.predict(
            test_dataset,
            **gen_kwargs,
        )
        with open(output_path, "w", encoding="utf8") as f:
            list_dict = []
            for i in range(len(predictions.predictions)):
                prediction = predictions.predictions[i]
                question = test_dataset[i]["input_ids"]
                gold = test_dataset[i]["labels"]
                list_dict.append(
                    {
                        "question": tokenizer.decode(
                            [x for x in question if x != -100], skip_special_tokens=True
                        ),
                        "gold_answer": tokenizer.decode(
                            [x for x in gold if x != -100], skip_special_tokens=True
                        ),
                        "prediction": tokenizer.decode(
                            [x for x in prediction if x != -100],
                            skip_special_tokens=True,
                        ),
                    }
                )

            json.dump(list_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    print(sys.argv)
    print(len(sys.argv))
    print(sys.argv[1].endswith(".yaml"))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and, it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )

    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script, and, it's the path to a yaml file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.train_files is not None:
        train_seq2seq(
            model_args,
            data_args,
            training_args,
        )
        clean_cache()

    if data_args.validation_files is not None:
        eval_seq2seq(
            model_args,
            data_args,
            training_args,
        )
        clean_cache()
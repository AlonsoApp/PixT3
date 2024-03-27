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

def get_output_path(val_path:str, output_dir:str):
    inferences_dir = "out/inferences"
    dataset_name = "l2t" if "Logic2Text" in val_path.split("/") else "totto"
    setting = output_dir.split("/")[output_dir.split("/").index("models")+1].split("__")[0]
    mode = os.path.splitext(os.path.basename(val_path))[0]
    return os.path.join(inferences_dir, dataset_name, "t5", f"{setting}_{mode}.txt")


def inf_seq2seq(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: Seq2SeqTrainingArguments):
    print("Loading model...")
    output_path = get_output_path(data_args.validation_files[0], training_args.output_dir)
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
        #output_path = os.path.join(
        #    training_args.output_dir,
        #    os.path.splitext(os.path.basename(val_path))[0] + ".predictions",
        #)

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
            for i in range(len(predictions.predictions)):
                prediction = predictions.predictions[i]
                line = tokenizer.decode([x for x in prediction if x != -100], skip_special_tokens=True)
                f.write(f"{line}\n")


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

    if data_args.validation_files is not None:
        inf_seq2seq(
            model_args,
            data_args,
            training_args,
        )
        clean_cache()
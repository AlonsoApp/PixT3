from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import TrainingArguments, GenerationConfig
from pathlib import Path


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use LoRA. If True, the model will be trained with "
            "LoRA: https://arxiv.org/abs/2106.09685"
        },
    )

    int8_quantization: bool = field(
        default=False,
        metadata={
            "help": "Whether to use int8 quantization. "
            "Requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes"
        },
    )
    lora_weights_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If the model has been trained with LoRA, "
            "path or huggingface hub name or local path to the pretrained weights."
        },
    )

    lora_r: Optional[int] = field(
        default=8,
    )
    lora_alpha: Optional[float] = field(
        default=16,
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
    )
    target_modules: Optional[str] = field(
        default_factory=list,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_files: List[str] = field(
        default=None, metadata={"help": "The input training data files (a jsonl file)."}
    )

    validation_files: List[str] = field(
        default=None,
        metadata={"help": "The input validation data files (a jsonl file)."},
    )

    test_files: List[str] = field(
        default=None, metadata={"help": "The input test data files (a jsonl file)."}
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use a *sortish sampler* or not. Only possible if the underlying datasets are *Seq2SeqDataset*
            for now but will become generally available in the near future.

            It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness
            for the training set.
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
    """

    sortish_sampler: bool = field(
        default=False, metadata={"help": "Whether to use SortishSampler or not."}
    )
    predict_with_generate: bool = field(
        default=False,
        metadata={
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."
        },
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )

    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use sampling ; use greedy decoding otherwise."
        },
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )

    top_k: int = field(
        default=50,
        metadata={
            "help": (
                "The number of highest probability vocabulary tokens to keep for top-k-filtering."
            )
        },
    )

    top_p: float = field(
        default=1.0,
        metadata={
            "help": (
                "If set to float < 1, only the most probable tokens "
                "with probabilities that add up to top_p or higher are kept for generation."
            )
        },
    )

    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": (
                "The parameter for repetition penalty. 1.0 means no penalty. See "
                "https://arxiv.org/abs/1909.05858 for more details."
            )
        },
    )

    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )

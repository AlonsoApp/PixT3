from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from typing import Optional
import logging
import os


def get_trainable_parameters(model: PreTrainedModel) -> (int, int, float):
    """
    Prints the number of trainable parameters in the model.
    :param model: The model to print the number of trainable parameters for.
    :return: The number of trainable parameters, the total number of parameters and the percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param, 100 * trainable_params / all_param


def load_model_for_training(
    model_weights_name_or_path: str,
    int8_quantization: bool = False,
    use_lora: bool = False,
    lora_weights_name_or_path: Optional[str] = None,
    lora_r: Optional[int] = 8,
    lora_alpha: Optional[int] = 16,
    lora_dropout: Optional[float] = 0.05,
) -> (PreTrainedModel, PreTrainedTokenizerBase):
    """
    Load any Decoder model for training.
    :param model_weights_name_or_path: The path to your local model weights and tokenizer or huggingface model name.
    :param int8_quantization: Whether to use int8 quantization.
                              Requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes
    :param use_lora: Whether to use LORA. See https://arxiv.org/pdf/2106.09685.pdf for more details.
                     Requires huggingface PEFT library: https://github.com/huggingface/peft
    :param lora_weights_name_or_path: The name or path to the pre-trained LORA model weights. You can also provide a
                                      huggingface hub model name to load the weights from there. If not provided, the
                                      weights will be initialized randomly, this requires training the model.
    :param lora_r: Lora attention dimension.
    :param lora_alpha: The alpha parameter for Lora scaling.
    :param lora_dropout: The dropout probability for Lora layers.
    """

    if int8_quantization and not use_lora:
        raise ValueError(
            "Training with Int8 quantization is only supported with LoRA. If you want"
            " to train in Int8, please add the flag --use_lora. You can only evaluate"
            " in Int8 without LoRA."
        )
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    logging.info(f"Device map: {device_map}")

    logging.info(f"Loading model model from {model_weights_name_or_path}")
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_weights_name_or_path,
        load_in_8bit=int8_quantization,
        device_map="auto"
        if "xl" in model_weights_name_or_path
        else (device_map if int8_quantization else None),
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_weights_name_or_path,
        add_eos_token=True,
    )

    if tokenizer.pad_token_id is None:
        logging.warning(
            "Your model does not have a pad token, we will use the ukn token as pad"
            " token."
        )
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if int8_quantization:
        from peft import prepare_model_for_int8_training

        model = prepare_model_for_int8_training(model)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel

        if lora_weights_name_or_path is None:
            logging.info(
                "No pretrained LORA weights provided, we will initialize the weights"
                " randomly."
            )

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=(
                    ["q_proj", "v_proj"]
                    if model.config.model_type == "llama"
                    else None  # Use PEFT default target modules. We specify here for Llama because the
                    # current version of PEFT does not have the Llama target modules yet.
                ),
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, config)
        else:
            logging.info(
                f"Loading pretrained LORA weights from {lora_weights_name_or_path}"
            )

            model = PeftModel.from_pretrained(model, lora_weights_name_or_path)

    trainable_params, total_params, trainable_percentage = get_trainable_parameters(
        model
    )
    logging.info(
        f"---> Trainable params: {trainable_params} || all params: {total_params} ||"
        f" trainable%: {round(trainable_percentage,6)}\n"
    )

    return model, tokenizer


def load_model_for_inference(
    weights_path: str,
    int8_quantization: bool = False,
    lora_weights_name_or_path: Optional[str] = None,
) -> (PreTrainedModel, PreTrainedTokenizerBase):
    """
    Load any Decoder model for inference.
    :param weights_path: The path to your local model weights and tokenizer.
                                You can also provide a huggingface hub model name.
    :param int8_quantization: Whether to use int8 quantization.
                              Requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes
    :param lora_weights_name_or_path: If the model has been trained with LoRA, path or huggingface hub name to the
                                      pretrained weights.

    """

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    logging.info(f"Device map: {device_map}")

    logging.info(f"Loading model from {weights_path}")
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=weights_path,
        load_in_8bit=int8_quantization,
        device_map="auto"
        if "xl" in weights_path
        else (device_map if int8_quantization else None),
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        weights_path,
        add_eos_token=True,
    )

    if tokenizer.pad_token_id is None:
        logging.warning(
            "Model does not have a pad token, we will use the ukn token as pad token."
        )
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if lora_weights_name_or_path:
        from peft import PeftModel

        logging.info(
            f"Loading pretrained LORA weights from {lora_weights_name_or_path}"
        )
        model = PeftModel.from_pretrained(model, lora_weights_name_or_path)

    return model, tokenizer

import argparse
import os
from typing import Any, Dict

import torch
import wandb
import yaml
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from utils import PartialMaskTokenizer, reformat_to_chat


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb_project = config["task_type"]
    wandb.init(project=wandb_project, config=config)


def setup_model_and_tokenizer(
    config: Dict[str, Any]
) -> tuple[AutoModelForCausalLM, AutoTokenizer, PartialMaskTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
        cache_dir=os.environ["HF_HOME"],  # FIXME:
    )
    partialMaskTokenizer = PartialMaskTokenizer(
        tokenizer, max_length=config["data"]["max_length"]
    )
    partialMaskTokenizer.format_tokenizer()
    tokenizer = partialMaskTokenizer.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=os.environ["HF_HOME"],
    )
    return model, tokenizer, partialMaskTokenizer


def prepare_dataset(
    config: Dict[str, Any], partialMaskTokenizer: PartialMaskTokenizer
) -> tuple[Dataset, Dataset]:
    dataset = load_from_disk(config["data"]["name"])
    dataset = Dataset.from_dict(
        {
            "messages": reformat_to_chat(
                input=dataset["instruction"], output=dataset["response"]
            )
        }
    )
    dataset = dataset.map(
        partialMaskTokenizer.preprocess,
        fn_kwargs={
            "mask_inputs": True,
            "add_generation_prompt": False,
        },
        batched=True,
    )

    split = dataset.train_test_split(
        test_size=config["data"]["train_test_split"], seed=config["data"]["seed"]
    )
    trainset, evalset = split["train"], split["test"]

    for ds in [trainset, evalset]:
        ds.set_format(columns=["input_ids", "attention_mask", "labels"])

    return trainset, evalset


def calculate_total_steps(dataset, config):
    return int(
        len(dataset)
        * config["training"]["num_epochs"]
        // (
            config["training"]["per_device_train_batch_size"]
            * config["training"]["gradient_accumulation_steps"]
            * config["num_gpus"]
        )
    )


def create_sft_config(config: Dict[str, Any], total_steps: int) -> SFTConfig:
    return SFTConfig(
        output_dir=config["model"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        evaluation_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        logging_strategy=config["training"]["logging_strategy"],
        logging_steps=config["training"]["logging_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        bf16=True,
        report_to="wandb",
        learning_rate=float(config["training"]["optimizer"]["learning_rate"]),
        weight_decay=config["training"]["optimizer"]["weight_decay"],
        lr_scheduler_type=config["training"]["scheduler"]["type"],
        warmup_steps=int(config["training"]["scheduler"]["warmup_ratio"] * total_steps),
        lr_scheduler_kwargs={
            "num_stable_steps": int(
                config["training"]["scheduler"]["stable_ratio"] * total_steps
            ),
            "num_decay_steps": int(
                config["training"]["scheduler"]["decay_ratio"] * total_steps
            ),
            "min_lr_ratio": config["training"]["scheduler"]["min_lr_ratio"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_file)
    setup_wandb(config)
    set_seed(config["data"]["seed"])

    model, tokenizer, partialMaskTokenizer = setup_model_and_tokenizer(config)
    trainset, evalset = prepare_dataset(config, partialMaskTokenizer)

    total_steps = calculate_total_steps(trainset, config)
    sft_config = create_sft_config(config, total_steps)

    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        args=sft_config,
        data_collator=default_data_collator,
    )

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir=config["model"]["output_dir"])
    tokenizer.save_pretrained(config["model"]["output_dir"])


if __name__ == "__main__":
    main()

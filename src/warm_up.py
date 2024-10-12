import argparse
from typing import Any, Dict, Tuple
import os
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
from trl import SFTConfig

from utils import PartialMaskTokenizer, reformat_to_chat, CustomSFTTrainer


def load_config(config_file: str) -> Dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb_project = config["task_type"]
    wandb.init(project=wandb_project, config=config)


def setup_tokenizer_and_model(config: Dict) -> Tuple[Any, Any, PartialMaskTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], cache_dir=os.environ["HF_HOME"]
    )
    partialMaskTokenizer = PartialMaskTokenizer(
        tokenizer, max_length=config["tokenizer_max_length"]
    )
    partialMaskTokenizer.format_tokenizer()
    tokenizer = partialMaskTokenizer.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=os.environ["HF_HOME"],
    )
    return tokenizer, model, partialMaskTokenizer


def prepare_dataset(
    config: Dict, partialMaskTokenizer: PartialMaskTokenizer
) -> Dataset:
    warm_up_dataset = load_from_disk(config["warm_up_dataset_path"])
    warm_up_dataset = warm_up_dataset.add_column(
        "messages",
        reformat_to_chat(
            input=warm_up_dataset["instruction"],
            output=warm_up_dataset["response"],
        ),
    )
    warm_up_dataset = warm_up_dataset.map(
        partialMaskTokenizer.preprocess,
        fn_kwargs={"mask_inputs": True, "add_generation_prompt": False},
        batched=True,
    )
    warm_up_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return warm_up_dataset


def calculate_total_steps(dataset, config):
    return int(
        len(dataset)
        * config["training"]["num_train_epochs"]
        // (
            config["training"]["per_device_train_batch_size"]
            * config["training"]["gradient_accumulation_steps"]
            * config["num_gpus"]
        )
    )


def create_trainer_args(config: Dict, total_steps: int) -> SFTConfig:
    args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
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
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    config = load_config(args.config_file)
    set_seed(42)

    setup_wandb(config)
    tokenizer, model, partialMaskTokenizer = setup_tokenizer_and_model(config)
    warm_up_dataset = prepare_dataset(config, partialMaskTokenizer)

    total_steps = calculate_total_steps(warm_up_dataset, config)
    trainer_args = create_trainer_args(config, total_steps)

    trainer = CustomSFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=warm_up_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    tokenizer.save_pretrained(config["output_dir"])
    trainer.accelerator.save_state(config["output_dir"])


if __name__ == "__main__":
    main()

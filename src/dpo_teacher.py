import argparse
import copy
import os
from pathlib import Path

from typing import Any, Dict

import torch
import wandb
import yaml
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

from utils import (
    PartialMaskTokenizer,
    construct_dpo_dataset,
    print_dataset_info,
)


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def setup_wandb(config: Dict[str, Any]) -> None:
    wandb.init(project=config["task_type"], config=config)


def prepare_tokenizer(config: Dict[str, Any]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    partialTokenizer = PartialMaskTokenizer(tokenizer)
    partialTokenizer.format_tokenizer()
    return partialTokenizer.tokenizer


def prepare_dataset(config: Dict[str, Any], tokenizer: AutoTokenizer) -> Dataset:
    score_dataset = load_from_disk(config["score_dataset_path"])
    score_dataset = score_dataset.to_pandas()
    baseline_ref_loss_path = os.path.join(
        Path(config["dpo_dataset_path"]).parent, "baseline_ref_loss.txt"
    )
    with open(baseline_ref_loss_path, "r") as f_r:
        baseline_ref_loss = float(f_r.read().strip())
    print(f"baseline reference loss: {baseline_ref_loss}")
    dpo_dataset_df = construct_dpo_dataset(score_dataset, baseline_ref_loss)
    dpo_dataset = Dataset.from_pandas(dpo_dataset_df)
    dpo_dataset.save_to_disk(config["dpo_dataset_path"])
    return dpo_dataset


def clean_prompt(prompt: str) -> str:
    prompt = prompt.replace(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>", ""
    )
    prompt = prompt.replace(
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>", ""
    )
    return prompt.strip()


def prepare_train_eval_datasets(
    dpo_dataset: Dataset, config: Dict[str, Any]
) -> tuple[Dataset, Dataset]:
    train_dataset = dpo_dataset.train_test_split(
        test_size=config["train_test_split"], shuffle=True
    )["train"]
    eval_dataset = dpo_dataset.train_test_split(
        test_size=config["train_test_split"], shuffle=True
    )["test"]

    new_prompt = [clean_prompt(p) for p in train_dataset["prompt"]]
    train_dataset = Dataset.from_dict(
        {
            "prompt": new_prompt,
            "chosen": train_dataset["chosen"],
            "rejected": train_dataset["rejected"],
        }
    )

    new_prompt = [clean_prompt(p) for p in eval_dataset["prompt"]]
    eval_dataset = Dataset.from_dict(
        {
            "prompt": new_prompt,
            "chosen": eval_dataset["chosen"],
            "rejected": eval_dataset["rejected"],
        }
    )

    return train_dataset, eval_dataset


def calculate_total_steps(train_dataset: Dataset, config: Dict[str, Any]) -> int:
    return int(
        len(train_dataset)
        * config["training"]["num_train_epochs"]
        // (
            config["training"]["batch_size"]
            * config["training"]["gradient_accumulation_steps"]
        )
    )


def prepare_dpo_config(config: Dict[str, Any], total_steps: int) -> DPOConfig:
    args = DPOConfig(
        output_dir=config["output_dir"],
        max_length=config["tokenizer_max_length"],
        max_prompt_length=config["max_prompt_length"],
        num_train_epochs=config["training"]["num_train_epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        evaluation_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        logging_strategy=config["training"]["logging_strategy"],
        logging_steps=config["training"]["logging_steps"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        bf16=True,
        report_to="wandb",
        beta=config["training"]["beta"],
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
        gradient_checkpointing=True,
    )
    return args


def prepare_model(
    config: Dict[str, Any], tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    config = load_config(args.config_file)
    setup_wandb(config)
    set_seed(42)

    tokenizer = prepare_tokenizer(config)
    dpo_dataset = prepare_dataset(config, tokenizer)
    train_dataset, eval_dataset = prepare_train_eval_datasets(dpo_dataset, config)

    print_dataset_info(train_dataset)
    print_dataset_info(eval_dataset)

    total_steps = calculate_total_steps(train_dataset, config)

    teacher_args = prepare_dpo_config(config, total_steps)
    model = prepare_model(config, tokenizer)
    reference_model = copy.deepcopy(model)

    trainer = DPOTrainer(
        model=model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=teacher_args,
    )

    trainer.train()
    trainer.save_model(output_dir=config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()

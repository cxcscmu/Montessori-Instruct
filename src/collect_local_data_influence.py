from pathlib import Path
import argparse
import random
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from utils import PartialMaskTokenizer, reformat_to_chat


def get_dataloaders(
    accelerator: Accelerator,
    train_args: Dict[str, any],
    eval_args: Dict[str, any],
    tokenizer_path: str,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, cache_dir=os.environ["HF_HOME"]
    )

    partialMaskTokenizer = PartialMaskTokenizer(tokenizer, max_length=1024)
    partialMaskTokenizer.format_tokenizer()
    tokenizer = partialMaskTokenizer.tokenizer

    # get train_dataloader
    raw_train_dataset = load_from_disk(train_args["data_path"])
    train_dataset = Dataset.from_dict(
        {
            "instruction": [
                ins
                for ins in raw_train_dataset["instruction"]
                for _ in range(train_args["num_gpus"])
            ],
            "response": [
                res
                for res in raw_train_dataset["response"]
                for _ in range(train_args["num_gpus"])
            ],
        }
    )
    # accelerator.print("original train dataset:\n")
    # for i in range(len(train_dataset)):
    #     accelerator.print(train_dataset[i])
    # train_dataset = train_dataset.shuffle(seed=42)
    # accelerator.print("train dataset after shuffle:\n")
    # for i in range(len(train_dataset)):
    #     accelerator.print(train_dataset[i])
    train_dataset = Dataset.from_dict(
        {
            "messages": reformat_to_chat(
                input=train_dataset["instruction"], output=train_dataset["response"]
            )
        }
    )
    with accelerator.main_process_first():
        tokenized_train_dataset = train_dataset.map(
            partialMaskTokenizer.preprocess,
            fn_kwargs={"mask_inputs": True, "add_generation_prompt": False},
            batched=True,
        )

    tokenized_train_dataset.set_format("torch")

    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=1, shuffle=False)

    # get eval dataloader
    eval_dataset = load_from_disk(eval_args["eval_dataset_path"])
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_args["eval_nums"]))
    # accelerator.print("size of eval dataset: ", len(eval_dataset))
    # accelerator.print("batch size of eval: ", eval_args["batch_size"])
    eval_dataset = Dataset.from_dict(
        {
            "messages": reformat_to_chat(
                input=eval_dataset["instruction"], output=eval_dataset["response"]
            )
        }
    )
    with accelerator.main_process_first():
        tokenized_eval_dataset = eval_dataset.map(
            partialMaskTokenizer.preprocess,
            fn_kwargs={"mask_inputs": True, "add_generation_prompt": False},
            batched=True,
        )

    tokenized_eval_dataset.set_format("torch")

    eval_dataloader = DataLoader(
        tokenized_eval_dataset, batch_size=eval_args["batch_size"], shuffle=False
    )

    return train_dataloader, eval_dataloader, raw_train_dataset


def one_step_train(
    model: AutoModelForCausalLM,
    optimizer: AdamW,
    accelerator: Accelerator,
    data_encodings: Dict[str, torch.Tensor],
    eval_dataloader: DataLoader,
    step: int,
    avg_loss_before: float,
) -> float:
    model.train()
    optimizer.zero_grad()
    data_encodings = {
        k: v.unsqueeze(0) if v.ndim == 1 else v
        for k, v in data_encodings.items()
        if k in ["input_ids", "attention_mask", "labels"]
    }
    outputs = model(**data_encodings)
    loss = outputs.loss
    loss = loss / accelerator.num_processes
    # if accelerator.is_local_main_process:
    #     print(f"{accelerator.device} loss: {loss:.10f}")
    # accelerator.print("train loss: ", loss.item())
    # print(f"train loss {loss.item()} on device {accelerator.device}")
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    # eval
    model.eval()
    total_loss = 0
    for index, batch in tqdm(
        enumerate(eval_dataloader),
        total=len(eval_dataloader),
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):
        batch = {
            k: v.unsqueeze(0) if v.ndim == 1 else v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        loss = accelerator.gather(loss).mean()
        total_loss += loss.item()
    avg_loss_after = total_loss / len(eval_dataloader)
    accelerator.print(f"eval loss after one step train: {avg_loss_after}")
    accelerator.print(f"eval loss diff: {avg_loss_after - avg_loss_before}")
    return avg_loss_after


def collect_local_data_influence(config: Dict[str, any]) -> None:
    set_seed(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    accelerator = Accelerator(mixed_precision="bf16")
    accelerator.print(
        f"datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True
    )

    raw_model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=os.environ["HF_HOME"],
    )
    model = accelerator.prepare(raw_model)

    train_args, eval_args, tokenizer_path = (
        config["train_args"],
        config["eval_args"],
        config["model_name_or_path"],
    )
    train_dataloader, eval_dataloader, raw_train_dataset = get_dataloaders(
        accelerator, train_args, eval_args, tokenizer_path
    )
    optimizer = AdamW(params=model.parameters())

    optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader
    )

    accelerator.load_state(config["warm_up_path"])

    # obtain baseline reference loss
    model.eval()
    total_loss = 0
    for index, batch in tqdm(
        enumerate(eval_dataloader),
        total=len(eval_dataloader),
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):
        batch = {
            k: v.unsqueeze(0) if v.ndim == 1 else v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        loss = accelerator.gather(loss).mean()
        total_loss += loss.item()
    avg_loss_before = total_loss / len(eval_dataloader)
    accelerator.print(f"baseline reference loss: {avg_loss_before}")
    # Write the baseline reference loss to a file
    baseline_ref_loss_path = os.path.join(
        Path(config["train_args"]["data_path"]).parent, "baseline_ref_loss.txt"
    )
    print(f"baseline reference loss path: {baseline_ref_loss_path}")
    with open(baseline_ref_loss_path, "w") as f:
        f.write(f"{avg_loss_before}")

    score_list = []
    for step, batch in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    ):
        accelerator.print(f"..........train batch: {step}...........")
        t1 = datetime.now()
        loss = one_step_train(
            model, optimizer, accelerator, batch, eval_dataloader, step, avg_loss_before
        )
        # accelerator.print(
        #     f"Epoch {epoch}, Step {step}, Loss on {accelerator.device}: {loss}"
        # )
        score_list.append(loss)
        accelerator.load_state(config["warm_up_path"])
        t2 = datetime.now()
        accelerator.print(f"time taken: {(t2 - t1).total_seconds()}")
    # print(score_list)
    raw_train_dataset = raw_train_dataset.add_column("scores", score_list)
    raw_train_dataset.save_to_disk(config["scores_dataset_path"])
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    with open(args.config_file, "r") as config_f:
        config = yaml.safe_load(config_f)
        collect_local_data_influence(config)

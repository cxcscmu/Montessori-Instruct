import argparse
import json
import os
from typing import Any, Dict, Tuple

import yaml
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from utils import (
    PartialMaskTokenizer,
    print_dataset_info,
)

# Constants
TOKENIZERS_PARALLELISM = "true"

os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as config_f:
        config = yaml.safe_load(config_f)
    return config["gen_responses"]


def setup_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    partialMaskTokenizer = PartialMaskTokenizer(tokenizer)
    partialMaskTokenizer.format_tokenizer()
    return partialMaskTokenizer.tokenizer


def setup_model_and_sample_params(
    config: Dict[str, Any], tokenizer: AutoTokenizer
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sample_params = {
        "n": config["n"],
        "best_of": config["best_of"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "presence_penalty": config["presence_penalty"],
        "skip_special_tokens": False,
        "stop": tokenizer.eos_token,
    }

    model_params = {
        "model": config["model_name_or_path"],
        "dtype": config["dtype"],
        "trust_remote_code": True,
        "max_model_len": config["max_model_len"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "seed": 42,
        "download_dir": os.environ["HF_HOME"],
        "swap_space": config["swap_space"],
    }

    return model_params, sample_params


def gen_responses(config: Dict[str, Any], tokenizer: AutoTokenizer) -> Dataset:
    seed_inst_dataset = load_from_disk(config["instruction_dataset_path"])
    if config["num_ins"] < len(seed_inst_dataset):
        seed_inst_dataset = seed_inst_dataset.select(range(config["num_ins"]))

    formatted_inst = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            tokenize=True,
        )
        for inst in seed_inst_dataset["instruction"]
    ]

    model_params, sample_params = setup_model_and_sample_params(config, tokenizer)

    sampling_params = SamplingParams(**sample_params)
    llm = LLM(**model_params)
    progress_bar = tqdm(total=len(formatted_inst))

    return_response = []
    batch_size = config["batch_size"]
    for start_index in range(0, len(formatted_inst), batch_size):
        end_index = start_index + batch_size
        batch = [
            TokensPrompt(prompt_token_ids=prompt)
            for prompt in formatted_inst[start_index:end_index]
        ]

        outputs = llm.generate(batch, sampling_params)
        for output in outputs:
            for generated_output in output.outputs:
                return_response.append(generated_output.text)
            progress_bar.update(1)

    inst_resp_dataset = {
        "seed_prompt": seed_inst_dataset["prompt"],
        "raw_inst": seed_inst_dataset["raw_inst"],
        "instruction": seed_inst_dataset["instruction"],
        "response": return_response,
    }

    return Dataset.from_dict(inst_resp_dataset)


def save_dataset(dataset: Dataset, config: Dict[str, Any]) -> None:
    with open(config["jsonl_save_path"], "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(
                json.dumps(
                    {"instruction": item["instruction"], "response": item["response"]}
                )
            )
            f.write("\n")

    print_dataset_info(dataset)
    dataset.save_to_disk(config["generated_responses_save_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    config = load_config(args.config_file)
    tokenizer = setup_tokenizer(config["model_name_or_path"])

    dataset = gen_responses(config, tokenizer)
    save_dataset(dataset, config)


if __name__ == "__main__":
    main()

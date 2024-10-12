import argparse
import json
import multiprocessing
import os
import random
from functools import partial
from typing import Any, Dict, List, Tuple

import yaml
from datasets import Dataset, load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from prompt import instruction_gen_prompt
from utils import (
    PartialMaskTokenizer,
    extract_instruction_from_raw,
    post_process,
    reformat_to_chat,
)

# Constants
TOKENIZERS_PARALLELISM = "true"
ROUGE_THRESHOLD = 0.7

os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as config_f:
        config = yaml.safe_load(config_f)
    return config["gen_instruction"]


def setup_tokenizer(config: Dict[str, Any]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], use_fast=False
    )
    partialMaskTokenizer = PartialMaskTokenizer(tokenizer)
    partialMaskTokenizer.format_tokenizer()
    return partialMaskTokenizer.tokenizer


def setup_llm(
    config: Dict[str, Any], tokenizer: AutoTokenizer
) -> Tuple[LLM, SamplingParams]:
    sample_params = SamplingParams(
        n=config["n"],
        best_of=config["best_of"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        presence_penalty=config["presence_penalty"],
        max_tokens=config["max_tokens"],
        skip_special_tokens=False,
        stop=tokenizer.eos_token,
    )

    model_params = {
        "model": config["model_name_or_path"],
        "dtype": config["dtype"],
        "trust_remote_code": True,
        "max_model_len": config["max_model_len"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "seed": 42,
        "download_dir": os.environ["HF_HOME"],
    }

    return LLM(**model_params), sample_params


def load_existing_instructions(config: Dict[str, Any]) -> List[str]:
    """
    load existing instructions from jsonl file (in case of being interrupted halfway)
    """
    instructions = []
    os.makedirs(os.path.dirname(config["jsonl_save_path"]), exist_ok=True)
    if os.path.exists(config["jsonl_save_path"]):
        with open(config["jsonl_save_path"], "r", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                instructions.append(data["instruction"])
    return instructions


def generate_instructions(
    config: Dict[str, Any],
    llm: LLM,
    sample_params: SamplingParams,
    tokenizer: AutoTokenizer,
    seed_dataset: Dataset,
    instructions: List[str],
    progress_bar: tqdm,
    scorer: rouge_scorer.RougeScorer,
) -> None:
    random_seed = len(instructions)
    duplicate_count = 0
    with open(config["jsonl_save_path"], "a", encoding="utf-8") as fout:
        while len(instructions) < config["num_ins"]:
            batch, used_seed_instruction = prepare_batch(
                config, seed_dataset, random_seed, instructions, tokenizer
            )
            random_seed += config["batch_size"]

            outputs = llm.generate(batch, sample_params)

            duplicate_count = process_outputs(
                outputs,
                tokenizer,
                instructions,
                used_seed_instruction,
                scorer,
                fout,
                progress_bar,
                duplicate_count,
            )
            print(f"Duplicate instructions count: {duplicate_count}")


def make_prompt(
    seed_dataset: Dataset,
    random_seed: int,
    new_instructions: List[str],
    seed_num: int = 8,
) -> Tuple[str, List[str]]:
    """
    Generate a prompt for instruction generation; few-shot examples randomly selected from seed data and previous instructions.
    """
    used_seed_data = []
    prompt = instruction_gen_prompt
    if len(new_instructions) < 2:
        random_choose_from_dataset = seed_dataset.shuffle(random_seed).select(
            range(seed_num)
        )
        random_choose_list = [
            seed_dataset_entry["instruction"]
            + (
                " " + seed_dataset_entry["input"]
                if "input" in seed_dataset.features and seed_dataset_entry["input"]
                else ""
            )
            for seed_dataset_entry in random_choose_from_dataset
        ]
    else:
        random.seed = random_seed
        random_choose_from_dataset = seed_dataset.shuffle(random_seed).select(
            range(seed_num - 2)
        )
        random_choose_list = [
            seed_dataset_entry["instruction"]
            + (
                " " + seed_dataset_entry["input"]
                if "input" in seed_dataset.features and seed_dataset_entry["input"]
                else ""
            )
            for seed_dataset_entry in random_choose_from_dataset
        ]
        random_choose_list += random.sample(new_instructions, 2)
    random.shuffle(random_choose_list)
    for i in range(len(random_choose_list)):
        instruction = random_choose_list[i]
        prompt += f"<instruction>{instruction}</instruction>\n"
        used_seed_data += [instruction]
    return prompt, used_seed_data


def prepare_batch(
    config: Dict[str, Any],
    seed_dataset: Dataset,
    random_seed: int,
    instructions: List[str],
    tokenizer: AutoTokenizer,
) -> Tuple[List[TokensPrompt], List[str]]:
    batch = []
    used_seed_instruction = []
    for _ in range(config["batch_size"]):
        prompt, used_seed_inst = make_prompt(
            seed_dataset, random_seed, instructions, config["seed_data_num"]
        )
        random_seed += 1
        prompt = reformat_to_chat(input=[prompt], output=None)[0]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation=prompt, add_generation_prompt=True, tokenize=True
        )
        token_prompt = TokensPrompt(prompt_token_ids=formatted_prompt)
        batch.append(token_prompt)
        used_seed_instruction.extend(used_seed_inst)
    return batch, used_seed_instruction


def process_outputs(
    outputs: List[Any],
    tokenizer: AutoTokenizer,
    instructions: List[str],
    used_seed_instruction: List[str],
    scorer: rouge_scorer.RougeScorer,
    fout: Any,
    progress_bar: tqdm,
    duplicate_count: int,
) -> int:
    for output in outputs:
        prompt = tokenizer.decode(output.prompt_token_ids, skip_special_tokens=False)
        for entry in output.outputs:
            text = entry.text
            inst = extract_and_process_instruction(text)
            if inst is None:
                continue

            if is_similar_instruction(
                inst, instructions, used_seed_instruction, scorer
            ):
                duplicate_count += 1
                continue

            save_instruction(inst, prompt, text, fout)
            instructions.append(inst)
            progress_bar.update(1)
    return duplicate_count


def extract_and_process_instruction(text: str) -> str:
    inst = extract_instruction_from_raw(text)
    if inst is None:
        print("Invalid format, skip.")
        return None
    inst = post_process(inst)
    if inst is None:
        print("One more invalid instruction.")
        return None
    return inst


def is_similar_instruction(
    inst: str,
    instructions: List[str],
    used_seed_instruction: List[str],
    scorer: rouge_scorer.RougeScorer,
) -> bool:
    inspection_inst = instructions + used_seed_instruction
    with multiprocessing.Pool(4) as p:
        rouge_scores = p.map(
            partial(scorer.score, inst),
            inspection_inst,
        )
    rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
    if max(rouge_scores) > ROUGE_THRESHOLD:
        # print(f"Find one similar instruction:")
        # print(f"Rouge score: {max(rouge_scores)}")
        return True
    return False


def save_instruction(inst: str, prompt: str, text: str, fout: Any) -> None:
    fout.write(
        json.dumps({"prompt": prompt, "instruction": inst, "raw_inst": text}) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    config = load_config(args.config_file)
    tokenizer = setup_tokenizer(config)
    llm, sample_params = setup_llm(config, tokenizer)

    progress_bar = tqdm(total=config["num_ins"])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    instructions = load_existing_instructions(config)
    progress_bar.update(len(instructions))
    print(f"Loaded {len(instructions)} existing instructions.")

    seed_dataset = load_dataset(
        config["seed_data_path"], split=config["split"]
    ).shuffle(seed=42)

    generate_instructions(
        config,
        llm,
        sample_params,
        tokenizer,
        seed_dataset,
        instructions,
        progress_bar,
        scorer,
    )

    dataset = Dataset.from_json(config["jsonl_save_path"])
    dataset.save_to_disk(config["generated_instructions_save_path"])


if __name__ == "__main__":
    main()

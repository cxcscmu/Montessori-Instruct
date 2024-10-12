import datetime
import re
import string
from contextlib import contextmanager
from typing import Dict, List

import pandas as pd
from transformers.trainer_pt_utils import LabelSmoother
from trl import SFTTrainer
from transformers.utils.import_utils import is_sagemaker_mp_enabled

IGNORE_INDEX = LabelSmoother.ignore_index


def print_dataset_info(dataset):
    print(f"Length of dataset: {len(dataset)}")
    print(f"Columns of dataset: {dataset.features}")
    print(f"First entry in dataset: {dataset[0]}")


def reformat_to_chat(input, output) -> List[Dict[str, str]]:
    """
    input: List[str] eg: ["How are you?"]
    output: List[str] eg: ["I'm doing well, how about you?"]

    Return: List[List[Dict[str, str]]] eg: [[{"role":"user", "content": "How are you?"}, {"role":"assistant", "content": "I'm doing well, how about you?"}]]
    """
    chat_data = []
    if output is None:
        for input_msg in input:
            chat_single = []
            chat_single.append({"role": "user", "content": input_msg})
            chat_data.append(chat_single)
    else:
        for user_msg, assistant_msg in zip(input, output):
            chat_single = []
            chat_single.append({"role": "user", "content": user_msg})
            chat_single.append({"role": "assistant", "content": assistant_msg})
            chat_data.append(chat_single)
    return chat_data


"""
This class wraps a tokenizer and provides functionality to:
1. Formats the tokenizer with a specific chat template;
2. Supports partial masking of input tokens, allowing for targeted loss computation.
"""


class PartialMaskTokenizer:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else 1024

    def format_tokenizer(self):
        """
        Use llama3 chat format
        """
        chat_template = open("./configs/chat_template.jinja").read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.tokenizer.chat_template = chat_template

        if self.tokenizer.bos_token == None:
            print("add bos token to the tokenizer")
            self.tokenizer.add_special_tokens({"bos_token": "<|begin_of_text|>"})
        if self.tokenizer.eos_token == None:
            print("add eos token to the tokenizer")
            self.tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(
        self,
        examples,
        mask_inputs=True,
        add_generation_prompt=True,
    ):
        """
        The map function;
        Need to have a column named "messages" in the dataset,
        representing the conversation between the user and the assistant.
        """

        def get_assistant_start_end_indices(messages, conversation_text):
            start_indices = []
            current_index = 0
            for message in messages:
                message_text = message["content"]
                match_index = conversation_text[current_index:].find(message_text)
                start_indices.append(current_index + match_index)
                current_index += match_index + len(message_text)
            end_indices = [
                (
                    len(conversation_text)
                    if i == len(start_indices) - 1
                    else start_indices[i + 1]
                )
                for i, x in enumerate(start_indices)
            ]
            roles = [message["role"] for message in messages]
            return [
                (s, e)
                for s, e, r in zip(start_indices, end_indices, roles)
                if r == "assistant"
            ]

        def get_masked_labels(conversation_ids, assistant_ranges):
            for id_, (id_s, id_e) in list(
                zip(
                    conversation_ids["input_ids"],
                    conversation_ids["offset_mapping"],
                ),
            ):
                if any(id_s >= s and id_e <= e for s, e in assistant_ranges):
                    yield id_
                else:
                    yield IGNORE_INDEX

        def get_masked_labels_only_on_attention_mask(conversation_ids):
            for id_, id_m in list(
                zip(
                    conversation_ids["input_ids"],
                    conversation_ids["attention_mask"],
                ),
            ):
                if id_m == 1:
                    yield id_
                else:
                    yield IGNORE_INDEX

        def tokenize_messages(
            messages,
            tokenizer,
            max_length=1024,
            mask_inputs=True,
            add_generation_prompt=True,
        ):
            conversation_text = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            conversation_ids = tokenizer(
                conversation_text,
                return_offsets_mapping=mask_inputs,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )
            if mask_inputs:
                assistant_ranges = get_assistant_start_end_indices(
                    messages, conversation_text
                )
                labels = get_masked_labels(conversation_ids, assistant_ranges)
                conversation_ids["labels"] = list(labels)
                del conversation_ids["offset_mapping"]
            else:
                labels = get_masked_labels_only_on_attention_mask(conversation_ids)
                conversation_ids["labels"] = list(labels)

            conversation_ids["input_texts"] = tokenizer.decode(
                conversation_ids["input_ids"], skip_special_tokens=False
            )
            return conversation_ids

        encodings_list = []
        for message in examples["messages"]:
            encodings = tokenize_messages(
                message,
                self.tokenizer,
                max_length=self.max_length,
                mask_inputs=mask_inputs,
                add_generation_prompt=add_generation_prompt,
            )
            encodings_list.append(encodings)
        return {
            "input_ids": [e["input_ids"] for e in encodings_list],
            "attention_mask": [e["attention_mask"] for e in encodings_list],
            "labels": [e["labels"] for e in encodings_list],
            # "input_texts": [e["input_texts"] for e in encodings_list], # reverse input_ids to human-readable text for format inspection
        }


def post_process(inst):
    """
    Post-process instructions to filter out invalid ones
    """

    def find_word_in_string(w, s):
        return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

    inst = re.sub(r"\s+", " ", inst).strip()
    inst = inst.strip().capitalize()
    if inst == "":
        print("empty instruction")
        return None
    # filter out too short or too long instructions
    if len(inst) <= 5 or len(inst) > 400:
        print("too long or too short instruction")
        print(len(inst))
        return None
    # filter based on keywords that are not suitable for language models.
    if any(
        find_word_in_string(word, inst)
        for word in [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
        ]
    ):
        print("contains unsuitable keywords")
        return None
    # filter those starting with punctuation
    if inst[0] in string.punctuation:
        print("starts with punctuation")
        return None
    # filter those starting with non-english character
    if not inst[0].isascii():
        print("starts with non-english character")
        return None
    # remove the "<input>" label or "</input>" label in the inst, leave the rest
    inst = re.sub(r"<input>|\s*<input>\s*|\s*</input>\s*|</input>", "", inst)
    return inst


def extract_instruction_from_raw(raw_data):
    """
    raw_data is a string. Find the content between <instruction> and </instruction>. If not found, return None.
    """

    pattern = r"<instruction>(.*?)</instruction>"
    match = re.findall(pattern, raw_data, re.DOTALL)
    if match:
        return match[-1].strip()
    else:
        return None


@contextmanager
def time_wrapper(label):
    t_start = datetime.datetime.now()
    yield
    t_end = datetime.datetime.now()
    print(
        f"[Time Statistics] ##{label}## : {(t_end-t_start).total_seconds():.3f}s.",
        flush=True,
    )


def construct_dpo_dataset(df: pd.DataFrame, baseline_ref_loss: float) -> pd.DataFrame:
    """
    Construct a DPO dataset for optimizing the teacher model.
    """
    preference_data = []

    for prompt in df["seed_prompt"].unique():
        prompt_records = df[df["seed_prompt"] == prompt]

        lower_scores = prompt_records[prompt_records["scores"] < baseline_ref_loss]
        higher_scores = prompt_records[prompt_records["scores"] > baseline_ref_loss]

        if lower_scores.empty or higher_scores.empty:
            continue

        for _, lower_record in lower_scores.iterrows():
            for _, higher_record in higher_scores.iterrows():
                entry = {
                    "prompt": prompt,
                    "chosen": lower_record["raw_inst"],
                    "rejected": higher_record["raw_inst"],
                }
                preference_data.append(entry)

    return pd.DataFrame(preference_data)


class CustomSFTTrainer(SFTTrainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            if self.args.weight_decay > 0.0:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.requires_grad
                        ],
                        "weight_decay": 0.0,
                    }
                ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

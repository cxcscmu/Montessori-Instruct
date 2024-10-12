from datasets import load_from_disk
import yaml
import argparse


def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()

config = load_yaml(args.config_file)

root_data_folder = config["root_data_folder"]
dataset = load_from_disk(config["gen_responses"]["generated_responses_save_path"])

warmup_data_num = config["warmup_data_num"]
probing_data_num = config["probing_data_num"]

warmup_dataset = dataset.select(range(warmup_data_num))
probing_dataset = dataset.select(
    range(warmup_data_num, warmup_data_num + probing_data_num)
)

warmup_dataset.save_to_disk(f"{root_data_folder}/warm_up_dataset")
probing_dataset.save_to_disk(f"{root_data_folder}/probing_dataset")

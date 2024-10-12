import os
import yaml
from pathlib import Path
import shutil


def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, file_path):
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# Load project config
project_config = load_yaml("./project_config.yml")

HOME = project_config["project_name"]
TEACHER_MODEL = project_config["teacher_model_name_or_path"]
STUDENT_MODEL = project_config["student_model_name_or_path"]
TRAINING_DATASET_NUM = project_config["training_dataset_num"]
PROBING_DATASET_NUM = project_config["probing_dataset_num"]
WARMUP_DATASET_NUM = project_config["warmup_dataset_num"]
MULTI_GPU_CONFIG = project_config["fsdp_or_ddp"]
GPUS = project_config["num_gpus"]

# Create necessary directories
for dir_name in ["configs", "logs", "data", "scripts"]:
    os.makedirs(f"./{dir_name}/{HOME}", exist_ok=True)

# Load and fill template configs
template_dir = Path("./configs/templates")
output_dir = Path(f"./configs/{HOME}")

for template_file in template_dir.glob("*.yml"):
    config = load_yaml(template_file)

    if template_file.name == "warm_up.yml":
        config["model_name_or_path"] = STUDENT_MODEL
        config["output_dir"] = f"./data/{HOME}/models/warm_up_checkpoint"
        config["warm_up_dataset_path"] = f"./data/{HOME}/warm_up_dataset"
        config["num_gpus"] = GPUS

    elif template_file.name == "dpo_teacher.yml":
        config["model_name_or_path"] = TEACHER_MODEL
        config["score_dataset_path"] = f"./data/{HOME}/score_dataset"
        config["dpo_dataset_path"] = f"./data/{HOME}/dpo_dataset"
        config["output_dir"] = f"./data/{HOME}/models/updated_teacher"
        config["num_gpus"] = GPUS

    elif template_file.name == "sft_student.yml":
        config["model"]["name"] = STUDENT_MODEL
        config["model"]["output_dir"] = f"./data/{HOME}/models/sft_student"
        config["data"]["name"] = f"./data/{HOME}/training_dataset"
        config["num_gpus"] = GPUS

    elif template_file.name == "gen_training_dataset.yml":
        config["gen_instruction"][
            "model_name_or_path"
        ] = f"./data/{HOME}/models/updated_teacher"
        config["gen_instruction"]["num_ins"] = TRAINING_DATASET_NUM
        config["gen_instruction"][
            "jsonl_save_path"
        ] = f"./data/{HOME}/jsonl/training_generated_instructions.jsonl"
        config["gen_instruction"][
            "generated_instructions_save_path"
        ] = f"./data/{HOME}/training_generated_instructions"

        config["gen_responses"][
            "model_name_or_path"
        ] = f"./data/{HOME}/models/updated_teacher"
        config["gen_responses"][
            "instruction_dataset_path"
        ] = f"./data/{HOME}/training_generated_instructions"
        config["gen_responses"]["num_ins"] = TRAINING_DATASET_NUM
        config["gen_responses"][
            "jsonl_save_path"
        ] = f"./data/{HOME}/jsonl/training_generated_responses.jsonl"
        config["gen_responses"][
            "generated_responses_save_path"
        ] = f"./data/{HOME}/training_dataset"
        config["gen_responses"]["num_gpus"] = GPUS

    elif template_file.name == "collect_local_data_influence.yml":
        config["model_name_or_path"] = STUDENT_MODEL
        config["warm_up_path"] = f"./data/{HOME}/models/warm_up_checkpoint"
        config["train_args"]["data_path"] = f"./data/{HOME}/probing_dataset"
        config["train_args"]["num_gpus"] = GPUS
        config["scores_dataset_path"] = f"./data/{HOME}/score_dataset"

    elif template_file.name == "gen_warmup_and_probing_dataset.yml":
        config["root_data_folder"] = f"./data/{HOME}"
        config["warmup_data_num"] = WARMUP_DATASET_NUM
        config["probing_data_num"] = PROBING_DATASET_NUM

        config["gen_instruction"]["model_name_or_path"] = TEACHER_MODEL
        config["gen_instruction"]["num_ins"] = WARMUP_DATASET_NUM + PROBING_DATASET_NUM
        config["gen_instruction"][
            "jsonl_save_path"
        ] = f"./data/{HOME}/jsonl/warmup_probing_generated_instructions.jsonl"
        config["gen_instruction"][
            "generated_instructions_save_path"
        ] = f"./data/{HOME}/warmup_probing_generated_instructions"

        config["gen_responses"]["model_name_or_path"] = TEACHER_MODEL
        config["gen_responses"][
            "instruction_dataset_path"
        ] = f"./data/{HOME}/warmup_probing_generated_instructions"
        config["gen_responses"]["num_ins"] = WARMUP_DATASET_NUM + PROBING_DATASET_NUM
        config["gen_responses"][
            "jsonl_save_path"
        ] = f"./data/{HOME}/jsonl/generated_responses_warmup_probing.jsonl"
        config["gen_responses"][
            "generated_responses_save_path"
        ] = f"./data/{HOME}/warmup_and_probing_dataset"
    # Save the filled config
    save_yaml(config, f"./configs/{HOME}/{template_file.name}")

print(
    f"Done! Configurations has been generated for project '{HOME}' in ./configs/{HOME}/"
)


def generate_project_scripts(project_name):
    script_templates = [
        "dpo_teacher.sh",
        "train_student.sh",
        "gen_training_dataset.sh",
        "collect_data_influence.sh",
        "warm_up.sh",
        "gen_warmup_probing_dataset.sh",
    ]

    scripts_dir = Path(f"./scripts/{project_name}")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for script in script_templates:
        source = Path(f"./scripts/templates/{script}")
        destination = f"{scripts_dir}/{script}"
        shutil.copy(source, destination)

        # Replace placeholders in the copied script
        with open(destination, "r") as f:
            content = f.read()

        content = content.replace("{project_name}", project_name)
        content = content.replace(
            "{multi_gpu_config_file}", project_config["fsdp_or_ddp"]
        )

        with open(destination, "w") as f:
            f.write(content)

    # Create run_all.sh
    run_all_content = f"""#!/bin/bash

# Set the project name
PROJECT_NAME="{project_name}"

# Function to run a script and handle errors
run_script() {{
    script_name="$1"
    echo "Running $script_name..."
    if bash "./scripts/$PROJECT_NAME/$script_name"; then
        echo "$script_name completed successfully."
    else
        echo "Error: $script_name failed. Exiting."
        exit 1
    fi
    echo
}}

# Run each script in the desired order
run_script "gen_warmup_probing_dataset.sh"
run_script "warm_up.sh"
run_script "collect_data_influence.sh"
run_script "dpo_teacher.sh"
run_script "gen_training_dataset.sh"
run_script "train_student.sh"

echo "All scripts have been executed successfully!"
"""

    run_all_path = scripts_dir / "run_all.sh"
    with open(run_all_path, "w") as f:
        f.write(run_all_content)

    # Make run_all.sh executable
    os.chmod(run_all_path, 0o755)

    print(f"Created and made executable: {run_all_path}")


generate_project_scripts(HOME)

print(
    f"Done! Scripts and run_all.sh have been generated for project '{HOME}' in ./scripts/{HOME}/"
)

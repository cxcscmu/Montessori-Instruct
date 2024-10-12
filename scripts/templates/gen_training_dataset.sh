#! /bin/bash

python ./src/gen_instructions.py --config_file ./configs/{project_name}/gen_training_dataset.yml > ./logs/{project_name}/gen_training_instructions.log 2>&1

if [ $? -eq 0 ]; then
    python ./src/gen_responses.py --config_file ./configs/{project_name}/gen_training_dataset.yml > ./logs/{project_name}/gen_training_responses.log 2>&1
else
    echo "Gen training dataset error: gen_instructions.py failed. Skipping gen_responses.py."
    exit 1
fi
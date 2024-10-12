#! /bin/bash

python ./src/gen_instructions.py --config_file ./configs/{project_name}/gen_warmup_and_probing_dataset.yml > ./logs/{project_name}/gen_warmup_probing_instructions.log 2>&1

if [ $? -eq 0 ]; then
    python ./src/gen_responses.py --config_file ./configs/{project_name}/gen_warmup_and_probing_dataset.yml > ./logs/{project_name}/gen_warmup_probing_responses.log 2>&1
else
    echo "Generate instructions error: gen_instructions.py failed. Skipping gen_responses.py."
    exit 1
fi

python ./src/divide_dataset.py --config_file ./configs/{project_name}/gen_warmup_and_probing_dataset.yml > ./logs/{project_name}/divide_dataset.log 2>&1
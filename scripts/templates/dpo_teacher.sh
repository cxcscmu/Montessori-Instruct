#! /bin/bash

python ./src/dpo_teacher.py --config_file ./configs/{project_name}/dpo_teacher.yml > ./logs/{project_name}/dpo_teacher.log 2>&1

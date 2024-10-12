#! /bin/bash

accelerate launch --config_file {multi_gpu_config_file} \
./src/sft_student.py --config_file ./configs/{project_name}/sft_student.yml \
> ./logs/{project_name}/sft_student.log 2>&1

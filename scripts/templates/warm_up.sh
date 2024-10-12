#! /bin/bash

accelerate launch --config_file {multi_gpu_config_file} ./src/warm_up.py \
--config_file ./configs/{project_name}/warm_up.yml \
> ./logs/{project_name}/warm_up.log 2>&1

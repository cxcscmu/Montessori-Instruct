#! /bin/bash

accelerate launch --config_file {multi_gpu_config_file} \
./src/collect_local_data_influence.py --config_file ./configs/{project_name}/collect_local_data_influence.yml \
1> ./logs/{project_name}/collect_data_influence.log \
2> ./logs/{project_name}/collect_data_influence_err.log

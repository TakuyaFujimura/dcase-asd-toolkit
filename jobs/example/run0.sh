#!/bin/bash

name="example"
version="dcase2023_baseline"
seed=0
gpu=0
infer_ver_list="epoch_12,epoch_14,epoch_16"
venv_dir="../.." # relative path from base.sh or absolute path

bash ./base.sh ${name} ${version} ${seed} ${gpu} ${infer_ver_list} ${venv_dir}


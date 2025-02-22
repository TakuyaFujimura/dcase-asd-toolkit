#!/bin/bash

name="example"
version="dcase2023_dis_baseline"
# Name of configuration file to use
# dcase-asd-library/config/train/experiments/${name}/${version}.yaml will be used

test_exp_yaml="dcase2023_dis_baseline"
# Name of configuration file to use for testing
# dcase-asd-library/config/test/experiments/${test_exp_yaml}.yaml will be used

seed=0
# Seed for random number generator

gpu=0
# ID of GPU to use

infer_ver_list="epoch_12,epoch_16"
# comma separated list of versions to inference. 
# Available versions: epoch_??, best, last
# epoch_??: Use model at the epoch (there must be a checkpoint file at the epoch)
# best: Use the model with the best loss (automatically selected)
# last: Use the model with the last epoch

bash ../base/base.sh ${name} ${version} ${test_exp_yaml} ${seed} ${gpu} ${infer_ver_list}


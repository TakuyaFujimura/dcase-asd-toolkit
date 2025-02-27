#!/bin/bash
########################
name="example"
version="dcase2023_dis_baseline"
seed=0
gpu=0
########################



cd ../../..

source "venv/bin/activate"

python -m asdlib.bin.train experiments="${name}/${version}" \
'name='${name}'' 'version='${version}_${seed}'' \
'trainer.devices='"[${gpu}]"'' 'seed='${seed}''


# for infer_ver in "${infer_ver_list[@]}"
# do
# 	python asdlib/bin/test.py experiments="${test_exp_yaml}" \
# 	'name='${name}'' 'version='${version}'' 'machine='${machine}'' \
# 	'infer_ver='${infer_ver}'' 'seed='${seed}'' 'device='"cuda:${gpu}"''
# done

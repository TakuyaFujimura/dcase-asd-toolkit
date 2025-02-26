#!/bin/bash
########################
name=$1
version=$2
machine=$3
test_exp_yaml=$4
seed=$5
gpu=$6

IFS=',' read -r -a infer_ver_list <<< "$7"
exp_yaml="${name}/${version}"
version="${version}_${seed}"
################################

cd ../../..

source "venv/bin/activate"

python asdlib/bin/train.py experiments="${exp_yaml}" \
'name='${name}'' 'version='${version}'' 'machine='${machine}'' \
'trainer.devices='"[${gpu}]"'' 'seed='${seed}''


for infer_ver in "${infer_ver_list[@]}"
do
	python asdlib/bin/test.py experiments="${test_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'machine='${machine}'' \
	'infer_ver='${infer_ver}'' 'seed='${seed}'' 'device='"cuda:${gpu}"''
done

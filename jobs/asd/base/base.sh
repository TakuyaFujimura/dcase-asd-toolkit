#!/bin/bash
########################
name=$1
version=$2
test_exp_yaml=$3
seed=$4
gpu=$5

IFS=',' read -r -a infer_ver_list <<< "$6"
exp_yaml="${name}/${version}"
version="${version}_${seed}"
################################

cd ../../..

source "venv/bin/activate"

python asdlib/bin/train.py experiments="${exp_yaml}" \
'name='${name}'' 'version='${version}'' \
'trainer.devices='"[${gpu}]"'' 'seed='${seed}''


for infer_ver in "${infer_ver_list[@]}"
do
	python asdlib/bin/test.py experiments="${test_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'infer_ver='${infer_ver}'' \
	'seed='${seed}'' 'device='"cuda:${gpu}"''
done

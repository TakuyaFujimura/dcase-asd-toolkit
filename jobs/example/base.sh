#!/bin/bash
########################
name=$1
version=$2
seed=$3
gpu=$4
IFS=',' read -r -a infer_ver_list <<< "$5"
venv_dir=$6
exp_yaml="${name}/${version}"
version="${version}_${seed}"
################################

source "${venv_dir}/venv/bin/activate"

cd ../..

python train.py experiments="${exp_yaml}" \
'name='${name}'' 'version='${version}'' \
'trainer.devices='"[${gpu}]"'' 'seed='${seed}''


for infer_ver in "${infer_ver_list[@]}"
do
	python test.py 'name='${name}'' 'version='${version}'' 'infer_ver='${infer_ver}'' \
	'seed='${seed}'' 'device='"cuda:${gpu}"''
done

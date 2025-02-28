#!/bin/bash
########################
name="example"
version="dcase2023_dis_baseline"
extract_exp_yaml="dis_baseline"
seed=0
########################


cd ../../..

source "venv/bin/activate"

python -m asdlib.bin.train experiments="${name}/${version}" 'seed='${seed}'' \
'name='${name}'' 'version='${version}''

for machine in "fan" "valve"; do
	python -m asdlib.bin.extract experiments="${extract_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}''
done

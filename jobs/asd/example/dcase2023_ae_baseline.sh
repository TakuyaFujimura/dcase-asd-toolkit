#!/bin/bash
########################
name="example"
version="dcase2023_ae_baseline"
extract_exp_yaml="ae_baseline"
score_exp_yaml="ae_baseline"
evaluate_exp_yaml="dcase2023"
ckpt_ver="last"
seed=0
########################


cd ../../..

source "venv/bin/activate"

for machine in "fan" "valve"; do

    python -m asdlib.bin.train experiments="${name}/${version}" 'seed='${seed}'' \
    'name='${name}'' 'version='${version}'' 'machine='${machine}''


	python -m asdlib.bin.extract experiments="${extract_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
    'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''

    python -m asdlib.bin.score experiments="${score_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''

    python -m asdlib.bin.evaluate experiments="${evaluate_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''
done

# TODO: Add more machines
# TODO: Add evaluation of all machines

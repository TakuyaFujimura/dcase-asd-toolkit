#!/bin/bash
########################
name="example"
version="dcase2023_dis_baseline"
extract_exp_yaml="dis_baseline"
score_exp_yaml="dis_baseline"
evaluate_exp_yaml="dcase2023"
umap_exp_yaml="cosine"
ckpt_ver="last"
seed=0
########################


cd ../../..

source "venv/bin/activate"

python -m asdlib.bin.train experiments="${name}/${version}" 'seed='${seed}'' \
'name='${name}'' 'version='${version}''

for machine in "fan" "valve"; do
	python -m asdlib.bin.extract experiments="${extract_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''

	python -m asdlib.bin.score experiments="${score_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''

	python -m asdlib.bin.evaluate experiments="${evaluate_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''

	python -m asdlib.bin.umap experiments="${umap_exp_yaml}" \
	'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
	'machine='${machine}'' 'ckpt_ver='${ckpt_ver}''
done

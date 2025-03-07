#!/bin/bash
# ----------------------------------------------- #
dcase="dcase2023"
name="example"
version="${dcase}_ae_baseline"
seed=0
extract_exp="machinewise"
score_exp="no_backend"
evaluate_exp="default"
umap_exp="euclid"
table_exp="default"
ckpt_ver_list="last"
# ----------------------------------------------- #
bash ../base/base.sh ${name} ${version} ${dcase} ${seed} ${extract_exp} ${score_exp} ${evaluate_exp} ${umap_exp} ${table_exp} ${ckpt_ver_list}

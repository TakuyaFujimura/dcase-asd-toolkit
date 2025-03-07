#!/bin/bash
# ----------------------------------------------- #
dcase="dcase2023"
name="example"
version="${dcase}_dis_baseline"
seed=0
extract_exp="shared"
score_exp="dis_baseline"
evaluate_exp="default"
umap_exp="cosine"
ckpt_ver_list="epoch_12,epoch_16"
# ----------------------------------------------- #
bash ../base/base.sh ${name} ${version} ${dcase} ${seed} ${extract_exp} ${score_exp} ${evaluate_exp} ${umap_exp} ${ckpt_ver_list}

#!/bin/bash

# This is a recipe for ASD system using BEATs features
# This directly uses the extracted BEATs features and does not require training process
# https://arxiv.org/abs/2409.05035

# ---------------------------- #
dcase="dcase2023"
seed="0"
name="recipe"
version="beats"
infer_ver="last"
metric="euclid"
# ---------------------------- #
# experiments_train=""
experiments_extract="scratch_${version}"
experiments_score="baseline"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap metric="${metric}"
done
asdit_table

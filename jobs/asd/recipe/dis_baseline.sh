#!/bin/bash

# This is a recipe for a discriminative ASD system
# Frontend models are shared for all machine types

# ---------------------------- #
dcase="dcase2023"
seed="0"
name="recipe"
version="dis_baseline"
infer_ver="last"
metric="cosine"
# ---------------------------- #
experiments_train="${version}"
experiments_extract="resume_shared"
experiments_score="baseline"
# ---------------------------- #
source ../base/base.sh

asdit_train
for machine in $machines; do
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap metric="${metric}"
done
asdit_table

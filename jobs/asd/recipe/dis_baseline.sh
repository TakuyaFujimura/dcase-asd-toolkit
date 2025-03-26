#!/bin/bash

# This is an example script for training and testing a discriminative baseline model
# Frontend models are created and shared for all machine types

# ---------------------------- #
dcase="dcase2023"
seed="0"
name="recipe"
version="dis_baseline"
infer_ver="last"
# ---------------------------- #
experiments_train="${name}/${version}"
experiments_extract="resume_shared"
experiments_score="dis_baseline"
experiments_umap="cosine"
# ---------------------------- #
source ../base/base.sh

asdit_train
for machine in $machines; do
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap
done
asdit_table

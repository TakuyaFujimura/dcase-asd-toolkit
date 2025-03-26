#!/bin/bash

# This is an example script for training and testing an autoencoder baseline model
# Frontend models are created for each machine types

# ---------------------------- #
dcase="dcase2023"
seed="0"
name="recipe"
version="ae_baseline"
infer_ver="last"
# ---------------------------- #
experiments_train="${name}/${version}"
experiments_extract="resume_machinewise"
experiments_score="no_backend"
experiments_umap="euclid"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdit_train machine="${machine}"
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap
done
asdit_table

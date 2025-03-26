#!/bin/bash

# This is an example script for ASD with spectrogram statistics like https://arxiv.org/pdf/2305.03328
# This is non-deep learning method and does not require training process

# ---------------------------- #
dcase="dcase2023"
seed="0"
name="recipe"
version="beats_pool"
infer_ver="mean"
# ---------------------------- #
# experiments_train=""
experiments_extract="${name}/${version}"
experiments_score="dis_baseline"
experiments_umap="euclid"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap
done
asdit_table

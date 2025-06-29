#!/bin/bash

# This is a recipe for ASD system using spectrogram features
# This is a non-deep learning method and does not require training process
# https://arxiv.org/abs/2305.03328

# ---------------------------- #
dcase=$1
seed=0
name="recipe"
version="spectrogram"
infer_ver="time_mean"
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
    asdit_visualize
done
asdit_table

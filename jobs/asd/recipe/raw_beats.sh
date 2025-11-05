#!/bin/bash

# This is a recipe for ASD system using BEATs features
# This does not require training neural networks
# https://arxiv.org/abs/2409.05035

# ---------------------------- #
dcase=$1
seed=0
name="recipe"
version="raw_beats"
infer_ver="last"
# ---------------------------- #
# experiments_train=""
experiments_extract="scratch/${version}"
experiments_score="default"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    asdkit_visualize
done
asdkit_table

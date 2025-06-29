#!/bin/bash

# This is a recipe for a discriminative ASD system
# Frontend models are shared for all machine types

# ---------------------------- #
dcase=$1
seed=$2
name="recipe"
version="dis_baseline"
infer_ver="last"
# ---------------------------- #
experiments_train="${version}"
experiments_extract="restore_shared_dcasesec"
experiments_score="baseline"
# ---------------------------- #
source ../base/base.sh

asdit_train
for machine in $machines; do
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_visualize
done
asdit_table

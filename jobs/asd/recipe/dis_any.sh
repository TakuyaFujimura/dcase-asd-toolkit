#!/bin/bash

# This is a recipe for a discriminative ASD system
# Frontend models are shared for all machine types

# ---------------------------- #
dcase=$1
seed=$2
name="recipe"
version=$3
infer_ver="last"
# ---------------------------- #
experiments_train="${version}"
experiments_extract="restore/shared"
experiments_score="default"
# ---------------------------- #
source ../base/base.sh

asdit_train experiments="${experiments_train}"
for machine in $machines; do
    asdit_extract experiments="${experiments_extract}"
    asdit_score experiments="${experiments_score}"
    asdit_evaluate
    asdit_visualize
done
asdit_table

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

asdkit_train experiments="${experiments_train}"
for machine in $machines; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    asdkit_visualize
done
asdkit_table

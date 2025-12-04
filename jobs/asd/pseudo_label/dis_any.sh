#!/bin/bash

# This is a recipe for a discriminative ASD system
# Frontend models are shared for all machine types

# ---------------------------- #
dcase=$1
seed=$2
name="pseudo_label"
experiments_train=$3
label_dict_path=$4
version=$5
infer_ver="last"
# ---------------------------- #
experiments_extract="restore/shared"
experiments_score="default"
# ---------------------------- #
source ../base/base.sh

asdkit_train experiments="${experiments_train}" label_dict_path.main="${label_dict_path}"
for machine in $machines; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    # asdkit_visualize
done
asdkit_table

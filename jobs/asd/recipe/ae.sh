#!/bin/bash

# This is a recipe for an autoencoder-based ASD system
# Different Frontend models are created for each machine type

# ---------------------------- #
dcase=$1
seed=$2
name="recipe"
version="ae"
infer_ver="last"
# ---------------------------- #
experiments_train="${version}"
experiments_extract="restore/machinewise"
experiments_score="no_backend"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdkit_train machine="${machine}" experiments="${experiments_train}"
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
    asdkit_visualize
done
asdkit_table

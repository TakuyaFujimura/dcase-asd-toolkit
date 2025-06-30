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
experiments_extract="restore/machinewise_allsec"
experiments_score="no_backend"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdit_train machine="${machine}" experiments="${experiments_train}"
    asdit_extract experiments="${experiments_extract}"
    asdit_score experiments="${experiments_score}"
    asdit_evaluate
    asdit_visualize
done
asdit_table

#!/bin/bash

# This is a recipe for an autoencoder-based ASD system
# Different Frontend models are created for each machine type

# ---------------------------- #
dcase=$1
seed=$2
name="recipe"
version="ae_baseline"
infer_ver="last"
metric="euclid"
# ---------------------------- #
experiments_train="${version}"
experiments_extract="resume_machinewise"
experiments_score="no_backend"
# ---------------------------- #
source ../base/base.sh

for machine in $machines; do
    asdit_train machine="${machine}"
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap metric="${metric}"
done
asdit_table

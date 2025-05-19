#!/bin/bash

# This is a recipe for a discriminative ASD system
# Frontend models are shared for all machine types

# ---------------------------- #
dcase=$1
seed=$2
name="recipe"
version=$3
infer_ver="last"
metric="cosine"
# ---------------------------- #
experiments_train="${version}"
experiments_extract="restore_shared"
experiments_score="baseline"
# ---------------------------- #
source ../base/base.sh

if [ $dcase = "dcase2020" ]; then
    additional_args="label_dict_path.main_label=labels/${dcase}/machine_section.json"
else
    additional_args="label_dict_path.main_label=labels/${dcase}/machine_section_attr_domain.json"
fi

asdit_train ${additional_args}
for machine in $machines; do
    asdit_extract
    asdit_score
    asdit_evaluate
    asdit_umap metric="${metric}"
done
asdit_table

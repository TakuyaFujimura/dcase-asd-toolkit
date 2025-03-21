#!/bin/bash

# This is an example script for training and testing a discriminative baseline model
# Frontend models are created and shared for all machine types

dcase="dcase2023"
seed="0"
name="example"
version="dis_baseline"

machines=$(bash ../base/get_machines.sh "${dcase}")

# set common args
common_args=(--name="${name}" --version="${version}" --dcase="${dcase}" --seed="${seed}" --infer_ver="last")

# train
bash ../base/base.sh "${common_args[@]}" \
--cfg_train="${name}/${version}" \

# test
for machine in $machines; do
    bash ../base/base.sh "${common_args[@]}" \
    --machine="${machine}" \
    --cfg_extract="resume_shared" \
    --cfg_score="dis_baseline" \
    --cfg_evaluate="default" \
    --cfg_umap="cosine"
done

# table
bash ../base/base.sh "${common_args[@]}" \
--cfg_table="default"

#!/bin/bash

# This is an example script for training and testing an autoencoder baseline model
# Frontend models are created for each machine types


dcase="dcase2023"
seed="0"
name="example"
version="ae_baseline"

machines=$(bash ../base/get_machines.sh "${dcase}")

# set common args
common_args=(--name="${name}" --version="${version}" --dcase="${dcase}" --seed="${seed}" --infer_ver="last")

# train and test
for machine in $machines; do
    bash ../base/base.sh "${common_args[@]}" \
    --machine="${machine}" \
    --cfg_train="${name}/${version}" \
    --cfg_extract="resume_machinewise" \
    --cfg_score="no_backend" \
    --cfg_evaluate="default" \
    --cfg_umap="euclid" \
    exit
done

# table
bash ../base/base.sh "${common_args[@]}" \
--cfg_table="default"

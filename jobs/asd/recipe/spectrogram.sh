#!/bin/bash

# This is an example script for ASD with spectrogram statistics like https://arxiv.org/pdf/2305.03328
# This is non-deep learning method and does not require training process

dcase="dcase2023"
seed="0"
name="recipe"
version="stft_mean"

machines=$(bash ../base/get_machines.sh "${dcase}")

# set common args
common_args=(--name="${name}" --version="${version}" --dcase="${dcase}" --seed="${seed}" --infer_ver="mean")

# train
# No training process

# test
for machine in $machines; do
    bash ../base/base.sh "${common_args[@]}" \
    --machine="${machine}" \
    --cfg_extract="${name}/${version}" \
    --cfg_score="dis_baseline" \
    --cfg_evaluate="default" \
    --cfg_umap="euclid"
done

# table
bash ../base/base.sh "${common_args[@]}" \
--cfg_table="default"

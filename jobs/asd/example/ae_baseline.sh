#!/bin/bash

dcase="dcase2023"
seed="0"
name="example"
version="${dcase}_ae_baseline"
common_args=(--name="${name}" --version="${version}" --dcase="${dcase}" --seed="${seed}")

machines=$(bash ../base/get_machines.sh "${dcase}")

# train and test
for machine in $machines; do
    bash ../base/base.sh "${common_args[@]}" \
    --machine="${machine}" \
    --cfg_train="${name}/${version}" \
    --cfg_extract="machinewise" \
    --cfg_score="no_backend" \
    --cfg_evaluate="default" \
    --cfg_umap="euclid" \
    --ckpt_ver="last"
    exit
done

# table
bash ../base/base.sh "${common_args[@]}" \
--cfg_table="default"

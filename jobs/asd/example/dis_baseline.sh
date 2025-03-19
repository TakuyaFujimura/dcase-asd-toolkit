#!/bin/bash

dcase="dcase2023"
seed="0"
name="example"
version="${dcase}_dis_baseline"
common_args=(--name="${name}" --version="${version}" --dcase="${dcase}" --seed="${seed}")

machines=$(bash ../base/get_machines.sh "${dcase}")


# train
bash ../base/base.sh "${common_args[@]}" \
--cfg_train="${name}/${version}" \


# test
for machine in $machines; do
    bash ../base/base.sh "${common_args[@]}" \
    --machine="${machine}" \
    --cfg_extract="shared" \
    --cfg_score="dis_baseline" \
    --cfg_evaluate="default" \
    --cfg_umap="cosine" \
    --ckpt_ver="last"
done

# table
bash ../base/base.sh "${common_args[@]}" \
--cfg_table="default"

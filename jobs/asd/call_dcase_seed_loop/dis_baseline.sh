#!/bin/bash
# ---------------------------- #
dcase_list=("dcase2023")
seed_list=("0" "1" "2" "3" "4")
# ---------------------------- #


recipe=$(basename $0)
cd ../recipe

for dcase in "${dcase_list[@]}"; do
    for seed in "${seed_list[@]}"; do
        bash "${recipe}" "${dcase}" "${seed}"
    done
done

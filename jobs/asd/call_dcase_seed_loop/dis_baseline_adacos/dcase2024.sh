#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L gpu=4
#PJM -L elapse=12:00:00
#PJM -j
# ---------------------------- #

dcase_list=("dcase2024")
# ---------------------------- #
recipe="dis_baseline_adacos"
cd ../../recipe

mkdir ../logs/${recipe}

for dcase in "${dcase_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 bash dis_any_version.sh "${dcase}" "0" "${recipe}" > "../logs/${recipe}/${dcase}_0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 bash dis_any_version.sh "${dcase}" "1" "${recipe}" > "../logs/${recipe}/${dcase}_1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 bash dis_any_version.sh "${dcase}" "2" "${recipe}" > "../logs/${recipe}/${dcase}_2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 bash dis_any_version.sh "${dcase}" "3" "${recipe}" > "../logs/${recipe}/${dcase}_3.log" 2>&1 &
    wait
done

#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L gpu=4
#PJM -L elapse=12:00:00
#PJM -j
# ---------------------------- #
dcase_list=("dcase2023")
# ---------------------------- #


recipe_stem="dis_baseline"
recipe="${recipe_stem}.sh"
cd ../../recipe

mkdir ../logs/${recipe_stem}

for dcase in "${dcase_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 bash "${recipe}" "${dcase}" "0" > "../logs/${recipe_stem}/${dcase}_0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 bash "${recipe}" "${dcase}" "1" > "../logs/${recipe_stem}/${dcase}_1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 bash "${recipe}" "${dcase}" "2" > "../logs/${recipe_stem}/${dcase}_2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 bash "${recipe}" "${dcase}" "3" > "../logs/${recipe_stem}/${dcase}_3.log" 2>&1 &
    wait
done

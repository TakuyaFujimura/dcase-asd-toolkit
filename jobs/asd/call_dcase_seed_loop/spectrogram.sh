#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -j
# ---------------------------- #

dcase_list=("23" "21" "22" "20" "24")
# ---------------------------- #
recipe="spectrogram"
cd ../recipe

for dcase in "${dcase_list[@]}"; do
    bash "${recipe}.sh" "dcase20${dcase}"
done

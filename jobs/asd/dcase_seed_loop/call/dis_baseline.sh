#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j
# ----------------------------------------------- #
cd ..

for dcase in dcase2023; do
    for seed in 1 2 3 4; do
    bash dis_baseline.sh ${dcase} ${seed}
    done
done

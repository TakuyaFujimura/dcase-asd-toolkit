#!/bin/bash
# ---------------------------- #
recipe="ae_baseline.sh"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash "${recipe}" "${dcase}" "${seed}"

#!/bin/bash
# ---------------------------- #
recipe="dis_baseline_adacos_trainable"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash dis_any_version.sh "${dcase}" "${seed}" "${recipe}"

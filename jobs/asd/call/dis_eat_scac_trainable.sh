#!/bin/bash
# ---------------------------- #
dcase="dcase2023"
seed="0"
recipe="dis_eat_scac_trainable"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash dis_any.sh "${dcase}" "${seed}" "${recipe}"

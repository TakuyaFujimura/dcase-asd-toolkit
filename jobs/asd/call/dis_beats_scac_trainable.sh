#!/bin/bash
# ---------------------------- #
dcase="dcase2023"
seed="0"
recipe="dis_beats_scac_trainable"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash dis_any.sh "${dcase}" "${seed}" "${recipe}"

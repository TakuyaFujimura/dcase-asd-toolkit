#!/bin/bash
# ---------------------------- #
dcase="dcase2023"
seed="0"
recipe="dis_multispec_scac_trainable"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash dis_any_version.sh "${dcase}" "${seed}" "${recipe}"

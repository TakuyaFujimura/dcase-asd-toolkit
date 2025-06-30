#!/bin/bash
# ---------------------------- #
dcase="dcase2023"
seed="0"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash "ae.sh" "${dcase}" "${seed}"

#!/bin/bash
# ---------------------------- #
dcase="dcase2023"
# ---------------------------- #

source ../../../venv/bin/activate
cd ../recipe
bash "raw_spec.sh" "${dcase}"

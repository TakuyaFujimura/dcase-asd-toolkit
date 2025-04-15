#!/bin/bash
# ---------------------------- #
recipe="dis_baseline_subspaceloss"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash dis_any_version.sh "${dcase}" "${seed}" "${recipe}"

#!/bin/bash
# ---------------------------- #
recipe="dis_multires_scac_trainable"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash dis_any_version.sh "${dcase}" "${seed}" "${recipe}"

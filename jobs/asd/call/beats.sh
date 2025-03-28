#!/bin/bash
# ---------------------------- #
recipe="beats.sh"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash "${recipe}" "${dcase}" "${seed}"

#!/bin/bash
# ---------------------------- #
recipe="beats"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash "${recipe}.sh" "${dcase}" "${seed}"

#!/bin/bash
# ---------------------------- #
recipe="spectrogram"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash "${recipe}.sh" "${dcase}" "${seed}"

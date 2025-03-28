#!/bin/bash
# ---------------------------- #
recipe="spectrogram.sh"
dcase="dcase2023"
seed="0"
# ---------------------------- #


cd ../recipe
bash "${recipe}" "${dcase}" "${seed}"

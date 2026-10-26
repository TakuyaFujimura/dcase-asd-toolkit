#!/usr/bin/env bash

# Run ASD with spectrogram or pretrained features; no frontend training is required.
# Arguments: DCASE RECIPE [EXTRACT_CONFIG] [SCORE_CONFIG].

dcase=$1
seed=0
name="recipe"
version=$2
infer_ver="default"

experiments_extract="${3:-from_config/${version}}"
experiments_score="${4:-default}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../base/base.sh"

for machine in "${machines[@]}"; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
done
asdkit_table

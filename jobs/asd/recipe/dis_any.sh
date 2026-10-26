#!/usr/bin/env bash

# Train one discriminative frontend shared across machine types.
# Arguments: DCASE SEED RECIPE [EXTRACT_CONFIG] [SCORE_CONFIG].

dcase=$1
seed=$2
name="recipe"
version=$3
infer_ver="default"

experiments_train="${version}"
experiments_extract="${4:-from_checkpoint/shared}"
experiments_score="${5:-default}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../base/base.sh"

asdkit_train experiments="${experiments_train}"
for machine in "${machines[@]}"; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
done
asdkit_table

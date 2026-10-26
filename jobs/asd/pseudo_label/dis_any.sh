#!/usr/bin/env bash
set -euo pipefail

# Train one discriminative frontend shared across machine types.
# Arguments: DCASE SEED TRAIN_CONFIG LABEL_JSON VERSION [EXTRACT_CONFIG] [SCORE_CONFIG].

dcase=$1
seed=$2
name="pseudo_label"
experiments_train=$3
label_dict_path=$4
version=$5
infer_ver="default"

experiments_extract="${6:-from_checkpoint/shared}"
experiments_score="${7:-default}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../base/base.sh"

asdkit_train experiments="${experiments_train}" label_dict_path.main="${label_dict_path}"
for machine in "${machines[@]}"; do
    asdkit_extract experiments="${experiments_extract}"
    asdkit_score experiments="${experiments_score}"
    asdkit_evaluate
done
asdkit_table

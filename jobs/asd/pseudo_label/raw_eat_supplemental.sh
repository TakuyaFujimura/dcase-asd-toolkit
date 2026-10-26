#!/usr/bin/env bash

# Extract supplemental EAT features for pseudo labeling (DCASE 2025).

dcase="dcase2025"
seed=0
name="recipe"
version="raw_eat"
infer_ver="default"

experiments_extract="from_config/raw_eat_preprocessed"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../base/base.sh"

for machine in "${machines[@]}"; do
    asdkit_extract experiments="${experiments_extract}" datamodule=supplemental overwrite=True
done

#!/usr/bin/env bash
set -euo pipefail
# Requires embeddings extracted by jobs/asd/run/raw_eat.sh.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${repo_root}/jobs/activate.sh"
cd "${script_dir}/.."

feature=raw_eat
config_name=kmeans_ratio_08
for dcase in dcase2023 dcase2024; do
    teacher_dir=${feature}/0/output/default
    python main.py --config-name="${config_name}" name=recipe dcase="${dcase}" teacher_dir="${teacher_dir}" output_label_name="${feature}_${config_name}"
    python main.py --config-name="${config_name}" name=recipe dcase="${dcase}" teacher_dir="${teacher_dir}" output_label_name="${feature}_${config_name}_all" machinewise=false
done

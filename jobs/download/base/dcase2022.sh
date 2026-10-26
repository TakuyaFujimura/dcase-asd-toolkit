#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/base.sh"
setup_download_paths "$1" "dcase2022" || exit 1

# Download development data.
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
    downoad_and_unzip "https://zenodo.org/record/6355122/files/dev_${machine_type}.zip" "${dev_dir}"
done

# Download evaluation data.
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
    downoad_and_unzip "https://zenodo.org/record/6462969/files/eval_data_${machine_type}_train.zip" "${eval_dir}"

    downoad_and_unzip "https://zenodo.org/record/6586456/files/eval_data_${machine_type}_test.zip" "${eval_dir}"
done

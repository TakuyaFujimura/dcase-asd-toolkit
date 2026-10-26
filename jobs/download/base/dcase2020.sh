#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/base.sh"
setup_download_paths "$1" "dcase2020" || exit 1

# Download development data.
for machine_type in ToyCar ToyConveyor fan valve slider pump; do
    downoad_and_unzip "https://zenodo.org/record/3678171/files/dev_data_${machine_type}.zip" "${dev_dir}"
done

# Download evaluation data.
for machine_type in ToyCar ToyConveyor fan valve slider pump; do
    downoad_and_unzip "https://zenodo.org/record/3727685/files/eval_data_train_${machine_type}.zip" "${eval_dir}"

    downoad_and_unzip "https://zenodo.org/record/3841772/files/eval_data_test_${machine_type}.zip" "${eval_dir}"
done

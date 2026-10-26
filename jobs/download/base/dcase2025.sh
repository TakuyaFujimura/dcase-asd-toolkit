#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/base.sh"
setup_download_paths "$1" "dcase2025" || exit 1

# Download development data.
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
    downoad_and_unzip "https://zenodo.org/records/15097779/files/dev_${machine_type}.zip" "${dev_dir}"
done

# Download evaluation data.
for machine_type in \
    ToyRCCar \
    ToyPet \
    HomeCamera \
    AutoTrash \
    Polisher \
    ScrewFeeder \
    BandSealer \
    CoffeeGrinder \
; do
    downoad_and_unzip "https://zenodo.org/records/15392814/files/eval_data_${machine_type}_train.zip" "${eval_dir}"
    downoad_and_unzip "https://zenodo.org/records/15519362/files/eval_data_${machine_type}_test.zip" "${eval_dir}"
done

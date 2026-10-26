#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/base.sh"
setup_download_paths "$1" "dcase2024" || exit 1

# Download development data.
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
    downoad_and_unzip "https://zenodo.org/record/10902294/files/dev_${machine_type}.zip" "${dev_dir}"
done

# Download evaluation data.
for machine_type in \
    3DPrinter_train_r2 \
    AirCompressor_train \
    Scanner_train \
    ToyCircuit_train \
    HoveringDrone_train \
    HairDryer_train \
    ToothBrush_train \
    RoboticArm_train_r2 \
    BrushlessMotor_train \
; do
    downoad_and_unzip "https://zenodo.org/records/11259435/files/eval_data_${machine_type}.zip" "${eval_dir}"
done

for machine_type in \
    3DPrinter \
    AirCompressor \
    Scanner \
    ToyCircuit \
    HoveringDrone \
    HairDryer \
    ToothBrush \
    RoboticArm \
    BrushlessMotor \
; do
    downoad_and_unzip "https://zenodo.org/records/11363076/files/eval_data_${machine_type}_test.zip" "${eval_dir}"
done

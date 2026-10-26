#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/base.sh"
setup_download_paths "$1" "dcase2026" || exit 1

# Download development data.
for machine_type in bearingEmu fan gearboxEmu sliderEmu ToyCar_r2 ToyCarEmu valveEmu; do
    downoad_and_unzip "https://zenodo.org/records/19336329/files/dev_${machine_type}.zip" "${dev_dir}"
done

# Download evaluation data.
for machine_type in \
    BlowerDustCollector \
    Sander \
    SewingMachine \
    ToothBrush \
    ToyDrone \
; do
    downoad_and_unzip "https://zenodo.org/records/20151556/files/eval_data_${machine_type}_train.zip" "${eval_dir}"
    downoad_and_unzip "https://zenodo.org/records/20437238/files/eval_data_${machine_type}_test.zip" "${eval_dir}"
done

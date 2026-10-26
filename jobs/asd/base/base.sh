#!/usr/bin/env bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || return 1

get_machines() {
    case "$1" in
        dcase2020)
            machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")
            ;;
        dcase2021)
            machines=("fan" "gearbox" "pump" "slider" "ToyCar" "ToyTrain" "valve")
            ;;
        dcase2022)
            machines=("bearing" "fan" "gearbox" "slider" "ToyCar" "ToyTrain" "valve")
            ;;
        dcase2023)
            machines=("bandsaw" "bearing" "fan" "gearbox" "grinder" "shaker" "slider" "ToyCar" "ToyDrone" "ToyNscale" "ToyTank" "ToyTrain" "Vacuum" "valve")
            ;;
        dcase2024)
            machines=("3DPrinter" "AirCompressor" "bearing" "BrushlessMotor" "fan" "gearbox" "HairDryer" "HoveringDrone" "RoboticArm" "Scanner" "slider" "ToothBrush" "ToyCar" "ToyCircuit" "ToyTrain" "valve")
            ;;
        dcase2025)
            machines=("AutoTrash" "BandSealer" "bearing" "CoffeeGrinder" "fan" "gearbox" "HomeCamera" "Polisher" "ScrewFeeder" "slider" "ToyCar" "ToyPet" "ToyRCCar" "ToyTrain" "valve")
            ;;
        dcase2026)
            machines=("bearingEmu" "BlowerDustCollector" "fan" "gearboxEmu" "Sander" "SewingMachine" "sliderEmu" "ToothBrush" "ToyCar" "ToyCarEmu" "ToyDrone" "valveEmu")
            ;;
        *)
            echo "Error: Invalid DCASE: $1" >&2
            return 1
            ;;
    esac
}

asdkit_train() (
    cd "${repo_root}" || exit 1
    source "${repo_root}/jobs/activate.sh" || exit 1
    python -m asdkit.bin.train \
        "name=${name}" \
        "version=${version}" \
        "dcase=${dcase}" \
        "seed=${seed}" \
        "$@"
)

asdkit_extract() (
    cd "${repo_root}" || exit 1
    source "${repo_root}/jobs/activate.sh" || exit 1
    python -m asdkit.bin.extract \
        "name=${name}" \
        "version=${version}" \
        "dcase=${dcase}" \
        "seed=${seed}" \
        "infer_ver=${infer_ver}" \
        "machine=${machine}" \
        "$@"
)

asdkit_score() (
    cd "${repo_root}" || exit 1
    source "${repo_root}/jobs/activate.sh" || exit 1
    python -m asdkit.bin.score \
        "name=${name}" \
        "version=${version}" \
        "dcase=${dcase}" \
        "seed=${seed}" \
        "infer_ver=${infer_ver}" \
        "machine=${machine}" \
        "$@"
)

asdkit_evaluate() (
    cd "${repo_root}" || exit 1
    source "${repo_root}/jobs/activate.sh" || exit 1
    python -m asdkit.bin.evaluate \
        "name=${name}" \
        "version=${version}" \
        "dcase=${dcase}" \
        "seed=${seed}" \
        "infer_ver=${infer_ver}" \
        "machine=${machine}" \
        "$@"
)

asdkit_visualize() (
    cd "${repo_root}" || exit 1
    source "${repo_root}/jobs/activate.sh" || exit 1
    python -m asdkit.bin.visualize \
        "name=${name}" \
        "version=${version}" \
        "dcase=${dcase}" \
        "seed=${seed}" \
        "infer_ver=${infer_ver}" \
        "machine=${machine}" \
        "$@"
)


asdkit_table() (
    cd "${repo_root}" || exit 1
    source "${repo_root}/jobs/activate.sh" || exit 1
    python -m asdkit.bin.table \
        "name=${name}" \
        "version=${version}" \
        "dcase=${dcase}" \
        "seed=${seed}" \
        "infer_ver=${infer_ver}" \
        "$@"
)


get_machines "${dcase}" || exit 1
echo "machines: ${machines[*]}"

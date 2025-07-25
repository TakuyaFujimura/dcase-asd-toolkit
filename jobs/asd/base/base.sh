#!/bin/bash

get_machines() {
    if [ "$1" = "dcase2020" ]; then
        machines=("fan"  "pump"  "slider"  "ToyCar"  "ToyConveyor"  "valve")
    elif [ "$1" = "dcase2021" ]; then
        machines=("fan"  "gearbox"  "pump"  "slider"  "ToyCar"  "ToyTrain"  "valve")
    elif [ "$1" = "dcase2022" ]; then
        machines=("bearing"  "fan"  "gearbox"  "slider"  "ToyCar"  "ToyTrain"  "valve")
    elif [ "$1" = "dcase2023" ]; then
        machines=("bandsaw" "bearing" "fan" "gearbox" "grinder" "shaker" "slider" "ToyCar" "ToyDrone" "ToyNscale" "ToyTank" "ToyTrain"  "Vacuum" "valve")
    elif [ "$1" = "dcase2024" ]; then
        machines=("3DPrinter" "AirCompressor" "bearing" "BrushlessMotor" "fan" "gearbox" "HairDryer" "HoveringDrone" "RoboticArm" "Scanner" "slider" "ToothBrush" "ToyCar" "ToyCircuit" "ToyTrain" "valve")
    else
        machines="InvalidDCASE"
    fi

    echo "${machines[@]}"
}

collect_args() {
    local vars=("$@")
    local args=()

    for var in "${vars[@]}"; do
        if [[ -n $var ]]; then
            args+=("$var=${!var}")
        fi
    done

    echo "${args[@]}"
}

asdkit_train() {
    local args=($(collect_args "name" "version" "dcase" "seed"))
    echo "${args[@]}"
    python -m asdkit.bin.train "${args[@]}" "$@"
}

asdkit_extract() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine"))
    python -m asdkit.bin.extract "${args[@]}" "$@"
}

asdkit_score() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine"))
    python -m asdkit.bin.score "${args[@]}" "$@"
}

asdkit_evaluate() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine"))
    python -m asdkit.bin.evaluate "${args[@]}" "$@"
}

asdkit_visualize() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine"))
    python -m asdkit.bin.visualize "${args[@]}" "$@"
}


asdkit_table() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver"))
    python -m asdkit.bin.table "${args[@]}" "$@"
}

# get machines
machines=$(get_machines "${dcase}")
if [ "$machines" = "InvalidDCASE" ]; then
    echo "Error: Invalid DCASE"
    exit 1
fi
echo "machines: $machines"

# change directory to project root
cd ../../..


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
        if [[ -z $var ]]; then
            echo "Error: $var is required but not defined"
            exit 1
        elif [[ $var == experiments_* ]]; then 
            args+=("experiments=${!var}")
        else
            args+=("$var=${!var}")
        fi
    done

    echo "${args[@]}"
}

asdit_train() {
    local args=($(collect_args "name" "version" "dcase" "seed" "experiments_train"))
    echo "${args[@]}"
    python -m asdit.bin.train "${args[@]}" "$@"
}

asdit_extract() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine" "experiments_extract"))
    python -m asdit.bin.extract "${args[@]}" "$@"
}

asdit_score() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine" "experiments_score"))
    python -m asdit.bin.score "${args[@]}" "$@"
}

asdit_evaluate() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine"))
    python -m asdit.bin.evaluate "${args[@]}" "$@"
}

asdit_umap() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver" "machine"))
    python -m asdit.bin.umap "${args[@]}" "$@"
}


asdit_table() {
    local args=($(collect_args "name" "version" "dcase" "seed" "infer_ver"))
    python -m asdit.bin.table "${args[@]}" "$@"
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

# activate virtual environment
if [ ! -d "venv" ]; then
    echo "venv not found in $(pwd). Did you run it from 'jobs/asd/base'?"
    exit 1
fi
source "venv/bin/activate"

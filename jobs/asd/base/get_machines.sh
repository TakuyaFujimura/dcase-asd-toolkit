#!/bin/bash
dcase=$1

if [ "$dcase" = "dcase2021" ]; then
    machines=("fan"  "gearbox"  "pump"  "slider"  "ToyCar"  "ToyTrain"  "valve")
elif [ "$dcase" = "dcase2022" ]; then
    machines=("bearing"  "fan"  "gearbox"  "slider"  "ToyCar"  "ToyTrain"  "valve")
elif [ "$dcase" = "dcase2023" ]; then
    machines=("bandsaw" "bearing" "fan" "gearbox" "grinder" "shaker" "slider" "ToyCar" "ToyDrone" "ToyNscale" "ToyTank" "ToyTrain"  "Vacuum" "valve")
elif [ "$dcase" = "dcase2024" ]; then
    machines=("3DPrinter" "AirCompressor" "bearing" "BrushlessMotor" "fan" "gearbox" "HairDryer" "HoveringDrone" "RoboticArm" "Scanner" "slider" "ToothBrush" "ToyCar" "ToyCircuit" "ToyTrain" "valve")
else
    echo "Invalid dcase"
    exit 1
fi

echo "${machines[@]}"

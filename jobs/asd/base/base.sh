#!/bin/bash
########################
name=$1
version=$2
dcase=$3
seed=$4
extract_exp=$5
score_exp=$6
evaluate_exp=$7
umap_exp=$8
table_exp=$9
IFS=',' read -r -a ckpt_ver_list <<< "${10}"
########################

# get machines
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


# activate virtual environment
cd ../../..
source "venv/bin/activate"



# training
if [ "${extract_exp}" = "shared" ]; then
    python -m asdlib.bin.train experiments="${name}/${version}" 'seed='${seed}'' \
    'name='${name}'' 'version='${version}''
elif [ "${extract_exp}" = "machinewise" ]; then
    for machine in "${machines[@]}"; do
        python -m asdlib.bin.train experiments="${name}/${version}" 'seed='${seed}'' \
        'name='${name}'' 'version='${version}'' 'machine='${machine}''
    done
fi

# testing
for ckpt_ver in "${ckpt_ver_list[@]}"; do
    for machine in "${machines[@]}"; do
        python -m asdlib.bin.extract experiments="${extract_exp}" \
        'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
        'ckpt_ver='${ckpt_ver}'' 'machine='${machine}''

        python -m asdlib.bin.score experiments="${score_exp}" \
        'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
        'ckpt_ver='${ckpt_ver}'' 'machine='${machine}''

        python -m asdlib.bin.evaluate experiments="${evaluate_exp}" \
        'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
        'ckpt_ver='${ckpt_ver}'' 'machine='${machine}'' 'dcase='${dcase}''

        python -m asdlib.bin.umap experiments="${umap_exp}" \
        'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
        'ckpt_ver='${ckpt_ver}'' 'machine='${machine}''
    done

    python -m asdlib.bin.table experiments="${table_exp}" \
    'name='${name}'' 'version='${version}'' 'seed='${seed}'' \
    'ckpt_ver='${ckpt_ver}'' dcase="${dcase}"
done



dst_dir=$1

dev_dir="${dst_dir}/dcase2024/dev_data/raw"
eval_dir="${dst_dir}/dcase2024/eval_data/raw"
mkdir -p ${dev_dir}
mkdir -p ${eval_dir}

# download dev data
cd ${dev_dir}
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/10902294/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd ${eval_dir}
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
wget "https://zenodo.org/records/11259435/files/eval_data_${machine_type}.zip"
unzip "eval_data_${machine_type}.zip"
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
wget "https://zenodo.org/records/11363076/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

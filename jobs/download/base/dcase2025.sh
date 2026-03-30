dst_dir=$1

dev_dir="${dst_dir}/dcase2025/dev_data/raw"
eval_dir="${dst_dir}/dcase2025/eval_data/raw"
mkdir -p ${dev_dir}
mkdir -p ${eval_dir}

# download dev data
cd ${dev_dir}
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
curl -L -O "https://zenodo.org/records/15097779/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd ${eval_dir}
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
curl -L -O "https://zenodo.org/records/15392814/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"
curl -L -O "https://zenodo.org/records/15519362/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

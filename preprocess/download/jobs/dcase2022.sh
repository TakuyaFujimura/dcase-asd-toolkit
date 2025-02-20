dst_dir=$1

dev_dir="${dst_dir}/dcase2022/dev_data/raw"
eval_dir="${dst_dir}/dcase2022/eval_data/raw"
mkdir -p ${dev_dir}
mkdir -p ${eval_dir}

# download dev data
cd ${dev_dir}
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/6355122/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd ${eval_dir}
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/6462969/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"

wget "https://zenodo.org/record/6586456/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

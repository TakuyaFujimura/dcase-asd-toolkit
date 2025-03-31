dst_dir=$1

dev_dir="${dst_dir}/dcase2021/dev_data/raw"
eval_dir="${dst_dir}/dcase2021/eval_data/raw"
mkdir -p ${dev_dir}
mkdir -p ${eval_dir}


# download dev data
cd ${dev_dir}
for machine_type in fan gearbox pump slider ToyCar ToyTrain valve; do
curl -L -O "https://zenodo.org/record/4562016/files/dev_data_${machine_type}.zip"
unzip "dev_data_${machine_type}.zip"
done

# download eval data
cd ${eval_dir}
for machine_type in fan gearbox pump slider ToyCar ToyTrain valve; do
curl -L -O "https://zenodo.org/record/4660992/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"

curl -L -O "https://zenodo.org/record/4884786/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

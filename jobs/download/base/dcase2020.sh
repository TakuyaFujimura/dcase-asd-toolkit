dst_dir=$1

dev_dir="${dst_dir}/dcase2020/dev_data/raw"
eval_dir="${dst_dir}/dcase2020/eval_data/raw"
mkdir -p ${dev_dir}
mkdir -p ${eval_dir}


# download dev data
cd ${dev_dir}
for machine_type in ToyCar ToyConveyor fan valve slider pump; do
curl -L -O "https://zenodo.org/record/3678171/files/dev_data_${machine_type}.zip"
unzip "dev_data_${machine_type}.zip"
done

# download eval data
cd ${eval_dir}
for machine_type in ToyCar ToyConveyor fan valve slider pump; do
curl -L -O "https://zenodo.org/record/3727685/files/eval_data_train_${machine_type}.zip"
unzip "eval_data_train_${machine_type}.zip"

curl -L -O "https://zenodo.org/record/3841772/files/eval_data_test_${machine_type}.zip"
unzip "eval_data_test_${machine_type}.zip"
done

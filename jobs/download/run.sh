data_dir="../../../data" # relative path from this script or absolute path
dcase="dcase2023"


absolute_data_dir=$(cd "$(dirname ${data_dir})" && pwd)/$(basename ${data_dir})
echo "absolute data_dir: ${absolute_data_dir}"
cd base
bash base.sh "${absolute_data_dir}" "${dcase}"

data_dir="../../../data" # relative path from this script or absolute path
dcase="dcase2022"


absolute_data_dir=$(cd "$(dirname ${data_dir})" && pwd)/$(basename ${data_dir})
echo "absolute data_dir: ${absolute_data_dir}"
base/base.sh "${absolute_data_dir}" "${dcase}"

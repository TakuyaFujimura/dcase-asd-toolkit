data_dir="../../../data" # relative path from this script or absolute path
dcase="dcase2023"
link_mode="symlink" # or "mv"

absolute_data_dir=$(cd "$(dirname ${data_dir})" && pwd)/$(basename ${data_dir})
echo "absolute data_dir: ${absolute_data_dir}"

cd ../..
source venv/bin/activate
python -m asdit.bin.format --data_dir="${absolute_data_dir}" --dcase="${dcase}" --link_mode="${link_mode}"

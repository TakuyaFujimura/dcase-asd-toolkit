#!/usr/bin/env bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."
data_dir="${repo_root}/../data" # absolute path to the data directory
dcase="dcase2023"
link_mode="symlink" # or "mv"

if [[ "${data_dir}" != /* ]]; then
    echo "Error: data_dir is a relative path: ${data_dir}. Relative paths are interpreted relative to repo_root (${repo_root}). Please use an absolute path for safety." >&2
    exit 1
fi

cd "${repo_root}"
source jobs/activate.sh || exit 1
python -m asdkit.bin.download_eval_data data_dir="${data_dir}" dcase="${dcase}" || exit 1
python -m asdkit.bin.format --data_dir="${data_dir}" --dcase="${dcase}" --link_mode="${link_mode}" || exit 1
bash jobs/labeling/base.sh "${dcase}" "${data_dir}" "${dcase}" || exit 1
bash jobs/labeling/base.sh "${dcase}" "${data_dir}" "${dcase}" true

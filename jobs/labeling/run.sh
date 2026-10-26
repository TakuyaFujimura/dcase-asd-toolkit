#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${script_dir}/../../../data"
dcase="dcase2025"
experiments="${dcase}"
machinewise=false

bash "${script_dir}/base.sh" "${dcase}" "${data_dir}" "${experiments}" "${machinewise}"

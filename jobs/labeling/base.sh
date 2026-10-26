#!/usr/bin/env bash

dcase=$1
data_dir=$2
experiments=$3
machinewise=${4:-false}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../.." || exit 1
source jobs/activate.sh || exit 1
python -m asdkit.bin.label experiments="${experiments}" dcase="${dcase}" data_dir="${data_dir}" machinewise="${machinewise}"

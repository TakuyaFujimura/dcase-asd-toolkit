#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${script_dir}/../../../data" # absolute path to the data directory
dcase="dcase2026"

case "$dcase" in
    dcase202[0-6]) bash "${script_dir}/base/${dcase}.sh" "${data_dir}" ;;
    *) echo "Unknown dcase: ${dcase}" >&2; exit 1 ;;
esac

#!/usr/bin/env bash

dcase="dcase2023"
seed="0"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/../recipe/gen_any.sh" "${dcase}" "${seed}" "dcase_ae"

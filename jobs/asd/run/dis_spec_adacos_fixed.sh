#!/usr/bin/env bash

dcase="dcase2023"
seed="0"
recipe="dis_spec_adacos_fixed"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/../recipe/dis_any.sh" "${dcase}" "${seed}" "${recipe}"

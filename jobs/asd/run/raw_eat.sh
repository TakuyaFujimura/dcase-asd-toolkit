#!/usr/bin/env bash

dcase="dcase2023"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/../recipe/raw_any.sh" "${dcase}" "raw_eat" \
    "from_config/raw_eat_preprocessed" "ssl_with_cls_preprocessed"

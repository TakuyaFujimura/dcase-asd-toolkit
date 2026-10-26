#!/usr/bin/env bash

dcase="dcase2023"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/../recipe/raw_any.sh" "${dcase}" "raw_beats" \
    "from_config/raw_beats_preprocessed" "ssl_preprocessed"

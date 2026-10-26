#!/usr/bin/env bash

dcase="dcase2023"
seed="0"
recipe="dis_beats_scac_trainable"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/../recipe/dis_any.sh" "${dcase}" "${seed}" "${recipe}" \
    "from_checkpoint/shared_ssl_preprocessed" "ssl_dis_preprocessed"

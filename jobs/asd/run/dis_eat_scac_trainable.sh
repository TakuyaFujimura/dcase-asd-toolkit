#!/usr/bin/env bash

dcase="dcase2023"
seed="0"
recipe="dis_eat_scac_trainable"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/../recipe/dis_any.sh" "${dcase}" "${seed}" "${recipe}" \
    "from_checkpoint/shared_ssl_with_cls_preprocessed" "ssl_dis_with_cls_preprocessed"

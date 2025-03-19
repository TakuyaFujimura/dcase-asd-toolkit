#!/bin/bash
########################
name="None"
version="None"
dcase="None"
seed="None"
machine="None"
cfg_train="None"
cfg_extract="None"
cfg_score="None"
cfg_evaluate="None"
cfg_umap="None"
cfg_table="None"
infer_ver="None"
########################

cd "$(dirname "$0")"

source ./parse_options.sh

# check required arguments
if [ "${name}" = "None" ] || [ "${version}" = "None" ] || [ "${dcase}" = "None" ] || [ "${seed}" = "None" ]; then
    echo "Missing required arguments"
    echo "name: ${name}, version: ${version}, dcase: ${dcase}, seed: ${seed}"
    exit 1
fi

cd ../../..

# activate virtual environment
if [ ! -d "venv" ]; then
    echo "venv not found in $(pwd)"
    exit 1
fi
source "venv/bin/activate"

# train
if [ "${cfg_train}" = "None" ]; then
    echo "Skipping training"
else
    echo "Starting training"
    if [ "${machine}" = "None" ]; then
        python -m asdit.bin.train experiments="${cfg_train}" \
        seed="${seed}" dcase="${dcase}" \
        name="${name}" version="${version}"
    else
        python -m asdit.bin.train experiments="${cfg_train}" \
        seed="${seed}" dcase="${dcase}" \
        name="${name}" version="${version}" machine="${machine}"
    fi
fi

# extract
if [ "${cfg_extract}" = "None" ]; then
    echo "Skipping extraction"
else
    echo "Starting extraction"
    python -m asdit.bin.extract experiments="${cfg_extract}" \
    seed="${seed}" dcase="${dcase}" \
    name="${name}" version="${version}" \
    infer_ver="${infer_ver}" machine="${machine}"
fi

# score
if [ "${cfg_score}" = "None" ]; then
    echo "Skipping scoring"
else
    echo "Starting scoring"
    python -m asdit.bin.score experiments="${cfg_score}" \
    seed="${seed}" dcase="${dcase}" \
    name="${name}" version="${version}" \
    infer_ver="${infer_ver}" machine="${machine}"
fi

# evaluate
if [ "${cfg_evaluate}" = "None" ]; then
    echo "Skipping evaluation"
else
    echo "Starting evaluation"
    python -m asdit.bin.evaluate experiments="${cfg_evaluate}" \
    seed="${seed}" dcase="${dcase}" \
    name="${name}" version="${version}" \
    infer_ver="${infer_ver}" machine="${machine}" dcase="${dcase}"
fi

# umap
if [ "${cfg_umap}" = "None" ]; then
    echo "Skipping umap"
else
    echo "Starting umap"
    python -m asdit.bin.umap experiments="${cfg_umap}" \
    seed="${seed}" dcase="${dcase}" \
    name="${name}" version="${version}" \
    infer_ver="${infer_ver}" machine="${machine}"
fi

# table
if [ "${cfg_table}" = "None" ]; then
    echo "Skipping table"
else
    echo "Starting table"
    python -m asdit.bin.table experiments="${cfg_table}" \
    seed="${seed}" dcase="${dcase}" \
    name="${name}" version="${version}" \
    infer_ver="${infer_ver}"
fi



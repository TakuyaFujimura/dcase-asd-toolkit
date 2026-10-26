#!/usr/bin/env bash

# Set shared paths for the calling yearly script after validating data_dir.
setup_download_paths() {
    local data_dir="$1"
    local dcase="$2"

    if [[ "$data_dir" != /* ]]; then
        echo "Error: data_dir is a relative path: ${data_dir}. Relative paths configured in run.sh must be relative to jobs/download/. Please use an absolute path for safety." >&2
        return 1
    fi

    # These variables are intentionally shared with the calling script.
    dst_dir="${data_dir}/original"
    dev_dir="${dst_dir}/${dcase}/dev_data/raw"
    eval_dir="${dst_dir}/${dcase}/eval_data/raw"
}

# Record completion only after both download and extraction succeed.
downoad_and_unzip() {
    local url="$1"
    local dst_dir="$2"
    local zip_name="${url##*/}"

    if [[ -f "${dst_dir}/.completed/${zip_name}.done" ]]; then
        echo "Already completed: ${zip_name}"
        return 0
    fi

    mkdir -p "${dst_dir}/.completed" || return 1
    (
        cd "$dst_dir" || exit 1
        curl --fail --location --output "${zip_name}.part" "$url" || exit 1
        mv "${zip_name}.part" "$zip_name" || exit 1
        # Overwrite files from an interrupted extraction without prompting.
        # Skip Windows zone metadata with inconsistent filenames in some archives.
        unzip -o "$zip_name" -x '*Zone.Identifier' || exit 1
        touch ".completed/${zip_name}.done"
    )
}

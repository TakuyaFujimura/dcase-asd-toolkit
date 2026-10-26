#!/usr/bin/env bash
# Source from job scripts. Customize this file to use a different environment.

asdkit_activate() {
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1

    if [[ -f "${repo_root}/.venv/bin/activate" ]]; then
        source "${repo_root}/.venv/bin/activate" || return 1
    else
        echo "No .venv found. Create .venv or customize jobs/activate.sh to use your own environment." >&2
        return 1
    fi
}

asdkit_activate

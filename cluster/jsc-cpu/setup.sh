#!/bin/bash
set -euo pipefail

SOURCE_PATH="${BASH_SOURCE[0]}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
PROJECT_ROOT="$(realpath "${ABSOLUTE_PATH}/../..")"

source "${ABSOLUTE_PATH}/config.sh"
source "${ABSOLUTE_PATH}/modules.sh"

if [[ -d "${ENV_DIR}" ]]; then
    echo "Removing existing JSC CPU venv at ${ENV_DIR}"
    rm -rf "${ENV_DIR}"
fi

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}/activate.sh"

python3 -m pip install --upgrade -r "${ABSOLUTE_PATH}/requirements.txt"
python3 -m pip install -e "${PROJECT_ROOT}"

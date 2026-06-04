SOURCE_PATH="${BASH_SOURCE[0]}"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1 )

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

export ENV_NAME="harpl-jsc-cpu"
export ENV_DIR="${ABSOLUTE_PATH}/venv"

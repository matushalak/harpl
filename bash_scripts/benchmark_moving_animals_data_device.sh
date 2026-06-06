#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_DEVICE="${HARPL_BENCH_TRAIN_DEVICE:-cuda}"
CPU_WORKERS="${HARPL_BENCH_CPU_WORKERS:-8}"
EPOCHS="${HARPL_BENCH_EPOCHS:-3}"
NUM_SEQUENCES="${HARPL_BENCH_NUM_SEQUENCES:-16000}"
BATCH_SIZE="${HARPL_BENCH_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${HARPL_BENCH_VAL_BATCH_SIZE:-128}"
SEED="${HARPL_BENCH_SEED:-42}"
RUN_ID="${HARPL_BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BENCH_ROOT="${HARPL_BENCH_ROOT:-${REPO_ROOT}/runs/moving_animals_data_device_${RUN_ID}}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${BENCH_ROOT}/logs" "${BENCH_ROOT}/checkpoints"

if ! "${PYTHON_BIN}" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
    echo "CUDA is not available in this Python environment; run this benchmark on a CUDA host." >&2
    exit 1
fi

run_case() {
    local name="$1"
    local spritevid_device="$2"
    local num_workers="$3"
    shift 3

    local log_path="${BENCH_ROOT}/logs/${name}.log"
    echo "=== ${name} ==="
    echo "spritevid_device=${spritevid_device} num_workers=${num_workers} log=${log_path}"

    (
        export HARPL_EPOCHS="${EPOCHS}"
        export HARPL_NUM_SEQUENCES="${NUM_SEQUENCES}"
        export HARPL_BATCH_SIZE="${BATCH_SIZE}"
        export HARPL_VAL_BATCH_SIZE="${VAL_BATCH_SIZE}"
        export HARPL_OFFLINE_BATCH_SIZE="${VAL_BATCH_SIZE}"
        export HARPL_OFFLINE_EPOCHS=0
        export HARPL_CHECKPOINT_EVERY=0
        export HARPL_NUM_WORKERS="${num_workers}"

        /usr/bin/time -p bash "${SCRIPT_DIR}/moving_animals.sh" \
            --device "${TRAIN_DEVICE}" \
            --seed "${SEED}" \
            --experiment_name "bench_${name}_${RUN_ID}" \
            --checkpoint_dir "${BENCH_ROOT}/checkpoints" \
            --nolog \
            --logger none \
            --online_eval_every 0 \
            --skip_final_eval \
            --offline_task none \
            --spritevid_device "${spritevid_device}" \
            "$@"
    ) 2>&1 | tee "${log_path}"
}

run_case "cpu_render_workers_${CPU_WORKERS}" "cpu" "${CPU_WORKERS}" --pin-memory
run_case "cuda_render_workers_0" "${TRAIN_DEVICE}" "0" --no-pin-memory

echo "Benchmark logs written to ${BENCH_ROOT}/logs"

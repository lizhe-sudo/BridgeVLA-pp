#!/usr/bin/env bash
# Astra RLBench launcher with the display and CoppeliaSim environment used by eval.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(dirname "${SCRIPT_DIR}")"
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"

export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export RLBENCH_DATA_FOLDER="${RLBENCH_DATA_FOLDER:-${BRIDGEVLA_DATA_ROOT}/RLBench}"
export RLBENCH_SIM_STACK="${RLBENCH_SIM_STACK-${FINETUNE_DIR}/bridgevla/libs/RLBench_peract587}"
export PYREP_SIM_STACK="${PYREP_SIM_STACK-${FINETUNE_DIR}/bridgevla/libs/PyRep_stepjam231}"

RLBENCH_CONDA_ENV="${RLBENCH_CONDA_ENV:-bridgevla_plus_rlbench}"
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ -z "${CONDA_PREFIX:-}" || "${ACTIVE_ENV}" != "${RLBENCH_CONDA_ENV}" ]]; then
    if [[ -z "${CONDA_BASE:-}" ]]; then
        if command -v conda >/dev/null 2>&1; then
            CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        elif [[ -n "${CONDA_EXE:-}" ]]; then
            CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
        fi
    fi
    if [[ -z "${CONDA_BASE:-}" || ! -f "${CONDA_BASE}/bin/activate" ]]; then
        echo "[eval_astra.sh] ERROR: activate the bridgevla_plus_rlbench conda environment first, or set CONDA_BASE." >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/bin/activate" "${RLBENCH_CONDA_ENV}"
fi

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}:/usr/lib/x86_64-linux-gnu/qt5/plugins"

XVFB_PID=""
cleanup_astra_xvfb() {
    if [[ -n "${XVFB_PID}" ]] && kill -0 "${XVFB_PID}" 2>/dev/null; then
        kill "${XVFB_PID}" 2>/dev/null || true
        wait "${XVFB_PID}" 2>/dev/null || true
    fi
}
trap cleanup_astra_xvfb EXIT

XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
if [[ -z "${DISPLAY:-}" ]] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    if xdpyinfo -display "${XVFB_DISPLAY}" >/dev/null 2>&1; then
        export DISPLAY="${XVFB_DISPLAY}"
    else
        Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
        sleep 2
        if ! kill -0 "${XVFB_PID}" 2>/dev/null; then
            echo "[eval_astra.sh] ERROR: Xvfb failed to start on ${XVFB_DISPLAY}" >&2
            exit 2
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    fi
fi

cd "${SCRIPT_DIR}"
python eval_astra.py "$@"

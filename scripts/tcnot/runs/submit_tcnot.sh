#!/bin/bash

# Run directly from a Yale login node. The login invocation validates the Python
# configuration and submits this same file as a one-based Slurm array.

#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=runjiang.bi@yale.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -euo pipefail

SCRIPT_RELATIVE_PATH="scripts/tcnot/runs/submit_tcnot.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
fi

SUBMIT_SCRIPT="$REPO_ROOT/$SCRIPT_RELATIVE_PATH"
RUN_SCRIPT="$REPO_ROOT/scripts/tcnot/runs/run_tcnot_sim.py"
CONDA_ENV="${CONDA_ENV:-surface-leakage}"

cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if ! command -v conda >/dev/null 2>&1; then
    ml miniconda
fi

run_python() {
    conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

verify_environment() {
    run_python -c \
        'import importlib.metadata, sys, surface_code_leakage_erasure, tesseract_decoder; expected="0.1.1.dev20260822020007"; actual=importlib.metadata.version("tesseract-decoder"); print(f"Python: {sys.executable}"); print(f"surface-code-leakage-erasure: {surface_code_leakage_erasure.__file__}"); print(f"Tesseract: {tesseract_decoder.__file__} ({actual})"); assert actual == expected, f"Expected tesseract-decoder {expected}, got {actual}"'
}

verify_environment

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p slurm
    NUM_SHARDS="$(run_python "$RUN_SCRIPT" --print-num-shards)"
    RUN_TAG="$(run_python "$RUN_SCRIPT" --print-run-tag)"
    run_python "$RUN_SCRIPT" --dry-run
    echo "Submitting ${NUM_SHARDS} shards for ${RUN_TAG}."
    sbatch \
        --array="1-${NUM_SHARDS}" \
        --chdir="$REPO_ROOT" \
        --job-name="$RUN_TAG" \
        "$SUBMIT_SCRIPT"
    exit 0
fi

run_python "$RUN_SCRIPT"

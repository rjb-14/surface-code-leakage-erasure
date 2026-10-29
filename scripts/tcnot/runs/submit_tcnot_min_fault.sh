#!/bin/bash

# Invoke with bash from the login node. Extra arguments are passed to sbatch, e.g.
# bash scripts/tcnot/runs/submit_tcnot_min_fault.sh \
#     --cpus-per-task=16 --time=08:00:00
# Resubmit the SAME command/tag to resume interrupted shards.

#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=runjiang.bi@yale.edu
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" || "${1:-}" == '--summarize-job' ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
fi
cd "$REPO_ROOT"
RUN_SCRIPT="$REPO_ROOT/scripts/tcnot/runs/run_tcnot_min_fault.py"
SUBMIT_SCRIPT="$REPO_ROOT/scripts/tcnot/runs/submit_tcnot_min_fault.sh"
CONDA_ENV="${CONDA_ENV:-surface-leakage}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-8}"
if [[ ! "$MAX_CONCURRENT_TASKS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'MAX_CONCURRENT_TASKS must be a positive integer.' >&2
    exit 1
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if ! command -v conda >/dev/null 2>&1; then
    ml miniconda
fi
# Activate once so the batch shell can exec Python directly. SLURM's B:USR1
# pre-timeout signal then reaches the coordinator instead of a conda wrapper.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ "${1:-}" == '--summarize-job' ]]; then
    exec python "$RUN_SCRIPT" --summarize
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    exec python "$RUN_SCRIPT" --worker
fi

mkdir -p slurm
python -c 'import sys, importlib.metadata, stim, tesseract_decoder; expected="0.1.1.dev20260822020007"; actual=importlib.metadata.version("tesseract-decoder"); print(f"Python: {sys.executable}"); print(f"Stim: {stim.__version__}"); print(f"Tesseract: {tesseract_decoder.__file__} ({actual})"); assert actual == expected, f"Expected tesseract-decoder {expected}, got {actual}"'
python "$RUN_SCRIPT" --prepare
NUM_SHARDS="$(python "$RUN_SCRIPT" --print-num-shards)"
RUN_TAG="$(python "$RUN_SCRIPT" --print-run-tag)"
SUBMITTED="$(sbatch --parsable --array="1-${NUM_SHARDS}%${MAX_CONCURRENT_TASKS}" \
    --chdir="$REPO_ROOT" --job-name="$RUN_TAG" "$@" "$SUBMIT_SCRIPT")"
JOB_ID="${SUBMITTED%%;*}"
echo "Submitted array $JOB_ID: $NUM_SHARDS shards, up to $MAX_CONCURRENT_TASKS concurrently."
sbatch --dependency="afterany:$JOB_ID" --chdir="$REPO_ROOT" \
    --job-name="${RUN_TAG}_summary" --ntasks=1 --cpus-per-task=1 \
    --mem-per-cpu=2G --time=00:10:00 \
    "$SUBMIT_SCRIPT" --summarize-job

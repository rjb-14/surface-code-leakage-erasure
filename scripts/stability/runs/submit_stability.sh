#!/bin/bash

# Invoke directly on a Yale login node. This file validates the configuration
# and submits itself as a one-based Slurm array, as submit_tcnot.sh does.

#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=runjiang.bi@yale.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:?Missing SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
fi

SUBMIT_SCRIPT="$REPO_ROOT/scripts/stability/runs/submit_stability.sh"
RUN_SCRIPT="$REPO_ROOT/scripts/stability/runs/run_stability_sim.py"
CONDA_ENV="${CONDA_ENV:-surface-leakage}"
export CONDA_ENV

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

# Upstream PyMatching silently ignores edge_reweights in decode(); checking
# only its version cannot distinguish it from the required fork.
run_python -c \
    'import sys, surface_code_leakage_erasure, pymatching; from surface_code_leakage_erasure.decoding import require_pymatching_edge_reweights; require_pymatching_edge_reweights(); print(f"Python: {sys.executable}"); print(f"surface-code-leakage-erasure: {surface_code_leakage_erasure.__file__}"); print(f"PyMatching: {pymatching.__file__} (edge reweighting verified)")'

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

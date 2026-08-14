#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=leak_only_stability
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=runjiang.bi@yale.edu
#SBATCH --output=scripts/stability/runs/log/slurm-%j.out
#SBATCH --error=scripts/stability/runs/log/slurm-%j.err

###########################################
# please DO NOT remove 'module load StdEnv'
###########################################
module load StdEnv
###########################################

module load GCC
module load uv

# IMPORTANT: submit this from the repo root (surface-code-leakage-erasure/),
# not from inside scripts/stability/runs/ -- e.g.:
#   sbatch scripts/stability/runs/submit_leak_only.sh static
# SLURM's relative --output/--error paths above and the .venv path below are
# both resolved relative to SLURM_SUBMIT_DIR, i.e. wherever `sbatch` was run
# from -- resolving paths via $0 instead is unreliable under SLURM, since it
# may execute a spooled copy of this script rather than the original file.
source .venv/bin/activate

# Usage (from repo root):
#   sbatch scripts/stability/runs/submit_leak_only.sh static
#   sbatch scripts/stability/runs/submit_leak_only.sh early -d 4 --rounds 2 4
#   sbatch scripts/stability/runs/submit_leak_only.sh late  --rounds 2 3 4 5
#
# $1 is the style (static/early/late); everything after it is passed
# straight through to run_leak_only_sim.py, so you can override distance,
# rounds, p_leak_list, etc. from the sbatch command line.
STYLE=$1
shift

uv run python scripts/stability/runs/run_leak_only_sim.py \
    --style "$STYLE" --outdir scripts/stability/runs/scratch "$@"

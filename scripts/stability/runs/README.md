# Leakage-only stability LER runs (Yale cluster)

Sweeps logical error rate (p_Pauli=0, leakage only) over rounds and p_leak for
the stability circuit, in static / early-swap / late-swap variants.

## One-time setup on the cluster

```bash
git clone git@github.com:rjb-14/surface-code-leakage-erasure.git
cd surface-code-leakage-erasure
git checkout runjiang/stability

module load StdEnv
module load GCC      # C++ toolchain, needed to compile the PyMatching fork
module load uv

uv venv
source .venv/bin/activate
uv pip install -e ".[analysis]"   # compiles the PyMatching fork -- can take a few minutes

mkdir -p scripts/stability/runs/log scripts/stability/runs/scratch
```

`.venv` ends up at the repo root (`surface-code-leakage-erasure/.venv`), which
is where `submit_leak_only.sh` expects it.

## Submitting jobs

Always submit from the **repo root** (not from inside this `runs/`
directory) -- `submit_leak_only.sh`'s relative paths (`.venv`, log output,
the python script itself) are resolved relative to wherever `sbatch` was
invoked from:

```bash
cd surface-code-leakage-erasure
sbatch scripts/stability/runs/submit_leak_only.sh static
sbatch scripts/stability/runs/submit_leak_only.sh early
sbatch scripts/stability/runs/submit_leak_only.sh late
```

The first argument is the style (`static`/`early`/`late`); anything after it
is forwarded to `run_leak_only_sim.py`, e.g. to override the defaults
(distance=2, rounds=[2,3,4,5], p_leak=[0.01,0.015,0.02,0.025,0.03]):

```bash
sbatch scripts/stability/runs/submit_leak_only.sh early  -d 4 --rounds 2 4
sbatch scripts/stability/runs/submit_leak_only.sh static --rounds 2 3 --min-samples 4096
```

Run `uv run python scripts/stability/runs/run_leak_only_sim.py --help` for
the full list of overridable parameters.

## Output

- SLURM logs: `scripts/stability/runs/log/slurm-<jobid>.out`/`.err`
- Results: `scripts/stability/runs/scratch/leak_only_style-<style>_d<distance>_depolarize_job<jobid>.dat`,
  one tab-separated line per (rounds, p_leak) pair:
  `style  noise_model  distance  p_leak  p_Pauli("inf" placeholder)  ec_sched  rounds  <format_results(...) columns>`

## Known limitation

`static` style (`StabilityCircuitBuilder`) currently only supports **even**
code distance -- odd `d` fails with a stim "non-deterministic observables"
error (a pre-existing limitation of the circuit construction, not something
introduced by these scripts). `early`/`late` (`WalkingStabilityCircuitBuilder`)
require even `d` too (enforced by `WalkingStabilityCircuitLayout`).

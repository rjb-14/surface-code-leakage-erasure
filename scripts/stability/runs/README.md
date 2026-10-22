# Single-patch stability cluster runs

`run_stability_sim.py` and `submit_stability.sh` provide the configurable,
resumable workflow corresponding to `10_interleaved_stability.py` in
`biased-noise-qec` and this repository's `run_tcnot_sim.py`. They use this
repository's **single-patch** static / early-walking / late-walking stability
circuits, with no transversal logical gate.

## Setup

Use the same Conda environment as the tCNOT runs. Tesseract is not needed for
this driver; erasure-aware single-patch decoding requires the PyMatching fork
declared in `pyproject.toml`:

```bash
ml miniconda
# Create this environment only if it does not already exist:
conda create -n surface-leakage python=3.12
conda run -n surface-leakage python -m pip install -e ".[dev]"
```

The submission wrapper verifies that PyMatching explicitly supports
`edge_reweights`. Upstream PyMatching reports the same version as the fork but
does not implement the required interface.

## Configure, inspect, and submit

Edit the configuration block in `run_stability_sim.py`:

- `STYLES`: any subset of `("static", "early", "late")`.
- `D_LIST`: positive even distances.
- `ROUNDS_MODE="explicit"`: full `STYLES × D_LIST × R_LIST × P_LEAK_LIST`
  sweep. With `"match_distance"`, use `rounds=d` for each distance instead.
- `BIAS=float("inf")`: leakage only. A finite bias sets
  `p_Pauli=p_leak/BIAS`, with uniform `DEPOLARIZE2(p_Pauli)` after two-qubit
  gates only. Initialization, measurement, and idle locations have no added
  Pauli noise.
- `DECODER_STRATEGY="marginal"` or `"quasi_mle"` (see below).
- `MAX_SHOTS`, `MAX_ERRORS`, `MIN_SHOTS`: per-point budgets. Zero `MAX_ERRORS`
  disables error stopping. Error stopping is checked between batches and may
  exceed the target by one batch; `MAX_SHOTS` is a hard cap.
- `RUN_TAG`: choose a new tag when the sweep, physics, or decoder settings
  change. Budgets can be increased under the same tag to extend a run.

The current physical model is Z-basis stability, `ec_sched=8`, skip-gate leakage,
and **one Bernoulli trial per two-qubit gate**, choosing either qubit uniformly.
Other leakage effects and per-qubit trials are rejected by configuration
validation rather than being silently substituted.

From the repository root:

```bash
conda run --no-capture-output -n surface-leakage \
  python scripts/stability/runs/run_stability_sim.py --dry-run

# Run a single one-based shard locally as a pilot. Lower budgets first.
conda run --no-capture-output -n surface-leakage \
  python scripts/stability/runs/run_stability_sim.py --task-index 1

# Run directly on a Yale login node; it invokes sbatch itself.
./scripts/stability/runs/submit_stability.sh
```

Set `CONDA_ENV=another-name` when using a different environment. The wrapper
uses 4 CPUs per array element; Python uses `SLURM_CPUS_PER_TASK` workers, or one
worker outside Slurm. Leakage-only runs group the full probability sweep by
`(style, d, rounds)` to reuse circuits and decoder caches. Finite-bias runs use
one physical point per shard. The checked-in configuration (early/late, d=4,
rounds=3/5/7, 21 probabilities) has **126 physical points in 6 shards**.

`--print-num-shards` and `--print-run-tag` support submission scripting. `--all`
explicitly runs every shard sequentially; invoking the driver without a shard
selection refuses to start the full production sweep.

## Decoding and statistics

Marginal decoding uses the existing `ErasureDecoder`. The driver supplements
its sparse Pauli graph with missing edges from the triggered erasure DEMs,
including their logical labels. Added edges use the existing decoder's `1e-9`
probability floor and are reweighted according to the erasure DEM. This avoids
the missing-edge failure of using a two-qubit-only Pauli graph directly while
keeping PyMatching's weight quantization accurate. The graph probability floor
is a decoding approximation; physical sampling uses the configured probability,
including exactly zero.

`quasi_mle` uses this repository's existing branch-and-bound search. Default
`DECODER_ARGS=None` resolves to:

```python
{"mle_time_limit": None, "max_queue_size": 4096}
```

Set a finite `mle_time_limit` in seconds to bound each search. Queue trimming
still applies; this is **not** the reference repository's exact `brute_force`
mode. If the search returns no valid solution, its marginal prediction is used,
matching the existing `sample_LER` convention. Only the selected strategy's
failures count toward the error budget and CSV.

Each shot belongs to the leakage/no-leakage population with a binomial draw
using `1-(1-p_leak)**N` for `N` gate trials. Leakage shots are sampled
conditionally on at least one event; no-leakage shots use the Pauli-only circuit.
This gives ordinary integer Monte Carlo failures/shots for Sinter, rather than
converting separately weighted LER estimates into fictitious error counts.
Each process keeps its own builder and decoder cache, and each resumed run draws
fresh randomness.

## Results and resuming

Each shard appends batch statistics to its own Sinter CSV under
`data/stability/runs/<RUN_TAG>/`. The output directory also contains
`identity.json`; incompatible configurations under the same tag are rejected.
Slurm logs go to `slurm/<RUN_TAG>-<array job>_<array index>.out` and `.err`.

Resubmit the same configuration or rerun the same `--task-index` to count
existing shots/errors toward the budgets. Completed points are skipped. You can
increase `MAX_SHOTS`, `MAX_ERRORS`, or `MIN_SHOTS` without changing the tag.
Keep this workflow separate from the older `.dat` outputs; they have different
statistics and are not imported as resume data.

---

# Legacy leakage-only stability LER runs (Yale cluster)

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

# Single-patch stability cluster runs

`run_stability_sim.py` and `submit_stability.sh` provide the configurable,
resumable workflow corresponding to `10_interleaved_stability.py` in
`biased-noise-qec` and this repository's `run_tcnot_sim.py`. They use this
repository's **single-patch** stability circuits, with no transversal logical
gate. The original Z-basis experiment supports static / early / late; the new
X-basis experiment uses the clipped-corner early-walking circuit.

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

- `STABILITY_EXPERIMENT="z"` (default) or `"x"`. Names refer to **data
  preparation/readout basis**: Z preserves the original experiment with final
  X-check parity; X uses the new circuit with final **Z-check parity**.
- `STYLES`: any subset of `("static", "early", "late")` for Z. X currently
  requires `STYLES=("early",)`; static/late X combinations are rejected.
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

Both experiments use `ec_sched=8`, skip-gate leakage,
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

For the new X-basis experiment, either edit the configuration block:

```python
STABILITY_EXPERIMENT = "x"
STYLES = ("early",)
RUN_TAG = "stability_x_early_leak_only_marginal"
```

or override the experiment and styles from the command line:

```bash
conda run --no-capture-output -n surface-leakage \
  python scripts/stability/runs/run_stability_sim.py \
  --stability-experiment x --style early --dry-run

# After lowering the configured budgets for a local pilot:
conda run --no-capture-output -n surface-leakage \
  python scripts/stability/runs/run_stability_sim.py \
  --stability-experiment x --style early --task-index 1

# On the cluster login node; arguments are forwarded to every array worker:
./scripts/stability/runs/submit_stability.sh \
  --stability-experiment x --style early
```

`--stability-experiment` accepts `x/X` and `z/Z`; `--style` accepts one or more
styles. X requires only `early`. The new X circuit preserves the d=2 special
geometry and clips the four data corners at d>=4. Both marginal and quasi-MLE,
leakage-only and finite-bias sampling use the selected experiment.

Set `CONDA_ENV=another-name` when using a different environment. The wrapper
uses 4 CPUs per array element; Python uses `SLURM_CPUS_PER_TASK` workers, or one
worker outside Slurm. Leakage-only runs group the full probability sweep by
`(style, d, rounds)` to reuse circuits and decoder caches. Finite-bias runs use
one physical point per shard. The checked-in configuration (early/late, d=4,
rounds=3/5/7, 21 probabilities) has **126 physical points in 6 shards**.
Selecting X with `--style early` gives **63 physical points in 3 shards** for
the same distances, rounds, and probabilities. Finite bias still gives one
physical point per shard.

`--print-num-shards`, `--print-run-tag`, and `--print-job-name` support submission
scripting. X job names have an `x_` prefix; `--print-run-tag` still returns the
configured tag unchanged. `--all`
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
matching the existing `sample_LER` convention. With `quasi_mle`, the same shots
also accumulate marginal failures using the prediction already returned by the
decoder; no additional sampling or decoding is needed. Only quasi-MLE failures
count toward the error budget, so both results have the same shot count.

Each shot belongs to the leakage/no-leakage population with a binomial draw
using `1-(1-p_leak)**N` for `N` gate trials. Leakage shots are sampled
conditionally on at least one event; no-leakage shots use the Pauli-only circuit.
This gives ordinary integer Monte Carlo failures/shots for Sinter, rather than
converting separately weighted LER estimates into fictitious error counts.
Each process keeps its own builder and decoder cache, and each resumed run draws
fresh randomness.

## Results and resuming

Each shard appends batch statistics to its own Sinter CSV. Original Z results
stay under `data/stability/runs/<RUN_TAG>/`; X results use
`data/stability/runs/x/<RUN_TAG>/` and an additional `_x_early_` filename label.
The output directory also contains
`identity.json`; incompatible configurations under the same tag are rejected.
Z Slurm logs retain `slurm/<RUN_TAG>-<array job>_<array index>.out` and `.err`;
X logs use `slurm/x_<RUN_TAG>-<array job>_<array index>.out` and `.err`.

X manifests/CSV metadata explicitly record `stability_experiment="x"`,
`basis="X"`, `mode="walking"`, `swap_time="early"`, the clipped-corner
geometry, and `observable="final_regular_z_check_parity"`. Original Z manifest
schemas and CSV identities remain unchanged, so existing Z runs can resume;
legacy metadata without `stability_experiment` retains its original Z meaning.
The worker cache key includes the experiment, preventing the two geometries
from sharing builders or decoder caches.

For `quasi_mle`, every batch writes two rows, distinguished by the CSV `decoder`
column (`quasi_mle` and `marginal`). The marginal row's metadata records
`paired_with="quasi_mle"` and `paired_decoder_args` to identify its source run.
Compute time is charged to the quasi-MLE row; the marginal row has `seconds=0`
to avoid double counting. Marginal-only runs still write one row per batch.

Resubmit the same configuration or rerun the same `--task-index` to count
existing shots/errors toward the budgets. Completed points are skipped. You can
increase `MAX_SHOTS`, `MAX_ERRORS`, or `MIN_SHOTS` without changing the tag.
Both decoder records resume together. Choose a new `RUN_TAG` for quasi-MLE runs
created before paired recording was added: their missing marginal counts cannot
be reconstructed, and the old manifest is deliberately rejected. Direct CSV
collection also rejects unequal quasi-MLE/marginal shot counts before appending.
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

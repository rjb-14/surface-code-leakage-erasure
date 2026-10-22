# Early-walking tCNOT cluster runs

This directory contains the unified memory/stability tCNOT Monte Carlo driver and
its Yale Slurm wrapper. The supported production model is deliberately fixed to:

- memory detectors: `both`;
- leakage effect: `skip_gate` (`skip gates` in the circuit builder);
- leakage granularity: `per_gate`;
- decoder: marginal Tesseract;
- Tesseract: `0.1.1.dev20260822020007`, matching `biased-noise-qec`.

## First cluster setup

Push the tested local commit, then on the cluster:

```bash
git pull
ml miniconda
conda create -n surface-leakage python=3.12
conda run -n surface-leakage python -m pip install -e ".[tesseract,dev]"
```

Verify the environment and pinned Tesseract version:

```bash
conda run -n surface-leakage python -c \
  'import importlib.metadata, sys, surface_code_leakage_erasure, tesseract_decoder; print(sys.executable); print(surface_code_leakage_erasure.__file__); print(tesseract_decoder.__file__); print(importlib.metadata.version("tesseract-decoder"))'
```

The final printed version must be `0.1.1.dev20260822020007`.

## Configure and inspect

Edit the configuration block at the top of `run_tcnot_sim.py`, including a new
`RUN_TAG` whenever the physical model, decoder configuration, or sweep changes.

```bash
conda run --no-capture-output -n surface-leakage \
  python scripts/tcnot/runs/run_tcnot_sim.py --dry-run
```

Run a single one-based shard as a pilot:

```bash
conda run --no-capture-output -n surface-leakage \
  python scripts/tcnot/runs/run_tcnot_sim.py --task-index 1
```

## Submit

From the repository checkout on the cluster:

```bash
./scripts/tcnot/runs/submit_tcnot.sh
```

The wrapper verifies the environment, checks the resolved shard count and run
identity, prints the dry-run configuration, and then submits a Slurm array. Set
`CONDA_ENV` only if an intentionally different environment name is required.
Each array element writes its own incremental Sinter CSV under
`data/tcnot/runs/<RUN_TAG>/`. Requeueing or submitting the same configuration again
resumes from prior shots/errors. Increasing `MAX_SHOTS` or `MAX_ERRORS` resumes the
same task; changing physics or decoder settings requires a new `RUN_TAG` because
the existing `identity.json` is checked fail-closed.

## Supplemental d=7 sweep: one job per probability

The original `run_tcnot_sim.py` and `submit_tcnot.sh` retain the bundled
memory/stability sweep. The independent per-probability version is in
`run_tcnot_sim_per_p.py` and `submit_tcnot_per_p.sh`.

The new driver runs only memory at `d=7`, with seven rounds per side. All 21
original leakage probabilities, from `0.002` to `0.2`, are retained. Each
probability has its own array task and CSV; task 1 uses the largest probability
and task 21 uses the smallest. The run tag is
`early_walking_tcnot_leak_only_memory_d7_r7_point_psweep_marginal_long_beam`.

Inspect and submit this new version with:

```bash
conda run --no-capture-output -n surface-leakage \
  python scripts/tcnot/runs/run_tcnot_sim_per_p.py --dry-run
./scripts/tcnot/runs/submit_tcnot_per_p.sh
```

Each task requests 8 CPUs and 24 GB total memory. By default all 21 tasks may
run concurrently when resources are available (168 CPUs for this array).
To change this limit, for example:

```bash
MAX_CONCURRENT_TASKS=2 ./scripts/tcnot/runs/submit_tcnot_per_p.sh
```

The supplemental run writes separate CSVs under `data/tcnot/runs/<RUN_TAG>/`.
Resubmitting the new version resumes its own CSVs; old sweep CSVs are not used.

The supplemental CSVs can later be combined with the old data after checking
that the physical model and decoder options match. Sum errors and accepted shots
for each matching physical point, then calculate the combined error rate; do not
average the two rates. The new `run_tag` differs, so readers that group by complete
metadata or require the old tag need to handle that explicitly. Read each sampled
record only once when combining the files. Old and new probability values may
differ by floating-point rounding (for example `0.2` versus
`0.20000000000000004`); match the intended grid points with a small tolerance.

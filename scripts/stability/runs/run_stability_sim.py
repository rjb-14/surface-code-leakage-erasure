"""Run single-patch static/early/late stability Monte Carlo sweeps.

Edit the configuration block, inspect with --dry-run, then run
./scripts/stability/runs/submit_stability.sh on a Yale login node.
Leakage-only shards bundle the probability sweep for each (style, d, rounds).
Finite-bias shards contain one physical point because their Pauli circuit varies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# =============================================================================
# EDIT THIS CONFIGURATION BLOCK BEFORE SUBMITTING
# =============================================================================

# Select any subset of ("static", "early", "late"). All require even distance.
STYLES = ("early", "late")
D_LIST = [4]
ROUNDS_MODE = "explicit"  # "explicit": D x R; "match_distance": rounds=d
R_LIST = [3, 5, 7]
P_LEAK_LIST = np.logspace(np.log10(2e-3), np.log10(2e-1), 21)

# eta_e = p_leak / p_Pauli. Finite bias adds uniform DEPOLARIZE2(p_Pauli)
# after two-qubit gates only; infinity means leakage only.
BIAS = float("inf")

# Uses the repository's erasure-aware PyMatching fork, not Tesseract.
DECODER_STRATEGY = "marginal"  # "marginal" or "quasi_mle"
# quasi_mle saves both quasi_mle and marginal results from the same shots.
# Shot/error budgets use quasi_mle; choose a new RUN_TAG for older quasi_mle runs.
# quasi_mle defaults: {"mle_time_limit": None, "max_queue_size": 4096}.
# This is the existing branch-and-bound search, with marginal fallback when no
# valid solution is found. It is not the reference repo's brute_force mode.
DECODER_ARGS = None

# Current single-patch builders implement per-gate trials and these model names.
LEAKAGE_EFFECT = "skip_gate"
LEAKAGE_GRANULARITY = "per_gate"

MAX_SHOTS = 10_000_000
MAX_ERRORS = 1_000  # 0 disables error stopping
MIN_SHOTS = 0
DECODER_CACHE_SIZE = 4096

# Change this when the sweep, physical model, or decoder settings change.
# Increasing shot/error budgets can reuse the tag and resumes existing CSVs.
RUN_TAG = "stability_early_plus_late_leak_only_marginal"

# =============================================================================
# END CONFIGURATION BLOCK
# =============================================================================


SUPPORTED_STYLES = ("static", "early", "late")
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunConfig:
    styles: tuple[str, ...]
    d_list: tuple[int, ...]
    rounds_mode: str
    r_list: tuple[int, ...]
    p_leak_list: tuple[float, ...]
    bias: float
    decoder_strategy: str
    decoder_args: dict | None
    leakage_effect: str
    leakage_granularity: str
    max_shots: int
    max_errors: int
    min_shots: int
    decoder_cache_size: int
    run_tag: str

    @property
    def leakage_only(self):
        return math.isinf(self.bias)

    @property
    def bias_label(self):
        return "inf" if self.leakage_only else format(self.bias, ".17g")


@dataclass(frozen=True)
class SweepTask:
    style: str
    d: int
    rounds: int
    p_idx: int
    p_leak: float


def _integer(name, value, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _integers(name, values, allow_empty=False):
    converted = tuple(_integer(name, value, 1) for value in values)
    if not converted and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    if len(converted) != len(set(converted)):
        raise ValueError(f"{name} must not contain duplicates.")
    return converted


def configured_run():
    styles = tuple(STYLES)
    if (
        not styles
        or len(styles) != len(set(styles))
        or any(style not in SUPPORTED_STYLES for style in styles)
    ):
        raise ValueError(
            f"STYLES must be a non-empty, unique subset of {SUPPORTED_STYLES}."
        )
    d_list = _integers("D_LIST", D_LIST)
    if any(d % 2 for d in d_list):
        raise ValueError("Every D must be positive and even for stability.")
    if ROUNDS_MODE not in ("explicit", "match_distance"):
        raise ValueError("ROUNDS_MODE must be 'explicit' or 'match_distance'.")
    r_list = _integers("R_LIST", R_LIST, allow_empty=ROUNDS_MODE == "match_distance")
    if any(isinstance(p, (bool, np.bool_)) for p in P_LEAK_LIST):
        raise ValueError("P_LEAK_LIST must contain probabilities, not booleans.")
    probabilities = tuple(float(p) for p in P_LEAK_LIST)
    if not probabilities or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError(
            "P_LEAK_LIST must be non-empty with finite probabilities in [0, 1]."
        )
    if len(probabilities) != len(set(probabilities)):
        raise ValueError("P_LEAK_LIST must not contain duplicates.")
    if isinstance(BIAS, (bool, np.bool_)):
        raise ValueError("BIAS must be positive or infinity, not bool.")
    bias = float(BIAS)
    if math.isnan(bias) or bias <= 0:
        raise ValueError("BIAS must be positive or infinity.")
    if any(p / bias > 1 for p in probabilities):
        raise ValueError("Finite bias requires p_leak / BIAS <= 1 at every point.")
    if DECODER_STRATEGY not in ("marginal", "quasi_mle"):
        raise ValueError("DECODER_STRATEGY must be 'marginal' or 'quasi_mle'.")
    decoder_args = None
    if DECODER_STRATEGY == "marginal":
        if DECODER_ARGS is not None:
            raise ValueError("DECODER_ARGS must be None for marginal decoding.")
    else:
        decoder_args = {"mle_time_limit": None, "max_queue_size": 4096}
        decoder_args.update(DECODER_ARGS or {})
        if set(decoder_args) != {"mle_time_limit", "max_queue_size"}:
            raise ValueError(
                "DECODER_ARGS accepts only mle_time_limit and max_queue_size."
            )
        decoder_args["max_queue_size"] = _integer(
            "max_queue_size", decoder_args["max_queue_size"], 1
        )
        limit = decoder_args["mle_time_limit"]
        if limit is not None:
            if (
                isinstance(limit, (bool, np.bool_))
                or not isinstance(limit, (int, float))
                or not math.isfinite(limit)
                or limit < 0
            ):
                raise ValueError(
                    "mle_time_limit must be non-negative and finite, or None."
                )
            decoder_args["mle_time_limit"] = float(limit)
    if LEAKAGE_EFFECT != "skip_gate":
        raise ValueError("LEAKAGE_EFFECT is fixed to 'skip_gate'.")
    if LEAKAGE_GRANULARITY != "per_gate":
        raise ValueError("LEAKAGE_GRANULARITY is fixed to 'per_gate'.")
    max_shots = _integer("MAX_SHOTS", MAX_SHOTS, 1)
    max_errors = _integer("MAX_ERRORS", MAX_ERRORS)
    min_shots = _integer("MIN_SHOTS", MIN_SHOTS)
    if min_shots > max_shots:
        raise ValueError("MIN_SHOTS cannot exceed MAX_SHOTS.")
    cache_size = _integer("DECODER_CACHE_SIZE", DECODER_CACHE_SIZE)
    if not isinstance(RUN_TAG, str) or not RUN_TAG_PATTERN.fullmatch(RUN_TAG):
        raise ValueError(
            "RUN_TAG must start alphanumeric and contain only letters, numbers, dot, underscore, or hyphen."
        )
    return RunConfig(
        styles,
        d_list,
        ROUNDS_MODE,
        r_list,
        probabilities,
        bias,
        DECODER_STRATEGY,
        decoder_args,
        LEAKAGE_EFFECT,
        LEAKAGE_GRANULARITY,
        max_shots,
        max_errors,
        min_shots,
        cache_size,
        RUN_TAG,
    )


def build_task_groups(config):
    groups = []
    for style in config.styles:
        for d in config.d_list:
            rounds_list = (
                (d,) if config.rounds_mode == "match_distance" else config.r_list
            )
            for rounds in rounds_list:
                tasks = tuple(
                    SweepTask(style, d, rounds, index, p)
                    for index, p in enumerate(sorted(config.p_leak_list, reverse=True))
                )
                groups.extend(
                    (tasks,) if config.leakage_only else ((task,) for task in tasks)
                )
    return groups


def _identity(config):
    identity = {
        "schema_version": 1,
        "experiment": "stability",
        "styles": list(config.styles),
        "d_list": list(config.d_list),
        "rounds_mode": config.rounds_mode,
        "r_list": list(config.r_list) if config.rounds_mode == "explicit" else None,
        "p_leak_list": sorted(config.p_leak_list, reverse=True),
        "bias": config.bias_label,
        "basis": "Z",
        "pauli_noise": (
            "none" if config.leakage_only else "uniform_two_qubit_depolarizing"
        ),
        "pauli_locations": "2-qubit gates",
        "erasure_check_schedule": 8,
        "leakage_effect": config.leakage_effect,
        "leakage_granularity": config.leakage_granularity,
        "decoder_backend": "pymatching_edge_reweights",
        "decoder_graph_probability_floor": 1e-9,
        "decoder_strategy": config.decoder_strategy,
        "decoder_args": config.decoder_args,
        "decoder_cache_size": config.decoder_cache_size,
        "run_tag": config.run_tag,
    }
    if config.decoder_strategy == "quasi_mle":
        # Older quasi_mle runs did not retain marginal counts and cannot resume
        # into paired statistics. Marginal-only identities remain unchanged.
        identity.update(schema_version=2, recorded_decoders=["quasi_mle", "marginal"])
    return identity


def _output_directory(config):
    return REPO_ROOT / "data" / "stability" / "runs" / config.run_tag


def _ensure_identity_manifest(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "identity.json"
    expected = _identity(config)
    try:
        with manifest.open("x", encoding="utf8") as file:
            file.write(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return
    except FileExistsError:
        pass
    # Another array element may still be writing the newly created manifest.
    for attempt in range(20):
        try:
            actual = json.loads(manifest.read_text(encoding="utf8"))
            break
        except (OSError, json.JSONDecodeError):
            if attempt == 19:
                raise
            time.sleep(0.05)
    if actual != expected:
        raise ValueError(
            f"RUN_TAG {config.run_tag!r} has an incompatible identity manifest at {manifest}. Choose a new RUN_TAG."
        )


def _shard_label(group):
    task = group[0]
    base = f"{task.style}_d{task.d}_r{task.rounds}"
    return base if len(group) > 1 else f"{base}_p{task.p_idx}"


def _metadata(config, task, p_pauli):
    metadata = _identity(config)
    for key in ("styles", "d_list", "r_list", "p_leak_list"):
        metadata.pop(key)
    metadata.update(
        circuit=(
            "StabilityCircuitBuilder"
            if task.style == "static"
            else "WalkingStabilityCircuitBuilder"
        ),
        style=task.style,
        d=task.d,
        rounds=task.rounds,
        r=task.rounds,
        p=p_pauli,
        p_leak=task.p_leak,
    )
    return metadata


def _worker_count():
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError(f"Invalid SLURM_CPUS_PER_TASK={raw!r}.") from error
    if count < 1:
        raise ValueError("SLURM_CPUS_PER_TASK must be positive.")
    return count


def _build_sampler(config, task, p_pauli, n_jobs):
    from surface_code_leakage_erasure.stability_sampler import StabilitySampler

    return StabilitySampler(
        style=task.style,
        distance=task.d,
        rounds=task.rounds,
        p_pauli=p_pauli,
        n_jobs=n_jobs,
        cache_size=config.decoder_cache_size,
    )


def _run_group(config, group, shard_index, output_directory):
    from surface_code_leakage_erasure.collect import collect

    path = (
        output_directory
        / f"{config.run_tag}_s{shard_index:04d}_{_shard_label(group)}.csv"
    )
    n_jobs = _worker_count()
    sampler = None
    current_key = None
    for task in group:
        p_pauli = 0.0 if config.leakage_only else task.p_leak / config.bias
        key = (task.style, task.d, task.rounds, p_pauli)
        if key != current_key:
            sampler = _build_sampler(config, task, p_pauli, n_jobs)
            current_key = key
        print(
            f"[{config.run_tag}] shard={shard_index} style={task.style} "
            f"d={task.d} r={task.rounds} p_leak={task.p_leak:.12g} "
            f"bias={config.bias_label} decoder={config.decoder_strategy} workers={n_jobs}",
            flush=True,
        )
        collect(
            sampler=sampler,
            p_leak=task.p_leak,
            max_shots=config.max_shots,
            max_errors=config.max_errors,
            min_shots=config.min_shots,
            decoder=config.decoder_strategy,
            decoder_strategy=config.decoder_strategy,
            decoder_args=config.decoder_args,
            save_file=path,
            json_metadata=_metadata(config, task, p_pauli),
            print_progress=True,
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-num-shards", action="store_true")
    parser.add_argument("--print-run-tag", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print config without sampling.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-index", type=int, help="Run one one-based shard.")
    selection.add_argument(
        "--all", action="store_true", help="Run all shards sequentially."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = configured_run()
    groups = build_task_groups(config)
    if args.print_num_shards:
        print(len(groups))
        return 0
    if args.print_run_tag:
        print(config.run_tag)
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    **_identity(config),
                    "max_shots": config.max_shots,
                    "max_errors": config.max_errors,
                    "min_shots": config.min_shots,
                    "num_physical_points": sum(len(group) for group in groups),
                    "num_slurm_shards": len(groups),
                    "output_directory": str(_output_directory(config)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    index = args.task_index
    if index is None and not args.all:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is not None:
            try:
                index = int(raw)
            except ValueError as error:
                raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID={raw!r}.") from error
    if index is not None and not 1 <= index <= len(groups):
        raise ValueError(f"Shard index must be in [1, {len(groups)}]; got {index}.")
    if index is None and not args.all:
        raise SystemExit(
            "Refusing to run the full production sweep implicitly. Use --dry-run, --task-index N, --all, or the Slurm wrapper."
        )
    output = _output_directory(config)
    _ensure_identity_manifest(output, config)
    selected = range(1, len(groups) + 1) if args.all else (index,)
    for index in selected:
        _run_group(config, groups[index - 1], index, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run early-walking memory and stability tCNOT Monte Carlo experiments.

Edit the configuration block, inspect it with ``--dry-run``, and submit it with
``./scripts/tcnot/runs/submit_tcnot.sh``.  This production driver intentionally
supports only the confirmed configuration subset: both memory detector families,
skip-gate leakage, per-gate leakage trials, and marginal Tesseract decoding.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# =============================================================================
# EDIT THIS CONFIGURATION BLOCK BEFORE SUBMITTING
# =============================================================================

EXPERIMENTS = ("memory", "stability")

MEMORY_D_LIST = [3, 5, 7]
MEMORY_ROUNDS_MODE = "match_distance"
MEMORY_R_LIST = [3, 5, 7]
MEMORY_DETECTORS = "both"

STABILITY_D_LIST = [6]
STABILITY_ROUNDS_MODE = "explicit"
STABILITY_ROUND_PAIRS = [(3, 0), (4, 0), (5, 0)]
STABILITY_OBSERVABLE_SELECTION = "both"

P_LEAK_LIST = np.logspace(np.log10(2e-3), np.log10(2e-1), 21)
BIAS = float("inf")

DECODER_STRATEGY = "marginal"
LEAKAGE_EFFECT = "skip_gate"
LEAKAGE_GRANULARITY = "per_gate"

MAX_SHOTS = 10_000_000
MAX_ERRORS = 100
MIN_SHOTS = 0
SEED = None

TCNOT_CACHE_SIZE = 4096
TESSERACT_COMPILE_CACHE_SIZE = 0
TESSERACT_VERSION = "0.1.1.dev20260822020007"
TESSERACT_PRESET = "long_beam"
TESSERACT_PRESET_OPTIONS = {
    "long_beam": {
        "pqlimit": 1_000_000,
        "det_beam": 20,
        "beam_climbing": True,
        "det_order_options": {
            "num_det_orders": 21,
            "method": "Index",
        },
        "no_revisit_dets": True,
    },
    "inf_beam": {
        "det_beam": 65_535,
    },
}

RUN_TAG = "early_walking_tcnot_leak_only_stability_d6_r345_memory_d357_rmatch_psweep_marginal_long_beam"

# =============================================================================
# END CONFIGURATION BLOCK
# =============================================================================


SUPPORTED_EXPERIMENTS = ("memory", "stability")
SUPPORTED_ROUNDS_MODES = ("explicit", "match_distance")
SUPPORTED_STABILITY_OBSERVABLES = ("control", "target", "both")
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunConfig:
    experiments: tuple[str, ...]
    memory_d_list: tuple[int, ...]
    memory_rounds_mode: str
    memory_r_list: tuple[int, ...]
    memory_detectors: str
    stability_d_list: tuple[int, ...]
    stability_rounds_mode: str
    stability_round_pairs: tuple[tuple[int, int], ...]
    stability_observable_selection: str
    p_leak_list: tuple[float, ...]
    bias: float
    decoder_strategy: str
    leakage_effect: str
    leakage_granularity: str
    max_shots: int
    max_errors: int
    min_shots: int
    seed: int | None
    tcnot_cache_size: int | None
    tesseract_compile_cache_size: int | None
    tesseract_version: str
    tesseract_preset: str
    tesseract_options: dict
    run_tag: str

    @property
    def leakage_only(self) -> bool:
        return math.isinf(self.bias)

    @property
    def bias_label(self) -> str:
        return "inf" if self.leakage_only else format(self.bias, ".12g")


@dataclass(frozen=True)
class SweepTask:
    experiment: str
    d: int
    p_idx: int
    p_leak: float
    rounds: int | None = None
    rounds_before: int | None = None
    rounds_after: int | None = None


def _as_experiments(values: Sequence[str]) -> tuple[str, ...]:
    converted = tuple(values)
    if not converted:
        raise ValueError("EXPERIMENTS must contain at least one experiment.")
    if len(converted) != len(set(converted)):
        raise ValueError("EXPERIMENTS must not contain duplicates.")
    unknown = set(converted) - set(SUPPORTED_EXPERIMENTS)
    if unknown:
        raise ValueError(
            f"EXPERIMENTS entries must be in {SUPPORTED_EXPERIMENTS}; "
            f"got {sorted(unknown)}."
        )
    return converted


def _as_int_tuple(
    name: str, values: Sequence[int], *, allow_empty: bool
) -> tuple[int, ...]:
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty when its experiment is enabled.")
    converted = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must contain integers; got {value!r}.")
        converted.append(int(value))
    if len(converted) != len(set(converted)):
        raise ValueError(f"{name} must not contain duplicate values.")
    return tuple(converted)


def _as_round_pairs(
    values: Sequence[tuple[int, int]], *, allow_empty: bool
) -> tuple[tuple[int, int], ...]:
    if not values and not allow_empty:
        raise ValueError(
            "STABILITY_ROUND_PAIRS must not be empty when stability is enabled."
        )
    converted = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Each stability round entry must be a two-item pair.")
        before, after = value
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            for item in (before, after)
        ):
            raise ValueError(f"Stability round counts must be integers; got {value!r}.")
        pair = int(before), int(after)
        if pair[0] < 0 or pair[1] < 0 or pair == (0, 0):
            raise ValueError(
                "Stability round counts must be non-negative and not both zero; "
                f"got {pair!r}."
            )
        converted.append(pair)
    if len(converted) != len(set(converted)):
        raise ValueError("STABILITY_ROUND_PAIRS must not contain duplicate pairs.")
    return tuple(converted)


def _as_probability_tuple(values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError("P_LEAK_LIST must not be empty.")
    for value in converted:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("P_LEAK_LIST values must be finite and in [0, 1].")
    if len(converted) != len(set(converted)):
        raise ValueError("P_LEAK_LIST must not contain duplicate values.")
    return converted


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _optional_nonnegative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(name, value)


def configured_run() -> RunConfig:
    experiments = _as_experiments(EXPERIMENTS)
    memory_enabled = "memory" in experiments
    stability_enabled = "stability" in experiments

    for name, mode in (
        ("MEMORY_ROUNDS_MODE", MEMORY_ROUNDS_MODE),
        ("STABILITY_ROUNDS_MODE", STABILITY_ROUNDS_MODE),
    ):
        if mode not in SUPPORTED_ROUNDS_MODES:
            raise ValueError(f"{name} must be one of {SUPPORTED_ROUNDS_MODES}.")

    memory_d_list = _as_int_tuple(
        "MEMORY_D_LIST", MEMORY_D_LIST, allow_empty=not memory_enabled
    )
    memory_r_list = _as_int_tuple(
        "MEMORY_R_LIST",
        MEMORY_R_LIST,
        allow_empty=not memory_enabled or MEMORY_ROUNDS_MODE == "match_distance",
    )
    stability_d_list = _as_int_tuple(
        "STABILITY_D_LIST", STABILITY_D_LIST, allow_empty=not stability_enabled
    )
    stability_round_pairs = _as_round_pairs(
        STABILITY_ROUND_PAIRS,
        allow_empty=not stability_enabled
        or STABILITY_ROUNDS_MODE == "match_distance",
    )

    if any(d <= 0 for d in memory_d_list):
        raise ValueError("Every MEMORY_D_LIST value must be positive.")
    if any(rounds <= 0 for rounds in memory_r_list):
        raise ValueError("Every MEMORY_R_LIST value must be positive.")
    if any(d <= 0 or d % 2 for d in stability_d_list):
        raise ValueError("Every STABILITY_D_LIST value must be positive and even.")
    if MEMORY_DETECTORS != "both":
        raise ValueError("MEMORY_DETECTORS is fixed to 'both' for this implementation.")
    if STABILITY_OBSERVABLE_SELECTION not in SUPPORTED_STABILITY_OBSERVABLES:
        raise ValueError(
            "STABILITY_OBSERVABLE_SELECTION must be 'control', 'target', or 'both'."
        )

    p_leak_list = _as_probability_tuple(P_LEAK_LIST)
    if isinstance(BIAS, (bool, np.bool_)):
        raise ValueError("BIAS must be a positive number or infinity, not bool.")
    bias = float(BIAS)
    if math.isnan(bias) or bias <= 0:
        raise ValueError("BIAS must be positive or infinity.")
    if DECODER_STRATEGY != "marginal":
        raise ValueError("DECODER_STRATEGY is fixed to 'marginal'.")
    if LEAKAGE_EFFECT != "skip_gate":
        raise ValueError("LEAKAGE_EFFECT is fixed to 'skip_gate'.")
    if LEAKAGE_GRANULARITY != "per_gate":
        raise ValueError("LEAKAGE_GRANULARITY is fixed to 'per_gate'.")

    max_shots = _nonnegative_int("MAX_SHOTS", MAX_SHOTS)
    max_errors = _nonnegative_int("MAX_ERRORS", MAX_ERRORS)
    min_shots = _nonnegative_int("MIN_SHOTS", MIN_SHOTS)
    if max_shots == 0:
        raise ValueError("MAX_SHOTS must be greater than zero.")
    if min_shots > max_shots:
        raise ValueError("MIN_SHOTS cannot exceed MAX_SHOTS.")

    seed = _optional_nonnegative_int("SEED", SEED)
    tcnot_cache_size = _optional_nonnegative_int(
        "TCNOT_CACHE_SIZE", TCNOT_CACHE_SIZE
    )
    compile_cache_size = _optional_nonnegative_int(
        "TESSERACT_COMPILE_CACHE_SIZE", TESSERACT_COMPILE_CACHE_SIZE
    )
    if TESSERACT_VERSION != "0.1.1.dev20260822020007":
        raise ValueError(
            "TESSERACT_VERSION must match biased-noise-qec: "
            "0.1.1.dev20260822020007."
        )
    if TESSERACT_PRESET not in TESSERACT_PRESET_OPTIONS:
        raise ValueError("Unknown TESSERACT_PRESET.")
    tesseract_options = copy.deepcopy(
        TESSERACT_PRESET_OPTIONS[TESSERACT_PRESET]
    )
    try:
        json.dumps(tesseract_options, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Tesseract options must be JSON-serializable.") from error

    if not isinstance(RUN_TAG, str) or not RUN_TAG_PATTERN.fullmatch(RUN_TAG):
        raise ValueError(
            "RUN_TAG must start alphanumeric and contain only letters, numbers, "
            "dot, underscore, or hyphen."
        )

    return RunConfig(
        experiments=experiments,
        memory_d_list=memory_d_list,
        memory_rounds_mode=MEMORY_ROUNDS_MODE,
        memory_r_list=memory_r_list,
        memory_detectors=MEMORY_DETECTORS,
        stability_d_list=stability_d_list,
        stability_rounds_mode=STABILITY_ROUNDS_MODE,
        stability_round_pairs=stability_round_pairs,
        stability_observable_selection=STABILITY_OBSERVABLE_SELECTION,
        p_leak_list=p_leak_list,
        bias=bias,
        decoder_strategy=DECODER_STRATEGY,
        leakage_effect=LEAKAGE_EFFECT,
        leakage_granularity=LEAKAGE_GRANULARITY,
        max_shots=max_shots,
        max_errors=max_errors,
        min_shots=min_shots,
        seed=seed,
        tcnot_cache_size=tcnot_cache_size,
        tesseract_compile_cache_size=compile_cache_size,
        tesseract_version=TESSERACT_VERSION,
        tesseract_preset=TESSERACT_PRESET,
        tesseract_options=tesseract_options,
        run_tag=RUN_TAG,
    )


def build_task_groups(config: RunConfig) -> list[tuple[SweepTask, ...]]:
    p_desc = tuple(reversed(config.p_leak_list))
    groups: list[tuple[SweepTask, ...]] = []

    if "memory" in config.experiments:
        for d in config.memory_d_list:
            round_values = (
                (d,)
                if config.memory_rounds_mode == "match_distance"
                else config.memory_r_list
            )
            for rounds in round_values:
                tasks = tuple(
                    SweepTask("memory", d, p_idx, p_leak, rounds=rounds)
                    for p_idx, p_leak in enumerate(p_desc)
                )
                groups.extend(
                    (tasks,)
                    if config.leakage_only
                    else ((task,) for task in tasks)
                )

    if "stability" in config.experiments:
        for d in config.stability_d_list:
            round_pairs = (
                ((d, d),)
                if config.stability_rounds_mode == "match_distance"
                else config.stability_round_pairs
            )
            for before, after in round_pairs:
                tasks = tuple(
                    SweepTask(
                        "stability",
                        d,
                        p_idx,
                        p_leak,
                        rounds_before=before,
                        rounds_after=after,
                    )
                    for p_idx, p_leak in enumerate(p_desc)
                )
                groups.extend(
                    (tasks,)
                    if config.leakage_only
                    else ((task,) for task in tasks)
                )
    return groups


def _identity(config: RunConfig) -> dict:
    return {
        "schema_version": 1,
        "experiments": list(config.experiments),
        "memory": {
            "circuit": "EarlyWalkingTCNOTBuilder",
            "d_list": list(config.memory_d_list),
            "rounds_mode": config.memory_rounds_mode,
            "r_list": list(config.memory_r_list)
            if config.memory_rounds_mode == "explicit"
            else None,
            "detectors": config.memory_detectors,
        },
        "stability": {
            "circuit": "EarlyWalkingStabilityTCNOTBuilder",
            "d_list": list(config.stability_d_list),
            "rounds_mode": config.stability_rounds_mode,
            "round_pairs": [list(pair) for pair in config.stability_round_pairs]
            if config.stability_rounds_mode == "explicit"
            else None,
            "observable_selection": config.stability_observable_selection,
        },
        "p_leak_list": list(config.p_leak_list),
        "bias": config.bias_label,
        "pauli_noise": "none"
        if config.leakage_only
        else "uniform_two_qubit_depolarizing",
        "pauli_locations": "2-qubit gates",
        "erasure_check_schedule": 8,
        "decoder_strategy": config.decoder_strategy,
        "leakage_effect": config.leakage_effect,
        "leakage_granularity": config.leakage_granularity,
        "seed": config.seed,
        "tcnot_cache_size": config.tcnot_cache_size,
        "tesseract_compile_cache_size": config.tesseract_compile_cache_size,
        "tesseract_version": config.tesseract_version,
        "tesseract_preset": config.tesseract_preset,
        "tesseract_options": config.tesseract_options,
        "run_tag": config.run_tag,
    }


def _output_directory(config: RunConfig) -> Path:
    return REPO_ROOT / "data" / "tcnot" / "runs" / config.run_tag


def _ensure_identity_manifest(output_directory: Path, config: RunConfig) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = output_directory / "identity.json"
    expected = _identity(config)
    serialized = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    try:
        with manifest.open("x", encoding="utf8") as file:
            file.write(serialized)
        return
    except FileExistsError:
        pass

    for attempt in range(20):
        try:
            actual = json.loads(manifest.read_text(encoding="utf8"))
            break
        except (json.JSONDecodeError, OSError):
            if attempt == 19:
                raise
            time.sleep(0.05)
    if actual != expected:
        raise ValueError(
            f"RUN_TAG {config.run_tag!r} has an incompatible identity manifest "
            f"at {manifest}. Choose a new RUN_TAG."
        )


def _task_base_label(task: SweepTask, config: RunConfig) -> str:
    if task.experiment == "memory":
        return f"mem_d{task.d}_r{task.rounds}_detboth"
    return (
        f"stab_d{task.d}_rb{task.rounds_before}_ra{task.rounds_after}_"
        f"obs{config.stability_observable_selection}"
    )


def _shard_label(group: tuple[SweepTask, ...], config: RunConfig) -> str:
    label = _task_base_label(group[0], config)
    return label if len(group) > 1 else f"{label}_p{group[0].p_idx}"


def _metadata(
    config: RunConfig, task: SweepTask, p_pauli: float, seed: int | None
) -> dict:
    metadata = {
        "circuit": "EarlyWalkingTCNOTBuilder"
        if task.experiment == "memory"
        else "EarlyWalkingStabilityTCNOTBuilder",
        "experiment": task.experiment,
        "geometry": "early_walking",
        "d": task.d,
        "p": p_pauli,
        "p_leak": task.p_leak,
        "bias": config.bias_label,
        "noise_model": "none"
        if config.leakage_only
        else "two_qubit_gates_only",
        "pauli_noise": "none"
        if config.leakage_only
        else "uniform_two_qubit_depolarizing",
        "pauli_locations": "2-qubit gates",
        "erasure_check_schedule": 8,
        "leakage_effect": config.leakage_effect,
        "leakage_granularity": config.leakage_granularity,
        "decoder_strategy": config.decoder_strategy,
        "tcnot_cache_size": config.tcnot_cache_size,
        "tesseract_compile_cache_size": config.tesseract_compile_cache_size,
        "tesseract_version": config.tesseract_version,
        "tesseract_preset": config.tesseract_preset,
        "tesseract_options": config.tesseract_options,
        "seed": seed,
        "run_tag": config.run_tag,
    }
    if task.experiment == "memory":
        metadata.update(
            rounds=task.rounds,
            rounds_before=task.rounds,
            rounds_after=task.rounds,
            rounds_mode=config.memory_rounds_mode,
            detectors="both",
        )
    else:
        metadata.update(
            rounds_before=task.rounds_before,
            rounds_after=task.rounds_after,
            rounds_mode=config.stability_rounds_mode,
            observable_selection=config.stability_observable_selection,
        )
    return metadata


def _worker_count() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        return 1
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(f"Invalid SLURM_CPUS_PER_TASK={raw!r}.") from error
    if workers <= 0:
        raise ValueError("SLURM_CPUS_PER_TASK must be positive.")
    return workers


def _task_seed(config: RunConfig, task: SweepTask) -> int | None:
    if config.seed is None:
        return None
    payload = json.dumps(
        {"base_seed": config.seed, "task": asdict(task)},
        separators=(",", ":"),
        sort_keys=True,
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def _build_sampler(
    config: RunConfig,
    task: SweepTask,
    p_pauli: float,
    point_seed: int | None,
    n_jobs: int,
):
    from surface_code_leakage_erasure import (
        EarlyWalkingStabilityTCNOTBuilder,
        EarlyWalkingTCNOTBuilder,
        TCNOTSampler,
    )

    common = {
        "p_pauli": p_pauli,
        "n_jobs": n_jobs,
        "seed": point_seed,
        "tcnot_cache_size": config.tcnot_cache_size,
        "tesseract_compile_cache_size": config.tesseract_compile_cache_size,
        "tesseract_options": config.tesseract_options,
        "leakage_effect": config.leakage_effect,
        "leakage_granularity": config.leakage_granularity,
    }
    if task.experiment == "memory":
        return TCNOTSampler(
            builder=EarlyWalkingTCNOTBuilder(task.d, seed=point_seed),
            rounds_per_side=task.rounds,
            circuit_kwargs={"basis": "X"},
            **common,
        )
    return TCNOTSampler(
        builder=EarlyWalkingStabilityTCNOTBuilder(task.d, seed=point_seed),
        rounds_before=task.rounds_before,
        rounds_after=task.rounds_after,
        circuit_kwargs={
            "basis": "Z",
            "observable_selection": config.stability_observable_selection,
        },
        **common,
    )


def _run_group(
    config: RunConfig,
    group: tuple[SweepTask, ...],
    shard_index: int,
    output_directory: Path,
) -> None:
    from surface_code_leakage_erasure.collect import collect

    save_file = output_directory / (
        f"{config.run_tag}_s{shard_index:04d}_{_shard_label(group, config)}.csv"
    )
    n_jobs = _worker_count()
    for task in group:
        p_pauli = 0.0 if config.leakage_only else task.p_leak / config.bias
        point_seed = _task_seed(config, task)
        sampler = _build_sampler(config, task, p_pauli, point_seed, n_jobs)
        detail = (
            f"r={task.rounds} detectors=both"
            if task.experiment == "memory"
            else f"rb={task.rounds_before} ra={task.rounds_after} "
            f"observable={config.stability_observable_selection}"
        )
        print(
            f"[{config.run_tag}] shard={shard_index} "
            f"experiment={task.experiment} d={task.d} {detail} "
            f"p_leak={task.p_leak:.12g} bias={config.bias_label} "
            f"decoder=marginal tesseract={config.tesseract_version} "
            f"preset={config.tesseract_preset} workers={n_jobs}",
            flush=True,
        )
        collect(
            sampler=sampler,
            p_leak=task.p_leak,
            max_shots=config.max_shots,
            max_errors=config.max_errors,
            min_shots=config.min_shots,
            save_file=save_file,
            json_metadata=_metadata(config, task, p_pauli, point_seed),
            decoder="marginal",
            decoder_strategy="marginal",
            decoder_args=None,
            print_progress=True,
        )


def _selected_shard(args: argparse.Namespace, n_shards: int) -> int | None:
    if args.task_index is not None:
        index = args.task_index
    else:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        try:
            index = int(raw) if raw is not None else None
        except ValueError as error:
            raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID={raw!r}.") from error
    if index is not None and not 1 <= index <= n_shards:
        raise ValueError(f"Shard index must be in [1, {n_shards}]; got {index}.")
    return index


def _print_summary(config: RunConfig, groups: list[tuple[SweepTask, ...]]) -> None:
    summary = {
        **_identity(config),
        "max_shots": config.max_shots,
        "max_errors": config.max_errors,
        "min_shots": config.min_shots,
        "num_physical_points": sum(len(group) for group in groups),
        "num_slurm_shards": len(groups),
        "shards_by_experiment": {
            experiment: sum(
                group[0].experiment == experiment for group in groups
            )
            for experiment in config.experiments
        },
        "output_directory": str(_output_directory(config)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-num-shards", action="store_true")
    parser.add_argument("--print-run-tag", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.task_index is not None and args.all:
        parser.error("--task-index and --all are mutually exclusive.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
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
        _print_summary(config, groups)
        return 0

    shard_index = _selected_shard(args, len(groups))
    if shard_index is None and not args.all:
        raise SystemExit(
            "Refusing to run the full production sweep implicitly. Use --dry-run, "
            "--task-index N, --all, or submit through the Slurm wrapper."
        )
    output_directory = _output_directory(config)
    _ensure_identity_manifest(output_directory, config)
    selected = range(1, len(groups) + 1) if shard_index is None else (shard_index,)
    for index in selected:
        _run_group(config, groups[index - 1], index, output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

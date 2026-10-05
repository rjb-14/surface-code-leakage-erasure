"""Incremental CSV collection and resumability for the marginal tCNOT sampler."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import stim
from sinter import Fit, TaskStats, fit_binomial, read_stats_from_csv_files
from sinter._data._csv_out import CSV_HEADER


class TCNOTSamplerProtocol(Protocol):
    baseline_circuit: stim.Circuit

    def sample(
        self,
        *,
        p_leak: float,
        max_shots: int,
        max_errors: int,
        min_shots: int,
        on_batch: Any,
        decoder_strategy: str,
        decoder_args: dict | None,
    ) -> tuple[int, int]: ...


def _strong_id(
    circuit: stim.Circuit,
    p_leak: float,
    decoder: str,
    json_metadata: Any,
) -> str:
    payload = json.dumps(
        {
            "circuit": str(circuit),
            "p_leak": p_leak,
            "decoder": decoder,
            "json_metadata": json_metadata,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def _prior_stats(save_file: Path | None, strong_id: str) -> tuple[int, int]:
    if save_file is not None and save_file.exists():
        for stat in read_stats_from_csv_files(save_file):
            if stat.strong_id == strong_id:
                return stat.errors, stat.shots
    return 0, 0


def _fit(shots: int, errors: int) -> Fit:
    return fit_binomial(
        num_shots=shots,
        num_hits=errors,
        max_likelihood_factor=9.0,
    )


def collect(
    *,
    sampler: TCNOTSamplerProtocol,
    p_leak: float,
    max_shots: int,
    max_errors: int = 0,
    min_shots: int = 0,
    save_file: str | Path | None = None,
    json_metadata: Any = None,
    decoder: str = "marginal",
    decoder_strategy: str = "marginal",
    decoder_args: dict | None = None,
    print_progress: bool = False,
) -> Fit:
    """Collect one marginal task, appending Sinter rows and resuming prior work."""
    if decoder_strategy != "marginal":
        raise ValueError("tCNOT collection supports decoder_strategy='marginal' only.")
    if decoder != "marginal":
        raise ValueError("tCNOT collection requires decoder='marginal'.")
    if decoder_args:
        raise ValueError("decoder_args are not accepted for marginal tCNOT collection.")
    for name, value in (
        ("max_shots", max_shots),
        ("max_errors", max_errors),
        ("min_shots", min_shots),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if min_shots > max_shots:
        raise ValueError("min_shots cannot exceed max_shots.")
    if not 0 <= p_leak <= 1:
        raise ValueError("p_leak must be in [0, 1].")

    save_path = Path(save_file) if save_file is not None else None
    strong_id = _strong_id(
        sampler.baseline_circuit,
        p_leak,
        decoder,
        json_metadata,
    )
    prior_errors, prior_shots = _prior_stats(save_path, strong_id)
    remaining_shots = max(max_shots - prior_shots, 0)
    remaining_errors = (
        max(max_errors - prior_errors, 0) if max_errors > 0 else 0
    )
    remaining_min_shots = max(min_shots - prior_shots, 0)

    if remaining_shots == 0 or (
        max_errors > 0
        and prior_shots >= min_shots
        and prior_errors >= max_errors
    ):
        return _fit(prior_shots, prior_errors)

    if save_path is not None and not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with save_path.open("x", encoding="utf8") as file:
                file.write(CSV_HEADER + "\n")
        except FileExistsError:
            pass

    started = time.perf_counter()
    running_shots = prior_shots
    running_errors = prior_errors

    def on_batch(batch_errors: int, batch_shots: int) -> None:
        nonlocal running_errors, running_shots
        if save_path is not None:
            stat = TaskStats(
                strong_id=strong_id,
                decoder=decoder,
                json_metadata=json_metadata,
                shots=batch_shots,
                errors=batch_errors,
                discards=0,
                seconds=time.perf_counter() - started,
            )
            with save_path.open("a", encoding="utf8") as file:
                file.write(stat.to_csv_line() + "\n")
        running_errors += batch_errors
        running_shots += batch_shots
        if print_progress:
            elapsed = time.perf_counter() - started
            rate = running_errors / running_shots if running_shots else 0.0
            print(
                f"[collect] shots={running_shots}/{max_shots} "
                f"errors={running_errors} rate={rate:.3e} elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )

    # If the error target was met before the minimum-shot floor, collect exactly
    # the missing floor shots with error stopping disabled. Otherwise a zero
    # remaining error target would mean "no error limit" and consume the entire
    # remaining max_shots budget.
    sample_shot_cap = remaining_shots
    if max_errors > 0 and prior_errors >= max_errors:
        sample_shot_cap = remaining_min_shots
        remaining_errors = 0

    new_errors, new_shots = sampler.sample(
        p_leak=p_leak,
        max_shots=sample_shot_cap,
        max_errors=remaining_errors,
        min_shots=remaining_min_shots,
        on_batch=on_batch if (save_path is not None or print_progress) else None,
        decoder_strategy=decoder_strategy,
        decoder_args=decoder_args,
    )
    return _fit(prior_shots + new_shots, prior_errors + new_errors)

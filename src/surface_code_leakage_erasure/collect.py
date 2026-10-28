"""Incremental Sinter CSV collection for tCNOT and single-patch stability."""

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
    decoder_args: dict | None = None,
) -> str:
    identity = {
        "circuit": str(circuit),
        "p_leak": p_leak,
        "decoder": decoder,
        "json_metadata": json_metadata,
    }
    # Preserve existing marginal tCNOT IDs; include search options for samplers
    # that explicitly support additional decoder strategies.
    if decoder_args:
        identity["decoder_args"] = decoder_args
    payload = json.dumps(
        identity,
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


def _paired_marginal_metadata(json_metadata, decoder, decoder_args):
    metadata = (
        dict(json_metadata)
        if isinstance(json_metadata, dict)
        else {"source_metadata": json_metadata}
    )
    metadata.update(
        decoder_strategy="marginal",
        decoder_args=None,
        paired_with=decoder,
        paired_decoder_args=decoder_args,
    )
    return metadata


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
    """Collect one task, appending Sinter rows and resuming prior work.

    Additional decoder strategies require an explicit sampler capability; the
    tCNOT sampler continues to accept marginal decoding only. Samplers declaring
    paired_marginal_strategies also append marginal rows from the same shots.
    Budgets and the returned fit always use the selected strategy.
    """
    supported = getattr(sampler, "supported_decoder_strategies", ("marginal",))
    if decoder_strategy not in supported:
        raise ValueError(f"Sampler collection supports decoder strategies {supported} only.")
    if decoder != decoder_strategy:
        raise ValueError("decoder must match decoder_strategy.")
    if decoder_strategy == "marginal" and decoder_args:
        raise ValueError("decoder_args are not accepted for marginal collection.")
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
        decoder_args,
    )
    prior_errors, prior_shots = _prior_stats(save_path, strong_id)
    paired_marginal = decoder_strategy in getattr(
        sampler, "paired_marginal_strategies", ()
    )
    prior_marginal_errors = 0
    if paired_marginal:
        marginal_metadata = _paired_marginal_metadata(
            json_metadata, decoder, decoder_args
        )
        marginal_id = _strong_id(
            sampler.baseline_circuit, p_leak, "marginal", marginal_metadata
        )
        prior_marginal_errors, marginal_shots = _prior_stats(save_path, marginal_id)
        if marginal_shots != prior_shots:
            raise ValueError(
                "Cannot resume paired quasi_mle/marginal results with unequal shot counts "
                f"({prior_shots} quasi_mle, {marginal_shots} marginal). "
                "Older quasi_mle-only CSVs cannot supply the missing marginal results; "
                "use a new RUN_TAG or save_file."
            )
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
    last_checkpoint = started
    running_shots = prior_shots
    running_errors = prior_errors
    running_marginal_errors = prior_marginal_errors

    def on_batch(batch_errors: int, batch_shots: int, marginal_errors=None) -> None:
        nonlocal running_errors, running_shots, running_marginal_errors, last_checkpoint
        now = time.perf_counter()
        if save_path is not None:
            stat = TaskStats(
                strong_id=strong_id,
                decoder=decoder,
                json_metadata=json_metadata,
                shots=batch_shots,
                errors=batch_errors,
                discards=0,
                seconds=now - last_checkpoint,
            )
            lines = stat.to_csv_line() + "\n"
            if paired_marginal:
                marginal_stat = TaskStats(
                    strong_id=marginal_id,
                    decoder="marginal",
                    json_metadata=marginal_metadata,
                    shots=batch_shots,
                    errors=marginal_errors,
                    discards=0,
                    # Sampling/decoding time is already charged to the primary row.
                    seconds=0,
                )
                lines += marginal_stat.to_csv_line() + "\n"
            # Append the pair together to keep checkpoints aligned on interruption.
            with save_path.open("a", encoding="utf8") as file:
                file.write(lines)
        running_errors += batch_errors
        running_shots += batch_shots
        if paired_marginal:
            running_marginal_errors += marginal_errors
        last_checkpoint = now
        if print_progress:
            elapsed = time.perf_counter() - started
            rate = running_errors / running_shots if running_shots else 0.0
            marginal_progress = (
                f" marginal_errors={running_marginal_errors} "
                f"marginal_rate={running_marginal_errors / running_shots:.3e}"
                if paired_marginal and running_shots
                else ""
            )
            print(
                f"[collect] shots={running_shots}/{max_shots} "
                f"errors={running_errors} rate={rate:.3e}{marginal_progress} elapsed={elapsed:.1f}s",
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

    callbacks = {"on_batch": None}
    if save_path is not None or print_progress:
        if paired_marginal:
            callbacks["on_paired_batch"] = lambda errors, marginal_errors, shots: on_batch(
                errors, shots, marginal_errors
            )
        else:
            callbacks["on_batch"] = on_batch

    new_errors, new_shots = sampler.sample(
        p_leak=p_leak,
        max_shots=sample_shot_cap,
        max_errors=remaining_errors,
        min_shots=remaining_min_shots,
        decoder_strategy=decoder_strategy,
        decoder_args=decoder_args,
        **callbacks,
    )
    return _fit(prior_shots + new_shots, prior_errors + new_errors)

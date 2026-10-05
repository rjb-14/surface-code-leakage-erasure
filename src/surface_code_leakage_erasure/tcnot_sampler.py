"""Monte Carlo sampling for early-walking memory and stability tCNOT circuits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections import OrderedDict
from typing import Callable, Mapping

import numpy as np
from joblib import Parallel, cpu_count, delayed
from stim import FlipSimulator

from .tcnot_decoder import EarlyWalkingTCNOTDecoder, TesseractHyperedgeDecoder


_WORKER_DECODER_CACHE_SIZE = 1
_decoder_cache: OrderedDict[str, EarlyWalkingTCNOTDecoder] = OrderedDict()
TESSERACT_DECODER_VERSION = "0.1.1.dev20260822020007"


def _require_tesseract_version() -> None:
    """Fail closed when production sampling uses a different decoder wheel."""
    try:
        actual = importlib.metadata.version("tesseract-decoder")
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError(
            "TCNOTSampler requires the optional tesseract-decoder package. "
            'Install it with `python -m pip install -e ".[tesseract]"`.'
        ) from error
    if actual != TESSERACT_DECODER_VERSION:
        raise RuntimeError(
            "TCNOTSampler requires tesseract-decoder "
            f"{TESSERACT_DECODER_VERSION}, got {actual}."
        )


def _round_kwargs(
    rounds_per_side: int | None,
    rounds_before: int | None,
    rounds_after: int | None,
) -> dict:
    if rounds_per_side is not None:
        return {"rounds_per_side": rounds_per_side}
    return {
        "rounds_before": rounds_before,
        "rounds_after": rounds_after,
    }


def _decoder_key(
    *,
    baseline_circuit,
    builder,
    round_kwargs: Mapping,
    circuit_kwargs: Mapping,
    p_pauli: float,
    tcnot_cache_size: int | None,
    tesseract_compile_cache_size: int | None,
    tesseract_options: Mapping,
) -> str:
    configuration = json.dumps(
        {
            "circuit": str(baseline_circuit),
            "builder": f"{type(builder).__module__}.{type(builder).__qualname__}",
            "round_kwargs": dict(round_kwargs),
            "circuit_kwargs": dict(circuit_kwargs),
            "p_pauli": p_pauli,
            "tcnot_cache_size": tcnot_cache_size,
            "tesseract_compile_cache_size": tesseract_compile_cache_size,
            "tesseract_options": dict(tesseract_options),
        },
        default=repr,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(configuration.encode()).hexdigest()


def _make_decoder(
    *,
    builder,
    rounds_per_side: int | None,
    rounds_before: int | None,
    rounds_after: int | None,
    p_pauli: float,
    circuit_kwargs: Mapping,
    tcnot_cache_size: int | None,
    tesseract_compile_cache_size: int | None,
    tesseract_options: Mapping,
) -> EarlyWalkingTCNOTDecoder:
    _require_tesseract_version()
    backend = TesseractHyperedgeDecoder(
        compile_cache_size=tesseract_compile_cache_size,
        **dict(tesseract_options),
    )
    return EarlyWalkingTCNOTDecoder(
        builder=builder,
        rounds_per_side=rounds_per_side,
        rounds_before=rounds_before,
        rounds_after=rounds_after,
        p_pauli=p_pauli,
        hyperedge_decoder=backend,
        circuit_kwargs=dict(circuit_kwargs),
        cache_size=tcnot_cache_size,
    )


def _get_worker_decoder(
    *,
    builder,
    decoder_key: str,
    rounds_per_side: int | None,
    rounds_before: int | None,
    rounds_after: int | None,
    p_pauli: float,
    circuit_kwargs: Mapping,
    tcnot_cache_size: int | None,
    tesseract_compile_cache_size: int | None,
    tesseract_options: Mapping,
) -> EarlyWalkingTCNOTDecoder:
    decoder = _decoder_cache.get(decoder_key)
    if decoder is not None:
        _decoder_cache.move_to_end(decoder_key)
        return decoder

    decoder = _make_decoder(
        builder=builder,
        rounds_per_side=rounds_per_side,
        rounds_before=rounds_before,
        rounds_after=rounds_after,
        p_pauli=p_pauli,
        circuit_kwargs=circuit_kwargs,
        tcnot_cache_size=tcnot_cache_size,
        tesseract_compile_cache_size=tesseract_compile_cache_size,
        tesseract_options=tesseract_options,
    )
    _decoder_cache[decoder_key] = decoder
    while len(_decoder_cache) > _WORKER_DECODER_CACHE_SIZE:
        _decoder_cache.popitem(last=False)
    return decoder


def _stim_seed(seed: np.random.SeedSequence) -> int:
    return int(seed.generate_state(1, dtype=np.uint64)[0])


def _single_tcnot_sample(
    *,
    p_leak: float,
    decoder: EarlyWalkingTCNOTDecoder,
    simulator_seed: int | None = None,
) -> int:
    """Sample one shot conditioned on at least one per-gate leakage event."""
    builder = decoder.builder
    leakage_locations = builder.generate_leakage_locations(
        decoder.total_rounds,
        p_leak,
        atleast_1=True,
        leak_locations="2-qubit gates",
    )
    circuit, erasure_checks = builder.get_circuit(
        p=decoder.p_pauli,
        leakage_circuit_locations=leakage_locations,
        use_cache=True,
        **decoder._round_kwargs,
        **decoder._physical_kwargs,
    )

    simulator = FlipSimulator(batch_size=1, seed=simulator_seed)
    simulator.do(circuit)
    syndrome = simulator.get_detector_flips(bit_packed=False).reshape(-1)
    observable_flips = simulator.get_observable_flips(bit_packed=False).reshape(-1)
    prediction = decoder.decode(
        syndrome=syndrome,
        erasure_checks=erasure_checks,
        decoder_strategy="marginal",
    )
    return int(np.any(prediction != observable_flips))


def _batched_tcnot_samples(
    *,
    batch_size: int,
    p_leak: float,
    builder,
    seed: np.random.SeedSequence,
    decoder_key: str,
    rounds_per_side: int | None,
    rounds_before: int | None,
    rounds_after: int | None,
    p_pauli: float,
    circuit_kwargs: Mapping,
    tcnot_cache_size: int | None,
    tesseract_compile_cache_size: int | None,
    tesseract_options: Mapping,
    decoder: EarlyWalkingTCNOTDecoder | None = None,
) -> tuple[int, int]:
    leakage_seed, simulator_seed = seed.spawn(2)
    simulator_rng = np.random.default_rng(simulator_seed)
    if decoder is None:
        decoder = _get_worker_decoder(
            builder=builder,
            decoder_key=decoder_key,
            rounds_per_side=rounds_per_side,
            rounds_before=rounds_before,
            rounds_after=rounds_after,
            p_pauli=p_pauli,
            circuit_kwargs=circuit_kwargs,
            tcnot_cache_size=tcnot_cache_size,
            tesseract_compile_cache_size=tesseract_compile_cache_size,
            tesseract_options=tesseract_options,
        )
    decoder.builder.rng = np.random.default_rng(leakage_seed)

    n_fails = 0
    for _ in range(batch_size):
        n_fails += _single_tcnot_sample(
            p_leak=p_leak,
            decoder=decoder,
            simulator_seed=int(simulator_rng.bit_generator.random_raw()),
        )
    return n_fails, batch_size


def _batched_tcnot_baseline_samples(
    *,
    batch_size: int,
    builder,
    seed: np.random.SeedSequence,
    decoder_key: str,
    rounds_per_side: int | None,
    rounds_before: int | None,
    rounds_after: int | None,
    p_pauli: float,
    circuit_kwargs: Mapping,
    tcnot_cache_size: int | None,
    tesseract_compile_cache_size: int | None,
    tesseract_options: Mapping,
) -> tuple[int, int]:
    decoder = _get_worker_decoder(
        builder=builder,
        decoder_key=decoder_key,
        rounds_per_side=rounds_per_side,
        rounds_before=rounds_before,
        rounds_after=rounds_after,
        p_pauli=p_pauli,
        circuit_kwargs=circuit_kwargs,
        tcnot_cache_size=tcnot_cache_size,
        tesseract_compile_cache_size=tesseract_compile_cache_size,
        tesseract_options=tesseract_options,
    )
    sampler = decoder.baseline_circuit.compile_detector_sampler(seed=_stim_seed(seed))
    syndromes, observable_flips = sampler.sample(
        shots=batch_size,
        separate_observables=True,
        bit_packed=False,
    )
    predictions = decoder.decode_batch(
        syndromes=syndromes,
        erasure_checks={},
        decoder_strategy="marginal",
    )
    failures = int(np.sum(np.any(predictions != observable_flips, axis=1)))
    return failures, batch_size


class TCNOTSampler:
    """Estimate early-walking tCNOT failure rates with marginal Tesseract decoding."""

    def __init__(
        self,
        *,
        builder,
        rounds_per_side: int | None = None,
        rounds_before: int | None = None,
        rounds_after: int | None = None,
        p_pauli: float = 0.0,
        circuit_kwargs: Mapping | None = None,
        n_jobs: int = 1,
        seed: int | None = None,
        tcnot_cache_size: int | None = 4096,
        tesseract_compile_cache_size: int | None = 0,
        tesseract_options: Mapping | None = None,
        leakage_effect: str = "skip_gate",
        leakage_granularity: str = "per_gate",
    ):
        if n_jobs == -1:
            n_jobs = cpu_count()
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs}.")
        if leakage_effect != "skip_gate":
            raise ValueError("TCNOTSampler currently supports leakage_effect='skip_gate' only.")
        if leakage_granularity != "per_gate":
            raise ValueError(
                "TCNOTSampler currently supports leakage_granularity='per_gate' only."
            )
        if not np.isfinite(p_pauli) or not 0 <= p_pauli <= 1:
            raise ValueError(f"p_pauli must be finite and in [0, 1], got {p_pauli}.")

        resolved_circuit_kwargs = {
            "Pauli_locations": "2-qubit gates",
            "leak_locations": "2-qubit gates",
            "ec_sched": 8,
            "leak_effect": "skip gates",
            **dict(circuit_kwargs or {}),
        }
        if resolved_circuit_kwargs["Pauli_locations"] != "2-qubit gates":
            raise ValueError("TCNOTSampler requires Pauli_locations='2-qubit gates'.")
        if resolved_circuit_kwargs["leak_locations"] != "2-qubit gates":
            raise ValueError("TCNOTSampler requires leak_locations='2-qubit gates'.")
        if resolved_circuit_kwargs["leak_effect"] != "skip gates":
            raise ValueError("TCNOTSampler requires builder leak_effect='skip gates'.")

        self.builder = builder
        self.rounds_per_side = rounds_per_side
        self.rounds_before = rounds_before
        self.rounds_after = rounds_after
        self.round_kwargs = _round_kwargs(
            rounds_per_side, rounds_before, rounds_after
        )
        self.p_pauli = float(p_pauli)
        self.circuit_kwargs = resolved_circuit_kwargs
        self.n_jobs = n_jobs
        self.tcnot_cache_size = tcnot_cache_size
        self.tesseract_compile_cache_size = tesseract_compile_cache_size
        self.tesseract_options = dict(tesseract_options or {})
        self.leakage_effect = leakage_effect
        self.leakage_granularity = leakage_granularity

        self.decoder = _make_decoder(
            builder=builder,
            rounds_per_side=rounds_per_side,
            rounds_before=rounds_before,
            rounds_after=rounds_after,
            p_pauli=self.p_pauli,
            circuit_kwargs=self.circuit_kwargs,
            tcnot_cache_size=tcnot_cache_size,
            tesseract_compile_cache_size=tesseract_compile_cache_size,
            tesseract_options=self.tesseract_options,
        )
        self.baseline_circuit = self.decoder.baseline_circuit
        self.num_leakage_trials = self.builder.count_leakage_locations(
            self.decoder.total_rounds,
            leak_locations="2-qubit gates",
        )[0]
        if self.num_leakage_trials <= 0:
            raise ValueError("The tCNOT circuit has no per-gate leakage trials.")

        self._decoder_key = _decoder_key(
            baseline_circuit=self.baseline_circuit,
            builder=builder,
            round_kwargs=self.round_kwargs,
            circuit_kwargs=self.circuit_kwargs,
            p_pauli=self.p_pauli,
            tcnot_cache_size=tcnot_cache_size,
            tesseract_compile_cache_size=tesseract_compile_cache_size,
            tesseract_options=self.tesseract_options,
        )
        self.parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        self._seed_sequence = np.random.SeedSequence(seed)
        self._subdivision_rng = np.random.default_rng(seed)
        self._baseline_sampler = None

    @staticmethod
    def _validate_decode_options(
        decoder_strategy: str, decoder_args: Mapping | None
    ) -> None:
        if decoder_strategy != "marginal":
            raise ValueError("TCNOTSampler supports decoder_strategy='marginal' only.")
        if decoder_args:
            raise ValueError("decoder_args are not accepted for marginal decoding.")

    def parallel_leakage_samples(
        self,
        *,
        p_leak: float,
        max_shots: int,
        on_batch: Callable[[int, int], None] | None = None,
        decoder_strategy: str = "marginal",
        decoder_args: Mapping | None = None,
    ) -> tuple[int, int]:
        self._validate_decode_options(decoder_strategy, decoder_args)
        if max_shots < 0:
            raise ValueError(f"max_shots must be non-negative, got {max_shots}.")
        if max_shots == 0:
            return 0, 0
        if not 0 < p_leak <= 1:
            raise ValueError(
                "parallel_leakage_samples conditions on a leakage event, so "
                f"p_leak must be in (0, 1], got {p_leak}."
            )

        n_batches = min(self.n_jobs, max_shots)
        quotient, remainder = divmod(max_shots, n_batches)
        batch_sizes = [
            quotient + (1 if index < remainder else 0)
            for index in range(n_batches)
        ]
        child_seeds = self._seed_sequence.spawn(n_batches)
        results = self.parallel(
            delayed(_batched_tcnot_samples)(
                batch_size=batch_size,
                p_leak=p_leak,
                builder=self.builder,
                seed=child_seed,
                decoder_key=self._decoder_key,
                rounds_per_side=self.rounds_per_side,
                rounds_before=self.rounds_before,
                rounds_after=self.rounds_after,
                p_pauli=self.p_pauli,
                circuit_kwargs=self.circuit_kwargs,
                tcnot_cache_size=self.tcnot_cache_size,
                tesseract_compile_cache_size=self.tesseract_compile_cache_size,
                tesseract_options=self.tesseract_options,
                decoder=self.decoder if self.n_jobs == 1 else None,
            )
            for batch_size, child_seed in zip(batch_sizes, child_seeds)
        )
        n_fails = 0
        n_done = 0
        for batch_fails, batch_done in results:
            n_fails += batch_fails
            n_done += batch_done
            if on_batch is not None:
                on_batch(batch_fails, batch_done)
        return n_fails, n_done

    parallel_erasure_samples = parallel_leakage_samples

    def sample_without_leakage(self, max_shots: int) -> tuple[int, int]:
        if max_shots < 0:
            raise ValueError(f"max_shots must be non-negative, got {max_shots}.")
        if max_shots == 0:
            return 0, 0
        if self.decoder.baseline_dem.num_errors == 0:
            return 0, max_shots

        if self.n_jobs == 1:
            if self._baseline_sampler is None:
                seed = self._seed_sequence.spawn(1)[0]
                self._baseline_sampler = self.baseline_circuit.compile_detector_sampler(
                    seed=_stim_seed(seed)
                )
            syndromes, observable_flips = self._baseline_sampler.sample(
                shots=max_shots,
                separate_observables=True,
                bit_packed=False,
            )
            predictions = self.decoder.decode_batch(
                syndromes=syndromes,
                erasure_checks={},
                decoder_strategy="marginal",
            )
            failures = int(np.sum(np.any(predictions != observable_flips, axis=1)))
            return failures, max_shots

        n_batches = min(self.n_jobs, max_shots)
        quotient, remainder = divmod(max_shots, n_batches)
        batch_sizes = [
            quotient + (1 if index < remainder else 0)
            for index in range(n_batches)
        ]
        child_seeds = self._seed_sequence.spawn(n_batches)
        results = self.parallel(
            delayed(_batched_tcnot_baseline_samples)(
                batch_size=batch_size,
                builder=self.builder,
                seed=child_seed,
                decoder_key=self._decoder_key,
                rounds_per_side=self.rounds_per_side,
                rounds_before=self.rounds_before,
                rounds_after=self.rounds_after,
                p_pauli=self.p_pauli,
                circuit_kwargs=self.circuit_kwargs,
                tcnot_cache_size=self.tcnot_cache_size,
                tesseract_compile_cache_size=self.tesseract_compile_cache_size,
                tesseract_options=self.tesseract_options,
            )
            for batch_size, child_seed in zip(batch_sizes, child_seeds)
        )
        n_fails = sum(batch_fails for batch_fails, _ in results)
        return n_fails, max_shots

    def _subdivide_shots(self, max_shots: int, p_noleak: float) -> tuple[int, int]:
        if p_noleak >= 1:
            return 0, max_shots
        if p_noleak <= 0:
            return max_shots, 0
        fractional_noleak = p_noleak * max_shots
        integer_noleak = int(fractional_noleak)
        remainder = fractional_noleak - integer_noleak
        n_noleak = integer_noleak + int(self._subdivision_rng.random() < remainder)
        return max_shots - n_noleak, n_noleak

    def sample(
        self,
        *,
        p_leak: float,
        max_shots: int,
        max_errors: int = 0,
        min_shots: int = 0,
        on_batch: Callable[[int, int], None] | None = None,
        decoder_strategy: str = "marginal",
        decoder_args: Mapping | None = None,
    ) -> tuple[int, int]:
        """Sample until the shot cap or marginal logical-error target is reached."""
        self._validate_decode_options(decoder_strategy, decoder_args)
        if not 0 <= p_leak <= 1:
            raise ValueError(f"p_leak must be in [0, 1], got {p_leak}.")
        if max_shots < 0 or max_errors < 0 or min_shots < 0:
            raise ValueError(
                "max_shots, max_errors, and min_shots must be non-negative."
            )
        if min_shots > max_shots:
            raise ValueError("min_shots cannot exceed max_shots.")

        p_noleak = (1 - p_leak) ** self.num_leakage_trials
        n_fails = 0
        n_done = 0
        batch_shots = self.n_jobs
        target_batch_seconds = 1.0

        while n_done < max_shots and not (
            max_errors > 0 and n_done >= min_shots and n_fails >= max_errors
        ):
            batch = min(batch_shots, max_shots - n_done)
            n_leak, n_noleak = self._subdivide_shots(batch, p_noleak)
            start = time.perf_counter()
            leak_fails, leak_done = self.parallel_leakage_samples(
                p_leak=p_leak,
                max_shots=n_leak,
            )
            noleak_fails, noleak_done = self.sample_without_leakage(n_noleak)
            elapsed = time.perf_counter() - start

            batch_fails = leak_fails + noleak_fails
            batch_done = leak_done + noleak_done
            n_fails += batch_fails
            n_done += batch_done
            if on_batch is not None:
                on_batch(batch_fails, batch_done)

            if batch_shots > self.n_jobs and elapsed > target_batch_seconds * 1.3:
                batch_shots = max(self.n_jobs, batch_shots // 2)
            for _ in range(4):
                if elapsed > target_batch_seconds * 0.3:
                    break
                batch_shots *= 2
                elapsed *= 2

        return n_fails, n_done

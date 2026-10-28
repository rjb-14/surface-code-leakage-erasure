"""Incremental, unconditioned Monte Carlo sampling for single-patch stability.

Unlike ``sample_LER``'s separately weighted leakage populations, this interface
returns integer failures and shots suitable for Sinter CSVs and budgeted resumes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pymatching
import stim
from joblib import Parallel, delayed

from .decoding import ErasureDecoder, require_pymatching_edge_reweights
from .circuit_builder import iterate_erasure_checks
from .stability_circuit import StabilityCircuitBuilder
from .walking_stability_builder import WalkingStabilityCircuitBuilder


STYLES = ("static", "early", "late")
DECODER_STRATEGIES = ("marginal", "quasi_mle")
GRAPH_PROBABILITY_FLOOR = 1e-9
CIRCUIT_KWARGS = {
    "basis": "Z",
    "Pauli_locations": "2-qubit gates",
    "leak_locations": "2-qubit gates",
    "ec_sched": 8,
    "leak_effect": "skip gates",
}


def resolve_decoder_args(strategy, args):
    if strategy not in DECODER_STRATEGIES:
        raise ValueError(f"decoder_strategy must be one of {DECODER_STRATEGIES}.")
    if strategy == "marginal":
        if args is not None:
            raise ValueError("decoder_args must be None for marginal decoding.")
        return None
    resolved = {"mle_time_limit": None, "max_queue_size": 4096}
    resolved.update(args or {})
    if set(resolved) != {"mle_time_limit", "max_queue_size"}:
        raise ValueError("quasi_mle accepts only mle_time_limit and max_queue_size.")
    limit = resolved["mle_time_limit"]
    if limit is not None and (
        isinstance(limit, (bool, np.bool_))
        or not isinstance(limit, (int, float))
        or not math.isfinite(limit)
        or limit < 0
    ):
        raise ValueError("mle_time_limit must be non-negative and finite, or None.")
    queue = resolved["max_queue_size"]
    if isinstance(queue, bool) or not isinstance(queue, int) or queue < 1:
        raise ValueError("max_queue_size must be a positive integer.")
    return resolved


@dataclass(frozen=True)
class _Model:
    style: str
    distance: int
    rounds: int
    p_pauli: float
    cache_size: int


class _Context:
    def __init__(self, model):
        if model.style == "static":
            self.builder = StabilityCircuitBuilder(model.distance)
        else:
            self.builder = WalkingStabilityCircuitBuilder(
                model.distance, swap_time=model.style
            )
        self.decoder = ErasureDecoder(self.builder, cache_size=model.cache_size)
        # Retain one baseline graph even when optional DEM caches are disabled:
        # erasure reweighting requires a persistent graph with leakage support.
        self.decoder.get_Pauli_matcher_graphdict_edgedict = lru_cache(maxsize=1)(
            self.decoder.get_Pauli_matcher_graphdict_edgedict
        )
        self.baseline_circuit, _ = self.builder.get_circuit(
            model.rounds, model.p_pauli, **CIRCUIT_KWARGS
        )
        dem = self.baseline_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        self.baseline_is_noiseless = dem.num_errors == 0
        self.baseline_matcher = pymatching.Matching.from_detector_error_model(dem)
        self.baseline_sampler = self.baseline_circuit.compile_detector_sampler()


@lru_cache(maxsize=1)
def _worker_context(model):
    # Cache builders inside each process: their bound functools.cache wrappers
    # cannot be pickled. The key intentionally excludes p_leak for circuit reuse.
    return _Context(model)


def _ensure_leakage_edges(context, model, checks):
    """Add missing erasure edges using the decoder's probability floor.

    Two-qubit Pauli noise alone does not span every leakage edge (for example
    static d=2's D5--D10 edge). The reweighting API cannot update absent edges.
    Obtain topology and logical labels from the actual erasure DEM. Use the
    existing decoder's 1e-9 graph regularization for otherwise absent edges.
    Very large inactive weights would degrade PyMatching's integer weight
    quantization for every other edge; infinite weights cannot be added at all.
    """
    matcher, graph, logicals = context.decoder.get_Pauli_matcher_graphdict_edgedict(
        model.rounds, model.p_pauli, **CIRCUIT_KWARGS
    )
    for rnd, gate, qubit in iterate_erasure_checks(checks):
        dem = context.decoder.get_single_ec_average_dem(
            model.rounds, rnd, gate, qubit, **CIRCUIT_KWARGS
        )
        for hyperedge in dem:
            for edge in hyperedge:
                detectors = tuple(term for term in edge if type(term) is int)
                if not detectors or detectors in logicals:
                    continue
                fault_ids = {
                    int(term[1:])
                    for term in edge
                    if isinstance(term, str) and term.startswith("L")
                }
                probability = GRAPH_PROBABILITY_FLOOR
                kwargs = dict(
                    fault_ids=fault_ids,
                    weight=math.log((1 - probability) / probability),
                    error_probability=probability,
                )
                if len(detectors) == 1:
                    matcher.add_boundary_edge(detectors[0], **kwargs)
                elif len(detectors) == 2:
                    matcher.add_edge(*detectors, **kwargs)
                else:
                    raise ValueError(
                        "Single-patch stability decoding requires graphlike edges."
                    )
                logicals[detectors] = frozenset(fault_ids)
                graph[(edge,)] = probability


def _leakage_batch(model, p_leak, shots, strategy, args):
    context = _worker_context(model)
    failures = marginal_failures = 0
    for _ in range(shots):
        locations = context.builder.generate_leakage_locations(
            model.rounds, p_leak, atleast_1=True, leak_locations="2-qubit gates"
        )
        circuit, checks = context.builder.get_circuit(
            model.rounds,
            model.p_pauli,
            leakage_circuit_locations=locations,
            **CIRCUIT_KWARGS,
        )
        simulator = stim.FlipSimulator(batch_size=1)
        simulator.do(circuit)
        syndrome = simulator.get_detector_flips(bit_packed=False).reshape(-1)
        truth = simulator.get_observable_flips(bit_packed=False).reshape(-1)
        _ensure_leakage_edges(context, model, checks)
        marginal, valid = context.decoder.decode(
            checks,
            syndrome,
            model.p_pauli,
            branch_and_bound=strategy == "quasi_mle",
            **(args or {}),
            **CIRCUIT_KWARGS,
        )
        # Match the existing sample_LER convention when search finds no valid
        # solution within its time/queue budget.
        marginal_error = int(np.any(marginal != truth))
        marginal_failures += marginal_error
        failures += (
            int(np.any(valid != truth))
            if strategy == "quasi_mle" and valid is not None
            else marginal_error
        )
    return failures, marginal_failures, shots


class StabilitySampler:
    """Sample Z-basis stability using the repository's erasure-aware PyMatching.

    Leakage is one Bernoulli trial per CNOT, choosing one of its qubits uniformly.
    Finite-bias Pauli noise is uniform DEPOLARIZE2 after two-qubit gates only.
    quasi_mle uses the existing bounded branch-and-bound search, not brute force.
    """

    supported_decoder_strategies = DECODER_STRATEGIES
    paired_marginal_strategies = ("quasi_mle",)

    def __init__(
        self, *, style, distance, rounds, p_pauli=0.0, n_jobs=1, cache_size=4096
    ):
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}.")
        for name, value in (
            ("distance", distance),
            ("rounds", rounds),
            ("n_jobs", n_jobs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if distance % 2:
            raise ValueError("Stability requires even distance.")
        if not math.isfinite(p_pauli) or not 0 <= p_pauli <= 1:
            raise ValueError("p_pauli must be finite and in [0, 1].")
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or cache_size < 0
        ):
            raise ValueError("cache_size must be a non-negative integer.")
        require_pymatching_edge_reweights()
        self.model = _Model(style, distance, rounds, float(p_pauli), cache_size)
        self.context = _worker_context(self.model)
        self.baseline_circuit = self.context.baseline_circuit
        self.num_leakage_trials = self.context.builder.count_leakage_locations(
            rounds, leak_locations="2-qubit gates"
        )[0]
        self.n_jobs = n_jobs
        self.parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        self.rng = np.random.default_rng()

    def _without_leakage(self, shots):
        if shots == 0 or self.context.baseline_is_noiseless:
            return 0
        syndrome, truth = self.context.baseline_sampler.sample(
            shots=shots, separate_observables=True
        )
        predictions = self.context.baseline_matcher.decode_batch(syndrome)
        return int(np.sum(np.any(predictions != truth, axis=1)))

    def sample(
        self,
        *,
        p_leak,
        max_shots,
        max_errors=0,
        min_shots=0,
        on_batch=None,
        decoder_strategy="marginal",
        decoder_args=None,
        on_paired_batch=None,
    ):
        """Return selected-strategy failures/shots, optionally reporting both.

        With quasi_mle, on_paired_batch receives (selected_errors,
        marginal_errors, shots) from the same shots. Error stopping and the
        existing two-argument on_batch callback use the selected strategy.
        """
        args = resolve_decoder_args(decoder_strategy, decoder_args)
        if (
            on_paired_batch is not None
            and decoder_strategy not in self.paired_marginal_strategies
        ):
            raise ValueError("on_paired_batch requires quasi_mle decoding.")
        if not math.isfinite(p_leak) or not 0 <= p_leak <= 1:
            raise ValueError("p_leak must be finite and in [0, 1].")
        for name, value in (
            ("max_shots", max_shots),
            ("max_errors", max_errors),
            ("min_shots", min_shots),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if min_shots > max_shots:
            raise ValueError("min_shots cannot exceed max_shots.")

        # Binomial population counts plus conditional leakage draws give ordinary
        # unconditioned Bernoulli shots, including the rare-leakage regime.
        p_any_leak = (
            1.0
            if p_leak == 1
            else -math.expm1(self.num_leakage_trials * math.log1p(-p_leak))
        )
        failures = done = 0
        window = 4 * self.n_jobs
        batch_size = window
        while done < max_shots and not (
            max_errors > 0 and done >= min_shots and failures >= max_errors
        ):
            shots = min(batch_size, max_shots - done)
            # Avoid taking a large adaptive batch past a required minimum floor.
            if done < min_shots:
                shots = min(shots, min_shots - done)
            n_leak = int(self.rng.binomial(shots, p_any_leak))
            started = time.perf_counter()
            batch_failures = self._without_leakage(shots - n_leak)
            # Both strategies use the same Pauli-only prediction on these shots.
            batch_marginal_failures = batch_failures
            if n_leak:
                chunks = min(n_leak, window)
                quotient, remainder = divmod(n_leak, chunks)
                results = self.parallel(
                    delayed(_leakage_batch)(
                        self.model,
                        p_leak,
                        quotient + int(index < remainder),
                        decoder_strategy,
                        args,
                    )
                    for index in range(chunks)
                )
                for count, marginal_count, _ in results:
                    batch_failures += count
                    batch_marginal_failures += marginal_count
            failures += batch_failures
            done += shots
            if on_batch is not None:
                on_batch(batch_failures, shots)
            if on_paired_batch is not None:
                on_paired_batch(batch_failures, batch_marginal_failures, shots)
            elapsed = time.perf_counter() - started
            if elapsed < 0.3:
                batch_size = min(batch_size * 2, 65536)
            elif elapsed > 1.3:
                batch_size = max(window, batch_size // 2)
        return failures, done

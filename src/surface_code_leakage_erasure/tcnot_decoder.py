"""Hyperedge-preserving leakage decoder for the early-walking tCNOT."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np
import stim

from .circuit_builder import iterate_erasure_checks
from .early_walking_transversal_cx import resolve_tcnot_round_schedule


DECODER_STRATEGIES = ("marginal", "quasi_mle")
GATE_EXCLUDING_EFFECTS = (
    "skip gates",
    "tailored then skip",
    "decode skip gates",
    "decode tailored then skip",
)


@dataclass(frozen=True)
class Hyperedge:
    """One atomic DEM mechanism, including all detector and logical flips."""

    detectors: frozenset[int]
    observables: frozenset[int]


HyperedgeModel = dict[Hyperedge, float]
LeakageLocation = tuple[int, int, int]
ErasureCheck = tuple[int, int, int]


class HyperedgeDecoder(Protocol):
    def decode(
        self,
        *,
        syndrome,
        model: Mapping[Hyperedge, float],
        num_detectors: int,
        num_observables: int,
    ):
        """Return predicted logical-observable flips."""


def _toggle(values: set[int], value: int):
    if value in values:
        values.remove(value)
    else:
        values.add(value)


def _add_independent_probability(
    model: HyperedgeModel,
    hyperedge: Hyperedge,
    probability: float,
):
    old_probability = model.get(hyperedge, 0.0)
    model[hyperedge] = (
        old_probability * (1 - probability)
        + (1 - old_probability) * probability
    )


def dem_to_hyperedge_model(dem: stim.DetectorErrorModel) -> HyperedgeModel:
    """Convert a Stim DEM without splitting atomic error instructions at ``^``."""

    model: HyperedgeModel = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                _toggle(detectors, target.val)
            elif target.is_logical_observable_id():
                _toggle(observables, target.val)
        hyperedge = Hyperedge(frozenset(detectors), frozenset(observables))
        _add_independent_probability(
            model,
            hyperedge,
            float(instruction.args_copy()[0]),
        )
    return model


def hyperedge_model_to_dem(
    model: Mapping[Hyperedge, float],
    num_detectors: int,
    num_observables: int,
) -> stim.DetectorErrorModel:
    """Serialize a complete atomic hyperedge model for Tesseract."""

    if num_detectors < 0 or num_observables < 0:
        raise ValueError("num_detectors and num_observables must be non-negative.")

    lines = []
    for hyperedge, probability in sorted(
        model.items(),
        key=lambda item: (
            tuple(sorted(item[0].detectors)),
            tuple(sorted(item[0].observables)),
            item[1],
        ),
    ):
        if not 0 <= probability <= 1:
            raise ValueError(
                f"Hyperedge probability must be in [0, 1], got {probability}."
            )
        if probability == 0:
            continue
        if any(d < 0 or d >= num_detectors for d in hyperedge.detectors):
            raise ValueError(f"Out-of-range detector in {hyperedge}.")
        if any(o < 0 or o >= num_observables for o in hyperedge.observables):
            raise ValueError(f"Out-of-range observable in {hyperedge}.")
        targets = [f"D{d}" for d in sorted(hyperedge.detectors)]
        targets.extend(f"L{o}" for o in sorted(hyperedge.observables))
        if targets:
            lines.append(f"error({probability!r}) " + " ".join(targets))

    if num_detectors:
        lines.append(f"detector D{num_detectors - 1}")
    if num_observables:
        lines.append(f"logical_observable L{num_observables - 1}")
    return stim.DetectorErrorModel("\n".join(lines))


def _average_disjoint_models(models: list[HyperedgeModel]) -> HyperedgeModel:
    if not models:
        return {}
    weight = 1 / len(models)
    averaged: HyperedgeModel = {}
    for model in models:
        for hyperedge, probability in model.items():
            averaged[hyperedge] = averaged.get(hyperedge, 0.0) + weight * probability
    return averaged


def _backtrack_cover(remaining_rows, uncovered, row_unions):
    """Find one candidate per erasure check covering all selected hyperedges."""

    stack = []
    result = None
    while True:
        while result is None:
            if not uncovered:
                result = (True, set())
                break
            if not remaining_rows:
                result = (False, uncovered)
                break

            current_row = remaining_rows[0]
            leftover_rows = remaining_rows[1:]
            leftover_unions = row_unions[1:]
            coverage = [len(uncovered & candidate) for candidate in current_row]
            candidate_rows = [
                current_row[index]
                for index in np.argsort(coverage)[::-1]
                if coverage[index] > 0
            ]
            if not candidate_rows:
                remaining_rows, row_unions = leftover_rows, leftover_unions
                continue

            stack.append(
                [candidate_rows, 0, uncovered, leftover_rows, leftover_unions, uncovered]
            )
            remaining_rows, row_unions = leftover_rows, leftover_unions
            uncovered = uncovered - candidate_rows[0]

        while stack:
            (
                candidate_rows,
                index,
                witness_so_far,
                leftover_rows,
                leftover_unions,
                frame_uncovered,
            ) = stack[-1]
            valid, witness = result
            if valid:
                stack.pop()
                continue
            if len(witness) < len(witness_so_far):
                witness_so_far = witness

            next_index = index + 1
            if not witness_so_far & set().union(
                *candidate_rows[next_index:], set()
            ):
                stack.pop()
                result = (False, witness_so_far)
                continue

            stack[-1][1] = next_index
            stack[-1][2] = witness_so_far
            remaining_rows, row_unions = leftover_rows, leftover_unions
            uncovered = frame_uncovered - candidate_rows[next_index]
            result = None
            break
        else:
            return result


class TesseractHyperedgeDecoder:
    """Adapter from :class:`HyperedgeModel` to Tesseract's Python decoder."""

    def __init__(
        self,
        compile_cache_size: int | None = 128,
        det_order_options: Mapping | None = None,
        **config_options,
    ):
        try:
            from tesseract_decoder import tesseract, utils as tesseract_utils
        except ImportError as error:
            raise ImportError(
                "TesseractHyperedgeDecoder requires the optional tesseract-decoder "
                'package. Install it with `python -m pip install -e ".[tesseract]"`.'
            ) from error
        if compile_cache_size is not None and compile_cache_size < 0:
            raise ValueError("compile_cache_size must be non-negative or None.")
        if det_order_options is not None and "det_orders" in config_options:
            raise ValueError("Pass either det_order_options or det_orders, not both.")
        self._tesseract = tesseract
        self._tesseract_utils = tesseract_utils
        self.config_options = dict(config_options)
        self.det_order_options = self._validate_det_order_options(det_order_options)
        self.compile_cache_size = compile_cache_size
        self._compile = lru_cache(maxsize=compile_cache_size)(self._compile)

    def _validate_det_order_options(self, options: Mapping | None) -> dict | None:
        """Validate the JSON-serializable per-DEM detector-order recipe."""
        if options is None:
            return None
        options = dict(options)
        unknown = set(options) - {"num_det_orders", "method"}
        if unknown:
            raise ValueError(
                f"Unknown det_order_options: {sorted(unknown)}. "
                "Supported: 'num_det_orders', 'method'."
            )
        if set(options) != {"num_det_orders", "method"}:
            raise ValueError(
                "det_order_options requires 'num_det_orders' and 'method'."
            )
        num_det_orders = options["num_det_orders"]
        if (
            isinstance(num_det_orders, bool)
            or not isinstance(num_det_orders, int)
            or num_det_orders < 1
        ):
            raise ValueError(
                "det_order_options.num_det_orders must be a positive integer."
            )
        method_name = options["method"]
        if not isinstance(method_name, str):
            raise ValueError(
                "det_order_options.method must be a DetectorOrderMethod name."
            )
        try:
            self._resolve_det_order_method(method_name)
        except AttributeError as error:
            raise ValueError(
                f"Unknown DetectorOrderMethod {method_name!r}."
            ) from error
        return {"num_det_orders": num_det_orders, "method": method_name}

    def _resolve_det_order_method(self, method_name: str):
        """Resolve detector-order enum names across Tesseract Python APIs."""
        detector_order_method = getattr(
            self._tesseract_utils, "DetectorOrderMethod", None
        )
        if detector_order_method is not None and hasattr(
            detector_order_method, method_name
        ):
            return getattr(detector_order_method, method_name)

        det_order = getattr(self._tesseract_utils, "DetOrder", None)
        legacy_name = {
            "Index": "DetIndex",
            "BFS": "DetBFS",
            "Coordinate": "DetCoordinate",
        }.get(method_name, method_name)
        if det_order is not None and hasattr(det_order, legacy_name):
            return getattr(det_order, legacy_name)
        raise AttributeError(method_name)

    @staticmethod
    def _model_key(model: Mapping[Hyperedge, float]) -> tuple:
        return tuple(
            sorted(
                (
                    tuple(sorted(hyperedge.detectors)),
                    tuple(sorted(hyperedge.observables)),
                    float(probability),
                )
                for hyperedge, probability in model.items()
            )
        )

    def _compile(self, model_key, num_detectors, num_observables):
        model = {
            Hyperedge(frozenset(detectors), frozenset(observables)): probability
            for detectors, observables, probability in model_key
        }
        dem = hyperedge_model_to_dem(model, num_detectors, num_observables)
        config_options = dict(self.config_options)
        if self.det_order_options is not None:
            method = self._resolve_det_order_method(
                self.det_order_options["method"]
            )
            config_options["det_orders"] = self._tesseract_utils.build_det_orders(
                dem=dem,
                num_det_orders=self.det_order_options["num_det_orders"],
                method=method,
            )
        return self._tesseract.TesseractConfig(
            dem=dem,
            **config_options,
        ).compile_decoder()

    def cache_info(self):
        return self._compile.cache_info()

    def cache_clear(self):
        self._compile.cache_clear()

    def compile_model(self, *, model, num_detectors, num_observables):
        return self._compile(
            self._model_key(model),
            num_detectors,
            num_observables,
        )

    @staticmethod
    def _validate_syndrome(syndrome, num_detectors):
        syndrome = np.asarray(syndrome)
        if syndrome.dtype != np.bool_:
            raise TypeError("Tesseract syndrome must be an unpacked NumPy boolean array.")
        if syndrome.ndim != 1 or syndrome.shape[0] != num_detectors:
            raise ValueError(
                f"Syndrome must have shape ({num_detectors},), got {syndrome.shape}."
            )
        return syndrome

    @staticmethod
    def _validate_syndromes(syndromes, num_detectors):
        syndromes = np.asarray(syndromes)
        if syndromes.dtype != np.bool_:
            raise TypeError("Tesseract syndromes must be unpacked NumPy boolean arrays.")
        if syndromes.ndim != 2 or syndromes.shape[1] != num_detectors:
            raise ValueError(
                f"Syndromes must have shape (shots, {num_detectors}), got {syndromes.shape}."
            )
        return syndromes

    def decode_compiled(
        self,
        *,
        decoder,
        syndrome,
        num_detectors,
        num_observables,
    ):
        syndrome = self._validate_syndrome(syndrome, num_detectors)
        prediction = np.asarray(decoder.decode(syndrome), dtype=np.bool_)
        if prediction.shape != (num_observables,):
            raise ValueError(
                "Tesseract returned an unexpected prediction shape: "
                f"expected ({num_observables},), got {prediction.shape}."
            )
        return prediction

    def decode_compiled_batch(
        self,
        *,
        decoder,
        syndromes,
        num_detectors,
        num_observables,
    ):
        syndromes = self._validate_syndromes(syndromes, num_detectors)
        predictions = np.asarray(decoder.decode_batch(syndromes), dtype=np.bool_)
        expected = (syndromes.shape[0], num_observables)
        if predictions.shape != expected:
            raise ValueError(
                f"Tesseract returned shape {predictions.shape}; expected {expected}."
            )
        return predictions

    def decode(self, *, syndrome, model, num_detectors, num_observables):
        decoder = self.compile_model(
            model=model,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )
        return self.decode_compiled(
            decoder=decoder,
            syndrome=syndrome,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )

    def decode_batch(self, *, syndromes, model, num_detectors, num_observables):
        decoder = self.compile_model(
            model=model,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )
        return self.decode_compiled_batch(
            decoder=decoder,
            syndromes=syndromes,
            num_detectors=num_detectors,
            num_observables=num_observables,
        )

    def solve_compiled(
        self,
        *,
        decoder,
        syndrome,
        num_detectors,
        num_observables,
    ):
        syndrome = self._validate_syndrome(syndrome, num_detectors)
        error_indices = decoder.decode_to_errors(syndrome)
        prediction = np.asarray(
            decoder.get_observables_from_errors(error_indices),
            dtype=np.bool_,
        )
        if prediction.shape != (num_observables,):
            raise ValueError(
                "Tesseract returned an unexpected prediction shape: "
                f"expected ({num_observables},), got {prediction.shape}."
            )
        hyperedges = [
            Hyperedge(
                frozenset(decoder.errors[index].symptom.detectors),
                frozenset(decoder.errors[index].symptom.observables),
            )
            for index in error_indices
        ]
        return prediction, float(decoder.cost_from_errors(error_indices)), hyperedges


class EarlyWalkingTCNOTDecoder:
    """Erasure-aware tCNOT decoder that retains genuine detector hyperedges."""

    def __init__(
        self,
        builder,
        rounds_per_side: int | None = None,
        p_pauli: float = 0.0,
        hyperedge_decoder: HyperedgeDecoder | None = None,
        circuit_kwargs: dict | None = None,
        cache_size: int | None = 4096,
        *,
        rounds_before: int | None = None,
        rounds_after: int | None = None,
    ):
        if cache_size is not None and cache_size < 0:
            raise ValueError("cache_size must be non-negative or None.")

        schedule = resolve_tcnot_round_schedule(
            rounds_per_side,
            rounds_before=rounds_before,
            rounds_after=rounds_after,
        )
        self.builder = builder
        self.rounds_per_side = (
            schedule.before if rounds_per_side is not None else None
        )
        self.rounds_before = schedule.before
        self.rounds_after = schedule.after
        self.total_rounds = schedule.total
        self._round_kwargs = (
            {"rounds_per_side": rounds_per_side}
            if rounds_per_side is not None
            else {
                "rounds_before": schedule.before,
                "rounds_after": schedule.after,
            }
        )
        self.p_pauli = p_pauli
        self.circuit_kwargs = {"basis": "X", **(circuit_kwargs or {})}
        if hyperedge_decoder is None:
            hyperedge_decoder = TesseractHyperedgeDecoder()
        if not callable(getattr(hyperedge_decoder, "decode", None)):
            raise TypeError("hyperedge_decoder must provide a callable decode() method.")
        self.hyperedge_decoder = hyperedge_decoder
        self.cache_size = cache_size

        self._physical_kwargs = builder.validate_circuit_kwargs(self.circuit_kwargs)
        self._envelope_kwargs = builder.validate_circuit_kwargs(
            self.circuit_kwargs,
            decode=True,
        )
        self.leak_effect = self._physical_kwargs["leak_effect"]
        self.ec_sched = self._physical_kwargs["ec_sched"]
        self.leak_locations = self._physical_kwargs["leak_locations"]

        self.baseline_circuit, _ = builder.get_circuit(
            p=p_pauli,
            p_leak=0,
            **self._round_kwargs,
            **self._physical_kwargs,
        )
        self.num_detectors = self.baseline_circuit.num_detectors
        self.num_observables = self.baseline_circuit.num_observables
        if self.num_observables not in {1, 2}:
            raise ValueError(
                "EarlyWalkingTCNOTDecoder expects one or two logical observables."
            )
        self.baseline_dem = self._circuit_dem(self.baseline_circuit)
        self.baseline_model = dem_to_hyperedge_model(self.baseline_dem)
        self._compiled_baseline_decoder = None

        if cache_size is not None and cache_size > 0:
            self.get_leakage_locations_from_erasure_check = lru_cache(cache_size)(
                self.get_leakage_locations_from_erasure_check
            )
            self.get_leakage_location_model = lru_cache(cache_size)(
                self.get_leakage_location_model
            )
            self.get_erasure_check_average_model = lru_cache(cache_size)(
                self.get_erasure_check_average_model
            )
            self.get_erasure_check_optimistic_model = lru_cache(cache_size)(
                self.get_erasure_check_optimistic_model
            )
            self.get_erasure_check_hyperedge_sets = lru_cache(cache_size)(
                self.get_erasure_check_hyperedge_sets
            )

    @staticmethod
    def _circuit_dem(circuit):
        return circuit.detector_error_model(
            decompose_errors=False,
            approximate_disjoint_errors=True,
            flatten_loops=True,
        )

    @staticmethod
    def flatten_erasure_checks(erasure_checks) -> list[ErasureCheck]:
        if erasure_checks is None:
            return []
        return list(iterate_erasure_checks(erasure_checks))

    def get_leakage_locations_from_erasure_check(
        self,
        ec_rnd: int,
        ec_gate: int,
        ec_qubit: int,
    ) -> tuple[LeakageLocation, ...]:
        locations = self.builder.single_ec_traceback(
            ec_rnd,
            ec_gate,
            ec_qubit,
            self.ec_sched,
            self.leak_locations,
        )
        return tuple(tuple(int(value) for value in location) for location in locations)

    def _single_location_pattern(self, location: LeakageLocation):
        leak_rnd, leak_gate, qubit = location
        return {leak_rnd: {leak_gate: {qubit}}}

    def get_leakage_location_model(
        self,
        leak_rnd: int,
        leak_gate: int,
        qubit: int,
    ) -> HyperedgeModel:
        circuit, _ = self.builder.get_circuit(
            p=0,
            leakage_circuit_locations=self._single_location_pattern(
                (leak_rnd, leak_gate, qubit)
            ),
            use_cache=False,
            **self._round_kwargs,
            **self._envelope_kwargs,
        )
        if (
            circuit.num_detectors != self.num_detectors
            or circuit.num_observables != self.num_observables
        ):
            raise ValueError("Leakage envelope changed the tCNOT detector/observable shape.")
        return dem_to_hyperedge_model(self._circuit_dem(circuit))

    def get_erasure_check_average_model(
        self,
        ec_rnd: int,
        ec_gate: int,
        ec_qubit: int,
    ) -> HyperedgeModel:
        locations = self.get_leakage_locations_from_erasure_check(
            ec_rnd,
            ec_gate,
            ec_qubit,
        )
        if not locations:
            raise ValueError(
                "Triggered erasure check has no candidate leakage locations: "
                f"{(ec_rnd, ec_gate, ec_qubit)}."
            )
        return _average_disjoint_models(
            [self.get_leakage_location_model(*location) for location in locations]
        )

    def get_erasure_check_optimistic_model(
        self,
        ec_rnd: int,
        ec_gate: int,
        ec_qubit: int,
    ) -> HyperedgeModel:
        optimistic: HyperedgeModel = {}
        for location in self.get_leakage_locations_from_erasure_check(
            ec_rnd,
            ec_gate,
            ec_qubit,
        ):
            for hyperedge, probability in self.get_leakage_location_model(
                *location
            ).items():
                optimistic[hyperedge] = max(
                    probability,
                    optimistic.get(hyperedge, 0),
                )
        return optimistic

    def get_erasure_check_hyperedge_sets(
        self,
        ec_rnd: int,
        ec_gate: int,
        ec_qubit: int,
    ):
        return tuple(
            frozenset(self.get_leakage_location_model(*location))
            for location in self.get_leakage_locations_from_erasure_check(
                ec_rnd,
                ec_gate,
                ec_qubit,
            )
        )

    def get_erasure_check_coverage_table(self, erasure_checks):
        rows = []
        coverage = {}
        for ec_index, check in enumerate(self.flatten_erasure_checks(erasure_checks)):
            hyperedge_sets = self.get_erasure_check_hyperedge_sets(*check)
            rows.append(hyperedge_sets)
            for location_index, hyperedge_set in enumerate(hyperedge_sets):
                for hyperedge in hyperedge_set:
                    coverage.setdefault(hyperedge, []).append(
                        (ec_index, location_index)
                    )
        return rows, coverage

    def check_solution_validity(
        self,
        decoded_hyperedges,
        erasure_checks,
        chosen_leakage_locations: list[LeakageLocation] | None = None,
    ):
        if self.leak_effect not in GATE_EXCLUDING_EFFECTS:
            return True, set()

        decoded = set(decoded_hyperedges)
        for location in chosen_leakage_locations or ():
            decoded -= set(self.get_leakage_location_model(*location))
        rows, coverage = self.get_erasure_check_coverage_table(erasure_checks)
        decoded &= coverage.keys()

        options_per_row = [0] * len(rows)
        for hyperedge in decoded:
            for ec_index in {index for index, _ in coverage[hyperedge]}:
                options_per_row[ec_index] += 1
        useful = np.argsort(options_per_row)[options_per_row.count(0) :]
        sorted_rows = [rows[index] for index in useful]
        row_unions = [set().union(*row) for row in sorted_rows]
        return _backtrack_cover(sorted_rows, decoded, row_unions)

    def calculate_hyperedge_reweights(self, erasure_checks) -> HyperedgeModel:
        return self._calculate_hyperedge_reweights_from_checks(
            self.flatten_erasure_checks(erasure_checks)
        )

    def _calculate_hyperedge_reweights_from_checks(
        self,
        checks,
    ) -> HyperedgeModel:
        model = dict(self.baseline_model)
        for check in checks:
            for hyperedge, probability in self.get_erasure_check_average_model(
                *check
            ).items():
                _add_independent_probability(model, hyperedge, probability)
        return model

    def calculate_search_reweights(
        self,
        fixed_leakage_locations,
        relaxed_erasure_checks,
    ) -> HyperedgeModel:
        model = dict(self.baseline_model)
        for location in fixed_leakage_locations:
            for hyperedge, probability in self.get_leakage_location_model(
                *location
            ).items():
                _add_independent_probability(model, hyperedge, probability)
        for check in relaxed_erasure_checks:
            for hyperedge, probability in self.get_erasure_check_optimistic_model(
                *check
            ).items():
                _add_independent_probability(model, hyperedge, probability)
        if any(probability > 0.5 + 1e-12 for probability in model.values()):
            raise ValueError(
                "Quasi-MLE relaxation requires every hyperedge probability <= 0.5."
            )
        return model

    def solve_model(self, model, syndrome):
        backend = self.hyperedge_decoder
        if not all(
            callable(getattr(backend, name, None))
            for name in ("compile_model", "solve_compiled")
        ):
            raise TypeError(
                "Quasi-MLE requires a backend with compile_model() and solve_compiled()."
            )
        prepared = backend.compile_model(
            model=model,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )
        return backend.solve_compiled(
            decoder=prepared,
            syndrome=syndrome,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )

    @staticmethod
    def _validate_decode_options(decoder_strategy, decoder_args):
        if decoder_strategy not in DECODER_STRATEGIES:
            raise ValueError(f"decoder_strategy must be one of {DECODER_STRATEGIES}.")
        if not decoder_args:
            return
        if decoder_strategy != "quasi_mle":
            raise ValueError(
                "decoder_args are only accepted for decoder_strategy='quasi_mle'."
            )
        unknown = set(decoder_args) - {"max_nodes", "mle_time_limit"}
        if unknown:
            raise ValueError(f"Unknown quasi-MLE decoder_args: {sorted(unknown)}.")

    def _supports_prepared_backend(self):
        return all(
            callable(getattr(self.hyperedge_decoder, name, None))
            for name in (
                "compile_model",
                "decode_compiled",
                "decode_compiled_batch",
            )
        )

    def _get_compiled_baseline_decoder(self):
        if self._compiled_baseline_decoder is None:
            self._compiled_baseline_decoder = self.hyperedge_decoder.compile_model(
                model=MappingProxyType(self.baseline_model),
                num_detectors=self.num_detectors,
                num_observables=self.num_observables,
            )
        return self._compiled_baseline_decoder

    def decode(
        self,
        syndrome,
        erasure_checks=None,
        decoder_strategy="marginal",
        decoder_args=None,
    ):
        """Decode one syndrome.

        Marginal decoding returns one Boolean prediction array.  Quasi-MLE
        returns ``(marginal_prediction, valid_prediction_or_none)``; the second
        value is ``None`` if the search cannot produce a valid leakage-location
        assignment, including when its search budget expires first.
        """
        self._validate_decode_options(decoder_strategy, decoder_args)
        if decoder_strategy == "quasi_mle":
            # Exact Tesseract bounds are a precondition for every quasi-MLE
            # result, including fast paths that do not enter the search.
            from .tcnot_mle_search import _require_exact_backend

            _require_exact_backend(self.hyperedge_decoder)

        flattened_checks = self.flatten_erasure_checks(erasure_checks)
        model = (
            MappingProxyType(self.baseline_model)
            if not flattened_checks
            else self._calculate_hyperedge_reweights_from_checks(flattened_checks)
        )

        if not flattened_checks and self._supports_prepared_backend():
            result = self.hyperedge_decoder.decode_compiled(
                decoder=self._get_compiled_baseline_decoder(),
                syndrome=syndrome,
                num_detectors=self.num_detectors,
                num_observables=self.num_observables,
            )
            return (result, result) if decoder_strategy == "quasi_mle" else result

        if decoder_strategy == "quasi_mle":
            from .tcnot_mle_search import find_valid_solution

            marginal, _, hyperedges = self.solve_model(model, syndrome)
            valid, _ = self.check_solution_validity(hyperedges, erasure_checks)
            if valid:
                return marginal, marginal
            return marginal, find_valid_solution(
                decoder=self,
                syndrome=syndrome,
                erasure_checks=erasure_checks,
                **(decoder_args or {}),
            )

        return self.hyperedge_decoder.decode(
            syndrome=syndrome,
            model=model,
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
        )

    def decode_batch(
        self,
        syndromes,
        erasure_checks=None,
        decoder_strategy="marginal",
        decoder_args=None,
    ):
        self._validate_decode_options(decoder_strategy, decoder_args)
        if decoder_strategy == "quasi_mle":
            raise ValueError("decode_batch does not support quasi-MLE.")
        syndromes = np.asarray(syndromes)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.num_detectors:
            raise ValueError(
                f"Syndromes must have shape (shots, {self.num_detectors}), "
                f"got {syndromes.shape}."
            )
        if syndromes.shape[0] == 0:
            return np.empty((0, self.num_observables), dtype=np.bool_)

        flattened_checks = self.flatten_erasure_checks(erasure_checks)
        no_erasure = not flattened_checks
        model = (
            MappingProxyType(self.baseline_model)
            if no_erasure
            else self._calculate_hyperedge_reweights_from_checks(flattened_checks)
        )
        if no_erasure and self._supports_prepared_backend():
            return self.hyperedge_decoder.decode_compiled_batch(
                decoder=self._get_compiled_baseline_decoder(),
                syndromes=syndromes,
                num_detectors=self.num_detectors,
                num_observables=self.num_observables,
            )
        backend_batch = getattr(self.hyperedge_decoder, "decode_batch", None)
        if callable(backend_batch):
            return backend_batch(
                syndromes=syndromes,
                model=model,
                num_detectors=self.num_detectors,
                num_observables=self.num_observables,
            )
        return np.asarray(
            [
                self.hyperedge_decoder.decode(
                    syndrome=syndrome,
                    model=model,
                    num_detectors=self.num_detectors,
                    num_observables=self.num_observables,
                )
                for syndrome in syndromes
            ],
            dtype=np.bool_,
        )


TCNOTDecoder = EarlyWalkingTCNOTDecoder

"""Moonwalking Z-parity stability with X-basis data boundaries."""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

from .early_walking_x_stability_circuit import EarlyWalkingXStabilityCircuitLayout
from .walking_stability_builder import WalkingStabilityCircuitBuilder


class EarlyWalkingXStabilityCircuitBuilder(WalkingStabilityCircuitBuilder):
    """Build the clipped-corner stability experiment.

    Leakage is one Bernoulli trial per nominal CX, choosing one endpoint.
    The erasure decoder completes its matching topology from the erasure DEM.
    """

    requires_erasure_graph_completion = True

    def __init__(self, d: int | EarlyWalkingXStabilityCircuitLayout, seed=None):
        layout = (
            d if isinstance(d, EarlyWalkingXStabilityCircuitLayout)
            else EarlyWalkingXStabilityCircuitLayout(d)
        )
        super().__init__(layout, seed=seed)
        # The inherited location counter assumes identical counts in both phases.
        phase_counts = [tuple(len(layer) for layer in phase) for phase in self.cnot_lists]
        if phase_counts[0] != phase_counts[1]:
            raise ValueError("Moonwalking phases must have equal CX counts.")
        for layers in self.cnot_lists:
            if any(len(layer) != len(set(layer)) for layer in layers):
                raise ValueError("A CX layer uses the same qubit more than once.")

    def __repr__(self):
        return f"EarlyWalkingXStabilityCircuitBuilder(d={self.layout.d})"

    def validate_circuit_kwargs(self, circuit_kwargs, decode=False):
        kwargs = dict(circuit_kwargs)
        kwargs.setdefault("basis", "X")
        kwargs.setdefault("leak_effect", "skip gates")
        validated = super().validate_circuit_kwargs(kwargs, decode=decode)
        for key, expected in (
            ("basis", "X"), ("ec_sched", 8), ("leak_locations", "2-qubit gates"),
        ):
            if validated[key] != expected:
                raise ValueError(f"This experiment requires {key}={expected!r}.")
        if validated["leak_effect"] not in ("skip gates", "decode skip gates"):
            raise ValueError("This experiment requires leak_effect='skip gates'.")
        return validated

    def _final_measurement_cache_key(self, p, Pauli_locations, basis):
        return (p, Pauli_locations, basis, self.final_round_parity)

    def _get_final_observable_text(self, rounds, basis):
        records = [
            self.tracker.get_curr_meas(q)
            for q in self.layout.get_stability_support((rounds-1) % 2)
        ]
        return "OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{r}]" for r in records)

    @staticmethod
    def _validate_rounds(rounds):
        if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 1:
            raise ValueError("rounds must be a positive integer.")
        return int(rounds)

    @staticmethod
    def _validate_leak_probability(p_leak):
        if (isinstance(p_leak, bool) or not isinstance(p_leak, Real)
                or not isfinite(p_leak) or not 0 <= p_leak <= 1):
            raise ValueError("p_leak must be finite and in [0, 1].")

    @staticmethod
    def _validate_leak_locations(leak_locations):
        if leak_locations != "2-qubit gates":
            raise ValueError("This experiment requires leak_locations='2-qubit gates'.")

    def _validate_injections(self, rounds, locations):
        if locations is None:
            return
        if not isinstance(locations, Mapping):
            raise ValueError("leakage_circuit_locations must map rounds to gate steps.")
        for rnd, steps in locations.items():
            if (isinstance(rnd, bool) or not isinstance(rnd, Integral)
                    or not 0 <= rnd < rounds or not isinstance(steps, Mapping)):
                raise ValueError("Invalid leakage round or gate-step mapping.")
            for step, qubits in steps.items():
                if (isinstance(step, bool) or not isinstance(step, Integral)
                        or not 0 <= step < 4):
                    raise ValueError("Leakage step must be an integer in [0, 3].")
                if not isinstance(qubits, (set, frozenset)):
                    raise ValueError("Leakage endpoints must be sets of qubit indices.")
                for q in qubits:
                    if (isinstance(q, bool) or not isinstance(q, Integral)
                            or q not in self.cnot_dicts[rnd % 2][step]):
                        raise ValueError("Leakage must occur on a nominal CX endpoint at that step.")

    def count_leakage_locations(self, rounds, leak_locations="2-qubit gates"):
        rounds = self._validate_rounds(rounds)
        self._validate_leak_locations(leak_locations)
        return super().count_leakage_locations(rounds, leak_locations)

    def ravel_leakage_locations(self, rounds, leak_locs_unraveled, leak_locations="2-qubit gates"):
        total, _, _ = self.count_leakage_locations(rounds, leak_locations)
        locations = list(leak_locs_unraveled)
        if any(isinstance(i, bool) or not isinstance(i, Integral)
               or not 0 <= i < total for i in locations):
            raise ValueError("Flat leakage indices must identify nominal CX trials.")
        return super().ravel_leakage_locations(rounds, locations, leak_locations)

    def generate_leakage_locations(
        self, rounds, p_leak, atleast_1=False, leak_locations="2-qubit gates",
    ):
        self.count_leakage_locations(rounds, leak_locations)
        self._validate_leak_probability(p_leak)
        if p_leak == 0:
            if atleast_1:
                raise ValueError("Cannot condition on leakage when p_leak=0.")
            return self.ravel_leakage_locations(rounds, [], leak_locations)
        return super().generate_leakage_locations(rounds, p_leak, atleast_1, leak_locations)

    def single_ec_traceback(self, ec_rnd, ec_gate, ec_qubit, ec_sched,
                            leak_locations="2-qubit gates"):
        self._validate_leak_locations(leak_locations)
        if ec_sched != 8 or ec_gate != 3:
            raise ValueError("This experiment reports erasures at round-end ec_sched=8 checks.")
        return super().single_ec_traceback(ec_rnd, ec_gate, ec_qubit, ec_sched, leak_locations)

    def get_circuit(
        self, rounds, p=0.0, p_leak=0.0,
        leakage_circuit_locations=None, use_cache=True, **circuit_kwargs,
    ):
        rounds = self._validate_rounds(rounds)
        self._validate_leak_probability(p_leak)
        self._validate_injections(rounds, leakage_circuit_locations)
        return super().get_circuit(
            rounds, p, p_leak,
            leakage_circuit_locations=leakage_circuit_locations,
            use_cache=use_cache, **circuit_kwargs,
        )

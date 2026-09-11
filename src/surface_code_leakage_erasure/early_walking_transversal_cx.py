"""Transversal CNOT between two early-walking surface-code patches."""

from dataclasses import dataclass

from .two_patch_layout import TwoPatchEarlyWalkingLayout
from .walking_circuit_builder import WalkingSCCircuitBuilder


@dataclass(frozen=True)
class TCNOTRoundSchedule:
    """Resolved syndrome-extraction rounds before and after the tCNOT."""

    before: int
    after: int

    @property
    def total(self) -> int:
        return self.before + self.after


def resolve_tcnot_round_schedule(
    rounds_per_side: int | None,
    *,
    rounds_before: int | None,
    rounds_after: int | None,
) -> TCNOTRoundSchedule:
    """Resolve the legacy symmetric API or the explicit asymmetric API."""

    if rounds_per_side is not None:
        if rounds_before is not None or rounds_after is not None:
            raise ValueError(
                "rounds_per_side cannot be combined with rounds_before or "
                "rounds_after."
            )
        rounds_before = rounds_after = rounds_per_side
    elif rounds_before is None or rounds_after is None:
        raise ValueError(
            "Provide rounds_per_side or both rounds_before and rounds_after."
        )

    for name, value in (
        ("rounds_before", rounds_before),
        ("rounds_after", rounds_after),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a non-negative integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    if rounds_before == 0 and rounds_after == 0:
        raise ValueError("rounds_before and rounds_after cannot both be zero.")
    return TCNOTRoundSchedule(rounds_before, rounds_after)


class EarlyWalkingTransversalCXBuilder(WalkingSCCircuitBuilder):
    """Build an X-basis transversal-CNOT experiment.

    Patch 1 is the control and patch 2 is the target. Syndrome-extraction round
    counts can be symmetric through ``rounds_per_side`` or independently set
    through ``rounds_before`` and ``rounds_after``. Observable 0 tracks the
    propagated control X logical
    (X_control * X_target), and observable 1 tracks the target X logical.

    Leakage uses the ordinary walking round/step representation, with
    :attr:`TCX_STEP` reserved for the transversal layer at ``cx_round``.
    """

    _supports_two_patch_layout = True
    _supports_odd_rounds = True
    _CX_DETECTOR_CACHE_TAG = "transversal-cx-boundary"
    TCX_STEP = -1

    def __init__(self, d: int | TwoPatchEarlyWalkingLayout, seed=None):
        layout = (
            d
            if isinstance(d, TwoPatchEarlyWalkingLayout)
            else TwoPatchEarlyWalkingLayout(d)
        )
        super().__init__(layout, seed=seed)
        self.rounds_per_side = None
        self.rounds_before = None
        self.rounds_after = None
        self.total_rounds = None
        self.cx_round = None
        self._emitting_real_round = False
        self._cx_detectors_active = False
        self._cx_final_detectors_active = False

    def __repr__(self):
        return f"EarlyWalkingTransversalCXBuilder(d={self.layout.d})"

    def _validate_experiment_basis(self, basis: str):
        if basis != "X":
            raise ValueError(
                "EarlyWalkingTransversalCXBuilder implements the X-basis "
                "experiment only."
            )

    def get_circuit(
        self,
        rounds_per_side: int | None = None,
        p: float = 0.0,
        p_leak: float = 0.0,
        leakage_circuit_locations: dict | None = None,
        use_cache: bool = True,
        *,
        rounds_before: int | None = None,
        rounds_after: int | None = None,
        **circuit_kwargs,
    ):
        """Construct a symmetric or explicitly asymmetric tCNOT circuit."""

        schedule = resolve_tcnot_round_schedule(
            rounds_per_side,
            rounds_before=rounds_before,
            rounds_after=rounds_after,
        )
        basis = circuit_kwargs.pop("basis", "X")
        self._validate_experiment_basis(basis)

        self.rounds_per_side = (
            schedule.before if schedule.before == schedule.after else None
        )
        self.rounds_before = schedule.before
        self.rounds_after = schedule.after
        self.total_rounds = schedule.total
        self.cx_round = schedule.before
        self._cx_detectors_active = False
        self._cx_final_detectors_active = False
        if leakage_circuit_locations:
            for rnd, steps in leakage_circuit_locations.items():
                if self.TCX_STEP in steps and rnd != self.cx_round:
                    raise ValueError(
                        "TCX_STEP leakage locations must be stored at "
                        f"round {self.cx_round}, not round {rnd}."
                    )
                invalid_tcx_qubits = (
                    set(steps.get(self.TCX_STEP, set())) - self.qubit_set
                )
                if invalid_tcx_qubits:
                    raise ValueError(
                        "Invalid transversal leakage qubits: "
                        f"{sorted(invalid_tcx_qubits)}."
                    )
                ordinary_steps = {
                    step
                    for step, qubits in steps.items()
                    if step != self.TCX_STEP and qubits
                }
                if rnd < 0 or rnd > self.total_rounds:
                    raise ValueError(
                        f"Leakage round {rnd} is outside [0, {self.total_rounds}]."
                    )
                if rnd == self.total_rounds and ordinary_steps:
                    raise ValueError(
                        "Only TCX_STEP leakage is allowed at the final round "
                        f"boundary {self.total_rounds}."
                    )
        return super().get_circuit(
            self.total_rounds,
            p,
            p_leak=p_leak,
            leakage_circuit_locations=leakage_circuit_locations,
            use_cache=use_cache,
            basis=basis,
            **circuit_kwargs,
        )

    def measure_stabilizers(self, p, rnd: int, **kwargs):
        """Emit one SE round, applying post-tCX role and detector changes."""

        # ``measure_stabilizers_no_leakage`` calls this method re-entrantly with
        # an effective cache phase instead of the real round number.  Never let
        # that internal call emit a second transversal layer.
        if self._emitting_real_round:
            return super().measure_stabilizers(p, rnd, **kwargs)

        self._emitting_real_round = True
        self._cx_detectors_active = (
            rnd == self.cx_round and self.cx_round < self.total_rounds
        )
        try:
            if self._cx_detectors_active:
                # The CX-boundary detector has its own form and must never be
                # stored under an ordinary two-round cache phase.
                kwargs["use_cache"] = False

                # The periodic walking role-change flags compare against the
                # previous ordinary round.  At this boundary the previous
                # interaction is the transversal CX instead.
                phase = rnd % 2
                original = self.cnot_dicts[phase]
                self.cnot_dicts[phase] = self._post_tcx_cnot_dicts(original)
                try:
                    return super().measure_stabilizers(p, rnd, **kwargs)
                finally:
                    self.cnot_dicts[phase] = original

            return super().measure_stabilizers(p, rnd, **kwargs)
        finally:
            self._cx_detectors_active = False
            self._emitting_real_round = False

    def _before_round_boundary(
        self,
        p,
        boundary: int,
        total_rounds: int,
        *,
        Pauli_locations: str,
        basis: str,
        ec_sched: int,
        leak_effect: str,
        leakage_circuit_locations: dict | None = None,
    ):
        if boundary != self.cx_round:
            return
        leakage_circuit_locations = leakage_circuit_locations or {}
        self._emit_transversal_cx(
            p=p,
            Pauli_locations=Pauli_locations,
            leak_effect=leak_effect,
            leakage_qubits=leakage_circuit_locations.get(self.TCX_STEP, set()),
        )
        if boundary < total_rounds:
            # The following round resets this set before its first syndrome CX.
            self.leaked -= self.reset_sets[boundary % 2]
        else:
            self._cx_final_detectors_active = True

    def _observable_cache_key(self, rounds, basis):
        return (
            rounds,
            basis,
            self.rounds_before,
            self.rounds_after,
        )

    def _active_sublattice_at_cx(self) -> int:
        if self.cx_round is None:
            raise RuntimeError("get_circuit must set cx_round before emitting the CX.")
        return (1 + self.cx_round) % 2

    def _transversal_pairs(self) -> list[tuple[int, int]]:
        shift = self.layout.index_shift
        active = self.dq_indexes[self._active_sublattice_at_cx()]
        control = [index for index in active if index < shift]
        return [(index, index + shift) for index in control]

    def _previous_role(self, qubit: int) -> int | None:
        """Return the qubit's last ordinary-CX role before the transversal CX."""

        for rnd in range(self.cx_round - 1, -1, -1):
            for step in range(3, -1, -1):
                entry = self.cnot_dicts[rnd % 2][step].get(qubit)
                if entry is not None:
                    return entry[1] % 2
        return None

    def _transversal_cnot_dict(self) -> dict[int, tuple[int, int, bool | None]]:
        result = {}
        for pair_index, (control, target) in enumerate(self._transversal_pairs()):
            control_role = 0
            target_role = 1
            previous_control_role = self._previous_role(control)
            previous_target_role = self._previous_role(target)
            result[control] = (
                target,
                2 * pair_index,
                None
                if previous_control_role is None
                else previous_control_role != control_role,
            )
            result[target] = (
                control,
                2 * pair_index + 1,
                None
                if previous_target_role is None
                else previous_target_role != target_role,
            )
        return result

    def _post_tcx_cnot_dicts(self, ordinary_dicts):
        """Override each tCNOT qubit's first post-CX role transition."""

        # Values are immutable tuples, so copying the four dictionaries is
        # sufficient and avoids repeatedly cloning their integer contents.
        result = [dict(cnot_dict) for cnot_dict in ordinary_dicts]
        tcx_roles = {
            qubit: entry[1] % 2
            for qubit, entry in self._transversal_cnot_dict().items()
        }
        unresolved = set(tcx_roles)
        for step in range(4):
            for qubit in list(unresolved):
                entry = result[step].get(qubit)
                if entry is None:
                    continue
                partner, index, _ = entry
                result[step][qubit] = (
                    partner,
                    index,
                    tcx_roles[qubit] != index % 2,
                )
                unresolved.remove(qubit)
        return result

    def _register_new_leakages(self, leakage_qubits, cnot_dict, leak_effect):
        newly_leaked = set(leakage_qubits) - self.leaked
        self.leaked.update(newly_leaked)

        if newly_leaked and leak_effect in (
            "decode skip gates",
            "decode tailored then skip",
        ):
            for qubit in sorted(newly_leaked):
                self.circuit.append(f"DEPOLARIZE1(0.75) {qubit}")

        if newly_leaked and leak_effect in (
            "tailored then skip",
            "decode tailored then skip",
        ):
            for qubit in sorted(newly_leaked & cnot_dict.keys()):
                other_qubit, index, _ = cnot_dict[qubit]
                q0, q1 = (
                    (qubit, other_qubit)
                    if index % 2 == 0
                    else (other_qubit, qubit)
                )
                self.circuit.extend(
                    self.apply_leakage_noise_model(
                        "tailored",
                        q0,
                        q1,
                        q0 in self.leaked,
                        q1 in self.leaked,
                    )
                )

    def _emit_transversal_cx(
        self,
        p: float,
        Pauli_locations: str,
        leak_effect: str,
        leakage_qubits: set[int],
    ):
        pairs = self._transversal_pairs()
        # Match the ordinary walking schedule's readable three-column format.
        targets = " ".join(f"{index:3d}" for pair in pairs for index in pair)
        cnot_dict = self._transversal_cnot_dict()
        self._register_new_leakages(leakage_qubits, cnot_dict, leak_effect)
        active = set(cnot_dict)
        idle = (
            " ".join(str(index) for index in sorted(self.qubit_set - active))
            if Pauli_locations == "all"
            else ""
        )
        self.make_stabilizer_gates(
            p,
            cnot_dict,
            targets,
            idle,
            leak_effect,
        )

    def define_detectors(self, rnd: int, basis: str = "Z"):
        if self._cx_detectors_active:
            if self.cx_round == 0:
                key = (self._CX_DETECTOR_CACHE_TAG, "initial", basis)
            else:
                # Tracker offsets depend on how many rounds precede the tCNOT,
                # not just on the walking phase.
                key = (
                    self._CX_DETECTOR_CACHE_TAG,
                    "internal",
                    self.cx_round % 2,
                    self.rounds_before,
                )
        else:
            key = rnd
        self.circuit.append(self._get_detector_text(key, basis))

    def _get_detector_text(self, rnd: int | tuple, basis: str):
        if isinstance(rnd, tuple) and rnd[:1] == (self._CX_DETECTOR_CACHE_TAG,):
            if rnd[1] == "initial":
                return self._initial_detector_text_after_tcx(basis)
            _, _, phase, _ = rnd
            return self._cx_boundary_detector_text(phase)
        return super()._get_detector_text(rnd, basis)

    def _initial_detector_text_after_tcx(self, basis: str) -> str:
        """Return first-round detectors when the tCNOT precedes all SE rounds."""

        return WalkingSCCircuitBuilder._get_detector_text(self, 0, "X")

    def _cx_boundary_detector_text(self, phase: int) -> str:
        """Detector layer immediately after the transversal CX.

        An early-walking detector tuple contains exactly one qubit that was
        measured in the preceding round instead of the current round.  The
        affected control-X and target-Z detectors acquire the paired patch's
        copy of that one pre-CX record.
        """

        detector_tuples = self.layout.detectors[phase]
        detector_types = self.layout.detector_check_types[phase]
        measured_now = self.measure_sets[phase]
        shift = self.layout.index_shift
        lines = []

        for index_tuple, check_type in zip(detector_tuples, detector_types):
            rec_indexes = [
                self.tracker.get_curr_meas(index) for index in index_tuple
            ]
            patch = self._detector_patch(index_tuple)
            crosses_cx = (
                (patch == 1 and check_type == "X")
                or (patch == 2 and check_type == "Z")
            )
            if crosses_cx:
                old_terms = [
                    index for index in index_tuple if index not in measured_now
                ]
                if len(old_terms) != 1:
                    raise ValueError(
                        "Expected one pre-CX measurement term in early detector "
                        f"{index_tuple}, found {old_terms}."
                    )
                old_index = old_terms[0]
                paired_old_index = (
                    old_index + shift if patch == 1 else old_index - shift
                )
                # paired_old_index was also not measured in this round, so its
                # current record is precisely the last record before the CX.
                rec_indexes.append(
                    self.tracker.get_curr_meas(paired_old_index)
                )

            recs = " ".join(f"rec[{index}]" for index in rec_indexes)
            lines.append(f"DETECTOR {recs}")

        return "\n".join(lines)

    def _detector_patch(self, index_tuple: tuple[int, ...]) -> int:
        patches = {self.layout.patch_of(index) for index in index_tuple}
        if len(patches) != 1:
            raise ValueError(
                "A detector tuple must contain indexes from exactly one patch: "
                f"{index_tuple}."
            )
        return patches.pop()

    def _final_sublattice(self) -> int:
        if self.total_rounds is None:
            raise RuntimeError("get_circuit must set total_rounds before readout.")
        return (1 + self.total_rounds) % 2

    def _last_round_phase(self) -> int:
        if not self.total_rounds:
            raise RuntimeError("A tCNOT circuit must contain at least one SE round.")
        return (self.total_rounds - 1) % 2

    def _final_measurement_cache_key(self, p, Pauli_locations, basis):
        return (p, Pauli_locations, basis, self._final_sublattice())

    def final_measurement(
        self,
        p,
        Pauli_locations: str,
        basis: str = "Z",
        use_cache=True,
    ):
        sublattice = self._final_sublattice()
        return self._final_measurement_impl(
            p,
            set(self.dq_indexes[sublattice]),
            self.dq_indexes_str[sublattice],
            Pauli_locations,
            basis,
            use_cache=use_cache,
        )

    def final_detectors(self, basis: str):
        detector_text = self._get_final_detector_text(
            basis,
            self._last_round_phase(),
            self._cx_final_detectors_active,
            self.rounds_before,
            self.rounds_after,
        )
        self.circuit.append(detector_text)

    def _get_final_detector_text(
        self,
        basis: str,
        last_round_phase: int,
        crosses_final_tcx: bool,
        rounds_before: int,
        rounds_after: int,
    ) -> str:
        detector_list = self.layout.final_detectors_early_swap(last_round_phase)[
            0 if basis == "Z" else 1
        ]
        final_data = set(self.dq_indexes[self._final_sublattice()])
        lines = []
        for index_tuple in detector_list:
            rec_indexes = [
                self.tracker.get_curr_meas(index) for index in index_tuple
            ]
            patch = self._detector_patch(index_tuple)
            affected = crosses_final_tcx and (
                (patch == 1 and basis == "X")
                or (patch == 2 and basis == "Z")
            )
            if affected:
                old_terms = [
                    index for index in index_tuple if index not in final_data
                ]
                if len(old_terms) != 1:
                    raise ValueError(
                        "Expected one pre-tCNOT term in final early detector "
                        f"{index_tuple}, found {old_terms}."
                    )
                paired_old_index = self.layout.pair(old_terms[0]).index
                rec_indexes.append(
                    self.tracker.get_curr_meas(paired_old_index)
                )
            recs = " ".join(f"rec[{index}]" for index in rec_indexes)
            lines.append(f"DETECTOR {recs}")
        return "\n".join(lines)

    def final_observable(self, rounds: int, basis: str = "Z"):
        self.circuit.append(self._get_final_observable_text(rounds, basis))

    def _get_final_observable_text(self, rounds: int, basis: str):
        if basis != "X":
            raise ValueError("The transversal-CX observables are defined in X basis.")

        control = self._logical_measurements(patch=1, basis="X")
        target = self._logical_measurements(patch=2, basis="X")
        observable_0 = " ".join(f"rec[{index}]" for index in control + target)
        observable_1 = " ".join(f"rec[{index}]" for index in target)
        return (
            f"OBSERVABLE_INCLUDE(0) {observable_0}\n"
            f"OBSERVABLE_INCLUDE(1) {observable_1}"
        )

    def _logical_measurements(self, patch: int, basis: str) -> list[int]:
        last_only, all_measurements = self.layout.logical_support(
            patch,
            basis,
            self._final_sublattice(),
        )
        if all_measurements:
            raise ValueError(
                "Early walking unexpectedly requires cross-round logical "
                "measurements; the transversal observable rule must be revisited."
            )
        return [self.tracker.get_curr_meas(index) for index in last_only]

    def _x_logical_measurements(self, patch: int) -> list[int]:
        return self._logical_measurements(patch, "X")

    def _tcx_detection_check(self, qubit: int, ec_sched: int):
        """Return the first erasure-check slot reached by a tCNOT leakage."""

        total_rounds = self.total_rounds
        if (
            self.cx_round < total_rounds
            and qubit in self.reset_sets[self.cx_round % 2]
        ):
            return None
        for rnd in range(self.cx_round, total_rounds):
            for step in range(4):
                if step < 3 and (step + 1) % ec_sched == 0:
                    return rnd, step
                if step == 3:
                    if ec_sched != 8 or qubit in self.measure_sets[rnd % 2]:
                        return rnd, step

        final_data = set(self.dq_indexes[self._final_sublattice()])
        if qubit in final_data:
            return total_rounds - 1, 3
        return None

    def single_ec_traceback(
        self,
        ec_rnd: int,
        ec_gate: int,
        ec_qubit: int,
        ec_sched: int,
        leak_locations: str = "2-qubit gates",
    ):
        possible = super().single_ec_traceback(
            ec_rnd,
            ec_gate,
            ec_qubit,
            ec_sched,
            leak_locations,
        )

        if leak_locations == "2-qubit gates":
            eligible = ec_qubit in {
                qubit for pair in self._transversal_pairs() for qubit in pair
            }
        elif leak_locations == "all":
            eligible = ec_qubit in self.qubit_set
        else:
            raise ValueError(f"Invalid leak_locations value: {leak_locations}")

        if eligible and self._tcx_detection_check(ec_qubit, ec_sched) == (
            ec_rnd,
            ec_gate,
        ):
            possible.append([self.cx_round, self.TCX_STEP, ec_qubit])

        return sorted(possible, key=lambda loc: (loc[0], loc[1], loc[2]))

    def count_leakage_locations(
        self,
        rounds: int,
        leak_locations: str = "2-qubit gates",
    ):
        if rounds != self.total_rounds:
            raise ValueError(
                "tCNOT leakage counting expects the complete configured circuit."
            )
        if leak_locations == "2-qubit gates":
            per_step = [len(self.cnot_lists[0][step]) // 2 for step in range(4)]
            tcx_locations = len(self._transversal_pairs())
        elif leak_locations == "all":
            per_step = [self.n_qubits] * 4
            tcx_locations = self.n_qubits
        else:
            raise ValueError(f"Invalid leak_locations value: {leak_locations}")
        per_round = sum(per_step)
        return rounds * per_round + tcx_locations, per_round, per_step

    def ravel_leakage_locations(
        self,
        rounds: int,
        leak_locs_unraveled: list,
        leak_locations: str = "2-qubit gates",
    ):
        total, per_round, per_step = self.count_leakage_locations(
            rounds,
            leak_locations,
        )
        result = {
            rnd: {step: set() for step in range(4)}
            for rnd in range(rounds)
        }
        result.setdefault(self.cx_round, {})[self.TCX_STEP] = set()

        pre_count = self.cx_round * per_round
        transversal_pairs = self._transversal_pairs()
        if leak_locations == "2-qubit gates":
            tcx_count = len(transversal_pairs)
        else:
            tcx_count = self.n_qubits
        tcx_stop = pre_count + tcx_count
        qubits = sorted(self.qubit_set)

        for raw_location in leak_locs_unraveled:
            location = int(raw_location)
            if location < 0 or location >= total:
                raise ValueError(f"Leakage location {location} is outside [0, {total}).")

            if pre_count <= location < tcx_stop:
                local = location - pre_count
                if leak_locations == "2-qubit gates":
                    qubit = transversal_pairs[local][self.rng.choice(2)]
                else:
                    qubit = qubits[local]
                result[self.cx_round][self.TCX_STEP].add(qubit)
                continue

            ordinary = location if location < pre_count else location - tcx_count
            rnd, in_round = divmod(ordinary, per_round)
            cumulative = 0
            for step, count in enumerate(per_step):
                if in_round < cumulative + count:
                    local = in_round - cumulative
                    if leak_locations == "2-qubit gates":
                        pair = self.cnot_lists[rnd % 2][step][2 * local : 2 * local + 2]
                        qubit = pair[self.rng.choice(2)]
                    else:
                        qubit = qubits[local]
                    result[rnd][step].add(qubit)
                    break
                cumulative += count

        return result


class EarlyWalkingTransversalCXBellBuilder(EarlyWalkingTransversalCXBuilder):
    """Build a Bell-correlation transversal-CNOT experiment.

    Patch 1 is initialized in the logical X basis and acts as the control.
    Patch 2 is initialized in the logical Z basis and acts as the target.  The
    ideal transversal CX therefore prepares a logical Bell state stabilized by
    both ``X_L1 * X_L2`` and ``Z_L1 * Z_L2``.  One circuit reads out one of
    these correlations, selected by ``basis``, as observable 0.
    """

    def __repr__(self):
        return f"EarlyWalkingTransversalCXBellBuilder(d={self.layout.d})"

    def _validate_experiment_basis(self, basis: str):
        if basis not in {"X", "Z"}:
            raise ValueError(
                "EarlyWalkingTransversalCXBellBuilder requires X- or Z-basis "
                "final readout."
            )

    def initialize_data_qubits(self, p, *, Pauli_locations, basis):
        initial_data = self.dq_indexes[1]
        initial_data_text = self.dq_indexes_str[1]
        control_data_text = " ".join(
            str(index)
            for index in initial_data
            if self.layout.patch_of(index) == 1
        )

        self.circuit.append(f"R {initial_data_text}")
        if Pauli_locations in {"all", "gates"} and p > 0.0:
            self.circuit.append(f"X_ERROR({p}) {initial_data_text}")
        self.circuit.append(f"H {control_data_text}")

    def _get_detector_text(self, rnd: int | tuple, basis: str):
        if rnd == 0:
            return self._mixed_initial_detector_text()
        return super()._get_detector_text(rnd, basis)

    def _initial_detector_text_after_tcx(self, basis: str) -> str:
        """Return Bell-state initial detectors after a boundary-zero tCNOT."""

        detector_tuples = []
        for check_type in ("X", "Z"):
            basis_index = 1 if check_type == "X" else 0
            control_tuples = [
                index_tuple
                for index_tuple in self.layout.initial_detectors[basis_index]
                if self._detector_patch(index_tuple) == 1
            ]
            for control_tuple in control_tuples:
                target_tuple = tuple(
                    self.layout.pair(index).index for index in control_tuple
                )
                detector_tuples.append(control_tuple + target_tuple)

        self.detectors_in_round_0 = len(detector_tuples)
        return "\n".join(
            "DETECTOR "
            + " ".join(
                f"rec[{self.tracker.get_curr_meas(index)}]"
                for index in index_tuple
            )
            for index_tuple in detector_tuples
        )

    def _mixed_initial_detector_text(self) -> str:
        detector_tuples = []
        for index_tuple in self.layout.initial_detectors[1]:
            if self._detector_patch(index_tuple) == 1:
                detector_tuples.append(index_tuple)
        for index_tuple in self.layout.initial_detectors[0]:
            if self._detector_patch(index_tuple) == 2:
                detector_tuples.append(index_tuple)

        self.detectors_in_round_0 = len(detector_tuples)
        lines = []
        for index_tuple in detector_tuples:
            recs = " ".join(
                f"rec[{self.tracker.get_curr_meas(index)}]"
                for index in index_tuple
            )
            lines.append(f"DETECTOR {recs}")
        return "\n".join(lines)

    def _get_final_observable_text(self, rounds: int, basis: str):
        if basis not in {"X", "Z"}:
            raise ValueError(
                "The Bell correlation observable requires X- or Z-basis readout."
            )

        control = self._logical_measurements(patch=1, basis=basis)
        target = self._logical_measurements(patch=2, basis=basis)
        records = " ".join(
            f"rec[{index}]" for index in control + target
        )
        return f"OBSERVABLE_INCLUDE(0) {records}"


# Shorter alias matching the common tCNOT terminology.
EarlyWalkingTCNOTBuilder = EarlyWalkingTransversalCXBuilder
EarlyWalkingTCNOTBellBuilder = EarlyWalkingTransversalCXBellBuilder

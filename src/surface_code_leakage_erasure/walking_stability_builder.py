from functools import cache
# from line_profiler import profile
from .walking_stability_circuit import WalkingStabilityCircuitLayout
from .circuit_builder import CircuitBuilder

class WalkingStabilityCircuitBuilder(CircuitBuilder):
    ### this class builds the circuit which implements the surface code

    def __init__(self, d : int | WalkingStabilityCircuitLayout, swap_time: str = "late", seed = None):
        if isinstance(d, WalkingStabilityCircuitLayout):
            layout = d
            d = layout.d
            swap_time = layout.swap_time
        else:
            layout = WalkingStabilityCircuitLayout(d, swap_time)

        super().__init__(layout, seed)
        self.swap_time = swap_time

        # cache frequently used attributes
        # data qubits
        self.dq_indexes = [
            sorted([qubit.index for qubit in dq_list]) for dq_list in self.layout.data_qubits]
        self.dq_indexes_str = [
            " ".join(str(q) for q in dq_list) for dq_list in self.dq_indexes]

        # plaquettes (for defining cnots)
        z_plaquettes = [sorted(z_plaqs) for z_plaqs in self.layout.z_plaquettes]  # need to be sorted for consistent detector definitions between get_circuit and get_decoding_circuit!
        x_plaquettes = [sorted(x_plaqs) for x_plaqs in self.layout.x_plaquettes]
        all_plaquettes = [x_plaqs + z_plaqs for (x_plaqs, z_plaqs) in zip(x_plaquettes, z_plaquettes)]

        # qubits to be reset/measured at start of round
        reset_indexes = self.layout.reset_indexes #[[plaq.ancilla.index for plaq in plaq_list] for plaq_list in all_plaquettes]
        self.reset_sets = [set(reset_list) for reset_list in reset_indexes]
        self.reset_indexes_str = [" ".join(str(q) for q in reset_list) for reset_list in reset_indexes]
        self.rx_indexes_str = [" ".join(str(q) for q in rx_list) for rx_list in self.layout.rx_indexes]
        measure_indexes = self.layout.measure_indexes
        self.measure_sets = [set(measure_list) for measure_list in measure_indexes]
        self.mx_indexes_str = [" ".join(str(q) for q in mx_list) for mx_list in self.layout.mx_indexes]

        # compute cnot qubit lists and strings
        self.cnot_lists : list[list[list[int]]] = []
        self.cnot_dicts : list[list[dict[int, tuple]]] = []
        self.cnot_strs : list[list[str]] = []
        self.idling_qubits_str: list[list[str]] = []
        all_qubit_indexes = set(q.index for q in self.qubits)
        qubit_roles = {}
        first_gate_per_qubit = {}  # {qubit index: (round number, step number)} for first gate the qubit is involved in
        for (rnd, plaq_list) in enumerate(all_plaquettes):
            cnot_lists_round = []
            cnot_dicts_round = []
            cnot_str_round = []
            idling_str_round = []
            for t in range(4):
                list_index = 0  # index of qubit in list of CNOT qubits for this step
                cnot_qubits_list = []
                cnot_qubits_dict = {}
                reverse_flag = (swap_time=="late" and t==3) or (swap_time=="early" and t==0)
                for plaquette in plaq_list:
                    cnot_qubits = self.single_plaquette_cnot_indexes(
                        plaquette, t, reverse=reverse_flag)
                    cnot_qubits_list.extend(cnot_qubits)
                    for (role, qubit) in enumerate(cnot_qubits):
                        if qubit not in first_gate_per_qubit:
                            first_gate_per_qubit[qubit] = (rnd, t)

                        if qubit in qubit_roles:
                            prev_role = qubit_roles[qubit][-1]
                            qubit_roles[qubit].append(role)
                            role_change = role != prev_role
                        else:
                            qubit_roles[qubit] = [role]
                            role_change = None  # will fix this after by comparing first role to last role

                        cnot_qubits_dict[qubit] = (cnot_qubits[1-role], list_index, role_change)
                        list_index += 1

                cnot_lists_round.append(cnot_qubits_list)
                cnot_dicts_round.append(cnot_qubits_dict)
                cnot_str_round.append(' '.join(f"{q:3d}" for q in cnot_qubits_list))
                idling_qubits = all_qubit_indexes - set(cnot_qubits_list)
                idling_str_round.append(' '.join(str(q) for q in idling_qubits))

            self.cnot_lists.append(cnot_lists_round)
            self.cnot_dicts.append(cnot_dicts_round)
            self.cnot_strs.append(cnot_str_round)
            self.idling_qubits_str.append(idling_str_round)
        # figure out if role-change in first gate per even round for each qubit
        for qubit, (rnd, t) in first_gate_per_qubit.items():
            first_role = qubit_roles[qubit][0]
            last_role = qubit_roles[qubit][-1]
            self.cnot_dicts[rnd][t][qubit] = (
                self.cnot_dicts[rnd][t][qubit][0],
                self.cnot_dicts[rnd][t][qubit][1],
                last_role != first_role)

        # cache the detector text generation since it repeats every 2 rounds
        self._get_detector_text = cache(self._get_detector_text)
        self._get_final_detector_text = cache(self._get_final_detector_text)
        self._get_final_observable_text = cache(self._get_final_observable_text)

        self.count_leakage_locations = cache(self.count_leakage_locations)

    def __repr__(self):
        return f"WalkingStabilityCircuitBuilder(d={self.layout.d}, swap_time={self.swap_time})"

    def initialize_data_qubits(self, p, * , Pauli_locations, basis):
        circuit = self.circuit  # local reference
        dq_indexes_str = self.dq_indexes_str[0] if self.swap_time == "late" else self.dq_indexes_str[1]
        circuit.append(f"R {dq_indexes_str}")
        if Pauli_locations in ["all", "gates"] and p > 0.0:
            circuit.append(f"X_ERROR({p}) {dq_indexes_str}")
        if basis == "X":
            # apply Hadamards to data qubits to switch to X basis
            circuit.append(f"H {dq_indexes_str}")

    def measure_stabilizers(
        self, p, rnd : int, *, Pauli_locations : str, basis : str,
        ec_sched : int, leak_effect : str,
        leakage_circuit_locations : dict | None = None,
        use_cache = True):
        if leakage_circuit_locations is None:
            leakage_circuit_locations = {}

        # check if there is or will be leakage. if not, bounce to no-leakage version of measure_stabilizers which can be cached
        if (len(self.leaked) == 0 and not any(leakage_circuit_locations.values())
            and use_cache):
            rnd_eff = 0 if rnd == 0 else ((rnd-1) % 2 + 1)  # the circuit is the same every 2 rounds after the first round
            return super().measure_stabilizers_no_leakage(
                p, rnd_eff, Pauli_locations=Pauli_locations, basis=basis, ec_sched=ec_sched)

        qubit_kwargs = dict(
            reset_indexes_str=self.reset_indexes_str[rnd%2],
            dq_indexes_str=self.dq_indexes_str[rnd%2],
            rx_indexes_str=self.rx_indexes_str[rnd%2],
            idling_qubits_str=self.idling_qubits_str[rnd%2],
            cnot_qubits=self.cnot_dicts[rnd%2],
            cnot_str=self.cnot_strs[rnd%2],
            measure_set=self.measure_sets[rnd%2],
            mx_indexes_str=self.mx_indexes_str[rnd%2],
        )
        circuit_kwargs = dict(
            Pauli_locations=Pauli_locations, basis=basis, ec_sched=ec_sched, leak_effect=leak_effect,
            leakage_circuit_locations = leakage_circuit_locations,
            )

        erasure_checks = super()._measure_stabilizers_impl(p, rnd, **qubit_kwargs, **circuit_kwargs)

        return erasure_checks

    # @profile
    def define_detectors(self, rnd : int, basis : str = "Z"):
        detector_text = self._get_detector_text(rnd, basis)
        self.circuit.append(detector_text)

    def _get_detector_text(self, rnd : int, basis : str):
        recs_list = []
        if rnd == 0:
            qubit_tuple_list = self.layout.initial_detectors[0] if basis == "Z" else self.layout.initial_detectors[1]
            for index_tuple in qubit_tuple_list:
                rec_idxs = [self.tracker.get_curr_meas(idx) for idx in index_tuple]
                recs_list.append(" ".join(f"rec[{rec_idx}]" for rec_idx in rec_idxs))
            self.detectors_in_round_0 = len(recs_list)
        elif rnd <= 2:
            for index_tuple in self.layout.detectors[rnd%2]:
                rec_idxs = [self.tracker.get_curr_meas(idx) for idx in index_tuple]
                recs_list.append(" ".join(f"rec[{rec_idx}]" for rec_idx in rec_idxs))
            self.detectors_per_round = len(recs_list)
        else:
            return self._get_detector_text(rnd-2, basis) # detectors repeat every 2 rounds

        return '\n'.join(f"DETECTOR {recs}" for recs in recs_list)

    def final_measurement(self, p, Pauli_locations : str, basis : str = "Z",
                          use_cache = True):
        final_rnd = getattr(self, "final_round_parity", 1)
        if self.swap_time == "late":
            data_idx = 1 - final_rnd
        else:  # swap_time == "early"
            data_idx = final_rnd
        dq_set = set(self.dq_indexes[data_idx])
        dq_indexes_str = self.dq_indexes_str[data_idx]
        return super()._final_measurement_impl(
            p, dq_set, dq_indexes_str, Pauli_locations, basis, use_cache=use_cache)

    def final_detectors(self, basis):
        final_rnd = getattr(self, "final_round_parity", 1)
        detector_text = self._get_final_detector_text(basis, final_rnd)
        self.circuit.append(detector_text)

    def _get_final_detector_text(self, basis : str, final_rnd : int = 1):
        recs_list = []
        final_detectors = self.layout.get_final_detectors(final_rnd)
        detector_list = final_detectors[0] if basis == "Z" else final_detectors[1]
        for index_tuple in detector_list:
            rec_idxs = [self.tracker.get_curr_meas(idx) for idx in index_tuple]
            recs_list.append(" ".join(f"rec[{rec_idx}]" for rec_idx in rec_idxs))

        detector_text = '\n'.join(f"DETECTOR {recs}" for recs in recs_list)
        return detector_text

    def final_observable(self, rounds, basis : str = "Z"):
        observable_text = self._get_final_observable_text(rounds, basis)
        self.circuit.append(observable_text)

    def _get_final_observable_text(self, rounds, basis : str):
        idx_list = []
        final_rnd = (rounds - 1) % 2
        observable = self.layout.get_logicals(final_rnd)
        # first element is tuple of qubits for which we only need last element
        for qubit_idx in observable[0]:
            idx_list.append(self.tracker.get_curr_meas(qubit_idx))
        # second element is tuple of qubits for which we need all measurements
        for qubit_idx in observable[1]:
            idx_list.extend(self.tracker.get_all_meas(qubit_idx))

        recs = " ".join(f"rec[{idx}]" for idx in idx_list)
        return f"OBSERVABLE_INCLUDE(0) {recs}"

    def get_circuit(
        self, rounds : int, p : float, p_leak : float = 0.0,
        leakage_circuit_locations : dict | None = None, **circuit_kwargs):
        if leakage_circuit_locations is None:
            leakage_circuit_locations = {}
        self.final_round_parity = (rounds - 1) % 2
        # NOTE: deliberately not cache_clear()-ing _get_final_detector_text /
        # _get_final_observable_text here (unlike the pre-port version this was
        # adapted from). Both are @cache-wrapped on (basis, final_rnd) /
        # (rounds, basis) respectively -- args that already fully determine
        # their output -- so the cache is correct without clearing, and clearing
        # it every build defeats the protection this caching is meant to give:
        # CircuitBuilder.get_circuit()'s newer _track_measurements optimization
        # (see stability_circuit.py's final_observable for the fuller
        # explanation) stops replaying measurement lists into self.tracker on
        # repeat get_circuit() calls with the same (rounds, basis) on one
        # shared builder instance -- exactly what ErasureDecoder does while
        # decoding a shot. Recomputing these texts from a tracker left stale
        # by that on every build (as cache_clear() forced) read missing/wrong
        # measurement records; reusing the first build's (already correct,
        # fully-tracked) cached text avoids that, the same way
        # SurfaceCodeCircuitBuilder's final_observable caching already does.

        return super().get_circuit(
            rounds, p, p_leak,
            leakage_circuit_locations = leakage_circuit_locations,
            **circuit_kwargs)

    def single_ec_traceback(self, ec_rnd : int, ec_gate : int, ec_qubit : int, ec_sched : int, leak_locations : str = "2-qubit gates"):

        ec_qubit_prev = None if ec_qubit in self.reset_sets[ec_rnd%2] else ec_qubit
        possible_leak_locs = super()._single_ec_traceback_impl(
            ec_rnd, ec_gate, ec_qubit, ec_sched, leak_locations,
            self.cnot_lists, ec_qubit_prev)

        return possible_leak_locs

    def count_leakage_locations(self, rounds : int, leak_locations : str = "2-qubit gates"):
        if leak_locations == "2-qubit gates":
            leakage_locations_per_step = [
                len(self.cnot_lists[0][t])//2 for t in range(4)]  # total 2-qubit gates per step
        elif leak_locations == "all":
            leakage_locations_per_step = [self.n_qubits for _ in range(4)]  # all qubits per step
        else:
            raise ValueError(f"Invalid leak_locations value: {leak_locations}")
        leakage_locations_per_round = sum(leakage_locations_per_step)
        total_leakage_locations = rounds*leakage_locations_per_round
        return total_leakage_locations, leakage_locations_per_round, leakage_locations_per_step

    def ravel_leakage_locations(self, rounds : int, leak_locs_unraveled : list, leak_locations : str = "2-qubit gates"):
        return super()._ravel_leakage_locations_impl(
            rounds, leak_locs_unraveled, leak_locations,
            cnot_qubits=self.cnot_lists)

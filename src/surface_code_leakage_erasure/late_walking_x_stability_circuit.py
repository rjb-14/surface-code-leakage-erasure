"""Late-walking Z-check-parity stability with explicit measurement frames."""

from numbers import Integral

from .surface_code import Coord, DataQubit, Plaquette, Qubit
from .walking_stability_circuit import WalkingStabilityCircuitLayout
from .x_stability_geometry import reference_x_stability_geometry


def _xor_indices(indices):
    """Cancel repeated measurement records, rather than keeping one copy."""
    parity = set()
    for index in indices:
        parity.symmetric_difference_update((index,))
    return tuple(sorted(parity))


class LateWalkingXStabilityCircuitLayout(WalkingStabilityCircuitLayout):
    """Reference clipped geometry, syndrome nodes, and late frame closures.

    data_qubits[phase] is incoming data; data_qubits[1-phase] is outgoing.
    Regular check ancillas with a last leg become outgoing data. Their
    syndromes are measured on that incoming last leg instead of the ancilla.
    """

    def __init__(self, d: int):
        if isinstance(d, bool) or not isinstance(d, Integral) or d < 2 or d % 2:
            raise ValueError("Distance must be a positive even integer (d >= 2).")
        self.d = int(d)
        self.swap_time = "late"
        self.experiment = "z-check-parity"
        self.preparation_basis = self.readout_basis = "X"
        self.define_layout()

    def define_layout(self):
        data, checks = reference_x_stability_geometry(self.d)
        self.reference_data_coords = data
        self.reference_checks = checks
        self.phase_shifts = (Coord(0, 0), Coord(1, 1))
        self.migration_deltas = tuple(
            self.phase_shifts[1-phase] - self.phase_shifts[phase] for phase in range(2)
        )
        data_coords = [frozenset(q + s for q in data) for s in self.phase_shifts]
        check_coords = [frozenset(a + s for a in checks) for s in self.phase_shifts]
        data_union = set().union(*data_coords)
        self.coord_to_qubit = {
            coord: (DataQubit if coord in data_union else Qubit)(i, coord)
            for i, coord in enumerate(sorted(set().union(*data_coords, *check_coords)))
        }
        self.qubits = frozenset(self.coord_to_qubit.values())
        self.index_to_qubit = {q.index: q for q in self.qubits}
        self.data_qubits = [frozenset(self.coord_to_qubit[q] for q in coords)
                            for coords in data_coords]
        self.plaquettes = []
        self.regular_plaquettes = []
        self.removal_plaquettes = []
        self.x_plaquettes = []
        self.z_plaquettes = []
        self.trailing_nodes = []
        self.removal_nodes = []
        self.reset_indexes = []
        self.rx_indexes = []
        self.measure_indexes = []
        self.mx_indexes = []
        self.syndrome_measurements = []
        # (outgoing data index, frame Pauli, controlling measurement index)
        self.frame_terms = []
        self.coord_to_plaquette = {}

        for phase, shift in enumerate(self.phase_shifts):
            delta = self.migration_deltas[phase]
            incoming, outgoing = data_coords[phase], data_coords[1-phase]
            trailing = check_coords[phase] - outgoing
            extra_outgoing = outgoing - check_coords[phase]
            regular, removals, frames = [], [], []
            measurement_nodes = {}
            rx, mx = set(), set()
            for a, (check_type, _) in sorted(checks.items()):
                coord = a + shift
                directions = self.X_DELTAS if check_type == "X" else self.Z_DELTAS
                if phase:
                    directions = directions[::-1]
                legs = [self.coord_to_qubit[coord + step]
                        if coord + step in incoming else None for step in directions]
                ancilla = self.coord_to_qubit[coord]
                regular.append(Plaquette(check_type, legs, ancilla, coord))
                node = legs[3] if legs[3] is not None else ancilla
                measurement_nodes[ancilla.index] = node.index
                if check_type == "X":
                    rx.add(ancilla.index)
                    mx.add(node.index)
                if legs[3] is not None:
                    if coord not in outgoing:
                        raise ValueError("A migrating check ancilla must be outgoing data.")
                    frames.append((ancilla.index, "Z" if check_type == "X" else "X", node.index))
                elif coord not in trailing or check_type != "Z":
                    raise ValueError("A check without a last leg must be a trailing Z check.")
            removal_nodes = set()
            for v in sorted(extra_outgoing):
                ancilla = self.coord_to_qubit[v]
                source = self.coord_to_qubit[v - delta]
                # RZ(v), CX(source,v), MX(source) gives Z^t on v.
                removals.append(Plaquette("X", [None, None, None, source], ancilla, v))
                removal_nodes.add(source)
                mx.add(source.index)
                frames.append((ancilla.index, "Z", source.index))
            self.regular_plaquettes.append(frozenset(regular))
            self.removal_plaquettes.append(frozenset(removals))
            self.plaquettes.append(frozenset(regular + removals))
            self.x_plaquettes.append(frozenset([p for p in regular if p.type == "X"] + removals))
            self.z_plaquettes.append(frozenset(p for p in regular if p.type == "Z"))
            self.coord_to_plaquette.update((p.coord, p) for p in regular + removals)
            self.trailing_nodes.append(frozenset(self.coord_to_qubit[q] for q in trailing))
            self.removal_nodes.append(frozenset(removal_nodes))
            self.reset_indexes.append(sorted(self.coord_to_qubit[q].index
                                             for q in check_coords[phase] | extra_outgoing))
            self.rx_indexes.append(sorted(rx))
            self.measure_indexes.append(sorted(self.coord_to_qubit[q].index for q in incoming | trailing))
            self.mx_indexes.append(sorted(mx))
            self.syndrome_measurements.append(measurement_nodes)
            self.frame_terms.append(tuple(frames))
            if {q for q, _, _ in frames} != {self.coord_to_qubit[q].index for q in outgoing}:
                raise ValueError("Late transfer frames must cover every outgoing data qubit.")
            if set(measurement_nodes.values()) | {q.index for q in removal_nodes} != set(self.measure_indexes[-1]):
                raise ValueError("Every late measurement must have a syndrome/removal role.")

        self.leakage_plaquettes = frozenset().union(*self.removal_plaquettes)
        self.index_to_plaquette = {p.ancilla.index: p for p in self.coord_to_plaquette.values()}
        if set(self.measure_indexes[0]) & set(self.measure_indexes[1]):
            raise ValueError("Current and previous syndrome record sublattices must be disjoint.")
        self.initial_detectors = [[], [
            (self.syndrome_measurements[0][p.index],)
            for p in sorted(self.regular_plaquettes[0]) if p.type == "X"
        ]]
        regular_by_coord = [{p.coord: p for p in plaquettes} for plaquettes in self.regular_plaquettes]
        self.detectors = []
        for phase in range(2):
            relations = []
            for p in sorted(self.regular_plaquettes[phase]):
                previous = regular_by_coord[1-phase][p.coord + self.migration_deltas[phase]]
                support = {q.index for q in p.ordered_data_qubits if q is not None}
                records = [self.syndrome_measurements[phase][p.index],
                           self.syndrome_measurements[1-phase][previous.index]]
                records.extend(self._frame_records(1-phase, support, p.type))
                relations.append(_xor_indices(records))
            self.detectors.append(relations)

    def _frame_records(self, phase, support, check_type):
        return [record for q, pauli, record in self.frame_terms[phase]
                if q in support and pauli != check_type]

    def get_final_detectors(self, final_rnd=1):
        relations = []
        delta = self.migration_deltas[final_rnd]
        for p in sorted(self.regular_plaquettes[final_rnd]):
            if p.type != "X":
                continue
            support = {self.coord_to_qubit[q.coord + delta].index
                       for q in p.ordered_data_qubits if q is not None}
            records = [self.syndrome_measurements[final_rnd][p.index], *support]
            records.extend(self._frame_records(final_rnd, support, "X"))
            relations.append(_xor_indices(records))
        return [[], relations]

    def get_stability_support(self, final_rnd):
        return tuple(self.syndrome_measurements[final_rnd][p.index]
                     for p in sorted(self.z_plaquettes[final_rnd]))

    def get_logicals(self, final_rnd=1):
        raise NotImplementedError("This experiment uses get_stability_support(), not logical strings.")

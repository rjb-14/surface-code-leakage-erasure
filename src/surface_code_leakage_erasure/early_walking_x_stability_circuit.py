"""Clipped-corner moonwalking geometry for Z-check-parity stability.

The X in the class name denotes data preparation/readout, not the observable.
Regular checks and teleportation removal nodes are deliberately kept separate.
"""

from numbers import Integral

from .surface_code import Coord, DataQubit, Plaquette, Qubit
from .walking_stability_circuit import WalkingStabilityCircuitLayout


class EarlyWalkingXStabilityCircuitLayout(WalkingStabilityCircuitLayout):
    """Two translated copies of the reference stability check geometry."""

    def __init__(self, d: int):
        if isinstance(d, bool) or not isinstance(d, Integral) or d < 2 or d % 2:
            raise ValueError("Distance must be a positive even integer (d >= 2).")
        self.d = int(d)
        self.swap_time = "early"
        self.experiment = "z-check-parity"
        self.preparation_basis = self.readout_basis = "X"
        self.define_layout()

    def define_layout(self):
        d = self.d
        corners = set() if d == 2 else {
            Coord(1, 1), Coord(2*d-1, 1),
            Coord(1, 2*d-1), Coord(2*d-1, 2*d-1),
        }
        data = {
            Coord(x, y) for x in range(1, 2*d, 2) for y in range(1, 2*d, 2)
        } - corners
        checks = {}
        for x in range(0, 2*d+1, 2):
            for y in range(0, 2*d+1, 2):
                a = Coord(x, y)
                support = frozenset(a + delta for delta in self.Z_DELTAS) & data
                if d == 2:
                    if a not in {Coord(2, 0), Coord(0, 2), Coord(2, 2),
                                 Coord(4, 2), Coord(2, 4)}:
                        continue
                    check_type = "X" if a == Coord(2, 2) else "Z"
                else:
                    interior = 0 < x < 2*d and 0 < y < 2*d
                    odd = ((x-y)//2) % 2 == 1
                    if len(support) < 2 or (not interior and odd):
                        continue
                    check_type = "X" if interior and odd else "Z"
                checks[a] = (check_type, support)

        self.reference_data_coords = frozenset(data)
        self.reference_checks = checks
        shifts = (Coord(1, 1), Coord(0, 0))
        self.phase_shifts = shifts
        data_coords = [frozenset(q + s for q in data) for s in shifts]
        check_coords = [frozenset(a + s for a in checks) for s in shifts]
        all_coords = set().union(*data_coords, *check_coords)
        data_union = set().union(*data_coords)
        self.coord_to_qubit = {
            coord: (DataQubit if coord in data_union else Qubit)(i, coord)
            for i, coord in enumerate(sorted(all_coords))
        }
        self.qubits = frozenset(self.coord_to_qubit.values())
        self.index_to_qubit = {q.index: q for q in self.qubits}
        self.data_qubits = [
            frozenset(self.coord_to_qubit[q] for q in coords) for coords in data_coords
        ]
        self.plaquettes = []
        self.regular_plaquettes = []
        self.x_plaquettes = []
        self.z_plaquettes = []
        self.leading_nodes = []
        self.removal_nodes = []
        self.removal_plaquettes = []
        self.reset_indexes = []
        self.rx_indexes = []
        self.measure_indexes = []
        self.mx_indexes = []
        self.coord_to_plaquette = {}

        for phase, shift in enumerate(shifts):
            delta = shift - shifts[1-phase]
            leading = check_coords[phase] - data_coords[1-phase]
            removal = data_coords[1-phase] - check_coords[phase]
            regular = []
            rx = set()
            for a, (check_type, _) in sorted(checks.items()):
                coord = a + shift
                directions = self.X_DELTAS if check_type == "X" else self.Z_DELTAS
                if phase:
                    directions = directions[::-1]
                legs = [
                    self.coord_to_qubit[coord + step]
                    if coord + step in data_coords[phase] else None
                    for step in directions
                ]
                regular.append(Plaquette(
                    check_type, legs, self.coord_to_qubit[coord], coord,
                ))
                if check_type == "X" and legs[0] is not None:
                    rx.add(legs[0].index)
            removals = []
            for u in sorted(removal):
                outgoing = self.coord_to_qubit[u + delta]
                # RX(outgoing), CX(outgoing, incoming), MZ(incoming) teleports
                # the state with X^t byproduct. This is not a regular MX check.
                removals.append(Plaquette(
                    "X", [outgoing, None, None, None], self.coord_to_qubit[u], u,
                ))
                rx.add(outgoing.index)
            self.regular_plaquettes.append(frozenset(regular))
            self.removal_plaquettes.append(frozenset(removals))
            self.plaquettes.append(frozenset(regular + removals))
            self.x_plaquettes.append(frozenset(
                [p for p in regular if p.type == "X"] + removals
            ))
            self.z_plaquettes.append(frozenset(p for p in regular if p.type == "Z"))
            self.coord_to_plaquette.update((p.coord, p) for p in regular + removals)
            self.leading_nodes.append(frozenset(self.coord_to_qubit[q] for q in leading))
            self.removal_nodes.append(frozenset(self.coord_to_qubit[q] for q in removal))
            self.reset_indexes.append(sorted(
                self.coord_to_qubit[q].index for q in data_coords[phase] | leading
            ))
            self.rx_indexes.append(sorted(rx))
            self.measure_indexes.append(sorted(
                self.coord_to_qubit[q].index for q in check_coords[phase] | removal
            ))
            self.mx_indexes.append(sorted(p.ancilla.index for p in regular if p.type == "X"))

        self.leakage_plaquettes = frozenset().union(*self.removal_plaquettes)
        self.index_to_plaquette = {p.ancilla.index: p for p in self.coord_to_plaquette.values()}
        self.initial_detectors = [[], [
            (p.ancilla.index,) for p in sorted(self.regular_plaquettes[0]) if p.type == "X"
        ]]
        self.detectors = []
        for phase in range(2):
            delta = shifts[phase] - shifts[1-phase]
            previous_checks = {p.coord for p in self.regular_plaquettes[1-phase]}
            current_measurements = set(self.measure_indexes[phase])
            relations = []
            for p in sorted(self.regular_plaquettes[phase]):
                previous = self.coord_to_qubit[p.coord - delta]
                # get_curr_meas(previous) must still resolve to the preceding
                # round, not a measurement emitted on this round's sublattice.
                if previous.coord not in previous_checks or previous.index in current_measurements:
                    raise ValueError("Previous syndrome must be unmeasured in the current phase.")
                records = [p.ancilla.index, previous.index]
                if p.type == "Z":
                    # The teleportation X byproducts flip exactly these Z checks.
                    support = {q.coord for q in p.ordered_data_qubits if q is not None}
                    records.extend(
                        u.index for u in sorted(self.removal_nodes[phase])
                        if u.coord + delta in support
                    )
                relations.append(tuple(records))
            self.detectors.append(relations)

    def get_final_detectors(self, final_rnd=1):
        return [[], [
            (p.ancilla.index,) + tuple(q.index for q in p.ordered_data_qubits if q is not None)
            for p in sorted(self.regular_plaquettes[final_rnd]) if p.type == "X"
        ]]

    def get_stability_support(self, final_rnd):
        """Only the final regular Z checks, excluding teleportation records."""
        return tuple(p.ancilla.index for p in sorted(self.z_plaquettes[final_rnd]))

    def get_logicals(self, final_rnd=1):
        raise NotImplementedError("This experiment uses get_stability_support(), not logical strings.")

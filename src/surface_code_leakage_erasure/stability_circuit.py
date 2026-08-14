from typing import Union
import stim
from .surface_code import Coord, DataQubit, Qubit, SurfaceCodeLayout
from .circuit_builder import CircuitBuilder, SurfaceCodeCircuitBuilder


def rotated_css_ancilla_coords(d: int) -> list[Coord]:
    """Even-even ancilla sites kept by the same perimeter rule as `rotated_css_circuit.py`."""
    grid_max = 2 * d + 1
    data_coords: list[Coord] = []
    for x in range(grid_max):
        for y in range(grid_max):
            if x % 2 == 1 and y % 2 == 1:
                data_coords.append(Coord(x, y))
    data_set = set(data_coords)

    ancilla_coords: list[Coord] = []
    for x in range(grid_max):
        for y in range(grid_max):
            if x % 2 != 0 or y % 2 != 0:
                continue
            neighbors = 0
            for dx in (-1, 1):
                for dy in (-1, 1):
                    if Coord(x + dx, y + dy) in data_set:
                        neighbors += 1
            if neighbors < 2:
                continue
            keep = True
            if neighbors < 4:
                at_left_or_right = x == 0 or x == 2 * d
                at_bottom_or_top = y == 0 or y == 2 * d
                on_perimeter = at_left_or_right or at_bottom_or_top
                if on_perimeter and (x + y) % 4 != 2:
                    keep = False
            if keep:
                ancilla_coords.append(Coord(x, y))
    ancilla_coords.sort()
    return ancilla_coords


class StabilityLayout(SurfaceCodeLayout):
    def define_layout(self):
        d = self.d
        ancilla_coord_list = rotated_css_ancilla_coords(d)

        # Data
        data_qubits: list[DataQubit] = []
        dq_coords: dict[Coord, DataQubit] = {}
        qubit_index = 0
        for i in range(d):
            for j in range(d):
                coord = Coord(1, 1) + Coord(2 * i, 2 * j)
                qubit = DataQubit(qubit_index, coord)
                data_qubits.append(qubit)
                dq_coords[coord] = qubit
                qubit_index += 1

        # Ancilla
        ancillas: list[Qubit] = []
        aq_coords: dict[Coord, Qubit] = {}
        n_data = len(data_qubits)
        for k, coord in enumerate(ancilla_coord_list):
            q = Qubit(n_data + k, coord)
            ancillas.append(q)
            aq_coords[coord] = q

        self.data_qubits = frozenset(data_qubits)
        self.qubits = frozenset(data_qubits + ancillas)
        self.coord_to_qubit = {**dq_coords, **aq_coords}
        self.index_to_qubit = {q.index: q for q in self.qubits}

        z_plaqs, x_plaqs, coord_to_plaq, index_to_plaq = self.layout_plaquettes()
        self.plaquettes = frozenset(z_plaqs + x_plaqs)
        self.z_plaquettes = frozenset(z_plaqs)
        self.x_plaquettes = frozenset(x_plaqs)
        self.coord_to_plaquette = coord_to_plaq
        self.index_to_plaquette = index_to_plaq

        z_logical, x_logical = self.get_logicals()
        self.z_logical = frozenset(z_logical)
        self.x_logical = frozenset(x_logical)


class StabilityCircuitBuilder(SurfaceCodeCircuitBuilder):
    """Static stability circuit builder.

    Inherits rotated-surface-code syndrome extraction from `SurfaceCodeCircuitBuilder`.
    For ``ec_sched=8``, each round applies erasure swap before the sparse end-of-round
    erasure check: leaked data qubits are bookkeeping-mapped onto ancilla partners
    (see ``erasure_swaps`` / ``erasure_unswaps`` set up in ``SurfaceCodeCircuitBuilder``).
    """

    def __init__(self, d: Union[int, StabilityLayout], seed=None):
        layout = d if isinstance(d, StabilityLayout) else StabilityLayout(d)
        super().__init__(layout, seed=seed)

    def _apply_erasure_swaps(self):
        """Map leaked data-qubit indices onto ancilla partners (ec_sched=8 only)."""
        if len(self.leaked) > 0:
            leaked_str = " ".join(str(q) for q in sorted(self.leaked))
            self.circuit.append(f"DEPOLARIZE1(0.75) {leaked_str}")
        self.leaked = {self.erasure_swaps[q] for q in self.leaked}

    def measure_stabilizers(
        self, p, rnd: int, *, Pauli_locations: str, basis: str,
        ec_sched: int, leak_effect: str,
        leakage_circuit_locations: dict | None = None,
        use_cache: bool = True,
    ):
        if leakage_circuit_locations is None:
            leakage_circuit_locations = {}

        if (len(self.leaked) == 0 and not any(leakage_circuit_locations.values())
                and use_cache):
            rnd_eff = 0 if rnd == 0 else 1
            return self.measure_stabilizers_no_leakage(
                p, rnd_eff, Pauli_locations=Pauli_locations, basis=basis, ec_sched=ec_sched)

        qubit_kwargs = dict(
            reset_indexes_str=self.x_indexes_str + " " + self.z_indexes_str,
            dq_indexes_str=self.dq_indexes_str,
            rx_indexes_str=self.x_indexes_str,
            idling_qubits_str=self.idling_qubits_str,
            cnot_qubits=self.cnot_dicts,
            cnot_str=self.cnot_strs,
            measure_set=self.stab_ancillas_set | set(self.erasure_ancillas),
            mx_indexes_str=self.x_indexes_str,
        )
        circuit_kwargs = dict(
            Pauli_locations=Pauli_locations, basis=basis, ec_sched=ec_sched,
            leak_effect=leak_effect, leakage_circuit_locations=leakage_circuit_locations,
        )

        if ec_sched == 8:
            self._apply_erasure_swaps()

        return CircuitBuilder._measure_stabilizers_impl(
            self, p, rnd, **qubit_kwargs, **circuit_kwargs)

    def single_ec_traceback(
        self, ec_rnd: int, ec_gate: int, ec_qubit: int, ec_sched: int,
        leak_locations: str = "2-qubit gates",
    ):
        return CircuitBuilder._single_ec_traceback_impl(
            self, ec_rnd, ec_gate, ec_qubit, ec_sched, leak_locations,
            [self.cnot_lists] * 2, self.erasure_unswaps.get(ec_qubit, None))

    def final_observable(self, rounds=0, basis: str = "Z"):
        # Cached like SurfaceCodeCircuitBuilder._get_final_observable_text (see
        # SurfaceCodeCircuitBuilder.__init__), rather than reading self.tracker
        # directly on every build: _observable_cache_key is inherited from
        # SurfaceCodeCircuitBuilder as (basis,), so get_circuit() stops
        # replaying measurement lists into the tracker on repeat builds with
        # the same (rounds, basis) once the tracker has been populated once
        # (e.g. across the repeated get_circuit() calls ErasureDecoder makes
        # on one shared builder instance while decoding a shot). A direct,
        # uncached tracker read here would then see a stale/incomplete
        # tracker on those later builds and emit a wrong rec[] reference in
        # OBSERVABLE_INCLUDE, which stim reports as a non-deterministic
        # observable. Routing through this cache -- populated once from a
        # build where the tracker is guaranteed complete -- avoids that.
        self.circuit.append(self._get_final_observable_text(basis))

    def _get_final_observable_text(self, basis: str):
        # Product of X-type stabilizer measurements from the last syndrome round (MRX block).
        idx_list = [self.tracker.get_curr_meas(plaq.index) for plaq in self.x_plaquettes]
        idx_list.sort()
        recs = " ".join(f"rec[{idx}]" for idx in idx_list)
        return f"OBSERVABLE_INCLUDE(0) {recs}" if recs else ""


def build_stability_circuit(
    distance: int,
    rounds: int,
    p: float = 0.0,
    p_leak: float = 0.0,
    **circuit_kwargs,
) -> tuple[stim.Circuit, list[list[set[int]]]]:
    """Stability: rotated-css ancilla topology + coords/indexing + syndrome extraction."""
    builder = StabilityCircuitBuilder(distance)
    return builder.get_circuit(rounds=rounds, p=p, p_leak=p_leak, **circuit_kwargs)


if __name__ == "__main__":
    circuit, _ = build_stability_circuit(
        distance=4,
        rounds=5,
        p=0.0,
        p_leak=0.0,
        basis="Z",
        Pauli_locations="all",
        leak_locations="2-qubit gates",
        ec_sched=8,
        leak_effect="depolarize",
    )
    print(circuit)

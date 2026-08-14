from .surface_code import Coord, Qubit, DataQubit, Plaquette, SurfaceCodeLayout

class WalkingStabilityCircuitLayout(SurfaceCodeLayout):
    def __init__(self, d: int, swap_time="late"):
        if d%2 != 0:
            raise ValueError("Distance must be even")
        if swap_time not in ["late", "early"]:
            raise ValueError("Swap time must be either 'late' or 'early'")
        self.swap_time = swap_time
        self.d = d
        self.define_layout(swap_time)

    def define_layout(self, swap_time):
        dq1, dq1_coords = self.layout_data_qubits(qubit_index=0, top_left=Coord(1,1))
        dq2, dq2_coords = self.layout_data_qubits(qubit_index=len(dq1), top_left=Coord(2,2))
        ancillas, aq_coords = self.layout_edge_ancillas()

        if swap_time == "late":
            self.data_qubits = [frozenset(dq1), frozenset(dq2)]
        else:
            self.data_qubits = [frozenset(dq2), frozenset(dq1)]
        self.qubits = frozenset(dq1 + dq2 + ancillas)

        self.coord_to_qubit = {**dq1_coords, **dq2_coords, **aq_coords}
        self.index_to_qubit = {q.index: q for q in self.qubits}

        ## define plaquettes including gate orders
        self.layout_plaquettes()

        ## define logical observable for basis="Z"
        logical = self.get_logicals()
        self.z_logical = logical
        self.x_logical = logical

        self.define_detectors()

    def layout_edge_ancillas(self):
        d = self.d
        coord_list = []

        top = Coord(2, 0)
        bottom = Coord(3, 2*d+1)
        left = Coord(0, 2)
        right = Coord(2*d+1, 3)
        starts = [top, bottom, left, right]
        for i in range(4):
            start = starts[i]
            if start == top or start == bottom:
                direction = Coord(4,0)
            else:
                direction = Coord(0,4)
            ran = d//2
            for j in range(ran):
                coord_list.append(start + direction*j)

        coord_list.sort()
        qubits = [Qubit(2*d**2 +i, coord) for i, coord in enumerate(coord_list)]
        coord_to_qubit ={coord: qubits[i] for i, coord in enumerate(coord_list)}
        return qubits, coord_to_qubit

    def layout_plaquettes(self):
         # initialize empty lists
        self.plaquettes = [] # list of sets, one per round, of all plaquettes

        self.bulk_plaquettes = []
        self.initial_plaquettes = []
        self.leading_plaquettes = []
        self.trailing_plaquettes = []

        self.z_plaquettes = []
        self.x_plaquettes = []

        self.reset_indexes = [] # list of lists, of qubits to be initialized at the start of round
        self.rx_indexes = [] # list of lists, of qubits to be initialized in + state
        self.measure_indexes = [] # list of lists, of qubits which are measured at end of round
        self.mx_indexes = [] # list of lists, of qubits tob be measured in X basis

        self.layout_bulk_plaquettes()
        self.layout_edge_plaquettes()
        if self.swap_time == "late":
            self.define_resets_measurements_late_swap()
        else:  # swap_time == "early"
            self.define_resets_measurements_early_swap()



    def layout_bulk_plaquettes(self):
        ## lay out plaquettes in the bulk
        d = self.d
        coord_to_plaquette: dict[Coord, Plaquette] = {}
        index_to_plaquette: dict[int, Plaquette] = {}

        z_deltas = self.Z_DELTAS  #[Coord(1,1), Coord(-1,1), Coord(1,-1), Coord(-1,-1)]
        x_deltas = self.X_DELTAS  #[Coord(1,1), Coord(1,-1), Coord(-1,1), Coord(-1,-1)]

        top_left_list = [Coord(2,2), Coord(3,3)]
        if self.swap_time == "early":
            top_left_list = top_left_list[::-1]

        for (rnd, top_left) in enumerate(top_left_list):
            z_plaquettes: list[Plaquette] = []
            x_plaquettes: list[Plaquette] = []
            for i in range(d-1):
                for j in range(d-1):
                    coord = top_left + Coord(j*2, i*2)
                    s_type = 'Z' if ((i+j)%2)==0 else 'X'
                    ancilla = self.coord_to_qubit[coord]
                    deltas = z_deltas if s_type == 'Z' else x_deltas
                    ordered_data_qubits = [self.coord_to_qubit.get(coord+delta) for delta in deltas]
                    plaquette = Plaquette(s_type, ordered_data_qubits, ancilla, coord)
                    for (t, dq) in enumerate(ordered_data_qubits):
                        if dq is not None:
                            dq.ordered_plaquettes[t] = plaquette
                    coord_to_plaquette[coord] = plaquette
                    index_to_plaquette[ancilla.index] = plaquette
                    if s_type == 'Z':
                        z_plaquettes.append(plaquette)
                    else:
                        x_plaquettes.append(plaquette)

            self.plaquettes.append(set(z_plaquettes + x_plaquettes))
            self.bulk_plaquettes.append(frozenset(z_plaquettes + x_plaquettes))
            self.z_plaquettes.append(set(z_plaquettes))
            self.x_plaquettes.append(set(x_plaquettes))

            # reverse gate order for next round
            z_deltas = z_deltas[::-1]
            x_deltas = x_deltas[::-1]

        self.coord_to_plaquette = coord_to_plaquette
        self.index_to_plaquette = index_to_plaquette


    def layout_edge_plaquettes(self):
        ## lay out plaquettes on the edges
        d = self.d
        swap_time = self.swap_time

        coord_to_plaquette: dict[Coord, Plaquette] = {}
        index_to_plaquette: dict[int, Plaquette] = {}

        z_deltas = self.Z_DELTAS
        x_deltas = self.X_DELTAS

        leading_plaquettes: list[list[Plaquette]] = [[],[]]
        trailing_plaquettes: list[list[Plaquette]] = [[],[]]

        ## regular edge plaquettes
        start_left_or_top = [Coord(0,2), Coord(2,0)]
        start_right_or_bottom = [Coord(2*d,2), Coord(2,2*d)]
        steps = [Coord(0,4), Coord(4,0)]

        for (stype, start1, start2, step) in zip(['X','X'], start_left_or_top, start_right_or_bottom, steps):
            for rnd in range(2):
                for start in [start1, start2]:
                    if swap_time == "early":
                        start += Coord(1,1)
                    deltas = x_deltas
                    plaq_list = self._one_edge_plaquettes(stype, rnd, start, step, deltas)

                    if start.isinside(Coord(1,1), Coord(d*2,d*2)) and swap_time == "late":
                        leading_plaquettes[rnd].extend(plaq_list)
                    elif not start.isinside(Coord(1,1), Coord(d*2,d*2)) and swap_time == "early":
                        leading_plaquettes[rnd].extend(plaq_list)
                    else:
                        trailing_plaquettes[rnd].extend(plaq_list)

                # update for second round
                if swap_time == "late":
                    start1 += Coord(1,1)
                    start2 += Coord(1,1)
                else:  # swap_time == "early"
                    start1 -= Coord(1,1)
                    start2 -= Coord(1,1)
                x_deltas = x_deltas[::-1]
                z_deltas = z_deltas[::-1]

         # initial plaquettes exclude special leakage-removal plaquettes
        self.initial_plaquettes = [sorted(self.z_plaquettes[0]), sorted(self.x_plaquettes[0])]

        ## special leakage-removal plaquettes
        leakage_plaquettes = []
        # if swap_time == "late":
        start_zs = [Coord(2*d, 4), Coord(1,1)]
        start_zs_1 = [Coord(4, 2*d), Coord(5,1)]
        delta = Coord(-1,-1)  # delta to data qubit leakage is removed from
        t = 3  # what time "swap" happens at
        if swap_time == "early":
            start_zs = start_zs[::-1]
            start_zs_1 = start_zs_1[::-1]
            t = 0
            delta = -1*delta

        for (rnd, start_z, start_z_1) in zip([0,1], start_zs, start_zs_1):
            seen_coords = set() #to avoid double counting plaquettes

            for (stype, start, step) in zip(['Z','Z'], [start_z, start_z_1], steps):
                plaq_list = []
                for i in range(d//2):
                    coord = start + step*i
                    if not coord.isinside(Coord(1,1), Coord(d*2,d*2)):
                        break

                    if coord in seen_coords:
                        continue
                    seen_coords.add(coord)

                    ancilla = self.coord_to_qubit[coord]
                    ordered_data_qubits = [None, None, None, None]
                    ordered_data_qubits[t] = self.coord_to_qubit.get(coord+delta)
                    plaquette = Plaquette(stype, ordered_data_qubits, ancilla, coord)
                    ordered_data_qubits[t].ordered_plaquettes[t] = plaquette
                    plaq_list.append(plaquette)

                    coord_to_plaquette[coord] = plaquette
                    index_to_plaquette[ancilla.index] = plaquette

                self.plaquettes[rnd] |= set(plaq_list)
                self.z_plaquettes[rnd] |= set(plaq_list)


                leakage_plaquettes.extend(plaq_list)
                if swap_time == "late":
                    leading_plaquettes[rnd].extend(plaq_list)
                # else:  # swap_time == "early"
                #     trailing_plaquettes[rnd].extend(plaq_list)      ######################

            delta = -1*delta

        self.trailing_plaquettes = [frozenset(plaq_list) for plaq_list in trailing_plaquettes]
        self.leading_plaquettes = [frozenset(plaq_list) for plaq_list in leading_plaquettes]
        self.leakage_plaquettes = frozenset(leakage_plaquettes)
        self.coord_to_plaquette.update(coord_to_plaquette)
        self.index_to_plaquette.update(index_to_plaquette)



    def _one_edge_plaquettes(self, stype, rnd, start, step, deltas):
        plaq_list = []

        for i in range(self.d//2):
            coord = start + step*i
            ancilla = self.coord_to_qubit[coord]

            ordered_data_qubits = []
            for delta in deltas:
                dq = self.coord_to_qubit.get(coord+delta)
                if dq in self.data_qubits[rnd]:
                    ordered_data_qubits.append(dq)
                else:
                    ordered_data_qubits.append(None)
            plaquette = Plaquette(stype, ordered_data_qubits, ancilla, coord)
            for (t, dq) in enumerate(ordered_data_qubits):
                if dq is not None:
                    dq.ordered_plaquettes[t] = plaquette

            plaq_list.append(plaquette)

        self.coord_to_plaquette.update({plaquette.coord: plaquette for plaquette in plaq_list})
        self.index_to_plaquette.update({plaquette.index: plaquette for plaquette in plaq_list})

        self.plaquettes[rnd] |= set(plaq_list)
        if stype == 'Z':
            self.z_plaquettes[rnd] |= set(plaq_list)
        else:
            self.x_plaquettes[rnd] |= set(plaq_list)

        return plaq_list


    def define_resets_measurements_late_swap(self):
        """ Define which qubits are reset and measured in each round for the late walking surface code """
        for rnd in range(2):
            reset_list = []
            rx_list = []
            measure_list = []
            mx_list = []

            for plaquette in self.plaquettes[rnd]:

                reset_list.append(plaquette.ancilla.index)
                if plaquette.type == 'X' and plaquette not in self.leakage_plaquettes:
                    # leakage plaquettes have opposite initialization from expected
                    rx_list.append(plaquette.ancilla.index)
                elif plaquette.type == 'Z' and plaquette in self.leakage_plaquettes:
                    # leakage plaquettes have opposite initialization from expected
                    rx_list.append(plaquette.ancilla.index)

                last_dq = plaquette.ordered_data_qubits[-1]
                if last_dq is not None:
                    measure_list.append(last_dq.index)
                    if plaquette.type == 'X':
                        mx_list.append(last_dq.index)

            for plaquette in self.trailing_plaquettes[rnd]:
                measure_list.append(plaquette.ancilla.index)
                if plaquette.type == 'X':
                    mx_list.append(plaquette.ancilla.index)

            self.reset_indexes.append(sorted(reset_list))
            self.rx_indexes.append(sorted(rx_list))
            self.measure_indexes.append(sorted(measure_list))
            self.mx_indexes.append(sorted(mx_list))


    def define_resets_measurements_early_swap(self):
        """ Define which qubits are reset and measured in each round for the early walking surface code """
        for rnd in range(2):
            reset_list = []
            rx_list = []
            measure_list = []
            mx_list = []

            for plaquette in self.plaquettes[rnd]:

                first_dq = plaquette.ordered_data_qubits[0]
                if first_dq is not None:
                    reset_list.append(first_dq.index)
                    if plaquette.type == 'X':
                        rx_list.append(first_dq.index)

                measure_list.append(plaquette.ancilla.index)
                if plaquette.type == 'X' and plaquette not in self.leakage_plaquettes:
                    mx_list.append(plaquette.ancilla.index)
                elif plaquette.type == 'Z' and plaquette in self.leakage_plaquettes:
                    mx_list.append(plaquette.ancilla.index)

            for plaquette in self.leading_plaquettes[rnd]:
                reset_list.append(plaquette.ancilla.index)
                if plaquette.type == 'X':
                    rx_list.append(plaquette.ancilla.index)


            self.reset_indexes.append(sorted(reset_list))
            self.rx_indexes.append(sorted(rx_list))
            self.measure_indexes.append(sorted(measure_list))
            self.mx_indexes.append(sorted(mx_list))


    # ---------------------------------------------------------------------
    # Detector definitions for WalkingStabilityCircuitLayout
    # ---------------------------------------------------------------------

    def _idx_at(self, coord: Coord) -> int:
        """Return qubit index at coord, with a useful error message."""
        q = self.coord_to_qubit.get(coord)
        if q is None:
            raise ValueError(f"No qubit at coord {coord}")
        return q.index

    def _maybe_idx_at(self, coord: Coord):
        """Return qubit index at coord, or None if no qubit exists there."""
        q = self.coord_to_qubit.get(coord)
        return None if q is None else q.index

    def _append_unique_detector(self, det_list: list[tuple[int, ...]], qids):
        """Append a detector tuple after removing None and duplicates."""
        qids = tuple(q for q in qids if q is not None)
        if len(qids) == 0:
            return
        # preserve order, remove duplicates
        qids = tuple(dict.fromkeys(qids))
        if len(qids) == 0:
            return
        det_list.append(qids)

    def define_detectors(self):
        """
        Layout-level detector definition.

        self.initial_detectors[type] is used for rnd == 0.
        self.detectors[rnd % 2] is used for later syndrome rounds.
        self.final_detectors[type] is used after final data measurement.

        type index:
            0 -> Z-basis memory / Z-type initial/final detectors
            1 -> X-basis memory / X-type initial/final detectors
        """
        self.detectors = [[], []]

        self.define_bulk_detectors()

        if self.swap_time == "late":
            self.initial_detectors = self.initial_detectors_late_swap()
            self.edge_detectors_late_swap()
            self.final_detectors = self.final_detectors_late_swap()
        else:
            self.initial_detectors = self.initial_detectors_early_swap()
            self.edge_detectors_early_swap()
            self.final_detectors = self.final_detectors_early_swap()

    def get_final_detectors(self, final_rnd=1):
        if self.swap_time == "late":
            return self.final_detectors_late_swap(final_rnd)
        return self.final_detectors_early_swap(final_rnd)

    # ---------------------------------------------------------------------
    # Initial detectors
    # ---------------------------------------------------------------------

    def initial_detectors_late_swap(self):
        """
        Late swap initial detectors.

        In late swap, the first syndrome round measures qubits from the
        initial data patch. For each initial plaquette, the local initial
        detector is the qubit at plaquette.coord + (-1,-1), if it exists.
        If not, fall back to the plaquette ancilla itself.

        This is still valid for the walking-stability layout because the
        bulk plaquette geometry is unchanged; only the edge types changed.
        """
        initial_detectors = []

        for plaq_list in self.initial_plaquettes:
            detector_qubits = []
            for plaquette in plaq_list:
                coord = plaquette.coord
                q = self.coord_to_qubit.get(coord + Coord(-1, -1))
                if q is None:
                    q = plaquette.ancilla
                detector_qubits.append((q.index,))
            initial_detectors.append(detector_qubits)

        return initial_detectors

    def initial_detectors_early_swap(self):
        """
        Early swap initial detectors for the walking-stability layout.

        Important difference from the original walking surface code:
        do NOT use the old almond/double-detector boundary rules.

        For the modified stability circuit, the local deterministic
        initial detectors are simply the ancilla measurement of the
        corresponding initial plaquette.

        For basis='Z', builder uses initial_detectors[0].
        For basis='X', builder uses initial_detectors[1].
        """
        initial_detectors_z = [
            (plaquette.ancilla.index,)
            for plaquette in self.initial_plaquettes[0]
        ]

        initial_detectors_x = [
            (plaquette.ancilla.index,)
            for plaquette in self.initial_plaquettes[1]
        ]

        return [initial_detectors_z, initial_detectors_x]

    # ---------------------------------------------------------------------
    # Bulk detectors
    # ---------------------------------------------------------------------

    def define_bulk_detectors(self):
        """
        Bulk detectors between neighboring walking rounds.

        Late swap:
            detector starts from plaquette.ordered_data_qubits[0].

        Early swap:
            detector starts from plaquette.ancilla.
            Some early-swap bulk detectors have one extra qubit near
            special rows/columns, handled by extra_bulk_detector_qubit().

        The detector list is stored in self.detectors[1-rnd] because the
        walking detector initialized in one round is closed/read out in the
        opposite round parity.
        """
        swap_time = self.swap_time

        step = Coord(1, 1)
        if swap_time == "early":
            step = -1 * step

        for rnd in range(2):
            detector_qubit_tuples = []

            for plaq in sorted(self.bulk_plaquettes[rnd]):
                qubit_list = []

                if swap_time == "late":
                    q0 = plaq.ordered_data_qubits[0]
                else:
                    q0 = plaq.ancilla

                if q0 is None:
                    continue

                qubit_list.append(q0.index)

                other_coord = q0.coord + step
                q1 = self.coord_to_qubit.get(other_coord)
                if q1 is None:
                    raise ValueError(
                        f"Bulk detector cannot find q1 at {other_coord}. "
                        f"rnd={rnd}, plaq={plaq}, q0={q0}"
                    )

                qubit_list.append(q1.index)

                if swap_time == "early":
                    extra_qubit = self.extra_bulk_detector_qubit(
                        rnd, plaq, other_coord
                    )
                    if extra_qubit is not None:
                        qubit_list.append(extra_qubit)

                detector_qubit_tuples.append(tuple(dict.fromkeys(qubit_list)))

            self.detectors[1 - rnd].extend(detector_qubit_tuples)
            step = -1 * step

    def extra_bulk_detector_qubit(self, rnd, plaq, other_coord):

        if self.swap_time != "early":
            return None

    # In the stability circuit, only X-type bulk detectors get these extra
    # closure qubits. Do not add Z-type extras.
        if plaq.type != "X":
            return None

        d = self.d
        sign = 1 if (rnd % 2 == 0) else -1

        x_delta = sign * Coord(2, 0)
        y_delta = sign * Coord(0, 2)

        low_boundary = 3
        high_boundary = 2 * (d - 1)

        candidates = []

    # Vertical-boundary closure: same as original X branch.
    # Example d=4: other_coord.x == 6 gives extra at other_coord + (2,0).
        if other_coord[0] in (low_boundary, high_boundary):
            candidates.append(other_coord + x_delta)

    # Horizontal-boundary closure: this was the original Z-branch geometry,
    # but in the stability layout it also belongs to X-type detectors.
    # Example d=4: other_coord=(4,6) gives extra=(4,8).
        if other_coord[1] in (low_boundary, high_boundary):
            candidates.append(other_coord + y_delta)

        for coord in candidates:
            q = self.coord_to_qubit.get(coord)
            if q is not None:
                return q.index

        return None
    # ---------------------------------------------------------------------
    # Edge detectors
    # ---------------------------------------------------------------------

    def edge_detectors_late_swap(self):
        """
        Late-swap edge detectors.

        There are two local edge-detector families:

        1. A trailing-edge plaquette gives a single-qubit detector on its ancilla
           in the same round parity.

        2. The same trailing-edge structure closes in the opposite round parity
           as a three-qubit detector along a diagonal line:
               coord, coord + step, coord + 2*step
        """
        # 1. single-qubit trailing-edge detectors
        for rnd in range(2):
            detector_qubit_tuples = []
            for plaquette in sorted(self.trailing_plaquettes[rnd]):
                detector_qubit_tuples.append((plaquette.ancilla.index,))
            self.detectors[rnd].extend(detector_qubit_tuples)

        # 2. three-qubit diagonal edge detectors
        step = Coord(1, 1)
        for rnd in range(2):
            detector_qubit_tuples = []

            for plaquette in sorted(self.trailing_plaquettes[rnd]):
                coord = plaquette.coord
                qids = [
                    self._maybe_idx_at(coord + i * step)
                    for i in range(3)
                ]
                self._append_unique_detector(detector_qubit_tuples, qids)

            self.detectors[1 - rnd].extend(detector_qubit_tuples)
            step = -1 * step

    def edge_detectors_early_swap(self):
        """
        Early-swap edge detectors.

        There are two local edge-detector families:

        1. Leading-edge plaquettes close as an almond-like detector consisting
           of the plaquette ancilla plus its existing data legs. This detector
           belongs to the opposite round parity.

        2. Trailing-edge plaquettes form a two-qubit detector between the
           trailing ancilla and the next plaquette ancilla attached through
           ordered_plaquettes[-1].
        """
        # 1. leading-edge almond-like detectors
        for rnd in range(2):
            detector_qubit_tuples = []

            for plaquette in sorted(self.leading_plaquettes[rnd]):
                qids = [plaquette.ancilla.index]
                qids.extend(
                    q.index for q in plaquette.ordered_data_qubits
                    if q is not None
                )
                self._append_unique_detector(detector_qubit_tuples, qids)

            self.detectors[1 - rnd].extend(detector_qubit_tuples)

        # 2. trailing-edge two-qubit detectors
        for rnd in range(2):
            detector_qubit_tuples = []

            for plaquette in sorted(self.trailing_plaquettes[rnd]):
                q0 = plaquette.ancilla

                # In the walking schedule, q0 should be connected to a plaquette
                # through its last ordered slot. If not, skip instead of crashing.
                linked_plaq = q0.ordered_plaquettes[-1]
                if linked_plaq is None:
                    continue

                q1_idx = linked_plaq.ancilla.index
                detector_qubit_tuples.append((q0.index, q1_idx))

            self.detectors[1 - rnd].extend(detector_qubit_tuples)

    # ---------------------------------------------------------------------
    # Final detectors
    # ---------------------------------------------------------------------

    def final_detectors_late_swap(self, final_rnd=1):
        """
        Final detectors for late swap.

        This follows the original walking logic, but uses the modified
        walking-stability plaquette sets. It is local and type-aware.
        """
        final_detectors_z = []
        final_detectors_x = []

        # Bulk final detectors.
        # For each round-1 bulk plaquette, use the final measured data qubits
        # around coord + (-1,-1).
        steps = self.Z_DELTAS
        final_step = Coord(-1, -1) if final_rnd == 1 else Coord(1, 1)
        for plaq in sorted(self.bulk_plaquettes[final_rnd]):
            qlist = []
            coord = plaq.coord + final_step

            q0 = self.coord_to_qubit.get(coord)
            if q0 is None:
                continue

            qlist.append(q0.index)
            for step in steps:
                q = self.coord_to_qubit.get(coord + step)
                if q is not None:
                    qlist.append(q.index)

            if plaq.type == "Z":
                final_detectors_z.append(tuple(dict.fromkeys(qlist)))
            else:
                final_detectors_x.append(tuple(dict.fromkeys(qlist)))

        # Leading/trailing edge overlap from initial plaquettes.
        plaq_list = list(
            set(plaq for plaq_list in self.initial_plaquettes for plaq in plaq_list)
            & set(self.trailing_plaquettes[1 - final_rnd])
        )
        for plaq in sorted(plaq_list):
            qlist = [
                q.index for q in plaq.ordered_data_qubits
                if q is not None
            ]
            if len(qlist) == 0:
                continue

            if plaq.type == "Z":
                final_detectors_z.append(tuple(dict.fromkeys(qlist)))
            else:
                final_detectors_x.append(tuple(dict.fromkeys(qlist)))

        # Trailing edge final detectors.
        step = Coord(-1, -1)
        for plaq0 in sorted(self.trailing_plaquettes[final_rnd]):
            coord = plaq0.coord
            q0 = self.coord_to_qubit.get(coord + step)
            if q0 is None:
                continue

            qlist = [q0.index]
            for plaq1 in q0.ordered_plaquettes:
                if plaq1 is None:
                    continue
                qlist.append(plaq1.ancilla.index)

            if plaq0.type == "Z":
                final_detectors_z.append(tuple(dict.fromkeys(qlist)))
            else:
                final_detectors_x.append(tuple(dict.fromkeys(qlist)))

        return [final_detectors_z, final_detectors_x]

    def final_detectors_early_swap(self, final_rnd=1):
        """
        Final detectors for early swap.

        In early swap, the final local detector for a non-leakage plaquette is
        the product of its existing data legs and its ancilla measurement.
        Leakage plaquettes are omitted here, matching the original walking code.
        """
        final_detectors_z = []
        final_detectors_x = []

        for plaq in sorted(self.plaquettes[final_rnd] - self.leakage_plaquettes):
            qlist = [
                q.index for q in plaq.ordered_data_qubits
                if q is not None
            ]
            qlist.append(plaq.ancilla.index)
            qlist = tuple(dict.fromkeys(qlist))

            if plaq.type == "Z":
                final_detectors_z.append(qlist)
            else:
                final_detectors_x.append(qlist)

        return [final_detectors_z, final_detectors_x]

    def get_logicals(self, final_rnd=1):
        """
        Return the walking-builder observable format:
            [qubits with only their latest measurement, qubits with all measurements]

        Only the basis='Z' stability observable is defined here. It is the
        product of the final X-basis plaquette measurement outcomes.
        """
        if self.swap_time == "late":
            logical_last = self.mx_indexes[final_rnd]
        else:
            logical_last = [
                plaq.ancilla.index
                for plaq in sorted(self.x_plaquettes[final_rnd])
                if plaq not in self.leakage_plaquettes
            ]

        return [tuple(logical_last), tuple()]

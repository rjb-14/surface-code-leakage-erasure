"""Two-patch layouts used by transversal logical gates.

The existing walking layout constructors assume that a patch starts at fixed
coordinates.  Instead of threading an offset through those constructors, this
module translates a fully constructed patch and merges the translated copy
back into the original layout.
"""

from .surface_code import Coord, DataQubit, Plaquette, Qubit
from .walking_surface_code import WalkingSurfaceCodeLayout


_SCALAR_ATTRIBUTES = frozenset({"d", "swap_time"})
_LOGICAL_ATTRIBUTES = frozenset({"x_logical", "z_logical"})
_TRANSLATED_ATTRIBUTES = frozenset(
    {
        "data_qubits",
        "qubits",
        "coord_to_qubit",
        "index_to_qubit",
        "plaquettes",
        "bulk_plaquettes",
        "initial_plaquettes",
        "leading_plaquettes",
        "trailing_plaquettes",
        "z_plaquettes",
        "x_plaquettes",
        "reset_indexes",
        "rx_indexes",
        "measure_indexes",
        "mx_indexes",
        "coord_to_plaquette",
        "index_to_plaquette",
        "leakage_plaquettes",
        "z_logical",
        "x_logical",
        "detectors",
        "initial_detectors",
        "final_detectors",
    }
)


def _collect_instances(value, kind, result):
    if isinstance(value, kind):
        result.add(value)
        return
    if isinstance(value, Coord):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_instances(key, kind, result)
            _collect_instances(item, kind, result)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_instances(item, kind, result)


def _layout_attributes(layout):
    unknown = set(vars(layout)) - _SCALAR_ATTRIBUTES - _TRANSLATED_ATTRIBUTES
    if unknown:
        raise TypeError(
            "Two-patch translation has no declared policy for layout attributes: "
            f"{sorted(unknown)}."
        )
    return {
        name: value
        for name, value in vars(layout).items()
        if name in _TRANSLATED_ATTRIBUTES
    }


def _translated_object_maps(layout, coord_shift, index_shift):
    qubits = set()
    plaquettes = set()
    for value in _layout_attributes(layout).values():
        _collect_instances(value, Qubit, qubits)
        _collect_instances(value, Plaquette, plaquettes)

    qubit_map = {
        qubit: (DataQubit if isinstance(qubit, DataQubit) else Qubit)(
            qubit.index + index_shift,
            qubit.coord + coord_shift,
        )
        for qubit in qubits
    }

    plaquette_map = {}
    for plaquette in plaquettes:
        translated_data = [
            None if qubit is None else qubit_map[qubit]
            for qubit in plaquette.ordered_data_qubits
        ]
        plaquette_map[plaquette] = Plaquette(
            plaquette.type,
            translated_data,
            qubit_map[plaquette.ancilla],
            plaquette.coord + coord_shift,
        )

    # Rebuild the reverse DataQubit -> Plaquette references.  Reusing the
    # original references would couple the supposedly independent patches.
    for plaquette in plaquettes:
        translated_plaquette = plaquette_map[plaquette]
        for step, qubit in enumerate(plaquette.ordered_data_qubits):
            if isinstance(qubit, DataQubit):
                translated_qubit = qubit_map[qubit]
                existing = translated_qubit.ordered_plaquettes[step]
                if existing is not None and existing is not translated_plaquette:
                    raise ValueError(
                        "Multiple plaquettes occupy translated data-qubit slot "
                        f"{translated_qubit.index}:{step}."
                    )
                translated_qubit.ordered_plaquettes[step] = translated_plaquette

    return qubit_map, plaquette_map


def _translate(value, qubit_map, plaquette_map, coord_shift, index_shift):
    # Order is significant: Coord subclasses tuple and bool subclasses int.
    if isinstance(value, Plaquette):
        return plaquette_map[value]
    if isinstance(value, Qubit):
        return qubit_map[value]
    if isinstance(value, Coord):
        return value + coord_shift
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value + index_shift

    args = (qubit_map, plaquette_map, coord_shift, index_shift)
    if isinstance(value, dict):
        return {
            _translate(key, *args): _translate(item, *args)
            for key, item in value.items()
        }
    if isinstance(value, frozenset):
        return frozenset(_translate(item, *args) for item in value)
    if isinstance(value, set):
        return {_translate(item, *args) for item in value}
    if isinstance(value, tuple):
        return tuple(_translate(item, *args) for item in value)
    if isinstance(value, list):
        return [_translate(item, *args) for item in value]
    raise TypeError(f"Cannot translate layout value of type {type(value).__name__}.")


def _merge_leaf(left, right):
    if isinstance(left, (set, frozenset)):
        merged = left | right
        if len(merged) != len(left) + len(right):
            raise ValueError("Translated layout sets overlap during merge.")
        return merged
    if isinstance(left, dict):
        overlap = left.keys() & right.keys()
        if overlap:
            raise ValueError(
                "Translated layout dictionaries have overlapping keys: "
                f"{sorted(overlap, key=repr)}."
            )
        return {**left, **right}
    if isinstance(left, (list, tuple)):
        return left + right
    raise TypeError(f"Cannot merge layout value of type {type(left).__name__}.")


def _merge_attribute(left, right):
    # Layout lists group corresponding categories (usually the two-round phase,
    # and for initial/final detectors the logical basis).  Merge matching
    # categories instead of concatenating the outer list.
    if isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("Per-round layout attributes have different lengths.")
        return [_merge_leaf(a, b) for a, b in zip(left, right)]
    return _merge_leaf(left, right)


def shift_and_merge(layout, coord_shift: Coord, index_shift: int):
    """Add a translated, independent copy of ``layout`` to itself in place."""

    qubit_map, plaquette_map = _translated_object_maps(
        layout, coord_shift, index_shift
    )
    original_indexes = {qubit.index for qubit in qubit_map}
    translated_indexes = {qubit.index for qubit in qubit_map.values()}
    if original_indexes & translated_indexes:
        raise ValueError("Translated qubit indexes overlap the original patch.")
    original_coords = {qubit.coord for qubit in qubit_map}
    translated_coords = {qubit.coord for qubit in qubit_map.values()}
    if original_coords & translated_coords:
        raise ValueError("Translated qubit coordinates overlap the original patch.")

    original_plaquette_coords = {plaquette.coord for plaquette in plaquette_map}
    translated_plaquette_coords = {
        plaquette.coord for plaquette in plaquette_map.values()
    }
    if original_plaquette_coords & translated_plaquette_coords:
        raise ValueError("Translated plaquette coordinates overlap the original patch.")

    originals = _layout_attributes(layout)
    translated = {
        name: _translate(
            value, qubit_map, plaquette_map, coord_shift, index_shift
        )
        for name, value in originals.items()
    }

    for name, value in originals.items():
        if name in _LOGICAL_ATTRIBUTES:
            setattr(layout, f"{name}_per_patch", [value, translated[name]])
        else:
            setattr(layout, name, _merge_attribute(value, translated[name]))

    layout.coord_shift = coord_shift
    layout.index_shift = index_shift


def early_walking_detector_check_types(layout) -> list[list[str]]:
    """Return the X/Z type corresponding to each early detector tuple.

    ``WalkingSurfaceCodeLayout.detectors`` stores only measurement-index
    tuples.  Replaying the fixed append order used by
    ``define_bulk_detectors`` and ``edge_detectors_early_swap`` recovers the
    source plaquette, and therefore the check type.
    """

    if layout.swap_time != "early":
        raise ValueError(
            "Detector type reconstruction is only defined for early walking."
        )

    result = []
    for phase in range(2):
        source_phase = 1 - phase
        types = (
            [p.type for p in sorted(layout.bulk_plaquettes[source_phase])]
            + [p.type for p in sorted(layout.leading_plaquettes[source_phase])]
            + [p.type for p in sorted(layout.trailing_plaquettes[source_phase])]
        )
        if len(types) != len(layout.detectors[phase]):
            raise ValueError(
                f"Could not align detector types in phase {phase}: "
                f"found {len(types)} types for {len(layout.detectors[phase])} detectors."
            )
        result.append(types)
    return result


class TwoPatchEarlyWalkingLayout(WalkingSurfaceCodeLayout):
    """Two translated copies of the ``swap_time='early'`` walking layout."""

    def __init__(self, d):
        super().__init__(d, swap_time="early")

    def define_layout(self, swap_time):
        super().define_layout(swap_time)
        single_patch_detector_types = early_walking_detector_check_types(self)

        shift_and_merge(
            self,
            coord_shift=Coord(2 * self.d + 2, 0),
            index_shift=len(self.qubits),
        )
        self.detector_check_types = [
            types + types for types in single_patch_detector_types
        ]

    def pair(self, qubit_or_index):
        """Return the corresponding qubit in the other patch."""

        index = (
            qubit_or_index.index
            if isinstance(qubit_or_index, Qubit)
            else int(qubit_or_index)
        )
        if index < 0 or index >= 2 * self.index_shift:
            raise ValueError(f"Qubit index {index} is outside the two-patch layout.")
        paired_index = (
            index + self.index_shift
            if index < self.index_shift
            else index - self.index_shift
        )
        return self.index_to_qubit[paired_index]

    def patch_of(self, qubit_or_index) -> int:
        """Return 1 for the control patch and 2 for the target patch."""

        index = (
            qubit_or_index.index
            if isinstance(qubit_or_index, Qubit)
            else int(qubit_or_index)
        )
        if index < 0 or index >= 2 * self.index_shift:
            raise ValueError(f"Qubit index {index} is outside the two-patch layout.")
        return 1 if index < self.index_shift else 2

    def logical_support(
        self,
        patch: int,
        basis: str,
        sublattice: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return one patch's logical support on an active data sublattice."""

        if patch not in (1, 2):
            raise ValueError("patch must be 1 or 2.")
        if basis == "X":
            logicals = self.x_logical_per_patch
        elif basis == "Z":
            logicals = self.z_logical_per_patch
        else:
            raise ValueError("basis must be 'X' or 'Z'.")
        if sublattice not in (0, 1):
            raise ValueError("sublattice must be 0 or 1.")

        support = logicals[patch - 1]
        if sublattice == 1:
            return tuple(tuple(indexes) for indexes in support)

        translated = []
        for indexes in support:
            translated_indexes = []
            for index in indexes:
                source = self.index_to_qubit[index]
                target = self.coord_to_qubit[source.coord + Coord(1, 1)]
                if self.patch_of(target) != patch:
                    raise ValueError(
                        "Logical-support translation crossed a patch boundary."
                    )
                translated_indexes.append(target.index)
            translated.append(tuple(translated_indexes))
        return tuple(translated)


# Backwards-compatible terminology used in the earlier design notes.
TwoPatchMoonwalkingLayout = TwoPatchEarlyWalkingLayout

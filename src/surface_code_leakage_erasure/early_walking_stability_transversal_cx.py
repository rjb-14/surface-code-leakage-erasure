"""Transversal CNOT between two early-walking stability-circuit patches."""

from .early_walking_transversal_cx import _EarlyWalkingTransversalCXMixin
from .two_patch_layout import TwoPatchEarlyWalkingStabilityLayout
from .walking_stability_builder import WalkingStabilityCircuitBuilder


class EarlyWalkingStabilityTransversalCXBuilder(
    _EarlyWalkingTransversalCXMixin,
    WalkingStabilityCircuitBuilder,
):
    """Build a Z-basis early-walking stability tCNOT experiment.

    Patch 1 is the control and patch 2 is the target.  The selected observables
    are patch-local products of the final round's X-check measurements, not
    propagated surface-code logical operators.
    """

    _layout_type = TwoPatchEarlyWalkingStabilityLayout
    _OBSERVABLE_PATCHES = {
        "control": (1,),
        "target": (2,),
        "both": (1, 2),
    }

    def __init__(
        self,
        d: int | TwoPatchEarlyWalkingStabilityLayout,
        seed=None,
    ):
        self.observable_selection = "both"
        super().__init__(d, seed=seed)

    def __repr__(self):
        return f"EarlyWalkingStabilityTransversalCXBuilder(d={self.layout.d})"

    @classmethod
    def _validate_observable_selection(cls, selection: str) -> str:
        if not isinstance(selection, str) or selection not in cls._OBSERVABLE_PATCHES:
            raise ValueError(
                "observable_selection must be 'control', 'target', or 'both'."
            )
        return selection

    def validate_circuit_kwargs(self, circuit_kwargs, decode=False):
        """Validate the builder-specific observable selector and shared kwargs."""

        circuit_kwargs = dict(circuit_kwargs)
        selection_was_provided = "observable_selection" in circuit_kwargs
        observable_selection = self._validate_observable_selection(
            circuit_kwargs.pop("observable_selection", "both")
        )
        validated = super().validate_circuit_kwargs(
            circuit_kwargs,
            decode=decode,
        )
        if selection_was_provided:
            validated["observable_selection"] = observable_selection
        return validated

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
        observable_selection: str = "both",
        **circuit_kwargs,
    ):
        """Construct the selected one- or two-observable stability tCNOT."""

        self.observable_selection = self._validate_observable_selection(
            observable_selection
        )
        circuit_kwargs.setdefault("basis", "Z")
        return super().get_circuit(
            rounds_per_side=rounds_per_side,
            p=p,
            p_leak=p_leak,
            leakage_circuit_locations=leakage_circuit_locations,
            use_cache=use_cache,
            rounds_before=rounds_before,
            rounds_after=rounds_after,
            **circuit_kwargs,
        )

    def _validate_experiment_basis(self, basis: str):
        if basis != "Z":
            raise ValueError(
                "EarlyWalkingStabilityTransversalCXBuilder implements the "
                "Z-basis stability experiment only."
            )

    def _observable_cache_key(self, rounds, basis):
        return (
            rounds,
            basis,
            self.rounds_before,
            self.rounds_after,
            self.observable_selection,
        )

    def _initial_detector_text_after_tcx(self, basis: str) -> str:
        """Use the ordinary local Z detectors after a boundary-zero tCNOT."""

        return WalkingStabilityCircuitBuilder._get_detector_text(
            self,
            0,
            basis,
        )

    def final_observable(self, rounds: int, basis: str = "Z"):
        self.circuit.append(
            self._get_final_observable_text(
                rounds,
                basis,
                self.observable_selection,
            )
        )

    def _get_final_observable_text(
        self,
        rounds: int,
        basis: str,
        observable_selection: str,
    ) -> str:
        if basis != "Z":
            raise ValueError(
                "The early-walking stability tCNOT observables are defined "
                "in Z-basis experiments only."
            )

        observable_patches = self._OBSERVABLE_PATCHES[
            self._validate_observable_selection(observable_selection)
        ]
        phase = self._last_round_phase()
        lines = []
        for observable_id, patch in enumerate(observable_patches):
            x_checks = [
                plaquette
                for plaquette in sorted(self.layout.x_plaquettes[phase])
                if plaquette not in self.layout.leakage_plaquettes
                and self.layout.patch_of(plaquette.ancilla) == patch
            ]
            records = " ".join(
                f"rec[{self.tracker.get_curr_meas(plaquette.ancilla.index)}]"
                for plaquette in x_checks
            )
            lines.append(f"OBSERVABLE_INCLUDE({observable_id}) {records}")
        return "\n".join(lines)


EarlyWalkingStabilityTCNOTBuilder = EarlyWalkingStabilityTransversalCXBuilder

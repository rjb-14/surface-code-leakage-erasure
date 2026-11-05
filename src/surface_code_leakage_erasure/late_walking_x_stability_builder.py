"""Late walking with X data boundaries and final Z-check parity."""

from .late_walking_x_stability_circuit import LateWalkingXStabilityCircuitLayout
from .walking_x_stability_builder import _WalkingXStabilityCircuitBuilder


class LateWalkingXStabilityCircuitBuilder(_WalkingXStabilityCircuitBuilder):
    """Build clipped-corner Z stability with one leakage trial per nominal CX.

    A successful trial chooses one endpoint uniformly. Physical circuits skip
    gates touching leaked qubits; decoding completes topology from erasure DEMs.
    """

    def __init__(self, d: int | LateWalkingXStabilityCircuitLayout, seed=None):
        layout = (
            d if isinstance(d, LateWalkingXStabilityCircuitLayout)
            else LateWalkingXStabilityCircuitLayout(d)
        )
        super().__init__(layout, seed=seed)

    def __repr__(self):
        return f"LateWalkingXStabilityCircuitBuilder(d={self.layout.d})"

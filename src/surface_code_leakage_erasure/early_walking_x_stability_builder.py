"""Early walking with X data boundaries and final Z-check parity."""

from .early_walking_x_stability_circuit import EarlyWalkingXStabilityCircuitLayout
from .walking_x_stability_builder import _WalkingXStabilityCircuitBuilder


class EarlyWalkingXStabilityCircuitBuilder(_WalkingXStabilityCircuitBuilder):
    """Build clipped-corner Z stability with one leakage trial per nominal CX.

    A successful trial chooses one endpoint uniformly. Physical circuits skip
    gates touching leaked qubits; decoding completes topology from erasure DEMs.
    """

    def __init__(self, d: int | EarlyWalkingXStabilityCircuitLayout, seed=None):
        layout = (
            d if isinstance(d, EarlyWalkingXStabilityCircuitLayout)
            else EarlyWalkingXStabilityCircuitLayout(d)
        )
        super().__init__(layout, seed=seed)

    def __repr__(self):
        return f"EarlyWalkingXStabilityCircuitBuilder(d={self.layout.d})"

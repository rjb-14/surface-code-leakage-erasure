"""Guard: the transversal-CX work must not change any existing builder's output.

The SHA-256 digests below were captured from the pre-change code. If one of
these fails, an existing file's behaviour was modified -- which the plan's
global constraints forbid.
"""
import hashlib

import pytest

from surface_code_leakage_erasure import (
    SurfaceCodeCircuitBuilder,
    WalkingSCCircuitBuilder,
)


def _digest(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    "make_builder, basis, expected",
    [
        (lambda: SurfaceCodeCircuitBuilder(3, seed=0), "Z", "d76cb5742b2c962a"),
        (lambda: SurfaceCodeCircuitBuilder(3, seed=0), "X", "85d0f9ba19df660a"),
        (lambda: WalkingSCCircuitBuilder(3, "late", seed=0), "X", "ed68fb4d86e8594c"),
        (lambda: WalkingSCCircuitBuilder(3, "early", seed=0), "X", "61db578ce50e5282"),
    ],
)
def test_existing_builders_unchanged(make_builder, basis, expected):
    circuit, _ = make_builder().get_circuit(4, 1e-3, 0.0, basis=basis)
    assert _digest(circuit) == expected


@pytest.mark.parametrize(
    "make_builder",
    [
        lambda: SurfaceCodeCircuitBuilder(3, seed=0),
        lambda: WalkingSCCircuitBuilder(3, "late", seed=0),
        lambda: WalkingSCCircuitBuilder(3, "early", seed=0),
    ],
)
def test_existing_builders_single_observable(make_builder):
    circuit, _ = make_builder().get_circuit(4, 1e-3, 0.0, basis="X")
    assert circuit.num_observables == 1

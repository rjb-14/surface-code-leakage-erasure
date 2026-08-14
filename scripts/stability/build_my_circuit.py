"""Build a standard rotated surface-code circuit.

Usage example:
    python build_my_circuit.py --distance 5 --rounds 5 --p 0.001 --basis Z
"""
import argparse
import stim
from surface_code_leakage_erasure import SurfaceCodeCircuitBuilder

def build_surface_code_circuit(
    distance: int,
    rounds: int | None = None,
    p: float = 0.0,
    basis: str = "Z",
) -> stim.Circuit:
    """Construct a standard (no-leakage) rotated surface-code circuit.

    Args:
        distance: Code distance d.
        rounds: Number of syndrome rounds. If None, use rounds=d.
        p: Physical Pauli error probability.
        basis: Logical basis ("Z" or "X").
    """
    if distance < 2:
        raise ValueError("distance must be at least 2")
    if rounds is None:
        rounds = distance
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if basis not in {"Z", "X"}:
        raise ValueError("basis must be 'Z' or 'X'")

    builder = SurfaceCodeCircuitBuilder(distance)
    circuit, _ = builder.get_circuit(
        rounds=rounds,
        p=p,
        p_leak=0.0,  # normal surface code: no leakage channel
        basis=basis,
        Pauli_locations="gates",
        leak_locations="2-qubit gates",
        ec_sched=8,
        leak_effect="depolarize",
    )
    return circuit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rotated surface-code stim circuit.")
    parser.add_argument("-d", "--distance", type=int, default=2, help="Code distance d.")
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=None,
        help="Number of syndrome rounds (default: rounds=distance).",
    )
    parser.add_argument("--p", type=float, default=0.001, help="Physical Pauli error rate.")
    parser.add_argument("--basis", choices=["Z", "X"], default="Z", help="Logical basis.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output file path for writing the circuit text.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    circuit = build_surface_code_circuit(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        basis=args.basis,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(str(circuit))
    else:
        print(circuit)

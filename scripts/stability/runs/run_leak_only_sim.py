"""Leakage-only LER sweep for the stability experiment (static / early / late).

Usage:
    python run_leak_only_sim.py --style static --distance 2 --rounds 2 3 4 5
    python run_leak_only_sim.py --style early  --distance 4
    python run_leak_only_sim.py --style late   --distance 2 --rounds 2 4

For each (rounds, p_leak) pair, estimates the logical error rate with
p_Pauli=0 (leakage only) via SurfaceCodeErasureSampler.sample_LER, and
appends one line per pair to a .dat file under --outdir.
"""
import argparse
import os

from surface_code_leakage_erasure import (
    SurfaceCodeErasureSampler,
    StabilityCircuitBuilder, StabilityLayout,
    WalkingStabilityCircuitBuilder, WalkingStabilityCircuitLayout,
)

DEFAULT_ROUNDS_LIST = [2, 3, 4, 5]
DEFAULT_P_LEAK_LIST = [0.01, 0.015, 0.02, 0.025, 0.03]


def build_layout_and_sampler(style: str, distance: int, n_jobs: int):
    """Construct the (layout, sampler) pair for the requested style.

    style="static" builds the static stability circuit (StabilityLayout /
    StabilityCircuitBuilder). style="early"/"late" build the walking
    stability circuit (WalkingStabilityCircuitLayout / WalkingStability-
    CircuitBuilder) with the corresponding swap_time.
    """
    if style == "static":
        layout = StabilityLayout(distance)
        builder_class = StabilityCircuitBuilder
    elif style in ("early", "late"):
        layout = WalkingStabilityCircuitLayout(distance, swap_time=style)
        builder_class = WalkingStabilityCircuitBuilder
    else:
        raise ValueError(f"Unknown style: {style!r} (expected 'static', 'early', or 'late')")

    sampler = SurfaceCodeErasureSampler(layout, n_jobs=n_jobs)
    # Override the circuit builder class to use the stability variant.
    sampler.circuit_builder_class = builder_class
    # SurfaceCodeErasureSampler.__init__ infers mode/swap_time from
    # isinstance(layout, WalkingSurfaceCodeLayout); WalkingStabilityCircuitLayout
    # doesn't subclass that (same design as the static StabilityLayout), so it's
    # always misdetected as "static". Fix up the metadata by hand so results
    # saved in ErasureSamplerResult.parameters correctly reflect the style.
    if style in ("early", "late"):
        sampler.mode = "walking"
        sampler.swap_time = style
    return layout, sampler


def logical_error_rate_leak_only(
    style: str,
    distance: int,
    rounds: int,
    p_leak: float,
    min_samples: int,
    max_samples: int,
    target_failure_count: int,
    n_jobs: int,
):
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    _, sampler = build_layout_and_sampler(style, distance, n_jobs)

    res = sampler.sample_LER(
        rounds=rounds,
        p_Pauli=0.0,
        p_leak=p_leak,
        min_samples=min_samples,
        max_samples=max_samples,
        target_failure_count=target_failure_count,
        basis="Z",
        Pauli_locations="all",
        leak_locations="2-qubit gates",
        ec_sched=8,
        leak_effect="depolarize",
        use_branch_and_bound=False,
    )

    return res


def format_results(res):
    has_mle = res.pfail_MLE is not None
    string_list = [
        f"{res.pfail_marginal.best:.4e}", f"{res.pfail_MLE.best:.4e}" if has_mle else "nan",
        f"{res.pfail_marginal.low:.4e}",  f"{res.pfail_MLE.low:.4e}"  if has_mle else "nan",
        f"{res.pfail_marginal.high:.4e}", f"{res.pfail_MLE.high:.4e}" if has_mle else "nan",
        f"{res.prob_no_leak:.4e}",
        str(res.num_samples), str(res.failures_marginal), str(res.failures_MLE),
        str(res.num_leakage_samples), str(res.leak_failures_marginal), str(res.leak_failures_MLE),
        str(res.num_noleak_samples), str(res.num_noleak_failures)]
    return string_list


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", choices=["static", "early", "late"], required=True,
                         help="Stability circuit variant: static, or walking with early/late swap.")
    parser.add_argument("-d", "--distance", type=int, default=2, help="Code distance d.")
    parser.add_argument("--rounds", type=int, nargs="+", default=DEFAULT_ROUNDS_LIST,
                         help=f"Round counts to sweep over (default: {DEFAULT_ROUNDS_LIST}).")
    parser.add_argument("--p-leak-list", type=float, nargs="+", default=DEFAULT_P_LEAK_LIST,
                         help=f"Leakage probabilities to sweep over (default: {DEFAULT_P_LEAK_LIST}).")
    parser.add_argument("--min-samples", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=1_000_000_000)
    parser.add_argument("--target-failure-count", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Passed to SurfaceCodeErasureSampler (-1 = all CPUs).")
    parser.add_argument("--outdir", type=str, default="scratch", help="Directory to write the .dat file to.")
    return parser.parse_args()


def main():
    args = _parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_suffix = f"_job{slurm_job_id}" if slurm_job_id else ""
    fname = f"{args.outdir}/leak_only_style-{args.style}_d{args.distance}_depolarize{job_suffix}.dat"
    print(f"Saving results to {fname}")

    for rounds in args.rounds:
        for p_leak in args.p_leak_list:
            res = logical_error_rate_leak_only(
                args.style, args.distance, rounds, p_leak,
                args.min_samples, args.max_samples, args.target_failure_count,
                args.n_jobs,
            )

            ler = res.pfail_marginal.best
            actual_shots = res.num_samples

            print(
                f"style={args.style} d={args.distance} rounds={rounds} p_leak={p_leak:.4f} "
                f"LER={ler:.5f} (shots={actual_shots})"
            )

            with open(fname, "a") as f:
                f.write("\t".join([
                    args.style, "depolarize", str(args.distance), f"{p_leak:.2e}", "inf", "8",
                    str(rounds)] +
                    format_results(res))
                    + "\n")


if __name__ == "__main__":
    main()

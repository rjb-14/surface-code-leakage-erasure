import os
from surface_code_leakage_erasure import (
    SurfaceCodeErasureSampler, StabilityCircuitBuilder, StabilityLayout,
)

# experiment parameters
DISTANCE = 2
ROUNDS_LIST = [5, 15, 25]
P_LIST = [0.01, 0.015, 0.02, 0.025, 0.03]
MIN_SAMPLES = 2048
MAX_SAMPLES = 1000000000
TARGET_FAILURE_COUNT = 100
RNG_SEED = 0
BASIS = "Z"


def logical_error_rate_pauli_only(
    distance: int,
    rounds: int,
    p: float,
    min_samples: int,
    max_samples: int,
    target_failure_count: int,
    seed: int,
):
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    layout = StabilityLayout(distance)
    sampler = SurfaceCodeErasureSampler(layout, n_jobs=-1)

    # We override the circuit builder class in the sampler to use our StabilityCircuitBuilder
    sampler.circuit_builder_class = StabilityCircuitBuilder

    res = sampler.sample_LER(
        rounds=rounds,
        p_Pauli=p,
        p_leak=0.0,
        min_samples=min_samples,
        max_samples=max_samples,
        target_failure_count=target_failure_count,
        basis=BASIS,
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

def main():
    scratchfolder = "scratch/"
    if not os.path.exists(scratchfolder):
        os.makedirs(scratchfolder)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_suffix = f"_job{slurm_job_id}" if slurm_job_id else ""
    fname = f"{scratchfolder}/pauli_only_d{DISTANCE}_depolarize{job_suffix}.dat"
    print(f"Saving results to {fname}")

    for rounds in ROUNDS_LIST:
        for i, p in enumerate(P_LIST):
            seed = RNG_SEED + rounds * 10000 + i
            res = logical_error_rate_pauli_only(
                DISTANCE, rounds, p, MIN_SAMPLES, MAX_SAMPLES, TARGET_FAILURE_COUNT, seed
            )

            ler = res.pfail_marginal.best
            actual_shots = res.num_samples

            print(
                f"d={DISTANCE} rounds={rounds} p={p:.4f} "
                f"LER={ler:.5f} (shots={actual_shots})"
            )

            # write results
            with open(fname, "a") as f:
                f.write("\t".join([
                    "static", "depolarize", str(DISTANCE), f"{p:.2e}", "inf", "8"] +
                    format_results(res))
                    + "\n")


if __name__ == "__main__":
    main()

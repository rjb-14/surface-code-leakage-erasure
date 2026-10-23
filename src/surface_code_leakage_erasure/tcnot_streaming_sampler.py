"""Continuous scheduling for the independent per-p early-walking tCNOT runner."""

from __future__ import annotations

import time

import numpy as np

from .streaming_sampling import StreamingSamplerMixin
from .tcnot_sampler import (
    TCNOTSampler,
    _batched_tcnot_baseline_samples,
    _batched_tcnot_samples,
    _get_worker_decoder,
)


_worker_state = None


def _initialize_worker(config):
    global _worker_state
    kwargs = config["decoder_kwargs"]
    _worker_state = dict(config, decoder=_get_worker_decoder(**kwargs))


def _sample_chunk(batch_size, seed):
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    state = _worker_state
    split_seed, leakage_seed, baseline_seed = seed.spawn(3)
    n_noleak = int(np.random.default_rng(split_seed).binomial(batch_size, state["p_noleak"]))
    n_leak = batch_size - n_noleak
    errors = 0
    if n_leak:
        errors, _ = _batched_tcnot_samples(
            batch_size=n_leak,
            p_leak=state["p_leak"],
            seed=leakage_seed,
            decoder=state["decoder"],
            **state["decoder_kwargs"],
        )
    if n_noleak and state["decoder"].baseline_dem.num_errors:
        baseline_errors, _ = _batched_tcnot_baseline_samples(
            batch_size=n_noleak,
            seed=baseline_seed,
            **state["decoder_kwargs"],
        )
        errors += baseline_errors
    return (
        errors,
        batch_size,
        time.process_time() - cpu_started,
        time.perf_counter() - wall_started,
    )


class StreamingTCNOTSampler(StreamingSamplerMixin, TCNOTSampler):
    """Reuse the original decoder with a bounded queue instead of batch barriers."""

    def sample(
        self,
        *,
        p_leak,
        max_shots,
        max_errors=0,
        min_shots=0,
        on_batch=None,
        decoder_strategy="marginal",
        decoder_args=None,
    ):
        self._validate_decode_options(decoder_strategy, decoder_args)
        self._validate_stream_budget(p_leak, max_shots, max_errors, min_shots)
        self.streaming_metrics = {}
        if self.n_jobs == 1:
            return super().sample(
                p_leak=p_leak, max_shots=max_shots, max_errors=max_errors,
                min_shots=min_shots, on_batch=on_batch,
                decoder_strategy=decoder_strategy, decoder_args=decoder_args,
            )
        if p_leak == 0 and self.decoder.baseline_dem.num_errors == 0:
            if max_shots and on_batch is not None:
                on_batch(0, max_shots)
            return 0, max_shots
        worker_config = {
            "p_leak": p_leak,
            "p_noleak": (1 - p_leak) ** self.num_leakage_trials,
            "decoder_kwargs": {
                "builder_type": self._worker_builder_type,
                "builder_layout": self._worker_builder_layout,
                "decoder_key": self._decoder_key,
                "rounds_per_side": self.rounds_per_side,
                "rounds_before": self.rounds_before,
                "rounds_after": self.rounds_after,
                "p_pauli": self.p_pauli,
                "circuit_kwargs": self.circuit_kwargs,
                "tcnot_cache_size": self.tcnot_cache_size,
                "tesseract_compile_cache_size": self.tesseract_compile_cache_size,
                "tesseract_options": self.tesseract_options,
            },
        }
        return self._run_stream(
            initializer=_initialize_worker, worker_config=worker_config,
            sample_worker=_sample_chunk, max_shots=max_shots,
            max_errors=max_errors, min_shots=min_shots, on_batch=on_batch,
        )

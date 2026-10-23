"""Scheduling regressions: no barriers, exact budgets, and complete draining."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from surface_code_leakage_erasure import streaming_sampling as scheduling


class ThreadExecutor(ThreadPoolExecutor):
    def __init__(self, *, env=None, **kwargs):
        super().__init__(**kwargs)

    def shutdown(self, wait=True, kill_workers=False):
        super().shutdown(wait=wait, cancel_futures=kill_workers)


def runner(monkeypatch, **kwargs):
    monkeypatch.setattr(scheduling, "ProcessPoolExecutor", ThreadExecutor)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    result = scheduling.StreamingSamplerMixin(
        checkpoint_shots=1, progress_interval=0, **kwargs
    )
    result.n_jobs = 2
    result._seed_sequence = np.random.SeedSequence(42)
    return result


def run(sampler, worker, *, max_shots, max_errors=0, min_shots=0, on_batch=None):
    return sampler._run_stream(
        initializer=lambda config: None,
        worker_config={},
        sample_worker=worker,
        max_shots=max_shots,
        max_errors=max_errors,
        min_shots=min_shots,
        on_batch=on_batch,
    )


def test_fast_worker_keeps_receiving_work_while_first_shot_is_slow(monkeypatch):
    sampler = runner(monkeypatch)
    release_slow = Event()
    fast_completed_before_release = []
    saved = []

    def worker(count, seed):
        if seed.spawn_key[-1] == 0:
            assert release_slow.wait(3), "Fast tasks stalled behind a batch barrier."
        else:
            fast_completed_before_release.append(not release_slow.is_set())
        return 0, count, 0.0, 0.0

    def checkpoint(errors, shots):
        saved.append((errors, shots))
        if sum(item[1] for item in saved) >= 8:
            release_slow.set()

    assert run(sampler, worker, max_shots=20, on_batch=checkpoint) == (0, 20)
    assert sum(fast_completed_before_release) >= 8
    assert sum(shots for _, shots in saved) == 20


def test_error_target_drains_slow_submitted_shots_instead_of_dropping_them(monkeypatch):
    sampler = runner(monkeypatch)
    release_slow = Event()
    indices = []
    saved = []

    def worker(count, seed):
        index = seed.spawn_key[-1]
        if index == 0:
            assert release_slow.wait(3)
        indices.append(index)
        return count, count, 0.0, 0.0

    def checkpoint(errors, shots):
        saved.append((errors, shots))
        release_slow.set()

    assert run(sampler, worker, max_shots=100, max_errors=1, on_batch=checkpoint) == (4, 4)
    assert sorted(indices) == [0, 1, 2, 3]
    assert sum(errors for errors, _ in saved) == 4
    assert sampler.streaming_metrics["submitted_shots"] == 4


def test_pending_work_counts_towards_shot_budget_and_seeds_are_unique(monkeypatch):
    sampler = runner(monkeypatch, chunk_shots=3)
    sizes = []
    seeds = []
    saved = []

    def worker(count, seed):
        sizes.append(count)
        seeds.append(seed.spawn_key)
        return 0, count, 0.0, 0.0

    assert run(sampler, worker, max_shots=17, on_batch=lambda e, n: saved.append(n)) == (0, 17)
    assert sorted(sizes) == [2, 3, 3, 3, 3, 3]
    assert len(set(seeds)) == len(seeds)
    assert sum(saved) == 17
    assert sampler.streaming_metrics["submitted_shots"] == 17


def test_minimum_shot_floor_is_respected_when_errors_arrive_early(monkeypatch):
    sampler = runner(monkeypatch)
    errors, shots = run(
        sampler, lambda n, seed: (n, n, 0.0, 0.0),
        max_shots=30, max_errors=1, min_shots=10,
    )
    assert 10 <= shots <= 13
    assert errors == shots
    assert sampler.streaming_metrics["submitted_shots"] == shots


def test_worker_exceptions_propagate(monkeypatch):
    sampler = runner(monkeypatch)

    def worker(count, seed):
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        run(sampler, worker, max_shots=5)


def test_slurm_deadline_stops_refilling_and_drains_all_submitted_work(monkeypatch):
    sampler = runner(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(scheduling.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(scheduling.time, "time", lambda: 1000.0)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1100")
    saved = []

    def checkpoint(errors, shots):
        saved.append(shots)
        now[0] = 100.0

    errors, shots = run(
        sampler, lambda n, seed: (0, n, 0.0, 0.0),
        max_shots=1000, on_batch=checkpoint,
    )
    assert errors == 0
    assert 4 < shots <= 8
    assert sum(saved) == shots == sampler.streaming_metrics["submitted_shots"]
    assert sampler.streaming_metrics["stop_reason"] == "slurm_deadline"


def test_expired_slurm_deadline_does_not_start_a_pool(monkeypatch):
    sampler = runner(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1")

    def unexpected_pool(**kwargs):
        pytest.fail("A worker pool was created after the deadline.")

    monkeypatch.setattr(scheduling, "ProcessPoolExecutor", unexpected_pool)
    assert run(sampler, None, max_shots=100) == (0, 0)


@pytest.mark.parametrize("option,value", [("chunk_shots", 0), ("pending_per_worker", 0)])
def test_invalid_queue_options_are_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        scheduling.StreamingSamplerMixin(**{option: value})

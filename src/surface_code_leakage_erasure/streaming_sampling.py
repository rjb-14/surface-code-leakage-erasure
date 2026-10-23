"""Bounded, continuously replenished process scheduling for Monte Carlo shots."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
import math
import os
import sys
import time

from joblib.externals.loky import ProcessPoolExecutor


class StreamingSamplerMixin:
    """Keep workers supplied while checkpointing results in the parent process.

    Error targets stop new submissions, then all submitted work is drained. In
    particular, slow decodes are never cancelled merely because faster shots
    have reached the error target. The shot budget includes pending work.
    """

    def __init__(
        self,
        *args,
        chunk_shots=1,
        pending_per_worker=2,
        checkpoint_shots=32,
        checkpoint_seconds=1.0,
        progress_interval=30.0,
        slurm_margin_seconds=60.0,
        **kwargs,
    ):
        for name, value in (
            ("chunk_shots", chunk_shots),
            ("pending_per_worker", pending_per_worker),
            ("checkpoint_shots", checkpoint_shots),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
            raise ValueError("checkpoint_seconds must be finite and positive.")
        if not math.isfinite(progress_interval) or progress_interval < 0:
            raise ValueError("progress_interval must be finite and non-negative.")
        if not math.isfinite(slurm_margin_seconds) or slurm_margin_seconds < 0:
            raise ValueError("slurm_margin_seconds must be finite and non-negative.")
        self.chunk_shots = chunk_shots
        self.pending_per_worker = pending_per_worker
        self.checkpoint_shots = checkpoint_shots
        self.checkpoint_seconds = checkpoint_seconds
        self.progress_interval = progress_interval
        self.slurm_margin_seconds = slurm_margin_seconds
        self.streaming_metrics = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_stream_budget(p_leak, max_shots, max_errors, min_shots):
        if not math.isfinite(p_leak) or not 0 <= p_leak <= 1:
            raise ValueError("p_leak must be finite and between 0 and 1.")
        for name, value in (
            ("max_shots", max_shots),
            ("max_errors", max_errors),
            ("min_shots", min_shots),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if min_shots > max_shots:
            raise ValueError("min_shots cannot exceed max_shots.")

    def _run_stream(
        self,
        *,
        initializer,
        worker_config,
        sample_worker,
        max_shots,
        max_errors,
        min_shots,
        on_batch,
    ):
        started = time.perf_counter()
        last_checkpoint = started
        last_progress = started
        submitted = completed = failures = chunks = 0
        buffered_shots = buffered_errors = 0
        worker_cpu = worker_wall = max_task_seconds = 0.0
        pending = {}
        max_pending = self.n_jobs * self.pending_per_worker
        stop_reason = None
        stop_at = None
        end_time = os.environ.get("SLURM_JOB_END_TIME")
        if end_time and self.slurm_margin_seconds:
            projected_end = float(end_time)
            if not math.isfinite(projected_end):
                raise ValueError("SLURM_JOB_END_TIME must be a finite UNIX timestamp.")
            stop_at = started + max(0, projected_end - time.time() - self.slurm_margin_seconds)

        def update_metrics():
            elapsed = max(time.perf_counter() - started, 1e-12)
            self.streaming_metrics = {
                "submitted_shots": submitted,
                "completed_shots": completed,
                "completed_chunks": chunks,
                "elapsed_seconds": elapsed,
                "shots_per_second": completed / elapsed,
                "worker_cpu_seconds": worker_cpu,
                "worker_wall_seconds": worker_wall,
                "worker_cpu_utilization": worker_cpu / (self.n_jobs * elapsed),
                "max_task_seconds": max_task_seconds,
                "stop_reason": stop_reason,
            }

        def checkpoint(force=False):
            nonlocal buffered_errors, buffered_shots, last_checkpoint, last_progress
            now = time.perf_counter()
            if buffered_shots and (
                force
                or buffered_shots >= self.checkpoint_shots
                or now - last_checkpoint >= self.checkpoint_seconds
            ):
                if on_batch is not None:
                    on_batch(buffered_errors, buffered_shots)
                buffered_errors = buffered_shots = 0
                last_checkpoint = now
            update_metrics()
            if (
                on_batch is not None
                and self.progress_interval > 0
                and now - last_progress >= self.progress_interval
            ):
                metrics = self.streaming_metrics
                print(
                    f'[stream] shots={completed} pending={len(pending)} '
                    f'rate={metrics["shots_per_second"]:.2f} shots/s '
                    f'worker_cpu={metrics["worker_cpu_utilization"]:.1%} '
                    f'max_task={max_task_seconds:.2f}s',
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now

        def fill_queue(executor):
            nonlocal submitted, stop_reason
            target_met = max_errors > 0 and completed >= min_shots and failures >= max_errors
            if target_met:
                stop_reason = "max_errors"
            elif submitted >= max_shots:
                stop_reason = stop_reason or "max_shots"
            elif stop_at is not None and time.perf_counter() >= stop_at:
                if stop_reason is None and on_batch is not None:
                    print(
                        f"[stream] Slurm time limit approaching; draining {len(pending)} "
                        "submitted chunks. Resubmit this point to continue its budget.",
                        file=sys.stderr, flush=True,
                    )
                stop_reason = stop_reason or "slurm_deadline"
            while stop_reason is None and len(pending) < max_pending and submitted < max_shots:
                count = min(self.chunk_shots, max_shots - submitted)
                seed = self._seed_sequence.spawn(1)[0]
                future = executor.submit(sample_worker, count, seed)
                pending[future] = count
                submitted += count

        if max_shots == 0:
            stop_reason = "max_shots"
            update_metrics()
            return 0, 0
        if stop_at is not None and time.perf_counter() >= stop_at:
            stop_reason = "slurm_deadline"
            update_metrics()
            return 0, 0

        executor = ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=initializer,
            initargs=(worker_config,),
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        try:
            fill_queue(executor)
            while pending:
                done, _ = wait(
                    pending,
                    timeout=self.checkpoint_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    expected_shots = pending.pop(future)
                    errors, shots, cpu_seconds, wall_seconds = future.result()
                    if shots != expected_shots or not 0 <= errors <= shots:
                        raise RuntimeError("Worker returned inconsistent shot/error counts.")
                    failures += errors
                    completed += shots
                    chunks += 1
                    buffered_errors += errors
                    buffered_shots += shots
                    worker_cpu += cpu_seconds
                    worker_wall += wall_seconds
                    max_task_seconds = max(max_task_seconds, wall_seconds)
                # Refill before CSV/log callbacks, so their I/O does not starve workers.
                fill_queue(executor)
                checkpoint()
            checkpoint(force=True)
        except BaseException:
            executor.shutdown(wait=True, kill_workers=True)
            raise
        else:
            executor.shutdown(wait=True)
            update_metrics()
        return failures, completed

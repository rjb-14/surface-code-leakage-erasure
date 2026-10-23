"""Real decoding and resumable collection through the streaming sampler."""

import pytest
import sinter

from surface_code_leakage_erasure.collect import collect
from surface_code_leakage_erasure.early_walking_transversal_cx import EarlyWalkingTCNOTBuilder
from surface_code_leakage_erasure.tcnot_streaming_sampler import StreamingTCNOTSampler


def test_real_worker_initialization_decoding_and_resume(tmp_path):
    pytest.importorskip("tesseract_decoder")
    sampler = StreamingTCNOTSampler(
        builder=EarlyWalkingTCNOTBuilder(3), rounds_per_side=1,
        circuit_kwargs={"basis": "X"}, n_jobs=2, seed=123,
        tesseract_options={"det_beam": 20}, progress_interval=0,
    )
    path = tmp_path / "samples.csv"
    kwargs = dict(
        sampler=sampler, p_leak=0.02, max_errors=0,
        save_file=path, json_metadata={"d": 3},
    )
    collect(max_shots=9, **kwargs)
    assert sampler.streaming_metrics["completed_shots"] == 9
    collect(max_shots=13, **kwargs)
    assert sampler.streaming_metrics["submitted_shots"] == 4
    (stats,) = sinter.read_stats_from_csv_files(path)
    assert stats.shots == 13
    assert 0 <= stats.errors <= 13


def test_noiseless_zero_leakage_finishes_without_starting_workers():
    pytest.importorskip("tesseract_decoder")
    sampler = StreamingTCNOTSampler(
        builder=EarlyWalkingTCNOTBuilder(3), rounds_per_side=1,
        circuit_kwargs={"basis": "X"}, n_jobs=2,
        tesseract_options={"det_beam": 20},
    )
    saved = []
    assert sampler.sample(p_leak=0, max_shots=11, on_batch=lambda e, n: saved.append((e, n))) == (0, 11)
    assert saved == [(0, 11)]


def test_pauli_noisy_baseline_is_decoded_in_workers():
    pytest.importorskip("tesseract_decoder")
    sampler = StreamingTCNOTSampler(
        builder=EarlyWalkingTCNOTBuilder(3), rounds_per_side=1,
        p_pauli=0.001, circuit_kwargs={"basis": "X"}, n_jobs=2,
        tesseract_options={"det_beam": 20}, progress_interval=0,
    )
    assert sampler.decoder.baseline_dem.num_errors > 0
    errors, shots = sampler.sample(p_leak=0, max_shots=4)
    assert shots == 4
    assert 0 <= errors <= shots

"""Real graph support, noise routing, and integer-budget stability sampling."""

import math

import numpy as np
import pymatching
import pytest
import stim
from sinter import read_stats_from_csv_files

from surface_code_leakage_erasure.collect import collect
from surface_code_leakage_erasure.stability_sampler import (
    CIRCUIT_KWARGS,
    GRAPH_PROBABILITY_FLOOR,
    StabilitySampler,
    _ensure_leakage_edges,
)


@pytest.mark.parametrize("style", ["static", "early", "late"])
@pytest.mark.parametrize("p_pauli", [0.0, 0.001])
def test_zero_leakage_and_finite_pauli_noise(style, p_pauli):
    sampler = StabilitySampler(style=style, distance=2, rounds=3, p_pauli=p_pauli)
    result = sampler.sample(p_leak=0.0, max_shots=12)
    assert result[1] == 12
    assert 0 <= result[0] <= 12
    channels = list(sampler.baseline_circuit.flattened())
    noise = [line for line in channels if line.name == "DEPOLARIZE2"]
    if p_pauli:
        assert noise and all(line.gate_args_copy() == [p_pauli] for line in noise)
        assert not any(
            line.name in ("DEPOLARIZE1", "X_ERROR", "Z_ERROR") for line in channels
        )
    else:
        assert result == (0, 12)


@pytest.mark.parametrize("style", ["static", "early", "late"])
@pytest.mark.parametrize("strategy", ["marginal", "quasi_mle"])
def test_real_leakage_shots_for_every_style_and_decoder(style, strategy):
    sampler = StabilitySampler(style=style, distance=2, rounds=3)
    # p=1 forces actual leakage decoding on every shot.
    failures, shots = sampler.sample(p_leak=1.0, max_shots=4, decoder_strategy=strategy)
    assert shots == 4
    assert 0 <= failures <= shots


def test_sparse_pauli_graph_gets_missing_leakage_edge_with_probability_floor():
    sampler = StabilitySampler(style="static", distance=2, rounds=3, cache_size=7)
    context, model = sampler.context, sampler.model
    matcher, graph, logicals = context.decoder.get_Pauli_matcher_graphdict_edgedict(
        3, 0.0, **CIRCUIT_KWARGS
    )
    # Find a concrete missing edge over all possible single-qubit leakage sites.
    found = None
    for rnd in range(3):
        for gate, qubits in enumerate(context.builder.cnot_lists):
            for qubit in qubits:
                locations = {rnd: {gate: {qubit}}}
                _, checks = context.builder.get_circuit(
                    3, 0.0, leakage_circuit_locations=locations, **CIRCUIT_KWARGS
                )
                before = set(logicals)
                _ensure_leakage_edges(context, model, checks)
                added = set(logicals) - before
                if added:
                    found = added.pop()
                    break
            if found:
                break
        if found:
            break
    assert found is not None
    data = (
        matcher.get_boundary_edge_data(found[0])
        if len(found) == 1
        else matcher.get_edge_data(*found)
    )
    assert data["error_probability"] == GRAPH_PROBABILITY_FLOOR
    assert data["weight"] == pytest.approx(
        math.log((1 - GRAPH_PROBABILITY_FLOOR) / GRAPH_PROBABILITY_FLOOR)
    )


@pytest.mark.parametrize("style", ["static", "early", "late"])
@pytest.mark.parametrize("p_pauli", [0.0, 0.001])
def test_reweighted_graph_matches_direct_averaged_dem_weights(style, p_pauli):
    sampler = StabilitySampler(
        style=style, distance=2, rounds=3, p_pauli=p_pauli, cache_size=17
    )
    context = sampler.context
    context.builder.rng = np.random.default_rng(321)
    for _ in range(10):
        locations = context.builder.generate_leakage_locations(3, 0.15, atleast_1=True)
        circuit, checks = context.builder.get_circuit(
            3, p_pauli, leakage_circuit_locations=locations, **CIRCUIT_KWARGS
        )
        _ensure_leakage_edges(context, sampler.model, checks)
        dem = stim.DetectorErrorModel(
            context.decoder.get_Pauli_dem_str(
                3, max(p_pauli, GRAPH_PROBABILITY_FLOOR), **CIRCUIT_KWARGS
            )
        )
        dem += stim.DetectorErrorModel(
            context.decoder.get_full_average_dem_str(checks, **CIRCUIT_KWARGS)
        )
        direct = pymatching.Matching.from_detector_error_model(dem)
        matcher, graph, _ = context.decoder.get_Pauli_matcher_graphdict_edgedict(
            3, p_pauli, **CIRCUIT_KWARGS
        )
        updates = context.decoder.calculate_edge_updates(
            graph, erasure_checks=checks, **CIRCUIT_KWARGS
        )
        simulator = stim.FlipSimulator(batch_size=1, seed=123)
        simulator.do(circuit)
        syndrome = simulator.get_detector_flips().reshape(-1)
        _, weight = matcher.decode(syndrome, edge_reweights=updates, return_weight=True)
        _, direct_weight = direct.decode(syndrome, return_weight=True)
        assert weight == pytest.approx(direct_weight, abs=1e-5)


def test_zero_cache_and_parallel_workers():
    sampler = StabilitySampler(
        style="early", distance=2, rounds=3, cache_size=0, n_jobs=2
    )
    assert sampler.sample(p_leak=1.0, max_shots=4)[1] == 4


def test_error_stopping_and_minimum_floor(monkeypatch):
    import surface_code_leakage_erasure.stability_sampler as module

    sampler = StabilitySampler(style="early", distance=2, rounds=3)
    monkeypatch.setattr(
        module, "_leakage_batch", lambda model, p, shots, strategy, args: (shots, shots)
    )
    batches = []
    assert sampler.sample(
        p_leak=1.0,
        max_shots=100,
        min_shots=7,
        max_errors=1,
        on_batch=lambda e, n: batches.append((e, n)),
    ) == (7, 7)
    assert sum(n for _, n in batches) == 7


def test_quasi_csv_identity_includes_search_arguments(tmp_path):
    sampler = StabilitySampler(style="late", distance=2, rounds=3)
    path = tmp_path / "quasi.csv"
    for queue in (64, 128):
        collect(
            sampler=sampler,
            p_leak=0.0,
            max_shots=4,
            save_file=path,
            decoder="quasi_mle",
            decoder_strategy="quasi_mle",
            decoder_args={"mle_time_limit": None, "max_queue_size": queue},
        )
    stats = list(read_stats_from_csv_files(path))
    assert len(stats) == 2
    assert all(stat.decoder == "quasi_mle" and stat.shots == 4 for stat in stats)


def test_collect_records_incremental_batch_time(monkeypatch, tmp_path):
    import surface_code_leakage_erasure.collect as module

    times = iter([10.0, 11.0, 13.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))

    class TwoBatchSampler:
        baseline_circuit = StabilitySampler(
            style="early", distance=2, rounds=3
        ).baseline_circuit

        def sample(self, *, on_batch, **kwargs):
            on_batch(0, 2)
            on_batch(0, 2)
            return 0, 4

    path = tmp_path / "timing.csv"
    collect(sampler=TwoBatchSampler(), p_leak=0.0, max_shots=4, save_file=path)
    stat = list(read_stats_from_csv_files(path))[0]
    assert stat.seconds == 3.0

"""Independent probability jobs and compatibility with the original sweep."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load(
    "scripts/tcnot/runs/run_tcnot_sim_per_p.py",
    "early_walking_tcnot_per_p_run",
)
LEGACY = _load(
    "scripts/tcnot/runs/run_tcnot_sim.py",
    "early_walking_tcnot_legacy_comparison",
)


def test_defaults_produce_21_distinct_d7_memory_jobs():
    cfg = RUN.configured_run()
    groups = RUN.build_task_groups(cfg)

    assert cfg.experiments == ("memory",)
    assert (cfg.max_shots, cfg.max_errors, cfg.min_shots) == (1_000_000, 100, 0)
    assert len(groups) == 21
    assert all(len(group) == 1 for group in groups)
    assert len({group[0] for group in groups}) == 21
    assert all((group[0].d, group[0].rounds) == (7, 7) for group in groups)
    assert [group[0].p_leak for group in groups] == list(reversed(cfg.p_leak_list))
    assert cfg.p_leak_list[0] == pytest.approx(0.002)
    assert cfg.p_leak_list[-1] == pytest.approx(0.2)
    labels = [RUN._shard_label(group, cfg) for group in groups]
    assert len(set(labels)) == 21
    assert labels[0] == "mem_d7_r7_detboth_p0"
    assert labels[-1] == "mem_d7_r7_detboth_p20"


@pytest.mark.parametrize("bias", [float("inf"), 50])
def test_every_memory_and_stability_point_gets_its_own_job(bias):
    cfg = replace(
        RUN.configured_run(),
        experiments=("memory", "stability"),
        memory_d_list=(3, 5),
        memory_rounds_mode="explicit",
        memory_r_list=(2, 3),
        stability_d_list=(2, 4),
        stability_round_pairs=((0, 1), (1, 2)),
        p_leak_list=(0.001, 0.01, 0.1),
        bias=bias,
    )
    groups = RUN.build_task_groups(cfg)

    assert len(groups) == 24
    assert all(len(group) == 1 for group in groups)
    assert len({group[0] for group in groups}) == 24
    assert [group[0].experiment for group in groups] == ["memory"] * 12 + ["stability"] * 12


@pytest.mark.parametrize("seed", [None, 123])
def test_d7_physics_and_seeds_match_original_sweep(seed):
    cfg = replace(RUN.configured_run(), seed=seed)
    original = replace(LEGACY.configured_run(), seed=seed)
    previous = {
        task.p_idx: task
        for group in LEGACY.build_task_groups(original)
        for task in group
        if task.experiment == "memory" and task.d == 7
    }
    assert cfg.run_tag != original.run_tag
    assert RUN._output_directory(cfg) != LEGACY._output_directory(original)
    assert (cfg.max_errors, cfg.min_shots) == (
        original.max_errors, original.min_shots
    )

    for (task,) in RUN.build_task_groups(cfg):
        old_task = previous[task.p_idx]
        assert asdict(task) == asdict(old_task)
        point_seed = RUN._task_seed(cfg, task)
        assert point_seed == LEGACY._task_seed(original, old_task)
        metadata = RUN._metadata(cfg, task, 0.0, point_seed)
        old_metadata = LEGACY._metadata(original, old_task, 0.0, point_seed)
        metadata.pop("run_tag")
        old_metadata.pop("run_tag")
        assert metadata == old_metadata


def test_selecting_one_array_index_runs_only_its_probability(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(RUN, "_output_directory", lambda config: tmp_path)
    monkeypatch.setattr(RUN, "_run_group", lambda *args: calls.append(args))

    assert RUN.main(["--task-index", "21"]) == 0
    assert len(calls) == 1
    cfg, group, index, directory = calls[0]
    assert index == 21
    assert directory == tmp_path
    assert len(group) == 1
    assert group[0].p_leak == pytest.approx(0.002)
    assert RUN._shard_label(group, cfg) == "mem_d7_r7_detboth_p20"

"""Single-patch stability configuration, dispatch, and Slurm wrapper tests."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from sinter import read_stats_from_csv_files


ROOT = Path(__file__).resolve().parents[1]
RUN_FILE = ROOT / "scripts/stability/runs/run_stability_sim.py"
SPEC = importlib.util.spec_from_file_location("single_patch_stability_run", RUN_FILE)
RUN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


def config(**kwargs):
    return replace(
        RUN.configured_run(),
        **{
            "styles": ("static", "early", "late"),
            "d_list": (2, 4),
            "r_list": (3, 5),
            "p_leak_list": (0.01, 0.1),
            "max_shots": 8,
            "max_errors": 0,
            "run_tag": "test",
            **kwargs,
        },
    )


def test_sharding_and_match_distance():
    cfg = config()
    groups = RUN.build_task_groups(cfg)
    assert len(groups) == 12
    assert all(len(group) == 2 for group in groups)
    assert [task.p_leak for task in groups[0]] == [0.1, 0.01]
    finite = RUN.build_task_groups(replace(cfg, bias=50))
    assert len(finite) == 24
    assert all(len(group) == 1 for group in finite)
    matched = RUN.build_task_groups(
        replace(cfg, rounds_mode="match_distance", r_list=())
    )
    assert len(matched) == 6
    assert all(task.d == task.rounds for group in matched for task in group)


@pytest.mark.parametrize(
    "name,value",
    [
        ("D_LIST", [3]),
        ("D_LIST", [True]),
        ("R_LIST", [0]),
        ("STYLES", ("early", "early")),
        ("STYLES", ("unknown",)),
        ("P_LEAK_LIST", [float("nan")]),
        ("P_LEAK_LIST", [0.1, 0.1]),
        ("P_LEAK_LIST", [True]),
        ("BIAS", 0),
        ("BIAS", 0.001),
        ("DECODER_STRATEGY", "unknown"),
        ("DECODER_ARGS", {}),
        ("LEAKAGE_GRANULARITY", "per_qubit"),
        ("LEAKAGE_EFFECT", "depolarize"),
        ("MIN_SHOTS", 10**10),
        ("MAX_SHOTS", 0),
        ("RUN_TAG", "../test"),
    ],
)
def test_invalid_configuration(monkeypatch, name, value):
    monkeypatch.setattr(RUN, name, value)
    with pytest.raises(ValueError):
        RUN.configured_run()


def test_quasi_defaults_and_brute_force_is_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(RUN, "DECODER_STRATEGY", "quasi_mle")
    cfg = RUN.configured_run()
    assert cfg.decoder_args == {"mle_time_limit": None, "max_queue_size": 4096}
    monkeypatch.setattr(RUN, "DECODER_ARGS", {"brute_force": True})
    with pytest.raises(ValueError, match="accepts only"):
        RUN.configured_run()


def test_manifest_resumes_budgets_but_rejects_physics_changes(tmp_path):
    cfg = config()
    RUN._ensure_identity_manifest(tmp_path, cfg)
    RUN._ensure_identity_manifest(tmp_path, replace(cfg, max_shots=200, max_errors=20))
    with pytest.raises(ValueError, match="incompatible"):
        RUN._ensure_identity_manifest(tmp_path, replace(cfg, bias=50))
    with pytest.raises(ValueError, match="incompatible"):
        RUN._ensure_identity_manifest(
            tmp_path, replace(cfg, decoder_strategy="quasi_mle")
        )


def test_dry_run_is_read_only_and_does_not_import_decoders(tmp_path):
    program = (
        "import runpy,sys; "
        f"sys.argv=[{str(RUN_FILE)!r},'--dry-run']; "
        f"runpy.run_path({str(RUN_FILE)!r},run_name='__main__'); "
        "assert 'pymatching' not in sys.modules; "
        "assert 'tesseract_decoder' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    groups = RUN.build_task_groups(RUN.configured_run())
    assert summary["num_slurm_shards"] == len(groups)
    assert summary["num_physical_points"] == sum(len(group) for group in groups)
    assert not list(tmp_path.iterdir())


def test_shard_selection_and_implicit_sweep_guard(monkeypatch, tmp_path):
    cfg = config()
    monkeypatch.setattr(RUN, "configured_run", lambda: cfg)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    with pytest.raises(SystemExit, match="Refusing to run"):
        RUN.main([])
    with pytest.raises(ValueError, match="Shard index"):
        RUN.main(["--task-index", "0"])
    monkeypatch.setattr(RUN, "_output_directory", lambda cfg: tmp_path)
    calls = []
    monkeypatch.setattr(
        RUN, "_run_group", lambda cfg, group, index, output: calls.append(index)
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    RUN.main([])
    RUN.main(["--task-index", "1"])
    RUN.main(["--all"])
    assert calls == [2, 1, *range(1, 13)]


def test_group_reuses_leakage_only_sampler(monkeypatch, tmp_path):
    import surface_code_leakage_erasure.collect as collector

    cfg = config(styles=("early",), d_list=(2,), r_list=(3,))
    built, collected = [], []
    monkeypatch.setattr(
        RUN, "_build_sampler", lambda *args: built.append(args) or object()
    )
    monkeypatch.setattr(collector, "collect", lambda **kwargs: collected.append(kwargs))
    RUN._run_group(cfg, RUN.build_task_groups(cfg)[0], 1, tmp_path)
    assert len(built) == 1
    assert len(collected) == 2
    assert collected[0]["sampler"] is collected[1]["sampler"]


def test_actual_csv_resumes_only_missing_shots(tmp_path):
    cfg = config(styles=("early",), d_list=(2,), r_list=(3,), p_leak_list=(0.0,))
    group = RUN.build_task_groups(cfg)[0]
    RUN._run_group(cfg, group, 1, tmp_path)
    RUN._run_group(replace(cfg, max_shots=13), group, 1, tmp_path)
    stats = list(read_stats_from_csv_files(*tmp_path.glob("*.csv")))
    assert len(stats) == 1
    assert (stats[0].shots, stats[0].errors) == (13, 0)
    assert stats[0].json_metadata["experiment"] == "stability"


def test_submit_wrapper_from_outside_repo_with_mock_scheduler(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    conda = commands / "conda"
    conda.write_text(f'#!/bin/bash\nshift 5\nexec "{sys.executable}" "$@"\n')
    conda.chmod(0o755)
    sbatch = commands / "sbatch"
    log = tmp_path / "submission.txt"
    sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$CONDA_ENV" "$@" > "$SUBMISSION_LOG"\n'
    )
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "PYTHONPATH": str(ROOT / "src"),
        "CONDA_ENV": "custom-stability",
        "SUBMISSION_LOG": str(log),
    }
    env.pop("SLURM_JOB_ID", None)
    subprocess.run(
        ["bash", str(RUN_FILE.with_name("submit_stability.sh"))],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = log.read_text().splitlines()
    assert args[0] == "custom-stability"
    count = len(RUN.build_task_groups(RUN.configured_run()))
    assert f"--array=1-{count}" in args
    assert f"--chdir={ROOT}" in args

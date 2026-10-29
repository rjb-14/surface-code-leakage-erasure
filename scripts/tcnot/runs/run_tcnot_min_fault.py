"""Exact early-walking tCNOT leakage weights on CPUs or a resumable SLURM array.

Edit the configuration, inspect --dry-run, then run submit_tcnot_min_fault.sh.
No Monte Carlo negatives are ever used to certify a minimum.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# Set these before importing NumPy/Stim, including in spawned child processes.
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from surface_code_leakage_erasure.early_walking_transversal_cx import (  # noqa: E402
    EarlyWalkingTransversalCXBuilder,
    )
from surface_code_leakage_erasure.early_walking_stability_transversal_cx import (  # noqa: E402
    EarlyWalkingStabilityTransversalCXBuilder,
    )
from surface_code_leakage_erasure.tcnot_fault_search import (  # noqa: E402
    FaultChecker, combination_range, leakage_options, shard_interval,
    )


# BEGIN USER CONFIGURATION
# One or both experiments; each distance is combined with each of its round pairs.
EXPERIMENTS = ('memory', 'stability')
MEMORY_D_LIST = [3]
MEMORY_ROUND_PAIRS = [(2, 2)]  # (r1, r2); either may be zero, not both
MEMORY_DETECTORS = 'both'  # fixed by the early-walking memory builder
STABILITY_D_LIST = [2]
STABILITY_ROUND_PAIRS = [(1, 1), (2, 0), (2, 1)]
STABILITY_OBSERVABLE_SELECTION = 'control'  # 'control', 'target', 'both'
EC_SCHED = 8  # 8: syndrome readout; 4, 2, 1: increasingly frequent checks

SEARCHES = ('undetectable', 'uncorrectable')
# Sequential: quasi-MLE only rechecks patterns that fail marginal. Use
# ('marginal',) to disable the follow-up. These do not create separate jobs.
DECODER_STRATEGIES = ('marginal', 'quasi_mle')
# Match the notebook's INF_DET_BEAM; remaining Tesseract options retain their defaults.
TESSERACT_OPTIONS = {'det_beam': 65535}
# Optional deterministic quasi-MLE node budget, e.g. {'max_nodes': 100000}.
# Wall-clock limits are forbidden for a proof of decoder behavior. None is unlimited.
DECODER_ARGS = None

MAX_WEIGHT = 4  # can be increased when resuming the same RUN_TAG
ENUMERATION_SHARDS = 4  # array jobs per circuit/search; independent of CPUs per job
CHUNK_SIZE = 256  # raw combinations per durable checkpoint/task
UNDETECTABLE_CHUNK_SIZE = 8192  # cheaper GF(2) checks need larger chunks to amortize IPC/I/O
# Probes quickly find an upper bound at higher weights. They NEVER certify negatives.
RANDOM_PROBES_PER_SHARD = 1000
PROBE_FROM_WEIGHT = 3
SEED = 0

# Resource limits, safe to raise when resuming. A limit hit stops without certification.
MAX_BRANCH_DIM = 18  # at most 2**18 physical outputs per unambiguous pattern
DECODE_BATCH_SIZE = 256
CACHE_SIZE = 4096  # bounded, per process
PROGRESS_SECONDS = 30
RUN_TAG = 'early_walking_tcnot_min_fault_control_marginal_first'
# END USER CONFIGURATION


_CHECKER = None
_CANCELLED = None
_STOP_REQUESTED = False
_STAGED = False
STAGED_STRATEGY = 'marginal_then_quasi_mle'


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return value


def configured_tasks():
    for name, values, allowed in (
            ('EXPERIMENTS', EXPERIMENTS, ('memory', 'stability')),
            ('SEARCHES', SEARCHES, ('undetectable', 'uncorrectable')),
            ):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError(f'{name} must contain distinct entries from {allowed}.')
    if 'uncorrectable' in SEARCHES:
        if tuple(DECODER_STRATEGIES) not in (('marginal',), ('marginal', 'quasi_mle')):
            raise ValueError('DECODER_STRATEGIES must be marginal, optionally followed by quasi_mle.')
    if MEMORY_DETECTORS != 'both':
        raise ValueError("The early-walking memory builder fixes MEMORY_DETECTORS to 'both'.")
    if isinstance(EC_SCHED, bool) or EC_SCHED not in (8, 4, 2, 1):
        raise ValueError('EC_SCHED must be 8, 4, 2, or 1.')
    if STABILITY_OBSERVABLE_SELECTION not in ('control', 'target', 'both'):
        raise ValueError('Invalid STABILITY_OBSERVABLE_SELECTION.')
    if DECODER_ARGS is not None and not isinstance(DECODER_ARGS, dict):
        raise ValueError('DECODER_ARGS must be a dictionary or None.')
    if set(DECODER_ARGS or {}) - {'max_nodes', 'mle_time_limit'}:
        raise ValueError('Unknown DECODER_ARGS option.')
    if (DECODER_ARGS or {}).get('mle_time_limit') is not None:
        raise ValueError('Use a deterministic max_nodes budget, not mle_time_limit.')
    if (DECODER_ARGS or {}).get('max_nodes') is not None:
        _integer('max_nodes', DECODER_ARGS['max_nodes'])
    for name in ('MAX_WEIGHT', 'ENUMERATION_SHARDS', 'CHUNK_SIZE', 'UNDETECTABLE_CHUNK_SIZE', 'DECODE_BATCH_SIZE',
                 'CACHE_SIZE', 'PROBE_FROM_WEIGHT', 'PROGRESS_SECONDS'):
        _integer(name, globals()[name])
    for name in ('RANDOM_PROBES_PER_SHARD', 'SEED', 'MAX_BRANCH_DIM'):
        _integer(name, globals()[name], 0)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', RUN_TAG):
        raise ValueError('RUN_TAG must be a safe single directory name.')
    tasks = []
    for experiment in EXPERIMENTS:
        distances = MEMORY_D_LIST if experiment == 'memory' else STABILITY_D_LIST
        pairs = MEMORY_ROUND_PAIRS if experiment == 'memory' else STABILITY_ROUND_PAIRS
        if not distances or not pairs:
            raise ValueError(f'{experiment} needs distances and round pairs.')
        if len(set(distances)) != len(distances):
            raise ValueError('Duplicate distances.')
        seen_pairs = set()
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('Round pairs must be (r1, r2).')
            for value in pair:
                _integer('round count', value, 0)
            if tuple(pair) == (0, 0) or tuple(pair) in seen_pairs:
                raise ValueError('Round pairs must be distinct and not (0, 0).')
            seen_pairs.add(tuple(pair))
        for distance in distances:
            _integer('distance', distance, 2)
            if experiment == 'stability' and distance % 2:
                raise ValueError('Stability distances must be even.')
            for r1, r2 in pairs:
                for search in SEARCHES:
                    strategy = (STAGED_STRATEGY if 'quasi_mle' in DECODER_STRATEGIES else 'marginal'
                                ) if search == 'uncorrectable' else None
                    tasks.append(dict(experiment=experiment, distance=distance,
                                      r1=r1, r2=r2, search=search, strategy=strategy))
    return tasks


def task_key(task):
    suffix = task['search'] + (f"_{task['strategy']}" if task['strategy'] else '')
    return (f"{task['experiment']}_d{task['distance']}_r1{task['r1']}_r2{task['r2']}_"
            f'{suffix}')


def build_experiment(task):
    rounds = dict(rounds_before=task['r1'], rounds_after=task['r2'])
    kwargs = dict(ec_sched=EC_SCHED, leak_effect='skip gates',
                  leak_locations='2-qubit gates', Pauli_locations='2-qubit gates')
    if task['experiment'] == 'memory':
        return EarlyWalkingTransversalCXBuilder(task['distance']), rounds, dict(kwargs, basis='X')
    return (EarlyWalkingStabilityTransversalCXBuilder(task['distance']), rounds,
            dict(kwargs, basis='Z', observable_selection=STABILITY_OBSERVABLE_SELECTION))


def build_circuit(task):
    builder, rounds, kwargs = build_experiment(task)
    return builder.get_circuit(p=0.0, **rounds, **kwargs)[0]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in (sorted(value) if isinstance(value, set) else value)]
    return value


def atomic_json(path, value):
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_identity():
    names = [str(path.relative_to(REPO_ROOT)) for path in sorted(
        (REPO_ROOT / 'src/surface_code_leakage_erasure').glob('*.py'))]
    hashes = {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in names}
    # Execution/config choices are recorded separately; include the scheduling implementation.
    source = Path(__file__).read_text()
    implementation = source.split('# BEGIN USER CONFIGURATION\n', 1)[0] + source.split(
        '# END USER CONFIGURATION\n', 1)[1]
    hashes['runner'] = hashlib.sha256(implementation.encode()).hexdigest()
    versions = {}
    for name in ('stim', 'numpy', 'tesseract-decoder'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    return dict(source_sha256=hashes, packages=versions)


def output_directory(output_root=None):
    return Path(output_root or REPO_ROOT / 'data/tcnot/min_fault') / RUN_TAG


def prepare(tasks, root):
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(
        schema=2, tasks=tasks, shards=ENUMERATION_SHARDS, chunk_size=CHUNK_SIZE,
        undetectable_chunk_size=UNDETECTABLE_CHUNK_SIZE,
        random_probes=RANDOM_PROBES_PER_SHARD, probe_from_weight=PROBE_FROM_WEIGHT,
        seed=SEED, memory_detectors=MEMORY_DETECTORS, ec_sched=EC_SCHED,
        stability_observables=STABILITY_OBSERVABLE_SELECTION,
        tesseract_options=TESSERACT_OPTIONS, decoder_args=DECODER_ARGS,
        decoder_strategies=list(DECODER_STRATEGIES),
        quasi_mle_scope='marginal_failed_patterns',
        leakage_effect='skip_gate', leakage_gates=['CX'],
        fault_weight='leaked_gate_endpoints', **_source_identity(),
        )
    with (root / '.manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError(f'{root}: incompatible configuration/source/dependencies. Use a new RUN_TAG.')
        else:
            atomic_json(path, identity)
    return identity


def _initialize_worker(task, circuit_kwargs, event, settings):
    global _CHECKER, _CANCELLED, _STAGED
    # Only the coordinator handles pre-timeout signals. Children stop cooperatively.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    _CANCELLED = event
    _STAGED = task['strategy'] == STAGED_STRATEGY
    builder, rounds, _ = build_experiment(task)
    _CHECKER = FaultChecker(builder, **rounds, circuit_kwargs=circuit_kwargs,
                            strategy='marginal' if _STAGED else task['strategy'] or 'marginal', **settings)
    if task['strategy'] and _CHECKER.uncorrectable({}) is not None:
        raise ValueError('Decoder fails on the zero-leak baseline; positive-weight search is invalid.')
    if _STAGED and _CHECKER.uncorrectable({}, strategy='quasi_mle') is not None:
        raise ValueError('Quasi-MLE fails on the zero-leak baseline.')


def _check_chunk(weight, phase, start, stop, seed, search):
    checker = _CHECKER
    n = len(checker.locations)
    total = math.comb(n, weight)
    if phase == 'enum':
        combinations = combination_range(n, weight, start, stop)
    else:
        # Stable direct draws: changing CPU count or restarting cannot change a probe.
        combinations = (
            (i, next(combination_range(n, weight, rank, rank + 1))[1])
            for i in range(start, stop)
            for rank in (random.Random(f'{seed}:{weight}:{i}').randrange(total),)
            )
    checked = rejected = quasi_checked = quasi_skipped = 0
    witnesses = {}

    def result(complete, **extra):
        return dict(complete=complete, checked=checked, rejected=rejected,
                    witnesses=witnesses, quasi_checked=quasi_checked,
                    quasi_skipped=quasi_skipped, **extra)

    def witness(details, rank, indices, locations):
        return dict(weight=weight, leakage_locations=locations, details=details,
                    phase=phase, index=rank, candidate_indices=indices)
    try:
        for rank, indices in combinations:
            if _CANCELLED.is_set():
                return result(False)
            locations = checker.candidate(indices)
            if locations is None:
                rejected += 1
                continue
            if _STAGED and search == 'uncorrectable':
                for strategy, details in checker.staged_uncorrectable(locations, _CANCELLED.is_set):
                    if strategy == 'marginal':
                        if details is None:
                            quasi_skipped += 1
                        else:
                            witnesses.setdefault('marginal', witness(details, rank, indices, locations))
                    else:
                        quasi_checked += 1
                        if details is not None:
                            witnesses['quasi_mle'] = witness(details, rank, indices, locations)
                            return result(False)
                checked += 1
                continue
            details = (checker.undetectable(indices) if search == 'undetectable'
                       else checker.uncorrectable(locations, _CANCELLED.is_set))
            checked += 1
            if details is not None:
                return result(False, witness=witness(details, rank, indices, locations))
    except InterruptedError:
        return result(False)
    except Exception as error:
        if not _STAGED:
            raise
        # Keep a marginal witness even if its quasi-MLE follow-up fails.
        return result(False, error=error)
    return result(True)


def best_witness(directory, strategy=None):
    pattern = f'witness_{strategy}_s*.json' if strategy else 'witness_s*.json'
    witnesses = [json.loads(path.read_text()) for path in directory.glob(pattern)]
    return min(witnesses, key=lambda w: w['weight']) if witnesses else None


def stopping_witness(task, directory):
    return best_witness(directory, 'quasi_mle' if task['strategy'] == STAGED_STRATEGY else None)


def worker_count(requested=None):
    allocation = os.environ.get('SLURM_CPUS_PER_TASK')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    limit = min(available, int(allocation)) if allocation else available
    count = requested if requested is not None else (limit if allocation else 1)
    _integer('workers', count)
    if count > limit:
        raise ValueError(f'Requested {count} workers but only {limit} allocated/available CPUs.')
    return count


def _run_phase(*, executor, event, state, layer, phase, end, weight, seed, directory,
               state_path, shard, workers):
    """A bounded dynamic queue, preserving holes when chunks complete out of order."""
    cursor_key = 'next_rank' if phase == 'enum' else 'next_probe'
    done_key = f'{phase}_done'
    done = {int(k): v for k, v in layer.setdefault(done_key, {}).items()}
    cursor = layer[cursor_key]
    scheduled = cursor
    pending = {}
    last_log = 0
    next_poll = 0
    terminal_error = None
    is_undetectable = state['task']['search'] == 'undetectable'
    staged = state['task']['strategy'] == STAGED_STRATEGY
    chunk_limit = UNDETECTABLE_CHUNK_SIZE if is_undetectable else CHUNK_SIZE
    # Keep several tasks per CPU even when a shard's interval is smaller than
    # the configured chunk limit. Persist the chosen grid so changing CPU count
    # on resume never changes existing chunk boundaries.
    chunk_size = layer.setdefault(f'{phase}_chunk_size', min(
        chunk_limit, max(64 if is_undetectable else 1,
                         (end - cursor + workers * 4 - 1) // (workers * 4)),
        ))
    while scheduled < end or pending:
        now = time.monotonic()
        if _STOP_REQUESTED:
            event.set()
        if now >= next_poll:
            witness = stopping_witness(state['task'], directory)
            if witness is not None and witness['weight'] <= weight:
                event.set()
            next_poll = now + 2
        while not event.is_set() and scheduled < end and len(pending) < workers * 2:
            stop = min(end, scheduled + chunk_size)
            if scheduled in done:
                scheduled = done[scheduled]
                continue
            future = executor.submit(_check_chunk, weight, phase, scheduled, stop, seed,
                                     state['task']['search'])
            pending[future] = (scheduled, stop)
            scheduled = stop
        if not pending:
            break
        finished, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
        for future in finished:
            start, stop = pending.pop(future)
            try:
                result = future.result()
            except Exception as error:
                terminal_error = error
                state['error'] = str(error)
                event.set()
                continue
            found = result.get('witnesses', {})
            if result.get('witness'):
                found = {None: result['witness']}
            for strategy, witness in found.items():
                witness['shard'] = shard
                suffix = f'{strategy}_' if strategy else ''
                path = directory / f'witness_{suffix}s{shard:04d}.json'
                previous = json.loads(path.read_text()) if path.exists() else None
                if previous is None or witness['weight'] < previous['weight']:
                    atomic_json(path, witness)
                    if staged:
                        state.setdefault('witness_weights', {})[strategy] = witness['weight']
                    else:
                        state['witness_weight'] = witness['weight']
                    print(f"WITNESS {task_key(state['task'])} decoder={strategy or state['task']['strategy']} "
                          f"weight={weight}: {witness['leakage_locations']}", flush=True)
                if not staged or strategy == 'quasi_mle':
                    event.set()
            if result['complete']:
                done[start] = stop
                layer['checked'] += result['checked']
                layer['rejected'] += result['rejected']
                for counter in ('quasi_checked', 'quasi_skipped'):
                    key = f'{phase}_{counter}'
                    layer[key] = layer.get(key, 0) + result.get(counter, 0)
            if result.get('error'):
                terminal_error = result['error']
                state['error'] = str(terminal_error)
                event.set()
        while cursor in done:
            cursor = done.pop(cursor)
        layer[cursor_key] = cursor
        layer[done_key] = done
        state['updated_at_unix'] = time.time()
        # Saving each received chunk bounds lost work on SIGKILL/requeue to in-flight chunks.
        atomic_json(state_path, state)
        if now - last_log >= PROGRESS_SECONDS:
            print(f"{task_key(state['task'])} shard={shard + 1}/{ENUMERATION_SHARDS} "
                  f'weight={weight} phase={phase} cursor={cursor}/{end} '
                  f"checked={layer['checked']} rejected={layer['rejected']}", flush=True)
            last_log = now
        if event.is_set():
            # Running checks see the shared event; queued work is cancelled without waiting.
            for future in list(pending):
                if future.cancel():
                    pending.pop(future)
            if not pending:
                break
    if terminal_error:
        raise terminal_error
    return cursor == end


def run_shard(task, shard, root, workers):
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    directory = root / task_key(task)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f'state_s{shard:04d}.json'
    # POSIX advisory lock is released by the OS on job death; no stale claim cleanup.
    with (directory / f'.shard_s{shard:04d}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This exact shard is already running in another job.') from error
        builder, rounds, kwargs = build_experiment(task)
        builder.get_circuit(p=0.0, **rounds, **kwargs)
        n = len(leakage_options(builder))
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            task=task, shard=shard, num_locations=n, weights={},
            )
        state.pop('error', None)
        state['status'] = 'running'
        context = mp.get_context('spawn')
        event = context.Event()

        def stopping(signum, frame):
            global _STOP_REQUESTED
            _STOP_REQUESTED = True
            # Event.set takes a process-shared lock. Let the scheduler set it
            # outside the signal handler to avoid interrupting that same lock.

        previous_handlers = {sig: signal.signal(sig, stopping)
                             for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
        settings = dict(tesseract_options=TESSERACT_OPTIONS, decoder_args=DECODER_ARGS,
                        max_branch_dim=MAX_BRANCH_DIM, batch_size=DECODE_BATCH_SIZE,
                        cache_size=CACHE_SIZE)
        print(f'{task_key(task)}: shard={shard + 1}/{ENUMERATION_SHARDS}, '
              f'{n} locations, {workers} worker processes', flush=True)
        atomic_json(state_path, state)
        try:
            with ProcessPoolExecutor(
                    max_workers=workers, mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(task, kwargs, event, settings),
                    ) as executor:
                for weight in range(1, MAX_WEIGHT + 1):
                    witness = stopping_witness(task, directory)
                    if _STOP_REQUESTED or witness is not None and witness['weight'] <= weight:
                        break
                    start, stop = shard_interval(math.comb(n, weight), shard, ENUMERATION_SHARDS)
                    layer = state['weights'].setdefault(str(weight), dict(
                        start=start, stop=stop, next_rank=start, next_probe=0,
                        checked=0, rejected=0, exhausted=False,
                        ))
                    if layer['exhausted']:
                        continue
                    seed = f"{SEED}:{task_key(task)}:{shard}:{task['search']}"
                    probes = (RANDOM_PROBES_PER_SHARD
                              if weight >= PROBE_FROM_WEIGHT and start < stop else 0)
                    if probes and not _run_phase(
                            executor=executor, event=event, state=state, layer=layer,
                            phase='probe', end=probes, weight=weight, seed=seed,
                            directory=directory, state_path=state_path, shard=shard, workers=workers):
                        break
                    if event.is_set():
                        break
                    complete = _run_phase(
                        executor=executor, event=event, state=state, layer=layer,
                        phase='enum', end=stop, weight=weight, seed=seed,
                        directory=directory, state_path=state_path, shard=shard, workers=workers)
                    layer['exhausted'] = complete
                    atomic_json(state_path, state)
                    if not complete:
                        break
            state['status'] = 'interrupted' if _STOP_REQUESTED else 'finished'
        except Exception as error:
            state.update(status='error', error=str(error))
            raise
        finally:
            atomic_json(state_path, state)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        return 75 if _STOP_REQUESTED else 0


def summarize(tasks, root):
    results = []
    for task in tasks:
        directory = root / task_key(task)
        states = [json.loads(path.read_text()) for path in sorted(directory.glob('state_s*.json'))]
        all_shards = {state['shard'] for state in states} == set(range(ENUMERATION_SHARDS))
        enumerated = 0
        for weight in range(1, MAX_WEIGHT + 1):
            if not all_shards or not all(
                    state['weights'].get(str(weight), {}).get('exhausted', False) for state in states):
                break
            enumerated = weight
        staged = task['strategy'] == STAGED_STRATEGY
        strategies = ('marginal', 'quasi_mle') if staged else (task['strategy'],)
        for strategy in strategies:
            witness = best_witness(directory, strategy if staged else None)
            restricted = staged and strategy == 'quasi_mle'
            through = enumerated
            if restricted:
                # Marginal success does not imply quasi-MLE success. Only fully
                # checked weights with zero skipped patterns certify a negative.
                through = 0
                for weight in range(1, enumerated + 1):
                    if any(state['weights'][str(weight)].get('enum_quasi_skipped', 0)
                           for state in states):
                        break
                    through = weight
            if witness is not None:
                through = min(through, witness['weight'] - 1)
            proven = witness is not None and through >= witness['weight'] - 1
            if proven:
                status = 'proven_minimum'
            elif witness:
                skipped_lower = restricted and any(
                    layer.get('enum_quasi_skipped', 0)
                    for state in states for weight, layer in state['weights'].items()
                    if int(weight) < witness['weight'])
                status = ('upper_bound_pending_unchecked_patterns' if skipped_lower
                          else 'upper_bound_pending_lower_weights')
            elif through >= MAX_WEIGHT:
                status = 'no_witness_through_max_weight'
            elif restricted and enumerated >= MAX_WEIGHT:
                status = 'no_witness_in_marginal_failures_through_max_weight'
            else:
                status = 'incomplete'
            result = dict(
                **dict(task, strategy=strategy),
                minimum_weight=witness['weight'] if proven else None,
                witness_weight=witness['weight'] if witness else None,
                proven_minimum=proven, certified_no_failure_through=through,
                enumerated_through=enumerated,
                search_scope='marginal_failed_patterns' if restricted else 'all_patterns',
                status=status, shards_present=len(states), shards_expected=ENUMERATION_SHARDS,
                errors=[state['error'] for state in states if state.get('error')], witness=witness,
                )
            results.append(result)
            label = 'd_leak' if task['search'] == 'undetectable' else f't_{strategy}'
            print(f"{task_key(task)}: {label}={result['minimum_weight']} "
                  f"status={result['status']} certified_negative_through={through} "
                  f"witness_weight={result['witness_weight']}", flush=True)
    atomic_json(root / 'results.json', results)
    csv_path = root / 'summary.csv'
    temporary = csv_path.with_name(f'{csv_path.name}.{os.getpid()}.tmp')
    fields = [key for key in results[0] if key not in ('witness', 'errors')]
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    os.replace(temporary, csv_path)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    mode.add_argument('--print-num-shards', action='store_true')
    mode.add_argument('--print-run-tag', action='store_true')
    parser.add_argument('--array-index', type=int, help='One-based array index; overrides SLURM_ARRAY_TASK_ID.')
    parser.add_argument('--workers', type=int, help='Defaults to SLURM_CPUS_PER_TASK, or 1 locally.')
    parser.add_argument('--output-root', type=Path, help='Parent directory containing RUN_TAG; must be shared across nodes.')
    args = parser.parse_args(argv)
    tasks = configured_tasks()
    if args.print_num_shards:
        print(len(tasks) * ENUMERATION_SHARDS)
        return 0
    if args.print_run_tag:
        print(RUN_TAG)
        return 0
    root = output_directory(args.output_root)
    if args.dry_run:
        print(json.dumps(dict(tasks=tasks, array_jobs=len(tasks) * ENUMERATION_SHARDS,
                              max_weight=MAX_WEIGHT, shards_per_search=ENUMERATION_SHARDS,
                              output_directory=str(root), certification='exact physical output enumeration',
                              max_branch_dim=MAX_BRANCH_DIM, ec_sched=EC_SCHED,
                              memory_detectors=MEMORY_DETECTORS,
                              stability_observables=STABILITY_OBSERVABLE_SELECTION,
                              decoder_strategies=DECODER_STRATEGIES,
                              decoder_execution='shared marginal-first workers',
                              quasi_mle_scope='marginal_failed_patterns',
                              tesseract_options=TESSERACT_OPTIONS,
                              decoder_args=DECODER_ARGS), indent=2))
        return 0
    prepare(tasks, root)
    if args.summarize:
        summarize(tasks, root)
        return 0
    if args.prepare:
        for task in tasks:
            builder, rounds, kwargs = build_experiment(task)
            circuit, _ = builder.get_circuit(p=0.0, **rounds, **kwargs)
            circuit.detector_error_model()
            n = len(leakage_options(builder))
            print(f'{task_key(task)}: locations={n}, raw_combinations=' + str({
                w: math.comb(n, w) for w in range(1, MAX_WEIGHT + 1)
                }))
        print(f'Results: {root}')
        return 0
    index = args.array_index
    if index is None and os.environ.get('SLURM_ARRAY_TASK_ID'):
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if index is None or not 1 <= index <= len(tasks) * ENUMERATION_SHARDS:
        parser.error('--worker needs a valid --array-index or SLURM_ARRAY_TASK_ID.')
    task_index, shard = divmod(index - 1, ENUMERATION_SHARDS)
    return run_shard(tasks[task_index], shard, root, worker_count(args.workers))


if __name__ == '__main__':
    raise SystemExit(main())

"""Exact leakage-only fault checks and directly indexed combination enumeration.

The undetectable check uses the notebook's decode=True Pauli envelope. The
uncorrectable check instead enumerates the physical skip_gate circuit's entire
reachable affine output space; it makes no sampling/coverage assumption.
"""

from __future__ import annotations

import itertools
import math
import re
from functools import lru_cache

import numpy as np
import stim

from .tcnot_decoder import EarlyWalkingTCNOTDecoder, TesseractHyperedgeDecoder


def leakage_options(builder):
    """Chronological (round, step, qubit) endpoints, including the tCNOT layer.

    Both endpoints of the same gate may be selected independently. Each leaked
    endpoint counts as one fault, matching the reference minimum-weight search.
    The builder must already have a configured round schedule.
    """
    options = []
    for rnd in range(builder.total_rounds + 1):
        if rnd == builder.cx_round:
            options.extend((rnd, builder.TCX_STEP, qubit)
                           for pair in builder._transversal_pairs() for qubit in pair)
        if rnd < builder.total_rounds:
            for step in range(4):
                options.extend((rnd, step, qubit)
                               for qubit in builder.cnot_lists[rnd % 2][step])
    return tuple(options)


def fold_locations(locations):
    result = {}
    for rnd, step, qubit in locations:
        result.setdefault(rnd, {}).setdefault(step, set()).add(qubit)
    return result


def leakage_lifetimes(builder, ec_sched):
    """Map each endpoint to its qubit's reset/erasure-check lifetime."""
    epochs = dict.fromkeys(builder.qubit_set, 0)
    result = {}

    def reset(qubits):
        for qubit in qubits:
            epochs[qubit] += 1

    for rnd in range(builder.total_rounds + 1):
        if rnd == builder.cx_round:
            for pair in builder._transversal_pairs():
                for qubit in pair:
                    result[rnd, builder.TCX_STEP, qubit] = (qubit, epochs[qubit])
            if rnd < builder.total_rounds:
                reset(builder.reset_sets[rnd % 2])
        if rnd == builder.total_rounds:
            break
        for step in range(4):
            for qubit in builder.cnot_lists[rnd % 2][step]:
                result[rnd, step, qubit] = (qubit, epochs[qubit])
            if step < 3 and (step + 1) % ec_sched == 0:
                reset(builder.qubit_set)
            elif step == 3:
                reset(builder.measure_sets[rnd % 2] if ec_sched == 8 else builder.qubit_set)
    return result


class BranchLimitError(RuntimeError):
    """A candidate has not been certified: its exact output space is too large."""


def combination_at(n: int, k: int, rank: int) -> tuple[int, ...]:
    """Unrank itertools.combinations(range(n), k), without scanning earlier tuples."""
    if not 0 <= k <= n or not 0 <= rank < math.comb(n, k):
        raise ValueError('Invalid combination dimensions or rank.')
    result = []
    low = 0
    for remaining in range(k, 0, -1):
        total = math.comb(n - low, remaining)
        left, right = low, n - remaining
        while left < right:
            middle = (left + right + 1) // 2
            skipped = total - math.comb(n - middle, remaining)
            if skipped <= rank:
                left = middle
            else:
                right = middle - 1
        rank -= total - math.comb(n - left, remaining)
        result.append(left)
        low = left + 1
    return tuple(result)


def combination_range(n: int, k: int, start: int, stop: int):
    """Yield only the requested lexicographic interval, using O(k) memory."""
    if not 0 <= start <= stop <= math.comb(n, k):
        raise ValueError('Invalid combination interval.')
    if start == stop:
        return
    current = list(combination_at(n, k, start))
    for rank in range(start, stop):
        yield rank, tuple(current)
        if rank + 1 == stop:
            break
        i = k - 1
        while current[i] == n - k + i:
            i -= 1
        current[i] += 1
        for j in range(i + 1, k):
            current[j] = current[j - 1] + 1


def shard_interval(total: int, shard: int, shards: int) -> tuple[int, int]:
    """Zero-based shard with disjoint, complete, nearly equal raw combination counts."""
    if not 0 <= shard < shards:
        raise ValueError('Invalid shard.')
    return total * shard // shards, total * (shard + 1) // shards


def independent_vectors(vectors):
    pivots = {}
    for vector in vectors:
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                break
            vector ^= pivots[pivot]
    return tuple(pivots.values())


def silent_logical(vectors, num_detectors: int) -> int:
    """Find a nonzero logical in the kernel of the detector projection, or zero."""
    mask = (1 << num_detectors) - 1
    pivots = {}
    for vector in vectors:
        while vector & mask:
            pivot = (vector & mask).bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                break
            vector ^= pivots[pivot]
        else:
            if vector:
                return vector
    return 0


def dem_vectors(dem, num_detectors: int):
    """Independent-mechanism support as Python bitsets, including observable bits."""
    for instruction in dem.flattened():
        if instruction.type != 'error':
            continue
        probability, = instruction.args_copy()
        if probability == 0:
            continue
        if not 0 < probability < 1:
            raise ValueError('Expected stochastic, non-deterministic mechanisms.')
        vector = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                vector ^= 1 << target.val
            elif target.is_logical_observable_id():
                vector ^= 1 << (num_detectors + target.val)
        if vector:
            yield vector


def bit_indices(value: int):
    result = []
    while value:
        bit = value & -value
        result.append(bit.bit_length() - 1)
        value ^= bit
    return result


def bit_rows(values, width: int):
    num_bytes = (width + 7) // 8
    packed = b''.join(value.to_bytes(num_bytes, 'little') for value in values)
    return np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8).reshape(len(values), num_bytes),
        axis=1, bitorder='little',
        )[:, :width].astype(np.bool_)


def affine_outputs(basis, offset: int):
    """Gray-code enumeration: one integer XOR per distinct reachable output."""
    current = offset
    yield current
    for index in range(1, 1 << len(basis)):
        current ^= basis[(index & -index).bit_length() - 1]
        yield current


class FaultChecker:
    def __init__(self, builder, *, rounds_before, rounds_after, circuit_kwargs=None,
                 strategy='marginal', tesseract_options=None,
                 decoder_args=None, max_branch_dim=18, batch_size=256,
                 cache_size=4096):
        if strategy not in ('marginal', 'quasi_mle'):
            raise ValueError('Unknown decoder strategy.')
        if (decoder_args or {}).get('mle_time_limit') is not None:
            raise ValueError('A wall-clock decoder budget cannot certify deterministic behavior.')
        self.builder = builder
        self.round_kwargs = dict(rounds_before=rounds_before, rounds_after=rounds_after)
        self.physical_kwargs = builder.validate_circuit_kwargs({
            'basis': 'X', 'leak_effect': 'skip gates', 'ec_sched': 8,
            **(circuit_kwargs or {}),
            })
        if (self.physical_kwargs['leak_effect'] != 'skip gates'
                or self.physical_kwargs['leak_locations'] != '2-qubit gates'):
            raise ValueError('The search requires skip gates leakage on two-qubit gates.')
        self.envelope_kwargs = builder.validate_circuit_kwargs(self.physical_kwargs, decode=True)
        circuit, _ = self.get_circuit({})
        if circuit.detector_error_model().num_errors:
            raise ValueError('The minimum-leakage search requires a noiseless baseline.')
        self.circuit = circuit
        self.locations = leakage_options(builder)
        lifetimes = leakage_lifetimes(builder, self.physical_kwargs['ec_sched'])
        self._lifetimes = tuple(lifetimes[location] for location in self.locations)
        self.nd = circuit.num_detectors
        self.no = circuit.num_observables
        self.strategy = strategy
        self.options = dict(tesseract_options or {'det_beam': 65535})
        self.decoder_args = decoder_args
        self.max_branch_dim = max_branch_dim
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._decoder = None
        self._converter = None
        self._logical_suffix = None
        self.envelope_vectors = lru_cache(maxsize=cache_size)(self.envelope_vectors)

    @property
    def decoder(self):
        if self._decoder is None:
            self._decoder = EarlyWalkingTCNOTDecoder(
                self.builder, **self.round_kwargs, p_pauli=0.0,
                circuit_kwargs=self.physical_kwargs,
                hyperedge_decoder=TesseractHyperedgeDecoder(compile_cache_size=1, **self.options),
                cache_size=self.cache_size,
                )
        return self._decoder

    def get_circuit(self, locations, *, decode=False):
        return self.builder.get_circuit(
            p=0.0, p_leak=0.0, leakage_circuit_locations=locations,
            **self.round_kwargs,
            **(self.envelope_kwargs if decode else self.physical_kwargs),
            )

    def candidate(self, indices):
        """Reject multiple initiations of the same unreset leakage lifetime."""
        lifetimes = [self._lifetimes[index] for index in indices]
        if len(set(lifetimes)) != len(lifetimes):
            return None
        return fold_locations(self.locations[index] for index in indices)

    def envelope_vectors(self, index):
        """Single-leak supports add exactly in the unchanged decode=True circuit.

        Every DEPOLARIZE1(0.75) reaches the full one-qubit Pauli group and every
        erased measurement is an independent fair coin. Taking the union of the
        single-leak generators therefore gives exactly the multi-leak span.
        """
        envelope, _ = self.get_circuit(
            fold_locations([self.locations[index]]), decode=True,
            )
        return independent_vectors(dem_vectors(envelope.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True, flatten_loops=True,
            ), self.nd))

    def undetectable(self, indices):
        vectors = tuple(itertools.chain.from_iterable(self.envelope_vectors(i) for i in indices))
        logical = silent_logical(vectors, self.nd)
        if not logical:
            return None
        mask = (1 << self.nd) - 1
        return {
            'detector_rank': len(independent_vectors(v & mask for v in vectors)),
            'full_rank': len(independent_vectors(vectors)),
            'silent_logical_observables': bit_indices(logical >> self.nd),
            'model': 'decode_true_envelope',
            }

    def _prepare_projection(self):
        # Promote logical parities to detectors so Stim can analyze their gauge
        # randomness too. Accumulate repeated OBSERVABLE_INCLUDE declarations.
        prefix = stim.Circuit()
        refs = [set() for _ in range(self.no)]
        for instruction in self.circuit.flattened():
            if instruction.name == 'OBSERVABLE_INCLUDE':
                logical = int(instruction.gate_args_copy()[0])
                for target in instruction.targets_copy():
                    if not target.is_measurement_record_target:
                        raise ValueError('Only measurement-record observables are supported.')
                    record = prefix.num_measurements + target.value
                    refs[logical].symmetric_difference_update({record})
            else:
                prefix.append(instruction)
        suffix = stim.Circuit()
        for records in refs:
            suffix.append('DETECTOR', [
                stim.target_rec(record - prefix.num_measurements) for record in sorted(records)
                ])
        self._logical_suffix = '\n' + str(suffix)
        self._converter = self.circuit.compile_m2d_converter()

    def physical_space(self, physical):
        """Exact affine output support in the ideal circuit's reference convention.

        skip gates removes Clifford gates and adds erased measurement flips and
        complete depolarization at erasure checks (independent fair X/Z coins).
        Stim's gauge analysis supplies the fair-coin
        generators for intrinsic measurement randomness. The physical reference
        sample supplies any affine offset relative to the *ideal* experiment.
        No disjoint-noise approximation is enabled on this physical path.
        """
        if self._logical_suffix is None:
            self._prepare_projection()
        if physical.num_measurements != self.circuit.num_measurements:
            raise ValueError('Leakage changed the measurement-record layout.')
        projected = stim.Circuit(re.sub(
            r'(?m)^OBSERVABLE_INCLUDE[^\n]*\n?', '', str(physical),
            ) + self._logical_suffix)
        dem = projected.detector_error_model(allow_gauge_detectors=True, flatten_loops=True)
        for instruction in dem.flattened():
            if instruction.type == 'error' and instruction.args_copy() != [0.5]:
                raise ValueError('Physical support certification requires only fair-coin noise.')
        basis = independent_vectors(dem_vectors(dem, projected.num_detectors))
        reference = self._converter.convert(
            measurements=physical.reference_sample()[None, :], append_observables=True,
            )[0]
        offset = int.from_bytes(np.packbits(reference, bitorder='little').tobytes(), 'little')
        return basis, offset

    def _predictions(self, syndromes, erasures, *, strategy=None):
        strategy = strategy or self.strategy
        if strategy == 'marginal':
            return self.decoder.decode_batch(syndromes=syndromes, erasure_checks=erasures)
        # Evaluate *every* syndrome, including ones marginal gets right. Quasi-MLE
        # can change those answers too, which the notebook's shortcut misses.
        predictions = []
        for syndrome in syndromes:
            marginal, valid = self.decoder.decode(
                syndrome=syndrome, erasure_checks=erasures,
                decoder_strategy='quasi_mle', decoder_args=self.decoder_args,
                )
            predictions.append(marginal if valid is None else valid)
        return np.asarray(predictions)

    def uncorrectable(self, locations, cancelled=lambda: False, *, strategy=None):
        physical, erasures = self.get_circuit(locations)
        basis, offset = self.physical_space(physical)
        return self._failure_in_space(
            locations, erasures, basis, offset, cancelled, strategy=strategy)

    def staged_uncorrectable(self, locations, cancelled=lambda: False):
        """Check marginal first; only failed patterns receive a quasi-MLE check.

        Reuse the physical output space and decoder caches. The follow-up checks
        every reachable output of the failed *pattern*, including outputs that
        marginal decoded correctly. Skipped marginal-success patterns remain
        unchecked for quasi-MLE, so this is a restricted quasi-MLE search.
        """
        physical, erasures = self.get_circuit(locations)
        basis, offset = self.physical_space(physical)
        marginal = self._failure_in_space(
            locations, erasures, basis, offset, cancelled, strategy='marginal')
        yield 'marginal', marginal
        if marginal is not None:
            yield 'quasi_mle', self._failure_in_space(
                locations, erasures, basis, offset, cancelled, strategy='quasi_mle')

    def _failure_in_space(self, locations, erasures, basis, offset, cancelled, *, strategy=None):
        ambiguous = silent_logical(basis, self.nd)
        if ambiguous:
            # Two actually reachable truths for one syndrome defeat any fixed
            # prediction. Decode once to return a concrete failing branch.
            outputs = iter((offset, offset ^ ambiguous))
        else:
            if len(basis) > self.max_branch_dim:
                raise BranchLimitError(
                    f'Physical output rank {len(basis)} exceeds MAX_BRANCH_DIM='
                    f'{self.max_branch_dim}; candidate {locations} remains uncertified. '
                    'Increase the limit and resume.'
                    )
            outputs = affine_outputs(basis, offset)
        checked = 0
        while True:
            if cancelled():
                raise InterruptedError('Another worker found a witness or the job is stopping.')
            values = list(itertools.islice(outputs, self.batch_size))
            if not values:
                return None
            rows = bit_rows(values, self.nd + self.no)
            predictions = self._predictions(rows[:, :self.nd], erasures, strategy=strategy)
            bad = np.flatnonzero(np.any(predictions != rows[:, self.nd:], axis=1))
            checked += len(values)
            if len(bad):
                shot = int(bad[0])
                return {
                    'syndrome': np.flatnonzero(rows[shot, :self.nd]).tolist(),
                    'actual_observables': rows[shot, self.nd:].astype(int).tolist(),
                    'predicted_observables': predictions[shot].astype(int).tolist(),
                    'erasure_checks': erasures,
                    'physical_output_rank': len(basis),
                    'num_reachable_outputs': 1 << len(basis),
                    'outputs_checked': checked,
                    'physical_self_ambiguous': bool(ambiguous),
                    'model': 'physical_skip_gate_exact',
                    }

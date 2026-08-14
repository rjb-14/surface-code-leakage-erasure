"""Modified MLE decoder: enumerate leak combos consistent with fired ECs,
then infer L0 by marginal posterior over those combos.

Designed for ec_sched=8 + skip-gates / decode-skip-gates models.

Decision rule (0-1 loss on L0):
  Z_ell = sum_C w_C * P(syn, L0=ell | C)
  L0_hat = argmax_ell Z_ell
with w_C = P(C) (uniform on valid combos when |C| is fixed / p_leak omitted).
MAP combo argmax_C w_C P(syn|C) is returned for explanation only.
"""
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from .decoding import iterate_erasure_checks, single_leak_loc


def active_qubits_in_round(builder, rnd: int) -> set[int]:
    cl = builder.cnot_lists
    if len(cl) == 2 and cl and isinstance(cl[0][0], list):
        parity = rnd % 2
        qs = set()
        for step in range(4):
            qs |= set(cl[parity][step])
        qs |= set(builder.measure_sets[parity])
        return qs
    qs = set()
    for step in range(4):
        qs |= set(cl[step])
    return qs


def _time(rnd: int, gate: int) -> int:
    return rnd * 4 + gate


def _gf2_contains(generators: list[int], target: int) -> bool:
    basis = [0] * 64

    def insert(x: int) -> None:
        while x:
            b = x.bit_length() - 1
            if basis[b]:
                x ^= basis[b]
            else:
                basis[b] = x
                return

    for g in generators:
        insert(g)
    x = target
    while x:
        b = x.bit_length() - 1
        if not basis[b]:
            return False
        x ^= basis[b]
    return True


def _mechs_from_stim_dem(stim_dem) -> list[tuple[int, int, float]]:
    """Each error(p) -> (det_mask, l0_bit, p). Separators ignored for flip set."""
    mechs: list[tuple[int, int, float]] = []
    for line in stim_dem:
        if line.type != "error":
            break
        p = float(line.args_copy()[0])
        det_mask, l0 = 0, 0
        for targ in line.targets_copy():
            if targ.is_relative_detector_id():
                det_mask ^= 1 << int(targ.val)
            elif targ.is_logical_observable_id() and int(targ.val) == 0:
                l0 ^= 1
        if det_mask or l0:
            mechs.append((det_mask, l0, p))
    return mechs


def _p_syn_l0(
    mechs: list[tuple[int, int, float]], syn_mask: int, n_det: int
) -> tuple[float, float, float]:
    """Walsh–Hadamard: (P(syn), P(syn,L0=0), P(syn,L0=1))."""
    n = n_det + 1
    M = 1 << n

    def prob_target(target: int) -> float:
        acc = 0.0
        for u in range(M):
            chi = 1.0
            for det_mask, l0, p in mechs:
                g = det_mask | (l0 << n_det)
                dot = bin(u & g).count("1") & 1
                chi *= (1.0 - p) + p * (1.0 - 2.0 * dot)
            sign = 1.0 - 2.0 * (bin(u & target).count("1") & 1)
            acc += sign * chi
        return acc / M

    p0 = prob_target(syn_mask)
    p1 = prob_target(syn_mask | (1 << n_det))
    return p0 + p1, p0, p1


class ModifiedMLEDecoder:
    """
    For each fired EC, pick one leak in (prev scheduled EC, this EC] that can cause
    the EC; keep combos whose DEM span contains the syndrome; infer L0 by
    marginalizing P(syn, L0 | C) over valid C (MAP-optimal under 0-1 loss).

    ``decode`` returns (pred, pred) to match ErasureDecoder's (marginal, mle) shape.
    """

    def __init__(self, circuit_builder, mode: str | None = None):
        self.circuit_builder = circuit_builder
        self.mode = mode or self._infer_mode(circuit_builder)
        self._mech_cache: dict[tuple, list[tuple[int, int, float]]] = {}
        self.last_result: dict[str, Any] | None = None

    @staticmethod
    def _infer_mode(builder) -> str:
        if hasattr(builder, "swap_time"):
            return "walking"
        if hasattr(builder, "measure_sets") and len(getattr(builder, "measure_sets", [])) == 2:
            return "walking"
        return "static"

    def _is_walking(self) -> bool:
        return self.mode in ("walking", "early", "late") or (
            hasattr(self.circuit_builder, "measure_sets")
            and len(self.circuit_builder.measure_sets) == 2
        )

    def _measure_set(self, rnd: int) -> set[int]:
        b = self.circuit_builder
        if hasattr(b, "measure_sets"):
            return b.measure_sets[rnd % len(b.measure_sets)]
        cl = b.cnot_lists
        if len(cl) == 2 and cl and isinstance(cl[0][0], list):
            return set().union(*(set(cl[rnd % 2][s]) for s in range(4)))
        return set().union(*(set(cl[s]) for s in range(4)))

    def _prev_ec_time(self, ec_rnd: int, ec_gate: int, qubit: int, rounds: int, ec_sched: int) -> int:
        assert ec_sched == 8, "ModifiedMLEDecoder currently assumes ec_sched=8"
        cur = _time(ec_rnd, ec_gate)
        prev = -1
        for rnd in range(rounds):
            tt = _time(rnd, 3)
            if tt >= cur:
                break
            if qubit in self._measure_set(rnd):
                prev = tt
        return prev

    def candidates_for_ec(
        self,
        rounds: int,
        ec_rnd: int,
        ec_gate: int,
        ec_qubit: int,
        *,
        ec_sched: int = 8,
        leak_locations: str = "all",
    ) -> list[tuple[int, int, int]]:
        raw = self.circuit_builder.single_ec_traceback(
            ec_rnd, ec_gate, ec_qubit, ec_sched, leak_locations
        )
        if self._is_walking():
            raw = [loc for loc in raw if loc[2] == ec_qubit]

        prev_t = self._prev_ec_time(ec_rnd, ec_gate, ec_qubit, rounds, ec_sched)
        cur_t = _time(ec_rnd, ec_gate)
        out: list[tuple[int, int, int]] = []
        for loc in raw:
            lr, lg, lq = int(loc[0]), int(loc[1]), int(loc[2])
            if not (prev_t < _time(lr, lg) <= cur_t):
                continue
            if lq not in active_qubits_in_round(self.circuit_builder, lr):
                continue
            out.append((lr, lg, lq))
        return out

    def _circuit_kwargs(self, **kwargs) -> dict:
        ck = dict(
            basis=kwargs.get("basis", "Z"),
            Pauli_locations=kwargs.get("Pauli_locations", "gates"),
            leak_locations=kwargs.get("leak_locations", "all"),
            ec_sched=kwargs.get("ec_sched", 8),
            leak_effect=kwargs.get("leak_effect", "skip gates"),
        )
        le = ck["leak_effect"]
        if "skip" in le and "decode" not in le:
            ck["leak_effect"] = "decode " + le
        return ck

    def _leak_mechs(self, rounds: int, loc: tuple[int, int, int], **kwargs):
        ck = self._circuit_kwargs(**kwargs)
        key = (rounds, loc, tuple(sorted(ck.items())))
        if key in self._mech_cache:
            return self._mech_cache[key]
        leak = single_leak_loc(rounds, *loc)
        cir, _ = self.circuit_builder.get_circuit(
            rounds, 0.0, leakage_circuit_locations=leak, **ck
        )
        stim_dem = cir.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        mechs = _mechs_from_stim_dem(stim_dem)
        self._mech_cache[key] = mechs
        return mechs

    def find_valid_combos(self, erasure_checks, syndrome, **kwargs) -> list[list[tuple[int, int, int]]]:
        rounds = len(erasure_checks)
        syn_bits = np.asarray(syndrome, dtype=bool).reshape(-1)
        syn_mask = sum(1 << i for i, b in enumerate(syn_bits) if b)
        ec_sched = kwargs.get("ec_sched", 8)
        leak_locations = kwargs.get("leak_locations", "all")

        fired = list(iterate_erasure_checks(erasure_checks))
        if not fired:
            return [[]] if syn_mask == 0 else []

        groups: list[list[tuple[int, int, int]]] = []
        for ec_rnd, ec_gate, ec_qubit in fired:
            cands = self.candidates_for_ec(
                rounds, ec_rnd, ec_gate, ec_qubit,
                ec_sched=ec_sched, leak_locations=leak_locations,
            )
            if not cands:
                return []
            groups.append(cands)

        valid: list[list[tuple[int, int, int]]] = []
        for choice in product(*groups):
            gens: list[int] = []
            for loc in choice:
                for det_mask, _l0, _p in self._leak_mechs(rounds, loc, **kwargs):
                    if det_mask:
                        gens.append(det_mask)
            if _gf2_contains(gens, syn_mask):
                valid.append(list(choice))
        return valid

    @staticmethod
    def _combo_prior_weight(combo: list[tuple[int, int, int]], p_leak: float | None) -> float:
        """Prior mass for C. Uniform (1) if p_leak is None; else p_leak**|C|."""
        if p_leak is None:
            return 1.0
        return float(p_leak) ** len(combo)

    def decode_detailed(self, erasure_checks, syndrome, **kwargs) -> dict[str, Any]:
        rounds = len(erasure_checks)
        syn_bits = np.asarray(syndrome, dtype=bool).reshape(-1)
        n_det = len(syn_bits)
        syn_mask = sum(1 << i for i, b in enumerate(syn_bits) if b)
        p_leak = kwargs.get("p_leak", None)

        combos = self.find_valid_combos(erasure_checks, syndrome, **kwargs)
        if not combos:
            result = dict(
                combo=None, p_syn=0.0, p_syn_l0_0=0.0, p_syn_l0_1=0.0,
                z0=0.0, z1=0.0, p_l0_0=0.0, p_l0_1=0.0,
                l0=0, l0_unique=False, n_valid=0, combo_scores=[],
                pred=np.array([0], dtype=np.uint8),
            )
            self.last_result = result
            return result

        scores: list[dict[str, Any]] = []
        z0 = 0.0
        z1 = 0.0
        best_combo = None
        best_score = -1.0
        best_p_syn = 0.0
        best_p0 = 0.0
        best_p1 = 0.0

        for combo in combos:
            mechs: list[tuple[int, int, float]] = []
            for loc in combo:
                mechs.extend(self._leak_mechs(rounds, loc, **kwargs))
            p_syn, p0, p1 = _p_syn_l0(mechs, syn_mask, n_det)
            w = self._combo_prior_weight(combo, p_leak)
            z0 += w * p0
            z1 += w * p1
            score = w * p_syn
            scores.append(dict(
                combo=combo, weight=float(w),
                p_syn=float(p_syn), p_syn_l0_0=float(p0), p_syn_l0_1=float(p1),
                score=float(score),
            ))
            if score > best_score + 1e-15:
                best_score = score
                best_combo = combo
                best_p_syn = p_syn
                best_p0 = p0
                best_p1 = p1

        z_tot = z0 + z1
        if z_tot <= 0:
            l0, unique = 0, False
            p_l0_0 = p_l0_1 = 0.0
        else:
            p_l0_0 = z0 / z_tot
            p_l0_1 = z1 / z_tot
            l0 = int(p_l0_1 >= p_l0_0)
            unique = min(p_l0_0, p_l0_1) < 1e-12

        result = dict(
            combo=best_combo,
            p_syn=float(best_p_syn),
            p_syn_l0_0=float(best_p0),
            p_syn_l0_1=float(best_p1),
            z0=float(z0),
            z1=float(z1),
            p_l0_0=float(p_l0_0),
            p_l0_1=float(p_l0_1),
            l0=l0,
            l0_unique=unique,
            n_valid=len(combos),
            combo_scores=scores,
            pred=np.array([l0], dtype=np.uint8),
        )
        self.last_result = result
        return result

    def decode(
        self,
        erasure_checks,
        syndrome,
        p_Pauli=0.0,
        branch_and_bound=False,
        mle_time_limit=None,
        verbose=False,
        **circuit_kwargs,
    ):
        """Return (pred, pred) so check_for_logical_error can fill marginal/mle slots."""
        detailed = self.decode_detailed(erasure_checks, syndrome, **circuit_kwargs)
        pred = detailed["pred"]
        return pred, pred

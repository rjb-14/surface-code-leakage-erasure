# Transversal CNOT 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在两块全等的 rotated surface code patch 之间构建 transversal CNOT 电路，支持 static / walking / moonwalking 三种码型，交付到"电路可构建 + stim 确定性验证通过"。

**Architecture:** 把两块 patch 合并成**一个**平移后的 layout 对象，现有 builder 的 `__init__` 从 `layout.plaquettes` 派生 CNOT 调度，因此两块 patch 自动在同一批 TICK 里并行跑 SE，无需改动 `_measure_stabilizers_impl`。一个 `TransversalCXMixin` 负责三件事：插入 CX 层、重映射 CX 边界轮的 detector、发两个 observable。static 与 walking 各接一个薄 builder 子类。

**Tech Stack:** Python ≥3.10, stim, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-09-04-transversal-cx-design.md`

## Global Constraints

- **只允许新增文件。** 不得修改任何现有文件的既有行为。`src/surface_code_leakage_erasure/__init__.py` 与 `pyproject.toml` **只做纯追加**（新增 import / `__all__` 条目 / 可选依赖），不得改动任何已有行。
- 新增文件仅限：`src/surface_code_leakage_erasure/two_patch_layout.py`、`src/surface_code_leakage_erasure/transversal_cx.py`、`tests/`（整个目录新建）。
- **距离验证不在范围内。** 不要写 `shortest_graphlike_error()` 或任何最小重量搜索的测试（spec 7.5 论证了 DEM 含 hyperedge，该函数不适用；spec 15 记录了这一延后）。
- **解码不在范围内。** `single_ec_traceback` 覆盖为 `NotImplementedError`；不碰 `erasure_sampler.py`、`decoding.py`、`modified_mle.py`。
- 实验语义固定：两块 patch 同步，`rounds = 2R`，CX 插在 round index `R = rounds // 2` 开始之前；两块均初始化到 `|+>`（`basis="X"`）；observable 0 = control 的 X logical，observable 1 = target 的 X logical。
- 平移量：`coord_shift = Coord(2*d + 2, 0)`，`index_shift = len(layout.qubits)`。合并后 `index < index_shift` ⟺ patch 1。
- 检测器约定（spec 7.2）：CX 后第一轮里，patch 1 的 X plaquette 与 patch 2 的 Z plaquette 的 detector 是 3 项，**第三项是配对 ancilla 在 round `R-1`（CX 前）的记录**。
- 每个 task 结束时提交。提交信息末尾加：
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/surface_code_leakage_erasure/two_patch_layout.py` | 纯 layout 层。把一个已构造的 layout 平移成第二份并与原份合并；暴露 `pair()` 与 per-patch logical。不含任何电路/噪声逻辑。 |
| `src/surface_code_leakage_erasure/transversal_cx.py` | 纯 builder 层。`TransversalCXMixin` + 两个具体 builder。不含 layout 构造逻辑。 |
| `tests/test_two_patch_layout.py` | layout 层单元测试 |
| `tests/test_transversal_cx.py` | builder 层与端到端测试 |
| `tests/test_regression_existing.py` | 现有 builder 未被改动的守卫 |

---

## Task 1: 测试脚手架与现有行为守卫

先建守卫再动任何代码——"只允许加入"这条约束必须是**可验证的**，不是口头保证。

**Files:**
- Create: `tests/__init__.py`（空文件）
- Create: `tests/test_regression_existing.py`
- Modify: `pyproject.toml`（**纯追加**一个 `dev` 可选依赖组）

**Interfaces:**
- Consumes: 无
- Produces: 可运行的 pytest 环境；后续所有 task 的回归守卫

- [ ] **Step 1: 追加 pytest 依赖**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 段**末尾追加**（不要改动已有的 `analysis` 组）：

```toml
# test suite only; the core simulation does not need this
dev = [
    "pytest",
]
```

- [ ] **Step 2: 建空的 tests 包**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 3: 写现有行为的守卫测试**

这些摘要是在**改动前**从当前 `main`/`runjiang/stability` 代码实测得到的。它们必须一直通过。

```python
# tests/test_regression_existing.py
"""Guard: the transversal-CX work must not change any existing builder's output.

The SHA-256 digests below were captured from the pre-change code. If one of
these fails, an existing file's behaviour was modified -- which the plan's
global constraints forbid.
"""
import hashlib

import pytest

from surface_code_leakage_erasure import (
    SurfaceCodeCircuitBuilder,
    WalkingSCCircuitBuilder,
)


def _digest(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    "make_builder, basis, expected",
    [
        (lambda: SurfaceCodeCircuitBuilder(3, seed=0), "Z", "d76cb5742b2c962a"),
        (lambda: SurfaceCodeCircuitBuilder(3, seed=0), "X", "85d0f9ba19df660a"),
        (lambda: WalkingSCCircuitBuilder(3, "late", seed=0), "X", "ed68fb4d86e8594c"),
        (lambda: WalkingSCCircuitBuilder(3, "early", seed=0), "X", "61db578ce50e5282"),
    ],
)
def test_existing_builders_unchanged(make_builder, basis, expected):
    circuit, _ = make_builder().get_circuit(4, 1e-3, 0.0, basis=basis)
    assert _digest(circuit) == expected


@pytest.mark.parametrize(
    "make_builder",
    [
        lambda: SurfaceCodeCircuitBuilder(3, seed=0),
        lambda: WalkingSCCircuitBuilder(3, "late", seed=0),
        lambda: WalkingSCCircuitBuilder(3, "early", seed=0),
    ],
)
def test_existing_builders_single_observable(make_builder):
    circuit, _ = make_builder().get_circuit(4, 1e-3, 0.0, basis="X")
    assert circuit.num_observables == 1
```

- [ ] **Step 4: 运行，确认全部通过**

Run: `python -m pytest tests/test_regression_existing.py -v`
Expected: 6 passed。**若有任何一条失败，停下**——说明基线摘要与当前代码不符，先查清原因再继续。

- [ ] **Step 5: 提交**

```bash
git add tests/ pyproject.toml
git commit -m "test: guard existing builder output before adding transversal CX

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: layout 平移与合并（static）

**Files:**
- Create: `src/surface_code_leakage_erasure/two_patch_layout.py`
- Create: `tests/test_two_patch_layout.py`

**Interfaces:**
- Consumes: `surface_code.Coord/Qubit/DataQubit/Plaquette/SurfaceCodeLayout`
- Produces:
  - `shift_and_merge(layout, coord_shift: Coord, index_shift: int) -> None`（就地改写 `layout`）
  - `TwoPatchSurfaceCodeLayout(d)`，属性 `coord_shift: Coord`、`index_shift: int`、`x_logical_per_patch: list`、`z_logical_per_patch: list`、方法 `pair(qubit) -> Qubit`、`patch_of(index) -> int`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_two_patch_layout.py
import pytest

from surface_code_leakage_erasure.surface_code import Coord, DataQubit
from surface_code_leakage_erasure.two_patch_layout import TwoPatchSurfaceCodeLayout


@pytest.fixture
def layout():
    return TwoPatchSurfaceCodeLayout(3)


def test_qubit_count_doubles(layout):
    assert len(layout.qubits) == 2 * 17          # d=3 single patch has 17 qubits
    assert len(layout.data_qubits) == 2 * 9
    assert len(layout.plaquettes) == 2 * 8


def test_index_ranges_split_by_patch(layout):
    assert layout.index_shift == 17
    indexes = sorted(q.index for q in layout.qubits)
    assert indexes == list(range(34))
    assert all(layout.patch_of(i) == 1 for i in range(17))
    assert all(layout.patch_of(i) == 2 for i in range(17, 34))


def test_patches_do_not_overlap_in_space(layout):
    xs1 = {q.coord[0] for q in layout.qubits if layout.patch_of(q.index) == 1}
    xs2 = {q.coord[0] for q in layout.qubits if layout.patch_of(q.index) == 2}
    assert max(xs1) < min(xs2)


def test_pair_is_a_translation_bijection(layout):
    patch1 = [q for q in layout.qubits if layout.patch_of(q.index) == 1]
    assert len(patch1) == 17
    paired = set()
    for q in patch1:
        p = layout.pair(q)
        assert p.coord == q.coord + layout.coord_shift
        assert p.index == q.index + layout.index_shift
        assert type(p) is type(q)
        paired.add(p.index)
    assert paired == set(range(17, 34))


def test_per_patch_logicals_are_disjoint_and_sized(layout):
    a, b = layout.x_logical_per_patch
    assert len(a) == 3 and len(b) == 3           # d=3 logical row
    assert {q.index for q in a}.isdisjoint({q.index for q in b})
    # the merged x_logical must not silently be the union of both patches
    assert len(layout.x_logical) == 3


def test_shifted_plaquettes_reference_shifted_qubits(layout):
    for plaq in layout.plaquettes:
        patch = layout.patch_of(plaq.ancilla.index)
        for dq in plaq.ordered_data_qubits:
            if dq is not None:
                assert layout.patch_of(dq.index) == patch


def test_back_references_are_not_shared_between_patches(layout):
    for dq in layout.data_qubits:
        if not isinstance(dq, DataQubit):
            continue
        for plaq in dq.ordered_plaquettes:
            if plaq is not None:
                assert layout.patch_of(plaq.ancilla.index) == layout.patch_of(dq.index)
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_two_patch_layout.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'surface_code_leakage_erasure.two_patch_layout'`

- [ ] **Step 3: 实现 two_patch_layout.py**

```python
# src/surface_code_leakage_erasure/two_patch_layout.py
"""Build a two-patch layout by translating an existing layout and merging it in.

The translation acts on a layout's *output structures* (the constructed Qubit
and Plaquette objects and the containers holding them) rather than on the
layout methods that produce them. Those methods hard-code boundary coordinates
(`SurfaceCodeLayout.layout_ancillas`, `WalkingSurfaceCodeLayout.layout_plaquettes`),
so threading an offset through them would be an invasive rewrite -- twice, once
per code family. Working on the outputs gives one implementation that serves the
static, walking and moonwalking layouts alike.
"""

from .surface_code import Coord, DataQubit, Plaquette, Qubit, SurfaceCodeLayout

# Attributes that describe the layout as a whole rather than one patch's
# contents. They are identical for both patches and must be left alone --
# shifting `d` or concatenating `swap_time` would be nonsense.
_SCALAR_ATTRS = frozenset({"d", "swap_time"})

# Logical operators are kept per patch instead of merged: merging them would
# produce the product of the two patches' logicals, but the experiment needs
# two independent observables.
_LOGICAL_ATTRS = frozenset({"z_logical", "x_logical"})


def _collect(value, out, kind):
    """Gather every instance of `kind` reachable from `value` into the set `out`."""
    if isinstance(value, kind):
        out.add(value)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _collect(k, out, kind)
            _collect(v, out, kind)
        return
    # Coord is a tuple subclass with no nested payload; skip it explicitly so we
    # don't iterate its two ints.
    if isinstance(value, Coord):
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _collect(v, out, kind)


def _layout_attrs(layout):
    """The layout's own data attributes, excluding the scalar descriptors."""
    return {k: v for k, v in vars(layout).items() if k not in _SCALAR_ATTRS}


def _build_qubit_map(layout, coord_shift, index_shift):
    qubits = set()
    for value in _layout_attrs(layout).values():
        _collect(value, qubits, Qubit)
    qmap = {}
    for q in qubits:
        cls = DataQubit if isinstance(q, DataQubit) else Qubit
        qmap[q] = cls(q.index + index_shift, q.coord + coord_shift)
    return qmap


def _build_plaquette_map(layout, qmap, coord_shift):
    plaquettes = set()
    for value in _layout_attrs(layout).values():
        _collect(value, plaquettes, Plaquette)
    pmap = {}
    for plaq in plaquettes:
        ordered = [None if dq is None else qmap[dq] for dq in plaq.ordered_data_qubits]
        pmap[plaq] = Plaquette(
            plaq.type, ordered, qmap[plaq.ancilla], plaq.coord + coord_shift)
    # Rebuild the DataQubit -> Plaquette back references on the shifted copies,
    # so the two patches never share a Plaquette object.
    for plaq in plaquettes:
        new_plaq = pmap[plaq]
        for t, dq in enumerate(plaq.ordered_data_qubits):
            if isinstance(dq, DataQubit):
                qmap[dq].ordered_plaquettes[t] = new_plaq
    return pmap


def _shift(value, qmap, pmap, coord_shift, index_shift):
    """Translate one attribute value onto patch 2."""
    # Order matters: Plaquette/Qubit before the generic containers, Coord before
    # tuple (Coord subclasses tuple), bool before int (bool subclasses int).
    if isinstance(value, Plaquette):
        return pmap[value]
    if isinstance(value, Qubit):
        return qmap[value]
    if isinstance(value, Coord):
        return value + coord_shift
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value + index_shift
    args = (qmap, pmap, coord_shift, index_shift)
    if isinstance(value, dict):
        return {_shift(k, *args): _shift(v, *args) for k, v in value.items()}
    if isinstance(value, frozenset):
        return frozenset(_shift(v, *args) for v in value)
    if isinstance(value, set):
        return {_shift(v, *args) for v in value}
    if isinstance(value, tuple):
        return tuple(_shift(v, *args) for v in value)
    if isinstance(value, list):
        return [_shift(v, *args) for v in value]
    raise TypeError(f"don't know how to shift {type(value).__name__}")


def _merge_leaf(a, b):
    if isinstance(a, frozenset):
        return a | b
    if isinstance(a, set):
        return a | b
    if isinstance(a, dict):
        return {**a, **b}
    if isinstance(a, (list, tuple)):
        return a + b
    raise TypeError(f"don't know how to merge {type(a).__name__}")


def _merge(a, b):
    """Merge patch 1's attribute value with patch 2's.

    A list-valued attribute is a per-round schedule (walking layouts index every
    such list by `rnd % 2`), so the two patches merge element by element. Every
    other container merges directly.
    """
    if isinstance(a, list):
        return [_merge_leaf(x, y) for x, y in zip(a, b)]
    return _merge_leaf(a, b)


def shift_and_merge(layout, coord_shift: Coord, index_shift: int):
    """Add a translated copy of `layout` to itself, in place.

    `layout.z_logical` / `layout.x_logical` are NOT merged: they are replaced by
    patch 1's values, and both patches' versions are exposed as
    `z_logical_per_patch` / `x_logical_per_patch`.
    """
    qmap = _build_qubit_map(layout, coord_shift, index_shift)
    pmap = _build_plaquette_map(layout, qmap, coord_shift)

    originals = _layout_attrs(layout)
    shifted = {
        name: _shift(value, qmap, pmap, coord_shift, index_shift)
        for name, value in originals.items()
    }

    for name, value in originals.items():
        if name in _LOGICAL_ATTRS:
            setattr(layout, f"{name}_per_patch", [value, shifted[name]])
            continue
        setattr(layout, name, _merge(value, shifted[name]))

    layout.coord_shift = coord_shift
    layout.index_shift = index_shift


class TwoPatchMixin:
    """Shared accessors for a layout holding two translated patches."""

    def pair(self, qubit):
        """Patch 2's counterpart of a patch-1 qubit (and vice versa is not defined)."""
        return self.coord_to_qubit[qubit.coord + self.coord_shift]

    def patch_of(self, index: int) -> int:
        """1 for the control patch, 2 for the target patch."""
        return 1 if index < self.index_shift else 2


class TwoPatchSurfaceCodeLayout(TwoPatchMixin, SurfaceCodeLayout):
    """Two translated copies of the static rotated surface code layout."""

    def define_layout(self):
        super().define_layout()
        shift_and_merge(
            self, coord_shift=Coord(2 * self.d + 2, 0),
            index_shift=len(self.qubits))
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_two_patch_layout.py -v`
Expected: 7 passed

- [ ] **Step 5: 确认现有行为守卫仍然通过**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/two_patch_layout.py tests/test_two_patch_layout.py
git commit -m "feat: add two-patch static surface code layout

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: layout 平移与合并（walking / moonwalking）

walking layout 有 25 个属性，形状比 static 复杂得多（按 `rnd%2` 索引的列表、索引元组构成的 detector 表）。Task 2 的通用平移器设计上就是为了同时吃下它，本 task 验证这一点。

**Files:**
- Modify: `src/surface_code_leakage_erasure/two_patch_layout.py`（追加 `TwoPatchWalkingLayout`）
- Modify: `tests/test_two_patch_layout.py`（追加 walking 用例）

**Interfaces:**
- Consumes: Task 2 的 `shift_and_merge`、`TwoPatchMixin`
- Produces: `TwoPatchWalkingLayout(d, swap_time="late")`，同样暴露 `coord_shift`、`index_shift`、`pair`、`patch_of`、`x_logical_per_patch`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_two_patch_layout.py`：

```python
from surface_code_leakage_erasure.two_patch_layout import TwoPatchWalkingLayout


@pytest.fixture(params=["late", "early"])
def walking_layout(request):
    return TwoPatchWalkingLayout(3, swap_time=request.param)


def test_walking_qubit_count_doubles(walking_layout):
    assert len(walking_layout.qubits) == 2 * 22   # d=3 walking patch has 22 qubits
    assert walking_layout.index_shift == 22
    assert sorted(q.index for q in walking_layout.qubits) == list(range(44))


def test_walking_per_round_lists_merge_elementwise(walking_layout):
    # every per-round schedule stays length 2 (indexed by rnd % 2) and doubles in
    # content, rather than becoming a 4-element list
    for name in ("data_qubits", "z_plaquettes", "x_plaquettes", "plaquettes",
                 "reset_indexes", "rx_indexes", "measure_indexes", "mx_indexes",
                 "detectors"):
        value = getattr(walking_layout, name)
        assert isinstance(value, list) and len(value) == 2, name


def test_walking_detectors_come_from_both_patches(walking_layout):
    for round_detectors in walking_layout.detectors:
        patches = {walking_layout.patch_of(idx)
                   for index_tuple in round_detectors for idx in index_tuple}
        assert patches == {1, 2}


def test_walking_scalars_untouched(walking_layout):
    assert walking_layout.d == 3
    assert walking_layout.swap_time in ("late", "early")


def test_walking_per_patch_logicals_disjoint(walking_layout):
    a, b = walking_layout.x_logical_per_patch
    idx_a = {i for group in a for i in group}
    idx_b = {i for group in b for i in group}
    assert idx_a and idx_b
    assert idx_a.isdisjoint(idx_b)
    assert all(walking_layout.patch_of(i) == 1 for i in idx_a)
    assert all(walking_layout.patch_of(i) == 2 for i in idx_b)


def test_walking_pair_is_translation(walking_layout):
    patch1 = [q for q in walking_layout.qubits
              if walking_layout.patch_of(q.index) == 1]
    for q in patch1:
        p = walking_layout.pair(q)
        assert p.index == q.index + walking_layout.index_shift
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_two_patch_layout.py -v -k walking`
Expected: FAIL，`ImportError: cannot import name 'TwoPatchWalkingLayout'`

- [ ] **Step 3: 实现**

在 `two_patch_layout.py` 顶部的 import 追加：

```python
from .walking_surface_code import WalkingSurfaceCodeLayout
```

文件末尾追加：

```python
class TwoPatchWalkingLayout(TwoPatchMixin, WalkingSurfaceCodeLayout):
    """Two translated copies of the walking / moonwalking layout.

    `swap_time="late"` gives the walking code, `"early"` the moonwalking code --
    the same split the single-patch layout uses.
    """

    def define_layout(self, swap_time):
        super().define_layout(swap_time)
        shift_and_merge(
            self, coord_shift=Coord(2 * self.d + 2, 0),
            index_shift=len(self.qubits))
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_two_patch_layout.py -v`
Expected: 全部通过（static 7 条 + walking 6 条 × 2 个 `swap_time` 参数）

若 `_shift` 抛 `TypeError: don't know how to shift ...`，说明 walking layout 有一个 Task 2 未覆盖的属性类型。把该类型加进 `_shift` 的分支，**不要**用 `except TypeError: return value` 兜底——静默跳过会产生一个两 patch 共享对象的 layout。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/two_patch_layout.py tests/test_two_patch_layout.py
git commit -m "feat: add two-patch walking and moonwalking layouts

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: 双 patch 并行电路（尚无 CX）

这是关键的中间检查点：它把"合并后的 layout 能否喂给现有 builder"与"CX 逻辑是否正确"**分离开**。此时电路是两块独立 patch 的 X memory，各自一个 observable，必然确定——若这一步就不确定，问题在 layout 而不在 CX。

**Files:**
- Create: `src/surface_code_leakage_erasure/transversal_cx.py`
- Create: `tests/test_transversal_cx.py`

**Interfaces:**
- Consumes: Task 2 的 `TwoPatchSurfaceCodeLayout`
- Produces:
  - `TransversalCXMixin`，属性 `cx_round: int`；方法 `final_observable(rounds, basis)`、`_observable_recs(patch) -> list[int]`（子类实现）、`_observable_cache_key(rounds, basis)`、`single_ec_traceback(...)`
  - `TransversalCXCircuitBuilder(d, seed=None)`，`get_circuit(rounds, p, p_leak=0.0, **kwargs) -> (stim.Circuit, erasure_checks)`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_transversal_cx.py
import pytest
import stim

from surface_code_leakage_erasure.transversal_cx import TransversalCXCircuitBuilder


def build(rounds=4, d=3, p=0.0, p_leak=0.0, **kwargs):
    builder = TransversalCXCircuitBuilder(d, seed=0)
    circuit, _ = builder.get_circuit(rounds, p, p_leak, basis="X", **kwargs)
    return builder, circuit


def test_two_observables():
    _, circuit = build()
    assert circuit.num_observables == 2


def test_qubit_count_is_two_patches():
    _, circuit = build()
    assert circuit.num_qubits == 2 * 17


def test_noiseless_circuit_is_deterministic():
    # stim raises if any declared detector or observable is not deterministic
    _, circuit = build()
    circuit.detector_error_model()


def test_odd_rounds_rejected():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    with pytest.raises(ValueError):
        builder.get_circuit(5, 0.0, 0.0, basis="X")


def test_cx_round_is_half_the_rounds():
    builder, _ = build(rounds=6)
    assert builder.cx_round == 3


def test_traceback_is_out_of_scope():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    with pytest.raises(NotImplementedError):
        builder.single_ec_traceback(0, 0, 0, 8)
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_transversal_cx.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'surface_code_leakage_erasure.transversal_cx'`

- [ ] **Step 3: 实现 transversal_cx.py**

本 step **不实现 CX 层，也不实现 detector 重映射**——只让两块 patch 并行跑并发两个 observable。

```python
# src/surface_code_leakage_erasure/transversal_cx.py
"""Transversal CNOT between two rotated surface code patches.

Patch 1 is the control and patch 2 the target. Both are prepared in |+>, run
`rounds // 2` rounds of stabilizer extraction, take a transversal CX, run
`rounds // 2` more, and are measured out in the X basis. Observable 0 is the
control's X logical and observable 1 the target's.

Everything here is an override on top of the existing builders: no existing file
changes behaviour. See docs/superpowers/specs/2026-09-04-transversal-cx-design.md
section 11 for why each override needs no base-class edit.
"""

from .circuit_builder import SurfaceCodeCircuitBuilder
from .two_patch_layout import TwoPatchSurfaceCodeLayout


class TransversalCXMixin:
    """Adds the CX layer, its detector remapping, and the two observables."""

    # A detector-phase key the base builders never produce. Their _get_detector_text
    # is keyed by round index (0, 1, 2, ...), so a negative key cannot collide with
    # an ordinary round -- which matters because the CX round's index CAN collide
    # with a cache key, see _init_transversal_cx_state.
    _CX_PHASE = -1

    def _init_transversal_cx_state(self):
        """Call at the end of a concrete builder's __init__."""
        self.cx_round = None
        # True while an outer, real round is being emitted. measure_stabilizers is
        # also called re-entrantly by measure_stabilizers_no_leakage to rebuild a
        # cache miss, and there its `rnd` argument is an rnd_eff cache key (0/1 for
        # static, 0/1/2 for walking), NOT a round index. Those keys collide with
        # cx_round whenever rounds <= 4, so the re-entrant call must never be
        # compared against cx_round.
        self._in_round = False
        # Whether the round currently being emitted is the CX round. Read by
        # define_detectors rather than re-deriving it from `rnd`, for the same reason.
        self._cx_detectors_active = False
        self._cx_leak_locs = set()

    def get_circuit(self, rounds: int, p: float, p_leak: float = 0.0,
                    leakage_circuit_locations: dict | None = None,
                    use_cache=True, **circuit_kwargs):
        if rounds % 2 != 0:
            raise ValueError(
                "rounds must be even for a transversal CX experiment: the CX sits "
                f"between two halves of rounds // 2 rounds each (got {rounds}).")
        self.cx_round = rounds // 2

        return super().get_circuit(
            rounds, p, p_leak,
            leakage_circuit_locations=leakage_circuit_locations,
            use_cache=use_cache, **circuit_kwargs)

    def _observable_cache_key(self, rounds, basis):
        # The CX position depends on `rounds`, so the observable text cannot be
        # cached across different round counts.
        return (rounds, basis)

    def final_observable(self, rounds, basis: str = "Z"):
        """Two observables: the control's X logical, then the target's."""
        for observable, patch in enumerate((1, 2)):
            recs = " ".join(f"rec[{idx}]" for idx in self._observable_recs(patch))
            self.circuit.append(f"OBSERVABLE_INCLUDE({observable}) {recs}")

    def _observable_recs(self, patch: int) -> list[int]:
        raise NotImplementedError("_observable_recs must be implemented in subclass")

    def single_ec_traceback(self, *args, **kwargs):
        raise NotImplementedError(
            "erasure-check traceback is not implemented for the transversal CX "
            "circuit; decoding is out of scope for this builder.")


class TransversalCXCircuitBuilder(TransversalCXMixin, SurfaceCodeCircuitBuilder):
    """Transversal CX between two static rotated surface code patches."""

    def __init__(self, d, seed=None):
        layout = d if isinstance(d, TwoPatchSurfaceCodeLayout) \
            else TwoPatchSurfaceCodeLayout(d)
        super().__init__(layout, seed)
        self._init_transversal_cx_state()

        # The base class sizes the erasure-swap ancilla pool as `d + 1`, which is
        # one more than a single patch needs. Two patches need 2*d of them, so the
        # base's zip() would silently drop the excess data qubits and leave them
        # without a swap partner. Rebuild the pool at the right size.
        remaining = sorted(set(self.dq_indexes) - set(self.erasure_swaps.keys()))
        self.erasure_ancillas = list(
            range(self.n_qubits, self.n_qubits + len(remaining)))
        self.erasure_swaps.update(zip(remaining, self.erasure_ancillas))
        self.erasure_unswaps.update(zip(self.erasure_ancillas, remaining))

    def __repr__(self):
        return f"TransversalCXCircuitBuilder(d={self.layout.d})"

    def _observable_recs(self, patch: int) -> list[int]:
        observable = self.layout.x_logical_per_patch[patch - 1]
        return [self.tracker.get_curr_meas(q.index) for q in sorted(observable)]
```

**注意**：基类 `measure_stabilizers` 传给 `_measure_stabilizers_impl` 的 `measure_set` 是 `self.stab_ancillas_set | set(self.erasure_ancillas)`，在 `__init__` 里重建 `erasure_ancillas` 之后这个表达式每轮重新求值，会自动用上新的池子，无需额外处理。

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_transversal_cx.py -v`
Expected: 6 passed

若 `test_noiseless_circuit_is_deterministic` 失败，问题在 Task 2 的 layout 合并（此时还没有任何 CX 逻辑），回到 Task 2 排查，**不要**在本文件里打补丁绕过。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/transversal_cx.py tests/test_transversal_cx.py
git commit -m "feat: add two-patch parallel circuit with two X observables

No transversal CX yet -- this isolates the merged layout from the CX logic.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: 插入 transversal CX 层

**Files:**
- Modify: `src/surface_code_leakage_erasure/transversal_cx.py`
- Modify: `tests/test_transversal_cx.py`

**Interfaces:**
- Consumes: Task 4 的 `TransversalCXMixin`
- Produces: `TransversalCXMixin.measure_stabilizers(...)`、`_cx_pairs() -> list[tuple[int, int]]`、`_emit_cx_layer(p, Pauli_locations)`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_transversal_cx.py`：

```python
def _cross_patch_cx_layers(circuit, index_shift):
    """Indices of the circuit instructions that are a cross-patch CX layer."""
    found = []
    for i, inst in enumerate(circuit):
        if inst.name != "CX":
            continue
        targets = [t.value for t in inst.targets_copy()]
        pairs = list(zip(targets[::2], targets[1::2]))
        if all(a < index_shift <= b for a, b in pairs):
            found.append((i, pairs))
    return found


def test_cx_layer_appears_exactly_once():
    builder, circuit = build(rounds=4)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1


def test_cx_layer_pairs_every_data_qubit():
    builder, circuit = build(rounds=4)
    (_, pairs), = _cross_patch_cx_layers(
        circuit.flattened(), builder.layout.index_shift)
    assert len(pairs) == 9                      # d=3 -> d**2 data qubits
    shift = builder.layout.index_shift
    assert all(b == a + shift for a, b in pairs)


@pytest.mark.parametrize("p_leak", [0.0, 0.05])
def test_cx_layer_not_duplicated_by_the_no_leakage_cache(p_leak):
    # p_leak=0 is the path that goes through measure_stabilizers_no_leakage, whose
    # cache-miss rebuild re-enters measure_stabilizers. If the CX round is not
    # excluded from that cache, the CX layer is emitted twice and the second copy
    # lands inside the cached text.
    builder, circuit = build(rounds=6, p=1e-3, p_leak=p_leak)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1


@pytest.mark.parametrize("rounds", [2, 4, 6, 8])
def test_cx_layer_appears_once_for_every_round_count(rounds):
    # rounds <= 4 make cx_round collide with an rnd_eff cache key (0/1 static),
    # which is exactly when a cache rebuild can be mistaken for the CX round.
    builder, circuit = build(rounds=rounds, p=1e-3, p_leak=0.0)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1


def test_cx_layer_survives_a_second_build_from_the_same_builder():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    for _ in range(2):
        circuit, _ = builder.get_circuit(4, 1e-3, 0.0, basis="X")
        layers = _cross_patch_cx_layers(
            circuit.flattened(), builder.layout.index_shift)
        assert len(layers) == 1
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_transversal_cx.py -v -k cx_layer`
Expected: 4 个用例 FAIL（`assert 0 == 1`——还没有任何 CX 层）

- [ ] **Step 3: 实现 CX 层**

在 `TransversalCXMixin` 中追加：

```python
    def measure_stabilizers(self, p, rnd: int, **kwargs):
        """Emit one round, preceded by the CX layer when this is the CX round.

        Two independent guards are needed here; neither replaces the other.

        `_in_round` distinguishes a real round from the re-entrant call that
        measure_stabilizers_no_leakage makes to rebuild a cache miss. That call
        passes an rnd_eff cache key (0/1 static, 0/1/2 walking) as `rnd`, which
        equals cx_round whenever rounds <= 4 -- for walking at rounds=4,
        cx_round is 2 and round 3 rebuilds under rnd_eff=2. Without this guard
        that rebuild would emit a CX layer, and the special CX detectors, into
        the shared cache entry.

        `use_cache=False` on the CX round keeps that round's 3-rec detectors out
        of the shared rnd_eff cache, where a later ordinary round with the same
        rnd_eff would pick them up.
        """
        if self._in_round:
            return super().measure_stabilizers(p, rnd, **kwargs)

        self._in_round = True
        self._cx_detectors_active = (rnd == self.cx_round)
        try:
            if self._cx_detectors_active:
                self._emit_cx_layer(p, kwargs["Pauli_locations"])
                kwargs["use_cache"] = False
            return super().measure_stabilizers(p, rnd, **kwargs)
        finally:
            self._in_round = False
            self._cx_detectors_active = False

    def _cx_data_qubits(self) -> list[int]:
        """Patch-1 data qubit indices holding the logical state at the CX."""
        raise NotImplementedError("_cx_data_qubits must be implemented in subclass")

    def _cx_pairs(self) -> list[tuple[int, int]]:
        shift = self.layout.index_shift
        return [(q, q + shift) for q in self._cx_data_qubits()]

    def _emit_cx_layer(self, p: float, Pauli_locations: str):
        """One transversal CX between the patches, occupying its own TICK.

        The instruction order mirrors make_stabilizer_gates: gates first, then
        Pauli noise. Leakage is added in a later task.
        """
        pairs = self._cx_pairs()
        gate_targets = " ".join(f"{q:3d}" for pair in pairs for q in pair)
        instructions = [f"CX {gate_targets}"]

        if p > 0.0:
            instructions.append(f"DEPOLARIZE2({p}) {gate_targets}")
            if Pauli_locations == "all":
                gated = {q for pair in pairs for q in pair}
                idling = sorted(self.qubit_set - gated)
                if idling:
                    instructions.append(
                        f"DEPOLARIZE1({p}) {' '.join(str(q) for q in idling)}")

        self.circuit.append("\n".join(instructions))
        self.circuit.append("TICK")
```

在 `TransversalCXCircuitBuilder` 中追加：

```python
    def _cx_data_qubits(self) -> list[int]:
        # The static code never moves its data qubits, so every patch-1 data
        # qubit holds state at the CX.
        shift = self.layout.index_shift
        return [q for q in self.dq_indexes if q < shift]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_transversal_cx.py -v`
Expected: 全部通过，**但 `test_noiseless_circuit_is_deterministic` 会失败**——CX 已经插入而 detector 尚未重映射，这正是预期。把该用例临时标记：

```python
@pytest.mark.xfail(reason="detector remapping lands in Task 6", strict=True)
def test_noiseless_circuit_is_deterministic():
    ...
```

`strict=True` 保证 Task 6 完成后这个 xfail 会因"意外通过"而报错，提醒把标记摘掉。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过（含 1 个 xfail）

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/transversal_cx.py tests/test_transversal_cx.py
git commit -m "feat: emit the transversal CX layer, guarded against cache re-entry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: CX 边界轮的 detector 重映射（static）

本 task 让确定性测试重新通过——它是整个设计的正确性判据。

**Files:**
- Modify: `src/surface_code_leakage_erasure/transversal_cx.py`
- Modify: `tests/test_transversal_cx.py`

**Interfaces:**
- Consumes: Task 5
- Produces: `TransversalCXMixin._get_detector_text(rnd, basis)`、`TransversalCXCircuitBuilder._cx_round_detector_text(basis)`

- [ ] **Step 1: 写失败的测试**

把 Task 5 加的 `xfail` 标记从 `test_noiseless_circuit_is_deterministic` 上摘掉，并追加：

```python
def _detector_sizes_per_round(circuit):
    """Number of rec targets in each DETECTOR, grouped by SHIFT_COORDS round."""
    rounds = [[]]
    for inst in circuit.flattened():
        if inst.name == "SHIFT_COORDS":
            rounds.append([])
        elif inst.name == "DETECTOR":
            rounds[-1].append(len(inst.targets_copy()))
    return rounds


@pytest.mark.parametrize("rounds", [4, 6])
def test_deterministic_with_cx(rounds):
    _, circuit = build(rounds=rounds)
    circuit.detector_error_model()


def test_cx_round_has_three_rec_detectors():
    builder, circuit = build(rounds=4)
    per_round = _detector_sizes_per_round(circuit)
    cx_round_sizes = per_round[builder.cx_round]
    # one 3-rec detector per patch-1 X plaquette and per patch-2 Z plaquette
    assert sorted(cx_round_sizes).count(3) == 2 * 4      # d=3: 4 X + 4 Z plaquettes
    assert set(cx_round_sizes) == {2, 3}


@pytest.mark.parametrize("rounds", [4, 8])
def test_rounds_after_the_cx_are_ordinary(rounds):
    builder, circuit = build(rounds=rounds)
    per_round = _detector_sizes_per_round(circuit)
    for rnd in range(builder.cx_round + 1, rounds):
        assert set(per_round[rnd]) == {2}, f"round {rnd} has non-ordinary detectors"


def test_rounds_before_the_cx_are_ordinary():
    builder, circuit = build(rounds=8)
    per_round = _detector_sizes_per_round(circuit)
    for rnd in range(1, builder.cx_round):
        assert set(per_round[rnd]) == {2}, f"round {rnd} has non-ordinary detectors"
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_transversal_cx.py -v -k "deterministic or cx_round or ordinary"`
Expected: FAIL——`test_deterministic_with_cx` 抛 stim 的非确定性错误，`test_cx_round_has_three_rec_detectors` 断言失败

- [ ] **Step 3: 实现 detector 重映射**

在 `TransversalCXMixin` 中追加：

```python
    def define_detectors(self, rnd: int, basis: str = "Z"):
        """Pick the detector phase from the flag, never from `rnd`.

        `rnd` may be an rnd_eff cache key rather than a round index (see
        measure_stabilizers), so it cannot be compared against cx_round here.
        _CX_PHASE also gives the CX round its own _get_detector_text cache entry,
        and puts it out of reach of the walking builder's `rnd - 2` recursion --
        that recursion walks 6 -> 4 -> 2 and can never reach a negative key.
        """
        key = self._CX_PHASE if self._cx_detectors_active else rnd
        self.circuit.append(self._get_detector_text(key, basis))

    def _get_detector_text(self, rnd: int, basis: str):
        if rnd == self._CX_PHASE:
            return self._cx_round_detector_text(basis)
        return super()._get_detector_text(rnd, basis)

    def _cx_round_detector_text(self, basis: str):
        raise NotImplementedError(
            "_cx_round_detector_text must be implemented in subclass")
```

在 `TransversalCXCircuitBuilder` 中追加：

```python
    def _cx_round_detector_text(self, basis: str):
        """Detectors for the round right after the transversal CX.

        Conjugating by the CX gives
            X-plaq_1(P) -> X-plaq_1(P) . X-plaq_2(P_bar)
            Z-plaq_2(P) -> Z-plaq_1(P_bar) . Z-plaq_2(P)
        so the control's X checks and the target's Z checks each pick up their
        partner's ancilla -- from the round BEFORE the CX, which is what keeps a
        single-qubit error after the CX flipping only two detectors. See spec
        sections 7.1-7.4.
        """
        shift = self.layout.index_shift
        recs_list = []
        for plaquette in self.plaquettes:
            patch = self.layout.patch_of(plaquette.index)
            idxs = [self.tracker.get_curr_meas(plaquette.index),
                    self.tracker.get_prev_meas(plaquette.index)]
            crosses = (patch == 1 and plaquette.type == "X") or \
                      (patch == 2 and plaquette.type == "Z")
            if crosses:
                partner = plaquette.index + shift if patch == 1 \
                    else plaquette.index - shift
                idxs.append(self.tracker.get_prev_meas(partner))
            recs_list.append(" ".join(f"rec[{idx}]" for idx in idxs))
        return "\n".join(f"DETECTOR {recs}" for recs in recs_list)
```

**为什么配对 ancilla 的索引就是 `index ± shift`**：两块 patch 是纯平移的全等副本，`shift_and_merge` 给 patch 2 的每个对象加上同一个 `index_shift`，所以 patch 1 的 ancilla `i` 与 patch 2 的 ancilla `i + shift` 位于对应位置。

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_transversal_cx.py -v`
Expected: 全部通过，无 xfail

若 `test_deterministic_with_cx` 仍失败，stim 的报错会指出**哪一个** detector 不确定。核对两点：(a) 交叉项用的是 `get_prev_meas` 而非 `get_curr_meas`；(b) `crosses` 的分支是"patch 1 的 X"与"patch 2 的 Z"，不是反过来。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/transversal_cx.py tests/test_transversal_cx.py
git commit -m "feat: remap detectors across the transversal CX boundary

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: walking / moonwalking builder

**本 task 是整个计划的风险集中点，含两个必须先解决的未知。不要跳过 Step 1 直接写实现。**

**未知 A —— walking detector 元组的分类。** spec 7.2 的规则用 plaquette 表述
（"patch 1 的 X check 吸收 target 的"），但 walking 的 detector 是 layout 预存的
**索引元组**，不带类型也不带 patch 元数据，长度有 1/2/3 三种。已实测确认：
`index_tuple[0]` **不能**用来判断 X/Z —— `swap_time="early"` 下每一条都无法分类
（`define_bulk_detectors` 中 late 取 `plaq.ordered_data_qubits[0]`、early 取
`plaq.ancilla`，且元组被追加到 `detectors[1-rnd]` 而不是 `detectors[rnd]`）。

**未知 B —— observable 形式**（spec 8.2）：walking 的 observable 会 XOR 进跨轮的
中间测量，而 CX 之后被追踪的算符已是 X_L(1)·X_L(2)。

两者的最终判据都是 `detector_error_model()` 的确定性检查，二值、不需要猜。

**Files:**
- Modify: `src/surface_code_leakage_erasure/transversal_cx.py`
- Modify: `tests/test_transversal_cx.py`
- Modify: `docs/superpowers/specs/2026-09-04-transversal-cx-design.md`（回填两个结论）

**Interfaces:**
- Consumes: Task 3 的 `TwoPatchWalkingLayout`、Task 6 的 mixin
- Produces: `TransversalCXWalkingBuilder(d, swap_time="late", seed=None)`

- [ ] **Step 1: 调查——确定 walking detector 元组的 (patch, 类型) 分类**

先写一个一次性脚本（放 scratchpad，不进仓库），把分类关系搞清楚再写实现。

```python
import sys; sys.path.insert(0, "src")
from surface_code_leakage_erasure.two_patch_layout import TwoPatchWalkingLayout

for swap_time in ("late", "early"):
    layout = TwoPatchWalkingLayout(3, swap_time)
    n = layout.index_shift
    for r in (0, 1):
        dets = layout.detectors[r]
        half = len(dets) // 2
        print(f"--- {swap_time} round {r}: {len(dets)} detectors ---")
        # Hypothesis 1: the merged list is patch1 ++ patch2, entry k <-> entry k+half
        for k in range(half):
            t1, t2 = dets[k], dets[k + half]
            print(f"  k={k:2d} patch1={t1} patch2={t2} "
                  f"shifted_match={tuple(i + n for i in t1) == t2}")
```

**要确认的三件事：**

1. 合并后的 `detectors[r]` 确实是 patch1 列表 ++ patch2 列表，且第 `k` 项与第
   `k + half` 项互为平移像（`shift_and_merge` 用的 `_merge` 对 list 做 zip 后
   逐元素拼接，`_shift` 保序，所以理应成立——但要实测确认，因为整个重映射
   依赖这个位置对应关系）。
2. 每条 detector 的 check 类型。按优先级尝试：
   - **(a) 从 plaquette 容器反推**：`define_bulk_detectors` / `edge_detectors_*`
     依次遍历 `sorted(self.bulk_plaquettes[rnd])`、`sorted(self.trailing_plaquettes[rnd])`、
     `sorted(self.leading_plaquettes[rnd])` 并 `extend` 到 `detectors[·]`，
     所以元组在列表中的**位置**与源 plaquette 一一对应。按同样的顺序重放一遍
     即可给每个位置打上类型标签。这是首选：纯确定性，无需跑电路。
   - **(b) 用 stim 经验分类**：构建只在 data qubit 上加 `Z_ERROR` 的无 CX 双 patch
     电路，取 `detector_error_model()`；出现的 detector 即为 X 型（Z 错误翻转 X
     稳定子）。layout 无关，但较重。
   - **(c) 若 (a)(b) 都不成立**：停下来报告，不要猜。
3. 把结论写成一个可复用的方法 `_detector_check_types(rnd) -> list[str]`，
   返回与 `layout.detectors[rnd]` 等长的 `"X"`/`"Z"` 列表。

**把调查结论写进 spec 7.6**（新增一小节说明 walking 的分类办法），再继续。

- [ ] **Step 2: 写失败的测试**

注意：walking 的 detector 元组长度**本来**就有 1/2/3 三种，所以不能像 static 那样
按"3 项 = 特殊"来断言。这里用确定性做主判据，用"CX 轮的元组变长了"做结构判据。

追加到 `tests/test_transversal_cx.py`：

```python
from surface_code_leakage_erasure.transversal_cx import TransversalCXWalkingBuilder


def build_walking(swap_time, rounds=4, d=3, p=0.0, p_leak=0.0, **kwargs):
    builder = TransversalCXWalkingBuilder(d, swap_time, seed=0)
    circuit, _ = builder.get_circuit(rounds, p, p_leak, basis="X", **kwargs)
    return builder, circuit


@pytest.mark.parametrize("swap_time", ["late", "early"])
def test_walking_two_observables(swap_time):
    _, circuit = build_walking(swap_time)
    assert circuit.num_observables == 2


@pytest.mark.parametrize("swap_time", ["late", "early"])
@pytest.mark.parametrize("rounds", [4, 8])
def test_walking_deterministic_with_cx(swap_time, rounds):
    # The oracle for both unknowns in this task.
    _, circuit = build_walking(swap_time, rounds=rounds)
    circuit.detector_error_model()


@pytest.mark.parametrize("swap_time", ["late", "early"])
@pytest.mark.parametrize("rounds", [4, 8])
def test_walking_cx_layer_appears_exactly_once(swap_time, rounds):
    # rounds=4 makes cx_round=2 collide with walking's rnd_eff cache keys {0,1,2}
    builder, circuit = build_walking(swap_time, rounds=rounds, p=1e-3)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1
    (_, pairs), = layers
    assert len(pairs) == 9


@pytest.mark.parametrize("swap_time", ["late", "early"])
def test_walking_cx_round_detectors_grew(swap_time):
    # Exactly half the detectors in the CX round pick up one extra rec: patch 1's
    # X checks and patch 2's Z checks.
    builder, circuit = build_walking(swap_time, rounds=8)
    per_round = _detector_sizes_per_round(circuit)
    cx_sizes = sorted(per_round[builder.cx_round])
    ordinary_sizes = sorted(per_round[builder.cx_round + 2])
    assert sum(cx_sizes) == sum(ordinary_sizes) + len(ordinary_sizes) // 2


@pytest.mark.parametrize("swap_time", ["late", "early"])
def test_walking_far_rounds_match_an_ordinary_round(swap_time):
    # cx_round = 4 here, so the base class's `rnd - 2` recursion passes through it
    builder, circuit = build_walking(swap_time, rounds=8)
    per_round = _detector_sizes_per_round(circuit)
    assert builder.cx_round == 4
    for rnd in (6, 7):
        assert sorted(per_round[rnd]) == sorted(per_round[rnd - 2])
```

- [ ] **Step 3: 运行，确认失败**

Run: `python -m pytest tests/test_transversal_cx.py -v -k walking`
Expected: FAIL，`ImportError: cannot import name 'TransversalCXWalkingBuilder'`

- [ ] **Step 4: 实现 walking builder**

`transversal_cx.py` 顶部 import 追加：

```python
from .two_patch_layout import TwoPatchWalkingLayout
from .walking_circuit_builder import WalkingSCCircuitBuilder
```

文件末尾追加。`_detector_check_types` 的**函数体由 Step 1 的调查结论填入**——
这是本计划唯一一处需要先调查再落笔的实现，其余全部照抄。

```python
class TransversalCXWalkingBuilder(TransversalCXMixin, WalkingSCCircuitBuilder):
    """Transversal CX between two walking (or moonwalking) surface code patches.

    `swap_time="late"` is the walking code, `"early"` the moonwalking code.
    """

    def __init__(self, d, swap_time: str = "late", seed=None):
        layout = d if isinstance(d, TwoPatchWalkingLayout) \
            else TwoPatchWalkingLayout(d, swap_time)
        super().__init__(layout, seed=seed)
        self._init_transversal_cx_state()

    def __repr__(self):
        return (f"TransversalCXWalkingBuilder(d={self.layout.d}, "
                f"swap_time={self.swap_time})")

    def _cx_sublattice(self) -> int:
        """Which data-qubit sublattice holds the state when the CX is applied.

        The state starts on sublattice 0 for `swap_time="late"` and 1 for
        `"early"` (matching initialize_data_qubits) and moves by one each round,
        so after `cx_round` rounds it sits on (start + cx_round) % 2. Sanity
        check: after `rounds = 2 * cx_round` rounds this returns to the start,
        which is exactly the sublattice final_measurement reads out.
        """
        start = 0 if self.swap_time == "late" else 1
        return (start + self.cx_round) % 2

    def _cx_data_qubits(self) -> list[int]:
        shift = self.layout.index_shift
        return [q for q in self.dq_indexes[self._cx_sublattice()] if q < shift]

    def _detector_check_types(self, rnd: int) -> list[str]:
        """'X' or 'Z' per entry of layout.detectors[rnd], same length and order.

        FILL IN FROM TASK 7 STEP 1. The walking layout stores detectors as bare
        index tuples with no type attached, and the tuple's first element does
        NOT identify the check -- verified false for swap_time="early" at every
        detector. Do not guess here.
        """
        raise NotImplementedError("fill in from the Task 7 Step 1 investigation")

    def _cx_round_detector_text(self, basis: str):
        """Detectors for the round right after the transversal CX.

        The merged detector list is patch 1's entries followed by patch 2's, with
        entry k and entry k + half being translation images of one another (Step 1
        confirms this). The cross term for entry k is that partner tuple's
        PREVIOUS-round records, which reconstruct the partner check's pre-CX value
        -- see spec 7.1.
        """
        rnd_phase = self.cx_round % 2
        detectors = self.layout.detectors[rnd_phase]
        types = self._detector_check_types(rnd_phase)
        shift = self.layout.index_shift
        half = len(detectors) // 2

        recs_list = []
        for k, index_tuple in enumerate(detectors):
            idxs = [self.tracker.get_curr_meas(i) for i in index_tuple]
            patch = 1 if k < half else 2
            crosses = (patch == 1 and types[k] == "X") or \
                      (patch == 2 and types[k] == "Z")
            if crosses:
                partner = tuple(i + shift if patch == 1 else i - shift
                                for i in index_tuple)
                idxs.extend(self.tracker.get_prev_meas(i) for i in partner)
            recs_list.append(" ".join(f"rec[{idx}]" for idx in idxs))
        return "\n".join(f"DETECTOR {recs}" for recs in recs_list)

    def _observable_recs(self, patch: int) -> list[int]:
        last_only, all_meas = self.layout.x_logical_per_patch[patch - 1]
        idxs = [self.tracker.get_curr_meas(idx) for idx in last_only]
        for idx in all_meas:
            idxs.extend(self.tracker.get_all_meas(idx))
        return idxs
```

- [ ] **Step 5: 运行测试；解决未知 B（observable 形式）**

Run: `python -m pytest tests/test_transversal_cx.py -v -k walking`

`test_walking_deterministic_with_cx` 有两种结果：

**(a) 通过** —— spec 8.2 的假设为假，patch 1 自己的 walking 记录已经确定。
把 spec 8.2 从"待验证"改为"已验证：不需要跨 patch 项"。

**(b) 失败**，stim 报某个 observable 不确定 —— 假设成立，observable 0 需要补上
patch 2 在 CX 之后的记录。**只**改 `_observable_recs`：

```python
    def _observable_recs(self, patch: int) -> list[int]:
        last_only, all_meas = self.layout.x_logical_per_patch[patch - 1]
        idxs = [self.tracker.get_curr_meas(idx) for idx in last_only]
        for idx in all_meas:
            idxs.extend(self.tracker.get_all_meas(idx))

        if patch == 1:
            # After the CX the tracked operator is X_L(1) . X_L(2), so the
            # control's observable picks up the target's records from the CX
            # onwards. Everything before the CX still reconstructs X_L(1) alone.
            partner_last, partner_all = self.layout.x_logical_per_patch[1]
            idxs.extend(self.tracker.get_curr_meas(i) for i in partner_last)
            for i in partner_all:
                idxs.extend(self._meas_after_cx(i))
        return idxs

    def _meas_after_cx(self, index: int) -> list[int]:
        """This qubit's measurement records from the CX round onwards."""
        history = self.tracker.history[index]
        return [t - self.tracker.t for t in history[self.cx_round:]]
```

若仍不确定，后续候选依次是：**只**加 `partner_last`；加 `partner_last` 与
`partner_all` 的全部记录。stim 的报错逐次收窄即可。**落在哪一个都要回填 spec 8.2。**

若 `test_walking_deterministic_with_cx` 在**所有**候选下都失败，说明 Step 1 的
分类或 spec 7.2 的规则在 walking 下不成立——**停下来报告**，不要继续试。

- [ ] **Step 6: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 7: 回填 spec 并提交**

编辑 `docs/superpowers/specs/2026-09-04-transversal-cx-design.md`：
7.6 补上 walking 的 detector 分类办法（Step 1 结论），8.2 改为实测结论，
第 14 节同步移除已消解的风险。

```bash
git add src/surface_code_leakage_erasure/transversal_cx.py tests/test_transversal_cx.py docs/superpowers/specs/2026-09-04-transversal-cx-design.md
git commit -m "feat: add walking and moonwalking transversal CX builders

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: CX 层的 leakage 位置记账

spec 第 9 节。CX 层带 leakage，所以扁平位置空间要扩容。

**Files:**
- Modify: `src/surface_code_leakage_erasure/transversal_cx.py`
- Modify: `tests/test_transversal_cx.py`

**Interfaces:**
- Consumes: Task 5 的 `_cx_pairs`
- Produces: `TransversalCXMixin.count_leakage_locations(rounds, leak_locations)`、`ravel_leakage_locations(rounds, leak_locs_unraveled, leak_locations)`；`leak_locs` 字典新增 `"cx"` 键

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_transversal_cx.py`：

```python
def test_cx_adds_leakage_locations():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    builder.cx_round = 2
    total, per_round, per_step = builder.count_leakage_locations(4, "2-qubit gates")
    base_total = sum(per_step) * 4
    assert total == base_total + 9              # d=3 -> d**2 transversal gates


def test_cx_leakage_locations_land_in_the_cx_key():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    builder.cx_round = 2
    total, per_round, _ = builder.count_leakage_locations(4, "2-qubit gates")
    # every index in the tail block belongs to the CX layer
    tail = list(range(4 * per_round, total))
    locs = builder.ravel_leakage_locations(4, tail, "2-qubit gates")
    assert set(locs["cx"]) and len(locs["cx"]) == 9
    shift = builder.layout.index_shift
    for q in locs["cx"]:
        assert q < shift or (q - shift) < shift


def test_round_leakage_locations_unchanged_by_the_cx_block():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    builder.cx_round = 2
    _, per_round, _ = builder.count_leakage_locations(4, "2-qubit gates")
    locs = builder.ravel_leakage_locations(4, [0, 1, per_round + 3], "2-qubit gates")
    assert locs["cx"] == set()
    assert sum(len(s) for r in range(4) for s in locs[r].values()) == 3


def test_skip_gates_rejected_on_the_cx_layer():
    builder = TransversalCXCircuitBuilder(3, seed=0)
    with pytest.raises(NotImplementedError):
        builder.get_circuit(4, 1e-3, 0.2, basis="X", leak_effect="skip gates")


def test_leaky_circuit_still_has_one_cx_layer():
    builder, circuit = build(rounds=4, p=1e-3, p_leak=0.05)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_transversal_cx.py -v -k leakage`
Expected: FAIL（`KeyError: 'cx'` 与位置总数断言失败）

- [ ] **Step 3: 实现**

在 `TransversalCXMixin` 中追加：

```python
    def _n_cx_leakage_locations(self, leak_locations: str) -> int:
        if leak_locations == "2-qubit gates":
            return len(self._cx_pairs())
        if leak_locations == "all":
            return self.n_qubits
        raise ValueError(f"Invalid leak_locations value: {leak_locations}")

    def count_leakage_locations(self, rounds: int,
                                leak_locations: str = "2-qubit gates"):
        """Round locations plus one extra block for the transversal CX layer.

        `per_round` and `per_step` are returned unchanged from the base class, so
        _ravel_leakage_locations_impl -- which uses only those two and never the
        total -- keeps working on the round block unmodified.
        """
        total, per_round, per_step = super().count_leakage_locations(
            rounds, leak_locations)
        return total + self._n_cx_leakage_locations(leak_locations), per_round, per_step

    def ravel_leakage_locations(self, rounds: int, leak_locs_unraveled: list,
                                leak_locations: str = "2-qubit gates"):
        _, per_round, _ = super().count_leakage_locations(rounds, leak_locations)
        boundary = rounds * per_round

        round_locs = [loc for loc in leak_locs_unraveled if loc < boundary]
        cx_locs = [loc - boundary for loc in leak_locs_unraveled if loc >= boundary]

        leak_locs = super().ravel_leakage_locations(
            rounds, round_locs, leak_locations)

        cx_leaked = set()
        if leak_locations == "2-qubit gates":
            pairs = self._cx_pairs()
            for loc in cx_locs:
                cx_leaked.add(pairs[loc][self.rng.choice(2)])
        else:
            cx_leaked.update(cx_locs)
        # A string key can never collide with get_circuit's integer round lookup.
        leak_locs["cx"] = cx_leaked
        return leak_locs
```

把 `TransversalCXMixin.get_circuit` **整体替换**为下面这版（Task 4 写的那版被完整取代）：

```python
    def get_circuit(self, rounds: int, p: float, p_leak: float = 0.0,
                    leakage_circuit_locations: dict | None = None,
                    use_cache=True, **circuit_kwargs):
        if rounds % 2 != 0:
            raise ValueError(
                "rounds must be even for a transversal CX experiment: the CX sits "
                f"between two halves of rounds // 2 rounds each (got {rounds}).")
        self.cx_round = rounds // 2

        # Generate the leakage locations here instead of letting the base class do
        # it. Our flat location space carries an extra block for the CX layer, and
        # its "cx" entry has to be split off before the round loop starts.
        # cx_round must already be set: counting the CX block needs _cx_pairs().
        if p_leak > 0.0 and not leakage_circuit_locations:
            leak_locations = circuit_kwargs.get(
                "leak_locations", self.kwargs_options["leak_locations"][0])
            leakage_circuit_locations = self.generate_leakage_locations(
                rounds, p_leak, leak_locations=leak_locations)
            p_leak = 0.0

        if leakage_circuit_locations:
            self._cx_leak_locs = set(leakage_circuit_locations.pop("cx", set()))
        else:
            self._cx_leak_locs = set()

        return super().get_circuit(
            rounds, p, p_leak,
            leakage_circuit_locations=leakage_circuit_locations,
            use_cache=use_cache, **circuit_kwargs)
```

最后把 leakage 效果接进 `_emit_cx_layer`。先把 `measure_stabilizers` 里的调用行

```python
                self._emit_cx_layer(p, kwargs["Pauli_locations"])
```

改成

```python
                self._emit_cx_layer(
                    p, kwargs["Pauli_locations"], kwargs["leak_effect"])
```

再把 `_emit_cx_layer` **整体替换**为：

```python
    def _emit_cx_layer(self, p: float, Pauli_locations: str, leak_effect: str):
        """One transversal CX between the patches, occupying its own TICK.

        The instruction order mirrors make_stabilizer_gates: gates, then Pauli
        noise, then leakage noise.
        """
        if leak_effect not in ("depolarize", "tailored") and self._cx_leak_locs:
            # The "skip gates" family removes gates involving a leaked qubit from
            # the circuit. Doing that to a transversal CX would drop one of the
            # d**2 pairs and silently change the logical operation, and the
            # detector remapping in _cx_round_detector_text assumes every pair is
            # present. Refuse rather than emit a wrong circuit; see the plan's
            # "Known limitation" note.
            raise NotImplementedError(
                f"leak_effect={leak_effect!r} is not supported on the transversal "
                "CX layer; use 'depolarize' or 'tailored'.")

        pairs = self._cx_pairs()
        gate_targets = " ".join(f"{q:3d}" for pair in pairs for q in pair)
        instructions = [f"CX {gate_targets}"]

        if p > 0.0:
            instructions.append(f"DEPOLARIZE2({p}) {gate_targets}")
            if Pauli_locations == "all":
                gated = {q for pair in pairs for q in pair}
                idling = sorted(self.qubit_set - gated)
                if idling:
                    instructions.append(
                        f"DEPOLARIZE1({p}) {' '.join(str(q) for q in idling)}")

        newly_leaked = self._cx_leak_locs - self.leaked
        self.leaked.update(newly_leaked)
        for q0, q1 in pairs:
            if q0 in newly_leaked or q1 in newly_leaked:
                instructions.extend(self.apply_leakage_noise_model(
                    leak_effect, q0, q1, q0 in self.leaked, q1 in self.leaked))

        self.circuit.append("\n".join(instructions))
        self.circuit.append("TICK")
```

**Known limitation（要写进 spec 第 15 节）**：`skip gates` / `tailored then skip`
及其 decode 变体在 CX 层上未实现，遇到就抛 `NotImplementedError`。这些模型会把
涉及泄漏比特的门从电路里删掉，而 transversal CX 少一对就改变了逻辑操作，
`_cx_round_detector_text` 也假设 `d²` 对全在。留给解码 spec 处理。

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_transversal_cx.py -v`
Expected: 全部通过

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/transversal_cx.py tests/test_transversal_cx.py
git commit -m "feat: account for leakage locations on the transversal CX layer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: 导出、hyperedge 可分解性、收尾

**Files:**
- Modify: `src/surface_code_leakage_erasure/__init__.py`（**纯追加**）
- Modify: `tests/test_transversal_cx.py`

**Interfaces:**
- Consumes: 前 8 个 task 的全部产出
- Produces: 包级导出 `TwoPatchSurfaceCodeLayout`、`TwoPatchWalkingLayout`、`TransversalCXCircuitBuilder`、`TransversalCXWalkingBuilder`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_transversal_cx.py`：

```python
def test_exported_from_package_root():
    import surface_code_leakage_erasure as pkg

    for name in ("TwoPatchSurfaceCodeLayout", "TwoPatchWalkingLayout",
                 "TransversalCXCircuitBuilder", "TransversalCXWalkingBuilder"):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__, name


@pytest.mark.parametrize("swap_time", [None, "late", "early"])
def test_hyperedges_are_decomposable(swap_time):
    """The DEM is not graphlike (see spec 7.5), but each 4-detector mechanism
    should split into two mechanisms the circuit actually contains. If this
    raises, the detector basis chosen in spec 7.4 needs revisiting."""
    if swap_time is None:
        _, circuit = build(rounds=4, p=1e-3)
    else:
        _, circuit = build_walking(swap_time, rounds=4, p=1e-3)
    circuit.detector_error_model(decompose_errors=True)
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_transversal_cx.py -v -k "exported or decomposable"`
Expected: `test_exported_from_package_root` FAIL（`AssertionError: TwoPatchSurfaceCodeLayout`）

- [ ] **Step 3: 追加导出**

在 `src/surface_code_leakage_erasure/__init__.py` 中，**只做追加**——不要改动任何已有的 import 或 `__all__` 条目。

在现有 import 块之后追加：

```python
from .two_patch_layout import (
    TwoPatchSurfaceCodeLayout,
    TwoPatchWalkingLayout,
)

from .transversal_cx import (
    TransversalCXCircuitBuilder,
    TransversalCXWalkingBuilder,
)
```

在 `__all__` 列表内追加（保持既有条目原样）：

```python
    # transversal CX
    "TwoPatchSurfaceCodeLayout",
    "TwoPatchWalkingLayout",
    "TransversalCXCircuitBuilder",
    "TransversalCXWalkingBuilder",
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_transversal_cx.py -v`
Expected: 全部通过

若 `test_hyperedges_are_decomposable` 失败：这不是导出问题，而是 spec 7.4 选基结论的反例。**停下来报告**，不要为了让测试通过而改测试——需要重新审视 detector 基。

- [ ] **Step 5: 全量测试并确认约束**

```bash
python -m pytest tests/ -v
```

Expected: 全部通过。

再确认"只允许加入"这条约束确实成立：

```bash
git diff --stat main...HEAD -- src/
```

Expected: `src/` 下只有 `two_patch_layout.py`、`transversal_cx.py` 两个**新增**文件，以及 `__init__.py` 的**纯追加**。若 `__init__.py` 出现删除行（`-` 开头），说明动了既有条目，必须改回。

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/__init__.py tests/test_transversal_cx.py
git commit -m "feat: export transversal CX builders and test DEM decomposability

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## 完成判据

- `python -m pytest tests/ -v` 全绿
- `git diff --stat main...HEAD -- src/` 只显示两个新文件加 `__init__.py` 的纯追加
- static / walking / moonwalking 三种码型的无噪声电路都通过 `detector_error_model()` 的确定性检查
- spec 8.2 的 walking observable 形式已回填实测结论

**不在完成判据内**（spec 第 15 节）：距离验证、任何解码路径、`erasure_sampler.py` 的双 observable 失败计数。

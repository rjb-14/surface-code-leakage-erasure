# Moonwalking Transversal CNOT 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在两块全等的 **moonwalking** surface code patch（`swap_time="early"`）之间构建 transversal CNOT 电路，交付到"电路可构建 + stim 确定性验证通过"。

**Architecture:** 把两块 patch 合并成**一个**平移后的 layout 对象。现有 `WalkingSCCircuitBuilder.__init__` 从 `layout.plaquettes` 派生 CNOT 调度，因此两块 patch 自动在同一批 TICK 里并行跑 SE，无需改动 `_measure_stabilizers_impl`。一个直接继承 `WalkingSCCircuitBuilder` 的 builder 负责三件事：插入 CX 层、重映射 CX 边界轮的 detector、发两个 observable。

**Tech Stack:** Python ≥3.10, stim, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-09-04-transversal-cx-design.md`

**与完整版计划的关系：** 本计划是 `2026-09-04-transversal-cx.md` 收窄到单一码型的版本。static 与 walking(late) 分支**不实现**。因为只有一个码型，不需要 mixin 抽象——直接继承即可。

## Global Constraints

- **只允许新增文件。** 不得修改任何现有文件的既有行为。`src/surface_code_leakage_erasure/__init__.py` 与 `pyproject.toml` **只做纯追加**。
- 新增文件仅限：`src/surface_code_leakage_erasure/two_patch_layout.py`、`src/surface_code_leakage_erasure/moonwalking_transversal_cx.py`、`tests/`（整个目录新建）。
- **只做 moonwalking**（`swap_time="early"`）。不实现 static、不实现 walking(late)。
- **距离验证不在范围内**（spec 15）。不要写 `shortest_graphlike_error()` 或任何最小重量搜索——spec 7.5 论证了 DEM 含 hyperedge，该函数不适用。
- **解码不在范围内。** `single_ec_traceback` 覆盖为 `NotImplementedError`；不碰 `erasure_sampler.py`、`decoding.py`、`modified_mle.py`。
- 实验语义：两块 patch 同步，`rounds = 2R`（walking 本就要求偶数轮），CX 插在 round index `R = rounds // 2` 开始之前；两块均初始化到 `|+>`（`basis="X"`）；observable 0 = control 的 X logical，observable 1 = target 的。
- 平移量：`coord_shift = Coord(2*d + 2, 0)`，`index_shift = len(layout.qubits)`。合并后 `index < index_shift` ⟺ patch 1。
- detector 约定（spec 7.2）：CX 后第一轮里，patch 1 的 **X** check 与 patch 2 的 **Z** check 各吸收配对 check 在 **round `R-1`（CX 前）** 的记录。
- 每个 task 结束时提交，提交信息末尾加：
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## 收窄带来的简化

| 完整版需要 | 本计划 | 原因 |
|---|---|---|
| `TransversalCXMixin` + 两个 builder | 一个 builder 直接继承 `WalkingSCCircuitBuilder` | 只有一个码型，不需要跨 layout 接口的抽象 |
| static + walking 两套 layout 子类 | 一个 `TwoPatchMoonwalkingLayout` | 同上 |
| 重建 `erasure_swaps` / `erasure_ancillas` | **不需要** | `WalkingSCCircuitBuilder` 没有 erasure swap 机制，`ec_sched == 8` 的 swap 只在 static builder 里 |
| 覆盖 `_observable_cache_key` | **不需要** | walking 的已经是 `(rounds, basis)` |
| 校验 `rounds` 为偶数 | **不需要额外写** | `WalkingSCCircuitBuilder.get_circuit` 已经在做 |
| detector 类型分类（完整版的最大未知） | **已解决**，见 Task 2 | 见下 |

**detector 类型分类已在计划阶段解决。** moonwalking 的 detector 是 layout 预存的索引元组，不带类型元数据，且首元素**不能**用来判断 X/Z（early 下每一条都不行）。但 `define_bulk_detectors` 与 `edge_detectors_early_swap` 的三个构造块都按固定顺序 `extend` 到 `detectors[1-rnd]`，所以位置与源 plaquette 一一对应：

```
detectors[r] 的类型序列
  = [p.type for p in sorted(bulk_plaquettes[1-r])]
  + [p.type for p in sorted(leading_plaquettes[1-r])]
  + [p.type for p in sorted(trailing_plaquettes[1-r])]
```

已在 d=3/5/7 上把元组构造完整重放并与 `layout.detectors[r]` **逐项比对通过**，不是猜测。Task 2 Step 1 的测试会把这个不变量固定下来。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/surface_code_leakage_erasure/two_patch_layout.py` | 纯 layout 层：把已构造的 layout 平移成第二份并合并；暴露 `pair()`、per-patch logical、detector 类型序列。无电路/噪声逻辑。 |
| `src/surface_code_leakage_erasure/moonwalking_transversal_cx.py` | 纯 builder 层：CX 层、detector 重映射、双 observable、leakage 记账。无 layout 构造逻辑。 |
| `tests/test_two_patch_layout.py` | layout 层单元测试 |
| `tests/test_moonwalking_transversal_cx.py` | builder 层与端到端测试 |
| `tests/test_regression_existing.py` | 现有 builder 未被改动的守卫 |

---

## Task 1: 测试脚手架与现有行为守卫

先建守卫再动任何代码——"只允许加入"必须是**可验证的**，不是口头保证。

**Files:**
- Create: `tests/__init__.py`（空文件）
- Create: `tests/test_regression_existing.py`
- Modify: `pyproject.toml`（**纯追加**一个 `dev` 可选依赖组）

**Interfaces:**
- Consumes: 无
- Produces: 可运行的 pytest 环境；后续所有 task 的回归守卫

- [ ] **Step 1: 追加 pytest 依赖**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 段**末尾追加**（不改动已有的 `analysis` 组）：

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

摘要是在**改动前**从当前 `runjiang/stability` 代码实测得到的。虽然本计划只碰 moonwalking，static 与 walking 的守卫一并保留——它们是"没有误伤"的证据。

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
Expected: 7 passed。**若有任何一条失败，停下**——基线摘要与当前代码不符，先查清原因。

- [ ] **Step 5: 提交**

```bash
git add tests/ pyproject.toml
git commit -m "test: guard existing builder output before adding transversal CX

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: 双 patch moonwalking layout

**Files:**
- Create: `src/surface_code_leakage_erasure/two_patch_layout.py`
- Create: `tests/test_two_patch_layout.py`

**Interfaces:**
- Consumes: `surface_code.Coord/Qubit/DataQubit/Plaquette`、`walking_surface_code.WalkingSurfaceCodeLayout`
- Produces:
  - `shift_and_merge(layout, coord_shift: Coord, index_shift: int) -> None`（就地改写）
  - `TwoPatchMoonwalkingLayout(d)`，属性 `coord_shift: Coord`、`index_shift: int`、`x_logical_per_patch: list`、`z_logical_per_patch: list`、`detector_check_types: list[list[str]]`；方法 `pair(qubit) -> Qubit`、`patch_of(index) -> int`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_two_patch_layout.py
import pytest

from surface_code_leakage_erasure.surface_code import DataQubit
from surface_code_leakage_erasure.two_patch_layout import TwoPatchMoonwalkingLayout


@pytest.fixture
def layout():
    return TwoPatchMoonwalkingLayout(3)


def test_qubit_count_doubles(layout):
    assert len(layout.qubits) == 2 * 22        # d=3 moonwalking patch has 22 qubits
    assert layout.index_shift == 22
    assert sorted(q.index for q in layout.qubits) == list(range(44))


def test_swap_time_is_early(layout):
    assert layout.swap_time == "early"
    assert layout.d == 3                        # scalars must not be shifted or merged


def test_patches_do_not_overlap_in_space(layout):
    xs1 = {q.coord[0] for q in layout.qubits if layout.patch_of(q.index) == 1}
    xs2 = {q.coord[0] for q in layout.qubits if layout.patch_of(q.index) == 2}
    assert max(xs1) < min(xs2)


def test_pair_is_a_translation_bijection(layout):
    patch1 = [q for q in layout.qubits if layout.patch_of(q.index) == 1]
    assert len(patch1) == 22
    paired = set()
    for q in patch1:
        p = layout.pair(q)
        assert p.coord == q.coord + layout.coord_shift
        assert p.index == q.index + layout.index_shift
        assert type(p) is type(q)
        paired.add(p.index)
    assert paired == set(range(22, 44))


def test_per_round_lists_merge_elementwise(layout):
    # every per-round schedule stays length 2 (indexed by rnd % 2) rather than
    # becoming a 4-element list
    for name in ("data_qubits", "z_plaquettes", "x_plaquettes", "plaquettes",
                 "reset_indexes", "rx_indexes", "measure_indexes", "mx_indexes",
                 "detectors"):
        value = getattr(layout, name)
        assert isinstance(value, list) and len(value) == 2, name


def test_detectors_are_patch1_then_patch2(layout):
    # The whole detector remapping relies on this positional correspondence.
    shift = layout.index_shift
    for round_detectors in layout.detectors:
        half = len(round_detectors) // 2
        assert half > 0
        for k in range(half):
            t1, t2 = round_detectors[k], round_detectors[k + half]
            assert tuple(i + shift for i in t1) == t2


def test_detector_check_types_align_with_detectors(layout):
    for rnd in (0, 1):
        types = layout.detector_check_types[rnd]
        assert len(types) == len(layout.detectors[rnd])
        assert set(types) == {"X", "Z"}
        # patch 2's half is the positional image of patch 1's, so the type list
        # is patch 1's doubled
        half = len(types) // 2
        assert types[:half] == types[half:]


def test_per_patch_logicals_disjoint(layout):
    a, b = layout.x_logical_per_patch
    idx_a = {i for group in a for i in group}
    idx_b = {i for group in b for i in group}
    assert idx_a and idx_b
    assert idx_a.isdisjoint(idx_b)
    assert all(layout.patch_of(i) == 1 for i in idx_a)
    assert all(layout.patch_of(i) == 2 for i in idx_b)


def test_back_references_are_not_shared_between_patches(layout):
    for dq_set in layout.data_qubits:
        for dq in dq_set:
            if not isinstance(dq, DataQubit):
                continue
            for plaq in dq.ordered_plaquettes:
                if plaq is not None:
                    assert (layout.patch_of(plaq.ancilla.index)
                            == layout.patch_of(dq.index))
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_two_patch_layout.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'surface_code_leakage_erasure.two_patch_layout'`

- [ ] **Step 3: 实现 two_patch_layout.py**

```python
# src/surface_code_leakage_erasure/two_patch_layout.py
"""Build a two-patch layout by translating an existing layout and merging it in.

The translation acts on a layout's *output structures* -- the constructed Qubit
and Plaquette objects and the containers holding them -- rather than on the
methods that produce them. Those methods hard-code boundary coordinates
(`WalkingSurfaceCodeLayout.layout_edge_ancillas`, `layout_plaquettes`), so
threading an offset through them would be an invasive rewrite of code this work
is not allowed to touch.
"""

from .surface_code import Coord, DataQubit, Plaquette, Qubit
from .walking_surface_code import WalkingSurfaceCodeLayout

# Attributes describing the layout as a whole rather than one patch's contents.
# They are identical for both patches: shifting `d` or concatenating `swap_time`
# would be nonsense.
_SCALAR_ATTRS = frozenset({"d", "swap_time"})

# Logical operators are kept per patch instead of merged. Merging them would give
# the product of the two patches' logicals, but the experiment needs two
# independent observables.
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
    # Coord is a tuple subclass holding two ints; don't descend into it.
    if isinstance(value, Coord):
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _collect(v, out, kind)


def _layout_attrs(layout):
    return {k: v for k, v in vars(layout).items() if k not in _SCALAR_ATTRS}


def _build_qubit_map(layout, coord_shift, index_shift):
    qubits = set()
    for value in _layout_attrs(layout).values():
        _collect(value, qubits, Qubit)
    return {
        q: (DataQubit if isinstance(q, DataQubit) else Qubit)(
            q.index + index_shift, q.coord + coord_shift)
        for q in qubits
    }


def _build_plaquette_map(layout, qmap, coord_shift):
    plaquettes = set()
    for value in _layout_attrs(layout).values():
        _collect(value, plaquettes, Plaquette)
    pmap = {}
    for plaq in plaquettes:
        ordered = [None if dq is None else qmap[dq] for dq in plaq.ordered_data_qubits]
        pmap[plaq] = Plaquette(
            plaq.type, ordered, qmap[plaq.ancilla], plaq.coord + coord_shift)
    # Rebuild the DataQubit -> Plaquette back references on the shifted copies so
    # the two patches never share a Plaquette object.
    for plaq in plaquettes:
        for t, dq in enumerate(plaq.ordered_data_qubits):
            if isinstance(dq, DataQubit):
                qmap[dq].ordered_plaquettes[t] = pmap[plaq]
    return pmap


def _shift(value, qmap, pmap, coord_shift, index_shift):
    """Translate one attribute value onto patch 2."""
    # Order matters: Plaquette/Qubit before the generic containers, Coord before
    # tuple (Coord subclasses tuple), bool/str before int (bool subclasses int).
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
    if isinstance(a, (frozenset, set)):
        return a | b
    if isinstance(a, dict):
        return {**a, **b}
    if isinstance(a, (list, tuple)):
        return a + b
    raise TypeError(f"don't know how to merge {type(a).__name__}")


def _merge(a, b):
    """Merge patch 1's attribute value with patch 2's.

    A list-valued attribute is a per-round schedule (walking layouts index every
    such list by `rnd % 2`), so the two patches merge element by element, and each
    element's contents concatenate: patch 1's entries first, then patch 2's. Every
    other container merges directly.
    """
    if isinstance(a, list):
        return [_merge_leaf(x, y) for x, y in zip(a, b)]
    return _merge_leaf(a, b)


def shift_and_merge(layout, coord_shift: Coord, index_shift: int):
    """Add a translated copy of `layout` to itself, in place.

    `z_logical` / `x_logical` are NOT merged: they keep patch 1's value, and both
    patches' versions are exposed as `z_logical_per_patch` / `x_logical_per_patch`.
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


def moonwalking_detector_check_types(layout) -> list[list[str]]:
    """'X' or 'Z' per entry of `layout.detectors[rnd]`, for a SINGLE-patch layout.

    The moonwalking layout stores detectors as bare index tuples with no type
    attached, and the tuple's first element does not identify the check (verified
    false at every detector for swap_time="early"). But the three blocks that
    build them -- define_bulk_detectors, then the leading and trailing blocks of
    edge_detectors_early_swap -- each iterate a sorted plaquette set and extend
    `detectors[1 - rnd]`, so an entry's position in the list identifies its source
    plaquette. Replaying that order tuple-for-tuple reproduces `layout.detectors`
    exactly at d = 3, 5 and 7.

    Must be called BEFORE shift_and_merge: afterwards `sorted()` interleaves the
    two patches by (type, index), which does not match the merged detector list's
    patch-1-then-patch-2 order.
    """
    types = []
    for rnd in range(2):
        src = 1 - rnd
        types.append(
            [p.type for p in sorted(layout.bulk_plaquettes[src])]
            + [p.type for p in sorted(layout.leading_plaquettes[src])]
            + [p.type for p in sorted(layout.trailing_plaquettes[src])])
    return types


class TwoPatchMoonwalkingLayout(WalkingSurfaceCodeLayout):
    """Two translated copies of the moonwalking (`swap_time="early"`) layout."""

    def __init__(self, d):
        super().__init__(d, swap_time="early")

    def define_layout(self, swap_time):
        super().define_layout(swap_time)

        # Compute the detector types on the single patch, then double them: patch
        # 2's detector list is the positional image of patch 1's.
        single = moonwalking_detector_check_types(self)

        shift_and_merge(
            self, coord_shift=Coord(2 * self.d + 2, 0),
            index_shift=len(self.qubits))

        self.detector_check_types = [types + types for types in single]

    def pair(self, qubit):
        """Patch 2's counterpart of a patch-1 qubit."""
        return self.coord_to_qubit[qubit.coord + self.coord_shift]

    def patch_of(self, index: int) -> int:
        """1 for the control patch, 2 for the target patch."""
        return 1 if index < self.index_shift else 2
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_two_patch_layout.py -v`
Expected: 9 passed

若 `_shift` 抛 `TypeError: don't know how to shift ...`，说明 layout 有一个未覆盖的属性类型。把该类型加进 `_shift` 的分支，**不要**用 `except TypeError: return value` 兜底——静默跳过会产生两 patch 共享对象的 layout。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/two_patch_layout.py tests/test_two_patch_layout.py
git commit -m "feat: add two-patch moonwalking layout

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: 双 patch 并行电路（尚无 CX）

关键的中间检查点：把"合并后的 layout 能否喂给现有 builder"与"CX 逻辑是否正确"**分离开**。此时电路是两块独立 patch 的 X memory，各自一个 observable，必然确定——若这一步就不确定，问题在 Task 2 而不在 CX。

**Files:**
- Create: `src/surface_code_leakage_erasure/moonwalking_transversal_cx.py`
- Create: `tests/test_moonwalking_transversal_cx.py`

**Interfaces:**
- Consumes: Task 2 的 `TwoPatchMoonwalkingLayout`
- Produces: `MoonwalkingTransversalCXBuilder(d, seed=None)`，属性 `cx_round: int | None`；方法 `get_circuit(rounds, p, p_leak=0.0, **kwargs) -> (stim.Circuit, erasure_checks)`、`final_observable(rounds, basis)`、`_observable_recs(patch) -> list[int]`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_moonwalking_transversal_cx.py
import pytest

from surface_code_leakage_erasure.moonwalking_transversal_cx import (
    MoonwalkingTransversalCXBuilder,
)


def build(rounds=4, d=3, p=0.0, p_leak=0.0, **kwargs):
    builder = MoonwalkingTransversalCXBuilder(d, seed=0)
    circuit, _ = builder.get_circuit(rounds, p, p_leak, basis="X", **kwargs)
    return builder, circuit


def test_two_observables():
    _, circuit = build()
    assert circuit.num_observables == 2


def test_qubit_count_is_two_patches():
    _, circuit = build()
    assert circuit.num_qubits == 2 * 22


def test_noiseless_circuit_is_deterministic():
    # stim raises if any declared detector or observable is not deterministic
    _, circuit = build()
    circuit.detector_error_model()


def test_odd_rounds_rejected():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    with pytest.raises(ValueError):
        builder.get_circuit(5, 0.0, 0.0, basis="X")


def test_cx_round_is_half_the_rounds():
    builder, _ = build(rounds=6)
    assert builder.cx_round == 3


def test_traceback_is_out_of_scope():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    with pytest.raises(NotImplementedError):
        builder.single_ec_traceback(0, 0, 0, 8)
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现**

本 step **不实现 CX 层，也不实现 detector 重映射**——只让两块 patch 并行跑并发两个 observable。

```python
# src/surface_code_leakage_erasure/moonwalking_transversal_cx.py
"""Transversal CNOT between two moonwalking surface code patches.

Patch 1 is the control and patch 2 the target. Both are prepared in |+>, run
`rounds // 2` rounds of stabilizer extraction, take a transversal CX, run
`rounds // 2` more, and are measured out in the X basis. Observable 0 is the
control's X logical and observable 1 the target's.

Everything here is an override on top of WalkingSCCircuitBuilder: no existing
file changes behaviour. See
docs/superpowers/specs/2026-09-04-transversal-cx-design.md section 11.
"""

from .two_patch_layout import TwoPatchMoonwalkingLayout
from .walking_circuit_builder import WalkingSCCircuitBuilder


class MoonwalkingTransversalCXBuilder(WalkingSCCircuitBuilder):
    """Transversal CX between two moonwalking (`swap_time="early"`) patches."""

    def __init__(self, d, seed=None):
        layout = d if isinstance(d, TwoPatchMoonwalkingLayout) \
            else TwoPatchMoonwalkingLayout(d)
        super().__init__(layout, seed=seed)

        self.cx_round = None
        # True while an outer, real round is being emitted. measure_stabilizers is
        # also called re-entrantly by measure_stabilizers_no_leakage to rebuild a
        # cache miss, and there its `rnd` argument is an rnd_eff cache key (0, 1 or
        # 2), NOT a round index. Those keys collide with cx_round whenever
        # rounds <= 4, so the re-entrant call must never be compared to cx_round.
        self._in_round = False
        # Whether the round being emitted is the CX round. Read by define_detectors
        # rather than re-derived from `rnd`, for the same reason.
        self._cx_detectors_active = False
        self._cx_leak_locs = set()

    def __repr__(self):
        return f"MoonwalkingTransversalCXBuilder(d={self.layout.d})"

    def get_circuit(self, rounds: int, p: float, p_leak: float = 0.0,
                    leakage_circuit_locations: dict | None = None,
                    use_cache=True, **circuit_kwargs):
        # WalkingSCCircuitBuilder.get_circuit already rejects odd `rounds`.
        self.cx_round = rounds // 2
        return super().get_circuit(
            rounds, p, p_leak,
            leakage_circuit_locations=leakage_circuit_locations,
            use_cache=use_cache, **circuit_kwargs)

    def final_observable(self, rounds, basis: str = "Z"):
        """Two observables: the control's X logical, then the target's."""
        for observable, patch in enumerate((1, 2)):
            recs = " ".join(f"rec[{idx}]" for idx in self._observable_recs(patch))
            self.circuit.append(f"OBSERVABLE_INCLUDE({observable}) {recs}")

    def _observable_recs(self, patch: int) -> list[int]:
        last_only, all_meas = self.layout.x_logical_per_patch[patch - 1]
        idxs = [self.tracker.get_curr_meas(idx) for idx in last_only]
        for idx in all_meas:
            idxs.extend(self.tracker.get_all_meas(idx))
        return idxs

    def single_ec_traceback(self, *args, **kwargs):
        raise NotImplementedError(
            "erasure-check traceback is not implemented for the transversal CX "
            "circuit; decoding is out of scope for this builder.")
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v`
Expected: 6 passed

若 `test_noiseless_circuit_is_deterministic` 失败，问题在 Task 2 的 layout 合并（此时还没有任何 CX 逻辑）。回 Task 2 排查，**不要**在本文件里打补丁绕过。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/moonwalking_transversal_cx.py tests/test_moonwalking_transversal_cx.py
git commit -m "feat: add two-patch moonwalking circuit with two X observables

No transversal CX yet -- this isolates the merged layout from the CX logic.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: 插入 transversal CX 层

**Files:**
- Modify: `src/surface_code_leakage_erasure/moonwalking_transversal_cx.py`
- Modify: `tests/test_moonwalking_transversal_cx.py`

**Interfaces:**
- Consumes: Task 3
- Produces: `measure_stabilizers(p, rnd, **kwargs)`、`_cx_sublattice() -> int`、`_cx_data_qubits() -> list[int]`、`_cx_pairs() -> list[tuple[int, int]]`、`_emit_cx_layer(p, Pauli_locations, leak_effect)`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_moonwalking_transversal_cx.py`：

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


@pytest.mark.parametrize("rounds", [2, 4, 6, 8])
def test_cx_layer_appears_exactly_once(rounds):
    # rounds <= 4 make cx_round collide with an rnd_eff cache key (0, 1 or 2),
    # which is exactly when a cache rebuild can be mistaken for the CX round.
    builder, circuit = build(rounds=rounds, p=1e-3, p_leak=0.0)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1


def test_cx_layer_pairs_every_active_data_qubit():
    builder, circuit = build(rounds=4)
    (_, pairs), = _cross_patch_cx_layers(
        circuit.flattened(), builder.layout.index_shift)
    assert len(pairs) == 9                      # d=3 -> d**2 data qubits
    shift = builder.layout.index_shift
    assert all(b == a + shift for a, b in pairs)


def test_cx_layer_survives_a_second_build_from_the_same_builder():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    for _ in range(2):
        circuit, _ = builder.get_circuit(4, 1e-3, 0.0, basis="X")
        layers = _cross_patch_cx_layers(
            circuit.flattened(), builder.layout.index_shift)
        assert len(layers) == 1


def test_skip_gates_rejected_on_the_cx_layer():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    with pytest.raises(NotImplementedError):
        builder.get_circuit(4, 1e-3, 0.2, basis="X", leak_effect="skip gates")
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v -k cx_layer`
Expected: FAIL（`assert 0 == 1`——还没有任何 CX 层）

- [ ] **Step 3: 实现**

在 `MoonwalkingTransversalCXBuilder` 中追加：

```python
    def measure_stabilizers(self, p, rnd: int, **kwargs):
        """Emit one round, preceded by the CX layer when this is the CX round.

        Two independent guards are needed; neither replaces the other.

        `_in_round` distinguishes a real round from the re-entrant call that
        measure_stabilizers_no_leakage makes to rebuild a cache miss. That call
        passes an rnd_eff cache key (0, 1 or 2) as `rnd`, which equals cx_round
        whenever rounds <= 4 -- at rounds=4, cx_round is 2 and round 3 rebuilds
        under rnd_eff=2. Without this guard that rebuild would emit a CX layer,
        and the special CX detectors, into the shared cache entry.

        `use_cache=False` on the CX round keeps that round's detectors out of the
        shared rnd_eff cache, where a later ordinary round with the same rnd_eff
        would otherwise pick them up.
        """
        if self._in_round:
            return super().measure_stabilizers(p, rnd, **kwargs)

        self._in_round = True
        self._cx_detectors_active = (rnd == self.cx_round)
        try:
            if self._cx_detectors_active:
                self._emit_cx_layer(
                    p, kwargs["Pauli_locations"], kwargs["leak_effect"])
                kwargs["use_cache"] = False
            return super().measure_stabilizers(p, rnd, **kwargs)
        finally:
            self._in_round = False
            self._cx_detectors_active = False

    def _cx_sublattice(self) -> int:
        """Which data-qubit sublattice holds the state when the CX is applied.

        Moonwalking starts with the state on sublattice 1 (see
        WalkingSCCircuitBuilder.initialize_data_qubits, which resets
        dq_indexes_str[1] for swap_time="early") and moves it by one each round,
        so after `cx_round` rounds it sits on (1 + cx_round) % 2. Sanity check:
        after rounds = 2 * cx_round this returns to 1, which is exactly the
        sublattice final_measurement reads out for "early".
        """
        return (1 + self.cx_round) % 2

    def _cx_data_qubits(self) -> list[int]:
        shift = self.layout.index_shift
        return [q for q in self.dq_indexes[self._cx_sublattice()] if q < shift]

    def _cx_pairs(self) -> list[tuple[int, int]]:
        shift = self.layout.index_shift
        return [(q, q + shift) for q in self._cx_data_qubits()]

    def _emit_cx_layer(self, p: float, Pauli_locations: str, leak_effect: str):
        """One transversal CX between the patches, occupying its own TICK.

        The instruction order mirrors make_stabilizer_gates: gates, then Pauli
        noise, then leakage noise.
        """
        if leak_effect not in ("depolarize", "tailored") and self._cx_leak_locs:
            # The "skip gates" family removes gates involving a leaked qubit from
            # the circuit. Dropping one of the d**2 transversal pairs would
            # silently change the logical operation, and the detector remapping
            # assumes every pair is present. Refuse rather than emit a wrong
            # circuit; recorded as out of scope in spec section 15.
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

`test_skip_gates_rejected_on_the_cx_layer` 依赖 `self._cx_leak_locs` 非空，
而它要到 Task 6 才被填充。本 task 先让该用例 xfail：

```python
@pytest.mark.xfail(reason="_cx_leak_locs is populated in Task 6", strict=True)
def test_skip_gates_rejected_on_the_cx_layer():
    ...
```

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v`
Expected: CX 层相关用例全部通过；**`test_noiseless_circuit_is_deterministic` 会失败**——CX 已插入而 detector 尚未重映射，这是预期。同样临时标记：

```python
@pytest.mark.xfail(reason="detector remapping lands in Task 5", strict=True)
def test_noiseless_circuit_is_deterministic():
    ...
```

`strict=True` 保证 Task 5 完成后这个 xfail 会因"意外通过"而报错，提醒摘掉标记。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过（含 2 个 xfail）

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/moonwalking_transversal_cx.py tests/test_moonwalking_transversal_cx.py
git commit -m "feat: emit the transversal CX layer, guarded against cache re-entry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: CX 边界轮的 detector 重映射

本 task 让确定性测试重新通过——它是整个设计的正确性判据。同时解决 spec 8.2 的 observable 未知。

**Files:**
- Modify: `src/surface_code_leakage_erasure/moonwalking_transversal_cx.py`
- Modify: `tests/test_moonwalking_transversal_cx.py`
- Modify: `docs/superpowers/specs/2026-09-04-transversal-cx-design.md`（回填 8.2 结论）

**Interfaces:**
- Consumes: Task 4
- Produces: `define_detectors(rnd, basis)`、`_get_detector_text(rnd, basis)`、`_cx_round_detector_text(basis)`；类属性 `_CX_PHASE = -1`

- [ ] **Step 1: 写失败的测试**

把 Task 4 加的 `xfail` 从 `test_noiseless_circuit_is_deterministic` 上摘掉，并追加：

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


@pytest.mark.parametrize("rounds", [4, 8])
def test_deterministic_with_cx(rounds):
    _, circuit = build(rounds=rounds)
    circuit.detector_error_model()


def test_cx_round_detectors_grew_by_exactly_half():
    # Half the detectors pick up their partner's records: patch 1's X checks and
    # patch 2's Z checks. Moonwalking detector tuples are 2 or 3 long to begin
    # with, so this counts total recs rather than asserting a fixed size.
    builder, circuit = build(rounds=8)
    per_round = _detector_sizes_per_round(circuit)
    cx_round = per_round[builder.cx_round]
    ordinary = per_round[builder.cx_round + 2]
    assert len(cx_round) == len(ordinary)
    assert sum(cx_round) > sum(ordinary)


def test_far_rounds_match_an_ordinary_round():
    # cx_round = 4 here, so the base class's `rnd - 2` recursion passes through it
    builder, circuit = build(rounds=8)
    per_round = _detector_sizes_per_round(circuit)
    assert builder.cx_round == 4
    for rnd in (6, 7):
        assert sorted(per_round[rnd]) == sorted(per_round[rnd - 2])
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v -k "deterministic or cx_round_detectors or far_rounds"`
Expected: FAIL——stim 报非确定性错误

- [ ] **Step 3: 实现**

在 `MoonwalkingTransversalCXBuilder` 中追加（`_CX_PHASE` 放在类定义顶部）：

```python
    # A detector-phase key WalkingSCCircuitBuilder never produces: its
    # _get_detector_text is keyed by round index (0, 1, 2, ...). A negative key
    # cannot collide with an ordinary round -- which matters because the CX
    # round's index CAN collide with an rnd_eff cache key.
    _CX_PHASE = -1

    def define_detectors(self, rnd: int, basis: str = "Z"):
        """Pick the detector phase from the flag, never from `rnd`.

        `rnd` may be an rnd_eff cache key rather than a round index (see
        measure_stabilizers), so it cannot be compared against cx_round here.
        _CX_PHASE also gives the CX round its own _get_detector_text cache entry
        and puts it out of reach of the base class's `rnd - 2` recursion, which
        walks 6 -> 4 -> 2 and can never reach a negative key.
        """
        key = self._CX_PHASE if self._cx_detectors_active else rnd
        self.circuit.append(self._get_detector_text(key, basis))

    def _get_detector_text(self, rnd: int, basis: str):
        if rnd == self._CX_PHASE:
            return self._cx_round_detector_text(basis)
        return super()._get_detector_text(rnd, basis)

    def _cx_round_detector_text(self, basis: str):
        """Detectors for the round right after the transversal CX.

        Conjugating by the CX gives
            X-plaq_1(P) -> X-plaq_1(P) . X-plaq_2(P_bar)
            Z-plaq_2(P) -> Z-plaq_1(P_bar) . Z-plaq_2(P)
        so the control's X checks and the target's Z checks each pick up their
        partner's records -- from the round BEFORE the CX, which is what keeps a
        single-qubit error after the CX flipping only two detectors. See spec
        sections 7.1-7.4.

        The merged detector list is patch 1's entries followed by patch 2's, entry
        k and entry k + half being translation images of one another, so the
        partner tuple is entry k's indices shifted.
        """
        rnd_phase = self.cx_round % 2
        detectors = self.layout.detectors[rnd_phase]
        types = self.layout.detector_check_types[rnd_phase]
        shift = self.layout.index_shift
        half = len(detectors) // 2

        recs_list = []
        for k, index_tuple in enumerate(detectors):
            idxs = [self.tracker.get_curr_meas(i) for i in index_tuple]
            patch = 1 if k < half else 2
            crosses = (patch == 1 and types[k] == "X") or \
                      (patch == 2 and types[k] == "Z")
            if crosses:
                partner = [i + shift if patch == 1 else i - shift
                           for i in index_tuple]
                idxs.extend(self.tracker.get_prev_meas(i) for i in partner)
            recs_list.append(" ".join(f"rec[{idx}]" for idx in idxs))
        return "\n".join(f"DETECTOR {recs}" for recs in recs_list)
```

- [ ] **Step 4: 运行测试；解决 observable 形式（spec 8.2）**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v`

`test_deterministic_with_cx` 有两种结果：

**(a) 通过** —— spec 8.2 的假设为假，patch 1 自己的 walking 记录已经确定。
把 spec 8.2 从"待验证"改为"已验证：不需要跨 patch 项"。

**(b) 失败**，stim 报某个 observable 不确定 —— 假设成立。**只**改 `_observable_recs`：

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
`partner_all` 的全部记录。stim 的报错是二值判据，逐次收窄即可。

若 detector 的错误也一并报出（不只是 observable），先核对两点：
(a) 交叉项用的是 `get_prev_meas` 而非 `get_curr_meas`；
(b) `crosses` 是"patch 1 的 X"与"patch 2 的 Z"，不是反过来。

**若所有候选都失败，停下报告**，不要继续试。

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过，`test_noiseless_circuit_is_deterministic` 无 xfail

- [ ] **Step 6: 回填 spec 并提交**

编辑 spec 第 8.2 节改为实测结论，同步第 14 节风险条目。

```bash
git add src/surface_code_leakage_erasure/moonwalking_transversal_cx.py tests/test_moonwalking_transversal_cx.py docs/superpowers/specs/2026-09-04-transversal-cx-design.md
git commit -m "feat: remap detectors across the transversal CX boundary

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: CX 层的 leakage 位置记账

spec 第 9 节。CX 层带 leakage，扁平位置空间要扩容。

**Files:**
- Modify: `src/surface_code_leakage_erasure/moonwalking_transversal_cx.py`
- Modify: `tests/test_moonwalking_transversal_cx.py`

**Interfaces:**
- Consumes: Task 4 的 `_cx_pairs`
- Produces: `count_leakage_locations(rounds, leak_locations)`、`ravel_leakage_locations(rounds, leak_locs_unraveled, leak_locations)`、`_n_cx_leakage_locations(leak_locations)`；`leak_locs` 字典新增 `"cx"` 键

- [ ] **Step 1: 写失败的测试**

把 Task 4 加在 `test_skip_gates_rejected_on_the_cx_layer` 上的 xfail 摘掉，并追加：

```python
def test_cx_adds_leakage_locations():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    builder.cx_round = 2
    total, per_round, per_step = builder.count_leakage_locations(4, "2-qubit gates")
    assert total == 4 * per_round + 9           # d=3 -> d**2 transversal gates


def test_cx_leakage_locations_land_in_the_cx_key():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    builder.cx_round = 2
    total, per_round, _ = builder.count_leakage_locations(4, "2-qubit gates")
    tail = list(range(4 * per_round, total))    # the whole CX block
    locs = builder.ravel_leakage_locations(4, tail, "2-qubit gates")
    assert len(locs["cx"]) == 9
    valid = {q for pair in builder._cx_pairs() for q in pair}
    assert locs["cx"] <= valid


def test_round_leakage_locations_unchanged_by_the_cx_block():
    builder = MoonwalkingTransversalCXBuilder(3, seed=0)
    builder.cx_round = 2
    _, per_round, _ = builder.count_leakage_locations(4, "2-qubit gates")
    locs = builder.ravel_leakage_locations(4, [0, 1, per_round + 3], "2-qubit gates")
    assert locs["cx"] == set()
    assert sum(len(s) for r in range(4) for s in locs[r].values()) == 3


def test_leaky_circuit_still_has_one_cx_layer():
    builder, circuit = build(rounds=4, p=1e-3, p_leak=0.05)
    layers = _cross_patch_cx_layers(circuit.flattened(), builder.layout.index_shift)
    assert len(layers) == 1


def test_leaky_circuit_still_builds():
    build(rounds=6, p=1e-3, p_leak=0.05)
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v -k "leakage or leaky or skip_gates"`
Expected: FAIL（`KeyError: 'cx'`、位置总数断言失败）

- [ ] **Step 3: 实现**

把 `get_circuit` **整体替换**为：

```python
    def get_circuit(self, rounds: int, p: float, p_leak: float = 0.0,
                    leakage_circuit_locations: dict | None = None,
                    use_cache=True, **circuit_kwargs):
        # WalkingSCCircuitBuilder.get_circuit already rejects odd `rounds`, but
        # cx_round has to be set before anything counts the CX leakage block.
        if rounds % 2 != 0:
            raise ValueError(
                "Number of rounds must be even for walking surface code.")
        self.cx_round = rounds // 2

        # Generate the leakage locations here instead of letting the base class do
        # it: our flat location space carries an extra block for the CX layer, and
        # its "cx" entry has to be split off before the round loop starts.
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

并追加：

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

        `per_round` and `per_step` come back unchanged from the base class, so
        _ravel_leakage_locations_impl -- which uses only those two and never the
        total -- keeps working on the round block untouched.
        """
        total, per_round, per_step = super().count_leakage_locations(
            rounds, leak_locations)
        return (total + self._n_cx_leakage_locations(leak_locations),
                per_round, per_step)

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

**注意**：`count_leakage_locations` 在 `WalkingSCCircuitBuilder.__init__` 里被
`cache()` 包装，包的是 MRO 解析后的绑定方法，也就是本类的覆盖版本。
缓存 key 是 `(rounds, leak_locations)`，而 `cx_round = rounds // 2` 由 `rounds`
唯一决定，所以不会有跨 `rounds` 的陈旧值。

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v`
Expected: 全部通过，无 xfail

- [ ] **Step 5: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/moonwalking_transversal_cx.py tests/test_moonwalking_transversal_cx.py
git commit -m "feat: account for leakage locations on the transversal CX layer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: 导出、hyperedge 可分解性、收尾

**Files:**
- Modify: `src/surface_code_leakage_erasure/__init__.py`（**纯追加**）
- Modify: `tests/test_moonwalking_transversal_cx.py`

**Interfaces:**
- Consumes: 前 6 个 task 的全部产出
- Produces: 包级导出 `TwoPatchMoonwalkingLayout`、`MoonwalkingTransversalCXBuilder`

- [ ] **Step 1: 写失败的测试**

```python
def test_exported_from_package_root():
    import surface_code_leakage_erasure as pkg

    for name in ("TwoPatchMoonwalkingLayout", "MoonwalkingTransversalCXBuilder"):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__, name


def test_hyperedges_are_decomposable():
    """The DEM is not graphlike (spec 7.5), but each 4-detector mechanism should
    split into two mechanisms the circuit actually contains. If this raises, the
    detector basis chosen in spec 7.4 needs revisiting."""
    _, circuit = build(rounds=4, p=1e-3)
    circuit.detector_error_model(decompose_errors=True)


def test_distance_five_builds_and_is_deterministic():
    _, circuit = build(rounds=6, d=5)
    circuit.detector_error_model()
    assert circuit.num_qubits == 2 * 58         # d=5 moonwalking patch has 58 qubits
```

- [ ] **Step 2: 运行，确认失败**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v -k "exported or decomposable or distance_five"`
Expected: `test_exported_from_package_root` FAIL

若 `test_distance_five_builds_and_is_deterministic` 的 qubit 数断言不符，
先跑 `len(WalkingSurfaceCodeLayout(5, "early").qubits)` 取实际值再改断言——
不要改成宽松断言绕过。

- [ ] **Step 3: 追加导出**

在 `src/surface_code_leakage_erasure/__init__.py` 中**只做追加**——不改动任何已有 import 或 `__all__` 条目。

现有 import 块之后追加：

```python
from .two_patch_layout import (
    TwoPatchMoonwalkingLayout,
)

from .moonwalking_transversal_cx import (
    MoonwalkingTransversalCXBuilder,
)
```

`__all__` 列表内追加（保持既有条目原样）：

```python
    # transversal CX (moonwalking)
    "TwoPatchMoonwalkingLayout",
    "MoonwalkingTransversalCXBuilder",
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_moonwalking_transversal_cx.py -v`
Expected: 全部通过

若 `test_hyperedges_are_decomposable` 失败：这不是导出问题，而是 spec 7.4 选基结论的反例。**停下报告**，不要为了让测试通过而改测试。

- [ ] **Step 5: 全量测试并确认约束**

```bash
python -m pytest tests/ -v
```

再确认"只允许加入"确实成立：

```bash
git diff --stat main...HEAD -- src/
```

Expected: `src/` 下只有 `two_patch_layout.py`、`moonwalking_transversal_cx.py` 两个**新增**文件，以及 `__init__.py` 的**纯追加**。

```bash
git diff main...HEAD -- src/surface_code_leakage_erasure/__init__.py | grep '^-[^-]'
```

Expected: 无输出。有输出说明动了既有条目，必须改回。

- [ ] **Step 6: 提交**

```bash
git add src/surface_code_leakage_erasure/__init__.py tests/test_moonwalking_transversal_cx.py
git commit -m "feat: export the moonwalking transversal CX builder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## 完成判据

- `python -m pytest tests/ -v` 全绿
- `git diff main...HEAD -- src/` 只显示两个新文件加 `__init__.py` 的纯追加
- d=3 与 d=5 的无噪声 moonwalking transversal CX 电路都通过 `detector_error_model()` 的确定性检查
- spec 8.2 的 observable 形式已回填实测结论

**不在完成判据内**（spec 第 15 节）：距离验证、任何解码路径、`erasure_sampler.py` 的双 observable 失败计数、static 与 walking(late) 分支、CX 层上的 `skip gates` 系列 leak_effect。

## 后续扩展路径

本计划刻意用直接继承而非 mixin。若之后要补 static / walking(late)：
`two_patch_layout.py` 的 `shift_and_merge` 是码型无关的，可直接复用；
`moonwalking_transversal_cx.py` 里与码型无关的部分（`measure_stabilizers` 的两道闸、
`_emit_cx_layer`、leakage 记账、`final_observable`）届时再上提为 mixin —— 
那时才有第二个使用者，抽象的形状也才看得清。

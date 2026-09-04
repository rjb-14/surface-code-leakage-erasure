# 设计：两块 rotated surface code patch 之间的 transversal CNOT

**日期：** 2026-09-04
**状态：** 设计已确认，待实现
**新增文件：** `src/surface_code_leakage_erasure/two_patch_layout.py`,
`src/surface_code_leakage_erasure/transversal_cx.py`, `tests/test_transversal_cx.py`

本文档自包含：一个全新的 session 应该能仅凭本文档 + `README.md` +
`src/surface_code_leakage_erasure/circuit_builder.py` +
`src/surface_code_leakage_erasure/walking_circuit_builder.py` 完成实现。

参考实现（不同码型，仅供对照约定）：`/Users/bryan/biased-noise-qec` 的
`src/circuits/tcnot.py`（foliated cluster state 版本）与
`tCNOT/noiseless_surface_code_trans_cx_X.stim`（circuit-based 参考电路）。

---

## 1. 目标与实验语义

在**两块全等的 rotated surface code patch** 之间做一次 transversal CNOT，
patch 1 是 control，patch 2 是 target。

时间结构（两块 patch **完全同步**，每轮并行做各自的 stabilizer measurement）：

```
两块 patch 均初始化到 |+>            (basis = "X")
round 0 .. R-1                      R 轮完整的 SE
---- transversal CX 层 ----          单独一个 TICK
round R .. 2R-1                     再 R 轮完整的 SE
final: 全部 data qubit MX
```

总轮数 `rounds = 2R`，CX 插在**第 R 轮跑完之后**（即 round index `R-1` 结束、
round index `R` 开始之前）。

两个 observable：`OBSERVABLE_INCLUDE(0)` 是 control 的 X logical，
`OBSERVABLE_INCLUDE(1)` 是 target 的 X logical。

**失败判据**：逐 shot OR —— 任一 observable 与预测不符即计一次失败，报一条 LER 曲线。

**必须同时支持三种码型**：

| 码型 | layout | builder |
|---|---|---|
| static | `SurfaceCodeLayout` | `SurfaceCodeCircuitBuilder` |
| walking | `WalkingSurfaceCodeLayout(d, swap_time="late")` | `WalkingSCCircuitBuilder` |
| moonwalking | `WalkingSurfaceCodeLayout(d, swap_time="early")` | `WalkingSCCircuitBuilder` |

---

## 2. 硬约束：只允许加入，不允许替代

**现有功能不得被改变。** 本设计的实现结果是：**零修改现有文件**，
只新增 `two_patch_layout.py`、`transversal_cx.py` 和 `tests/`。
所有对现有行为的调整都通过 mixin 覆盖完成，覆盖只在新的双 patch builder 上生效。

第 11 节逐条论证了每个覆盖点为什么不需要动基类。

---

## 3. 现有框架的相关事实

实现前必须知道的既有行为，全部已在代码中核实：

**`MeasurementTracker`**（`tracker.py`）内部存**绝对**测量时刻
`history[item]`，只在 `get_curr_meas()` 时才 `- self.t` 转成相对 `rec[]` 偏移。
因此跨 patch、跨轮引用测量记录不需要额外的绝对索引机制——两块 patch 共用一个
tracker 即可。

**static 与 walking 的 layout 接口形状不同：**

| | static | walking |
|---|---|---|
| `data_qubits` | 扁平 `frozenset` | `[frozenset, frozenset]`，按 `rnd%2` 索引 |
| `plaquettes` | 扁平 `frozenset` | 同上，list of 2 |
| detector | builder 直接遍历 `self.plaquettes` | layout 预存**索引元组** `layout.detectors[rnd%2]` |
| logical | `frozenset[Qubit]` | `(只取最后一次测量的 idx, 需要全部测量的 idx)` 二元组 |
| 轮次周期 | rnd 0 特殊，其余相同 | 周期 2，且要求 `rounds` 为偶数 |

`WalkingSCCircuitBuilder` **直接继承 `CircuitBuilder`**，不是
`SurfaceCodeCircuitBuilder` 的子类。所以"写一个继承某个 builder 的双 patch 类"
覆盖不了三种码型。

**`_measure_stabilizers_impl`（基类）末尾会调 `self.define_detectors(rnd, basis)`。**
这意味着 `measure_stabilizers_no_leakage` 缓存的整段电路文本**包含 detector**，
缓存 key 里的 `rnd_eff` 必须能区分 detector 形态不同的轮次。

**`measure_stabilizers_no_leakage` 在缓存未命中时会用 `rnd_eff` 当作 `rnd`
回调 `self.measure_stabilizers(p, rnd_eff, ...)`**（`circuit_builder.py` 中
`self.measure_stabilizers(p, rnd_eff, ...)` 那一行）。所以 `rnd_eff` 取值
同时充当"缓存 key"和"detector 形态选择器"，两者天然一致。

**`erasure_ancillas` 的既有行为**：`SurfaceCodeCircuitBuilder.__init__` 里
`self.erasure_ancillas = list(range(self.n_qubits, self.n_qubits + d + 1))`，
而 `len(remaining_qubits) == d`（d=3..9 已逐一核实）。即单 patch 下多分配了
**一个从不使用的** ancilla，`zip(remaining_qubits, self.erasure_ancillas)`
不会截断。**现有代码没有 bug，不要改这一行。** 但双 patch 下
`len(remaining_qubits) == 2d > d+1`，`zip` 会静默截断，导致部分 data qubit
没有 erasure swap 伙伴，`ec_sched == 8` 时 `self.erasure_swaps[q]` 抛 KeyError。
必须在 mixin 里重建（见第 10 节）。

walking builder **没有** erasure swap 机制（`WalkingSCCircuitBuilder` 里没有
`erasure_swaps`，其 `measure_stabilizers` 也不做 `ec_sched == 8` 的 swap），
所以第 10 节只对 static 分支适用。

---

## 4. 架构

**融合 layout + 共享 mixin。**

把两块 patch 合并成**一个** layout 对象（patch 2 的坐标和 qubit index 整体平移）。
现有 builder 的 `__init__` 全部从 `self.layout.plaquettes` 等结构派生 CNOT 调度，
所以两块 patch 会**自动在同一批 TICK 里并行**跑 4 步 SE —— 不需要动
`_measure_stabilizers_impl` 一行，也不会出现"顺序调用两个 builder 变成 8 个时间步"
的问题。

```
two_patch_layout.py
    shift_layout(layout, coord_shift, index_shift)   平移已构造好的输出结构
    TwoPatchSurfaceCodeLayout(SurfaceCodeLayout)
    TwoPatchWalkingLayout(WalkingSurfaceCodeLayout)

transversal_cx.py
    TransversalCXMixin                               CX 层 + detector 重映射 + 双 observable
    TransversalCXCircuitBuilder(TransversalCXMixin, SurfaceCodeCircuitBuilder)
    TransversalCXWalkingBuilder(TransversalCXMixin, WalkingSCCircuitBuilder)
```

`SurfaceCodeCircuitBuilder.__init__` 接受 `d : int | SurfaceCodeLayout` 并用
`isinstance` 判断，`TwoPatchSurfaceCodeLayout` 作为子类可以直接传入；
walking 侧同理（`isinstance(d, WalkingSurfaceCodeLayout)`）。

---

## 5. Layout：平移合并

### 5.1 为什么作用在输出结构上

`SurfaceCodeLayout.layout_ancillas()` 把边界 ancilla 的坐标硬编码
（`top_left = Coord(2,2)`、`starts`、`directions` 等），
`get_logicals()` 硬编码 `coord[0] == 1` / `coord[1] == 1`。
给每个 `layout_*` 方法都加 offset 参数是一次侵入式重构，且 walking 那套
（edge ancilla、按轮 plaquette、预存 detector 索引元组）要再来一遍。

改为对**已构造好的** `Qubit` / `Plaquette` 对象做平移拷贝：一套合并逻辑同时
吃 static 的扁平 frozenset 和 walking 的 `[rnd%2]` 列表，只需要一个形状感知的
map helper（"要么是集合，要么是集合的列表"）。

### 5.2 平移量

```python
coord_shift = Coord(2*d + 2, 0)     # 水平错开，两 patch 之间留一列空隙
index_shift = len(layout.qubits)    # patch 1 的真实 qubit 数（data + ancilla）
```

于是合并后 layout 的真实 qubit index 是 `0 .. 2*index_shift - 1`，
`index < index_shift` 即属于 patch 1。

erasure ancilla 不参与这里的平移：它是**虚拟** index（不进 stim 电路），
由 builder 在合并后的 `self.n_qubits` 之上分配，见第 10 节。

### 5.3 需要平移/重建的结构

static：`data_qubits`, `qubits`, `coord_to_qubit`, `index_to_qubit`,
`plaquettes`, `z_plaquettes`, `x_plaquettes`, `coord_to_plaquette`,
`index_to_plaquette`, `z_logical`, `x_logical`。

walking：以上全部，外加 `reset_indexes`, `rx_indexes`, `measure_indexes`,
`mx_indexes`, `initial_detectors`, `detectors`, `final_detectors`
（后四者是索引元组的列表，平移就是对每个 index 加 `index_shift`）。

`Plaquette.ordered_data_qubits` 持有 `Qubit` 对象引用、`DataQubit.ordered_plaquettes`
反向引用——平移时必须先建好平移后的 `Qubit` 对象映射表，再用它构造平移后的
`Plaquette`，最后回填反向引用，避免两份 layout 共享对象。

### 5.4 必须覆盖 `get_logicals`

static 版按 `coord[1] == 1` 取 X logical。patch 2 是**水平**平移的，
y 坐标不变，所以它的 X logical 行同样满足 `coord[1] == 1`——直接沿用会把
两块的 X logical 并成**一个**集合，得到的是两者的乘积，不是我们要的两个独立
observable。

双 patch layout 必须额外暴露：

```python
self.x_logical_per_patch = [<patch 1 的>, <patch 2 的>]
```

walking 的元素是 `(last_only_tuple, all_meas_tuple)` 二元组，static 的是
`frozenset[Qubit]`。

### 5.5 配对

```python
def pair(q):
    return self.coord_to_qubit[q.coord + coord_shift]
```

纯平移，对任何 qubit 都成立。**walking 里因此不需要关心 CX 那一刻哪个子晶格
是活跃的**——直接对当前活跃的 data qubit 集合逐个平移即可。

---

## 6. transversal CX 层

插在 round index `R-1` 结束、round index `R = rounds // 2` 开始之前，独占一个 TICK。
噪声顺序照抄 `make_stabilizer_gates` 的"先门、再 Pauli、再 leakage"：

```
CX  q1_0 q2_0  q1_1 q2_1 ...        d² 对（当前活跃 data qubit 全部）
DEPOLARIZE2(p) <同一个 pair 串>
DEPOLARIZE1(p) <idling qubit>       仅当 Pauli_locations == "all"
<leakage 噪声字符串>                 见第 9 节
TICK
```

`DEPOLARIZE2` 在 `Pauli_locations` 为 `"2-qubit gates"`、`"gates"`、`"all"`
时都要加（和 `make_stabilizer_gates` 一致：那里 `p > 0.0` 就加）。
`idling qubit` 是全部 qubit 减去参与 CX 的 data qubit，即两块 patch 的全部 ancilla。

**实现位置**：mixin 覆盖 `measure_stabilizers`，当 `rnd == self.cx_round`
时先把 CX 层 append 进 `self.circuit`，再调 `super().measure_stabilizers(...)`。
物理上等价于"round R-1 之后、round R 的 ancilla reset 之前"，且不需要重写
`get_circuit` 的轮次循环。

### 6.1 必须对 CX 轮禁用 no-leakage 缓存

**这是一个会真出错的陷阱。** `measure_stabilizers_no_leakage` 在缓存未命中时
会用 `rnd_eff` 当作 `rnd` **回调 `self.measure_stabilizers(p, rnd_eff, ...)`**
来重建电路。经过 MRO，这次回调命中的是 **mixin 的覆盖版本**；若此时
`rnd_eff == cx_round`，CX 层会被**再发一遍**，而且这一遍落在
`self.circuit[circuit_starting_len:]` 切片**内部**，被写进缓存。
之后每次命中该缓存都会多出一层 CX。

修法：mixin 的 `measure_stabilizers` 在 `rnd == self.cx_round` 时
**强制 `use_cache=False`** 再委托 `super()`。这样 CX 轮永不进入
`measure_stabilizers_no_leakage`，重入路径不存在。

代价：`2R` 轮里有 1 轮不走缓存，可忽略。

副作用（正面）：CX 轮永远走真实 `rnd`，所以 `_get_detector_text` 只会收到
`rnd == cx_round` 这一个取值，不需要额外的 `rnd_eff` 哨兵值。

---

## 7. Detector 重映射（核心）

### 7.1 推导

CX（control `a`，target `b`）的 Heisenberg 共轭：

$$X_a \to X_a X_b,\quad X_b \to X_b,\quad Z_a \to Z_a,\quad Z_b \to Z_a Z_b$$

transversal CX 把 patch 1 的每个 data qubit $q$ 与 patch 2 的 $\bar q$ 配对。
patch 1 支撑集 $S$ 上的 X 稳定子：

$$\prod_{q\in S} X_q^{(1)} \;\to\; \prod_{q\in S} X_q^{(1)} X_{\bar q}^{(2)}
 = X_S^{(1)}\cdot X_{\bar S}^{(2)}$$

因为两块 patch 是**平移全等**的，$\bar S$ 恰好是 patch 2 对应位置那个 X 稳定子的
支撑集。所以：

$$X\text{-plaq}_1(P) \to X\text{-plaq}_1(P)\cdot X\text{-plaq}_2(\bar P)$$
$$Z\text{-plaq}_2(P) \to Z\text{-plaq}_1(\bar P)\cdot Z\text{-plaq}_2(P)$$
$$Z\text{-plaq}_1 \to Z\text{-plaq}_1,\qquad X\text{-plaq}_2 \to X\text{-plaq}_2$$

设 CX 前最后一轮（round `R-1`）的测量结果为 $m_{R-1}$，CX 后第一轮
（round `R`）为 $m_R$。因为 CX 自逆（$U^\dagger = U$），
patch 1 的 X 稳定子在 CX 后的取值由**两个 CX 前**的测量决定：

$$U^\dagger X\text{-plaq}_1(P)\, U = X\text{-plaq}_1(P)\cdot X\text{-plaq}_2(\bar P)
\;\Longrightarrow\;
m_R^{X(1)}(P) = m_{R-1}^{X(1)}(P) \oplus m_{R-1}^{X(2)}(\bar P)$$

$$U^\dagger Z\text{-plaq}_2(P)\, U = Z\text{-plaq}_1(\bar P)\cdot Z\text{-plaq}_2(P)
\;\Longrightarrow\;
m_R^{Z(2)}(P) = m_{R-1}^{Z(2)}(P) \oplus m_{R-1}^{Z(1)}(\bar P)$$

**交叉项在 round `R-1`（CX 前），不是 round `R`。** 见 7.4：把交叉项放在
round `R` 也是确定性的，但会把 hyperedge 挂到 CX 层的主导噪声上。

### 7.2 结论

**只有 CX 后第一轮（round `R`）的 detector 特殊**，且只有两类要改：

| 稳定子 | detector | rec 数 |
|---|---|---|
| patch 1 的 **X** plaq $P$ | $m_R^{X(1)}(P) \oplus m_{R-1}^{X(1)}(P) \oplus m_{R-1}^{X(2)}(\bar P)$ | 3 |
| patch 2 的 **Z** plaq $P$ | $m_R^{Z(2)}(P) \oplus m_{R-1}^{Z(2)}(P) \oplus m_{R-1}^{Z(1)}(\bar P)$ | 3 |
| patch 1 的 Z plaq | 普通两项 | 2 |
| patch 2 的 X plaq | 普通两项 | 2 |

round `R+1` 及以后恢复正常（帧在 CX 之后不再变化）。

口诀：**X 沿 control → target 传播，所以 control 的 X check 吸收 target 的；
Z 沿 target → control 传播，所以 target 的 Z check 吸收 control 的——
且吸收的是 CX 前那一轮的记录。**

实现上三个 rec 全部来自基类已经在用的两个调用：
`get_curr_meas(自己的 ancilla)`、`get_prev_meas(自己的 ancilla)`、
`get_prev_meas(配对 ancilla)`。

### 7.3 验算

记 CX 前 patch 2 数据比特 $\bar q$ 相邻的两个 X plaquette 为 $\bar P_1,\bar P_2$，
patch 1 对应的为 $P_1, P_2$。

| 错误 | 触发的 detector | 数量 |
|---|---|---|
| CX **后** patch 1 上 $Z_q$ | $m_R^{X(1)}(P_i)$ 翻转 → patch 1 的两个 3 项 detector | 2 ✓ |
| CX **后** patch 2 上 $Z_{\bar q}$ | $m_R^{X(2)}(\bar P_i)$ 翻转 → patch 2 的两个 2 项 detector | 2 ✓ |
| CX **前** patch 1 上 $Z_q$（$Z_q^{(1)}\to Z_q^{(1)}$） | 同上第一行 | 2 ✓ |
| CX **前** patch 2 上 $Z_{\bar q}$（$\to Z_q^{(1)}Z_{\bar q}^{(2)}$） | patch 1 的两个 + patch 2 的两个 | **4，hyperedge** |

X 侧（Z detector）完全对称。

最后一行是**唯一**的 hyperedge 来源，且它本身就是一个两比特关联错误：
单比特 $Z_{\bar q}$ 经 CX 传播成 $Z_q^{(1)}Z_{\bar q}^{(2)}$。这与普通 SE 里
CNOT 上 `DEPOLARIZE2` 产生 $Z\otimes Z$ 属同一类，stim 例行分解
（它可拆成"CX 后 patch 1 上 $Z_q$"⊕"CX 后 patch 2 上 $Z_{\bar q}$"，
两者都是电路中真实存在的机制）。

### 7.4 为什么交叉项取 CX 前而不是 CX 后

把交叉项放在 round `R` 同样是确定性的：

$$U^\dagger\big(X\text{-plaq}_1(P)X\text{-plaq}_2(\bar P)\big)U = X\text{-plaq}_1(P)
\;\Longrightarrow\; m_R^{X(1)}(P)\oplus m_R^{X(2)}(\bar P) = m_{R-1}^{X(1)}(P)$$

两种写法相差一个 patch 2 自己的普通 detector，张成同一个空间，都合法——
**这是选基问题，不是对错问题。** 但错误特征完全不同：

| 错误 | 交叉项 @ round `R` | 交叉项 @ round `R-1`（采用） |
|---|---|---|
| CX 后 patch 2 上 $Z$ | **4，hyperedge** | 2 ✓ |
| CX 后 patch 1 上 $Z$ | 2 | 2 ✓ |
| CX 前 patch 1 上 $Z$ | 2 | 2 ✓ |
| CX 前 patch 2 上 $Z$ | 2 ✓ | **4，hyperedge** |

CX 层的 `DEPOLARIZE2` 在门**之后**施加（照 `make_stabilizer_gates` 的
"先门、再 Pauli"顺序），其 $I\otimes Z$ 分量正是"CX 后 patch 2 上 $Z$"。
交叉项取 round `R` 会把 hyperedge 挂在 CX 层的**主导噪声**上；取 round `R-1`
则只剩下"CX 前 patch 2 上的错误"这一类**本来就是关联错误**的机制。
**采用 round `R-1`。**

### 7.5 DEM 不是 graphlike

无论选哪种基，7.3 表格最后一行的 hyperedge 都无法消除——
它反映的是"单比特错误经 transversal CX 传播成跨 patch 两比特错误"这一物理事实。
因此：

- **`shortest_graphlike_error()` 不是本电路的正确距离工具**，它只看 ≤2 detector
  的机制，会漏掉需要 hyperedge 的最小重量逻辑错误。距离验证方案见第 13 节。
- hyperedge 应当是**可分解**的（`decompose_errors=True`），因为每个 4-detector
  机制都能拆成两个电路中真实存在的 2-detector 机制。这一点需要测试确认
  （`detector_error_model(decompose_errors=True)` 不抛异常）。
- 解码侧的影响留到解码 spec；本次范围内只需保证距离验证用对工具。

### 7.6 实现

mixin 覆盖 `_get_detector_text(rnd, basis)`，**并且必须自己把 `rnd` 归约到
基类的周期取值，不能让基类的递归自己走**：

```python
def _get_detector_text(self, rnd, basis):
    if rnd == self.cx_round:
        return <特殊文本>
    if rnd == 0:
        return super()._get_detector_text(0, basis)
    return super()._get_detector_text(1 + (rnd - 1) % 2, basis)
```

**为什么不能直接 `super()._get_detector_text(rnd, basis)`**：
`WalkingSCCircuitBuilder._get_detector_text` 对 `rnd > 2` 做
`return self._get_detector_text(rnd - 2, basis)`，而 `self.` 会再次派发到
**mixin 的覆盖版本**。若 `cx_round = 4`，查询 `rnd = 6` 会递归到 `rnd = 4`
被 mixin 拦截，**把 CX 边界轮的特殊 detector 文本用到了一个普通轮上**。
上面的归约把 `rnd ≥ 1` 一次性压到 `{1, 2}`（正是 walking 递归的收敛值），
递归因此不会经过 `cx_round`。

static 侧同样适用：基类只用 `{0, 1}`，`1 + (rnd-1)%2 ∈ {1, 2}` 中的 `2`
会被基类折叠回 `1`，行为不变。

配对查找用 `self.layout.pair()`（第三个 rec 用配对 ancilla 的 `get_prev_meas`）。static 侧遍历 `self.x_plaquettes` / `self.z_plaquettes`
并按 patch 归属分流；walking 侧的 detector 是 layout 预存的索引元组，
需要按 ancilla index 判断归属（`index < index_shift` 即 patch 1）并追加配对项。

`__init__` 里的 `self._get_detector_text = cache(self._get_detector_text)`
包的是 MRO 解析后的**绑定方法**，也就是 mixin 的版本，所以覆盖后仍然被缓存。
`rnd == cx_round` 的相对偏移在每次 build 中都相同，缓存它是安全的。

---

## 8. Observable

### 8.1 static

`OBSERVABLE_INCLUDE(0)` = patch 1 的 X logical 的最终 MX 记录，
`OBSERVABLE_INCLUDE(1)` = patch 2 的。与参考电路
`tCNOT/noiseless_surface_code_trans_cx_X.stim` 的约定一致（d=3 时各 3 个 rec）。

两块都在 $|+\rangle$，$CX|{+}{+}\rangle = |{+}{+}\rangle$，所以两者各自确定。

**为什么不用 Heisenberg 传播后的形式**：记
$A = X_L^{(1)}$、$B = X_L^{(2)}$、$A' = X_L^{(1)}X_L^{(2)} = A\cdot B$。
$\{A, B\}$ 与 $\{A', B\}$ 生成**同一个群**，在"任一翻转即失败"的判据下
失败率完全相同。static 下两种约定等价，取记录数更少的那个。

### 8.2 walking / moonwalking：待验证

walking 的 observable 不只是最终测量：`_get_final_observable_text` 会对
`observable[1]` 里的 qubit 调 `tracker.get_all_meas()`，把**跨轮的中间测量**
全部 XOR 进来（data qubit 走动时被测量并重置）。这些中间测量横跨 CX 前后，
而 CX 之后被追踪的算符已经是 $X_L^{(1)}X_L^{(2)}$ 而不是 $X_L^{(1)}$。

因此这里**不是"选哪组基"的问题**（8.1 的等价性论证不适用），而是
**直接拼 patch 1 的 walking 记录很可能根本不确定**。

**假设（待验证）**：

> observable 0 = [patch 1 的完整 walking 记录] ⊕ [patch 2 walking 记录中
> **落在 CX 之后**的那一部分]
>
> observable 1 = [patch 2 的完整 walking 记录]

**验证方式**：无噪声电路调 `stim.Circuit.detector_error_model()`。
observable 不确定时 stim 直接报错，所以这是一个二值判定，不需要猜。
若假设为假，按 stim 的报错逐步收窄（候选变体：只加 CX 后 target 的
最终测量、或连同 CX 后所有中间测量）。

static 没有这个问题（只有最终测量，不跨 CX）。

### 8.3 实现

mixin 覆盖 `final_observable(self, rounds, basis)`——static 与 walking 的
**这个方法签名相同**，所以一份实现即可（不要覆盖 `_get_final_observable_text`，
两者签名不同：static 是 `(basis)`，walking 是 `(rounds, basis)`）。

`_observable_cache_key` 也要覆盖：CX 位置依赖 `rounds`，所以必须返回
`(rounds, basis)`，不能沿用 static 的 `(basis,)`。

---

## 9. leakage 位置记账

用户已确认：**CX 层带 Pauli + leakage**。

现有 `_ravel_leakage_locations_impl` 假设每轮位置数均匀（`loc // leak_locs_per_round`）。
插入 CX 层会破坏这个均匀划分。

**做法**：把扁平索引区间切成两段。

```
[0, rounds * per_round)                    普通轮，原样委托 super()
[rounds * per_round, + n_cx)               CX 层，mixin 自己映射
```

mixin 覆盖 `count_leakage_locations`，返回
`(rounds * per_round + n_cx, per_round, per_step)`——
**`per_round` 和 `per_step` 与基类返回值保持一致**，只有 `total` 变大。

这样是安全的，因为 `_ravel_leakage_locations_impl` 在循环里只用
`leak_locs_per_round` 和 `leak_locs_per_step`，**不用 `total_leak_locs`**。

`n_cx` 取值：

- `leak_locations == "2-qubit gates"`：CX 门对数（当前活跃 data qubit 数，即 d²）
- `leak_locations == "all"`：`self.n_qubits`（idling qubit 也能泄漏）

mixin 覆盖 `ravel_leakage_locations`：把低段列表交给
`super().ravel_leakage_locations(...)`，高段自己映射到 CX 门（哪一侧 qubit 泄漏
用 `self.rng.choice(2)` 随机选，与基类一致），存进返回 dict 的
**专用 key `"cx"`**。

`get_circuit` 用 `leakage_circuit_locations.get(rnd, {})` 按整数轮次取值，
字符串 key `"cx"` 不会被轮次循环误取。mixin 的 `get_circuit` 在委托 `super()`
之前把它取出存到 `self._cx_leak_locs`，供第 6 节的 CX 层使用。

`count_leakage_locations` 在 `__init__` 里被 `cache()` 包装，覆盖后同样保持缓存。

**注意**：CX 层引入的 leakage 会置位 `self.leaked`，使 round `R` 的
`measure_stabilizers` 走非缓存路径（基类里 `len(self.leaked) == 0` 的判断）。
这是正确行为。

---

## 10. erasure swap（仅 static 分支）

mixin 的 `__init__` 在 `super().__init__()` 之后重建：

```python
n_remaining = len(remaining_qubits)          # 双 patch 下是 2d
self.erasure_ancillas = list(range(self.n_qubits, self.n_qubits + n_remaining))
self.erasure_swaps.update(zip(remaining_qubits, self.erasure_ancillas))
self.erasure_unswaps.update(zip(self.erasure_ancillas, remaining_qubits))
```

并同步更新 `measure_stabilizers` 传给 `_measure_stabilizers_impl` 的
`measure_set`（基类是 `self.stab_ancillas_set | set(self.erasure_ancillas)`）。

walking builder 没有 erasure swap 机制，此节不适用。

---

## 11. 为什么不需要修改现有文件

逐条论证"只允许加入"如何满足：

| 需要的行为改变 | 手段 | 为什么不用动基类 |
|---|---|---|
| 插入 CX 层 | mixin 覆盖 `measure_stabilizers` | `rnd == cx_round` 时先发 CX 再调 `super()`，轮次循环不用重写 |
| CX 边界轮的 detector | mixin 覆盖 `_get_detector_text` | `__init__` 的 `cache()` 包的是 MRO 解析后的绑定方法，自动包到 mixin 版本 |
| CX 轮避开缓存重入 | mixin 在该轮强制 `use_cache=False` | `use_cache` 本来就是两个基类 `measure_stabilizers` 的既有参数（见 6.1） |
| 双 observable | mixin 覆盖 `final_observable` + `_observable_cache_key` | 两个基类的 `final_observable` 签名相同 |
| leakage 位置 | mixin 覆盖 `count_leakage_locations` / `ravel_leakage_locations` | 保持 `per_round`/`per_step` 不变，低段委托 `super()` |
| erasure ancilla 数量 | mixin `__init__` 内重建 | 现有单 patch 行为本来就正确（多一个不用的 ancilla），不碰 |
| `single_ec_traceback` | mixin 覆盖为 `NotImplementedError` | 解码侧，本次范围外，显式挡住而不是静默给错结果 |

`get_circuit`：mixin 覆盖，作用只有三件——校验 `rounds` 为偶数、
设 `self.cx_round = rounds // 2`、取出 `leakage_circuit_locations["cx"]`，
然后 `return super().get_circuit(...)`。

---

## 12. 文件清单

**新增**

- `src/surface_code_leakage_erasure/two_patch_layout.py`
  - `shift_layout(layout, coord_shift, index_shift)`
  - `TwoPatchSurfaceCodeLayout(SurfaceCodeLayout)`
  - `TwoPatchWalkingLayout(WalkingSurfaceCodeLayout)`
  - 两者均暴露 `x_logical_per_patch`、`z_logical_per_patch`、`pair()`、
    `coord_shift`、`index_shift`
- `src/surface_code_leakage_erasure/transversal_cx.py`
  - `TransversalCXMixin`
  - `TransversalCXCircuitBuilder(TransversalCXMixin, SurfaceCodeCircuitBuilder)`
  - `TransversalCXWalkingBuilder(TransversalCXMixin, WalkingSCCircuitBuilder)`
- `tests/test_transversal_cx.py`

**修改**

- `src/surface_code_leakage_erasure/__init__.py`：只**追加** import 和 `__all__` 条目
- `pyproject.toml`：追加 `[project.optional-dependencies]` 的 `dev = ["pytest"]`

除以上两处纯追加外，不修改任何现有文件。

---

## 13. 验证计划

`tests/test_transversal_cx.py`，pytest（与 `/Users/bryan/biased-noise-qec/tests/` 的习惯对齐）。

1. **确定性**：`p=0`、`p_leak=0` 下 `circuit.detector_error_model()` 不抛异常
   （detector 与 observable 均确定）。static / walking(late) / moonwalking(early)
   各一个用例。**这是第 8.2 节假设的判定器。**
2. **距离**：`p>0` 且 Pauli-only（`p_leak=0`）下最小不可探测逻辑错误的重量等于 `d`。

   **不能用 `shortest_graphlike_error()`**——7.5 已论证本电路的 DEM 含 hyperedge，
   该函数只看 ≤2 detector 的机制，会漏掉需要 hyperedge 的最小重量逻辑错误，
   给出的距离可能偏大。

   改用仓库里已有的 house 方法（`scripts/min_fault_weight_pauli.ipynb`）：
   `flatten_dem_lines(circuit.detector_error_model(decompose_errors=False))`
   拿到每个独立物理噪声机制的 `(detector 集合, logical 集合)`，
   然后按重量递增搜索"合并后 detector 集合为空、logical 集合非空"的组合。
   未分解的 DEM 天然包含 hyperedge 行，所以这个搜索是 hyperedge-safe 的。

   组合爆炸的控制：穷举 `C(n, w)` 在 d=3、`rounds = 2d` 下的 `w ≤ 3` 是否可接受
   需实测。若太慢，退到 `circuit.search_for_undetectable_logical_errors(...)`
   （stim 自带、支持 degree > 2 的边），并在 d=3 最小配置下用穷举法交叉验证一次。
   **这两条路都要在实现期实测确认，不要预设。**

3. **hyperedge 可分解**：`circuit.detector_error_model(decompose_errors=True)`
   不抛异常。7.5 论证了每个 4-detector 机制都能拆成两个电路中真实存在的
   2-detector 机制，但这是论证不是验证，需要测试兜住。
4. **observable 数量**：`circuit.num_observables == 2`。
5. **CX 层存在性**：CX 后第一轮之前恰好有一层 `d²` 个跨 patch 的 `CX`，
   且两个 qubit index 分属两块 patch（`< index_shift` 与 `>= index_shift`）。
6. **detector 形态**：CX 后第一轮里，patch 1 的 X detector 与 patch 2 的 Z
   detector 是 3 项（第三项是**配对 ancilla 在 CX 前那一轮**的记录），
   其余是 2 项。
7. **回归**：现有 `SurfaceCodeCircuitBuilder` / `WalkingSCCircuitBuilder`
   在相同参数下生成的电路与引入本次改动前**逐字节相同**——
   直接兑现第 2 节"只允许加入"的约束。
8. **CX 层不重复**（6.1 的陷阱）：整个电路里跨 patch 的 CX 层**恰好出现一次**。
   失效模式是缓存重入把 CX 发两遍，且第二遍进了缓存——静默产生错误电路，
   必须有测试兜住。同时对 `p_leak > 0` 和 `p_leak = 0` 各测一次
   （后者才会走缓存路径）。
9. **CX 后远端轮的 detector 正常**（7.6 的陷阱）：取 `rounds` 使
   `cx_round ≥ 2` 且存在 `rnd > cx_round + 1` 的轮次（如 walking d=3、
   `rounds = 8`，`cx_round = 4`），断言 round `cx_round + 2` 的 detector
   全是 2 项。失效模式是 walking 的 `rnd-2` 递归撞进 `cx_round`，
   把特殊 detector 文本用到普通轮上——同样是静默错误。

---

## 14. 未决与风险

- **8.2 的 walking observable 形式**是本设计的主要未知。判定器现成
  （stim 的确定性检查），若假设为假需按报错收窄，属于实现期的小幅迭代，
  不影响架构。
- **距离搜索的可行性**（13 节第 2 条）：穷举未分解 DEM 行的组合是精确且
  hyperedge-safe 的，但 `C(n, w)` 可能爆炸。实现期先实测 d=3 的规模，
  再决定是否退到 `search_for_undetectable_logical_errors`。**不要预设哪条路可行。**
- **hyperedge 的可分解性**（13 节第 3 条）：7.5 给的是论证不是验证。
  若 `decompose_errors=True` 抛异常，说明存在无法拆成两个真实 2-detector
  机制的 hyperedge，需要回头重新审视 detector 基的选择。
- **walking 下 CX 时刻的活跃子晶格**：设计上用纯平移配对绕开了这个问题
  （5.5 节），但实现时需确认 `rounds // 2` 落点处两块 patch 确实同相位——
  两者同步推进，应当自动成立，测试 1 会覆盖。
- **距离是否真为 d**：R 轮 + CX + R 轮，R = rounds/2。若 R 太小距离会被时间
  方向截断，测试时取 `rounds >= 2*d`。

---

## 15. 范围外（下一个 spec）

- 全部解码内容：`ErasureDecoder` / `ModifiedMLEDecoder` 对双 observable
  与跨 patch 超图的适配
- `single_ec_traceback`（本次显式 `NotImplementedError`）
- `erasure_sampler.py` 的双 observable 失败计数
  （`np.sum(predictions != observable_flips)` 需改成逐 shot OR：
  `np.any(pred != flips, axis=1).sum()`）——判据已定，实现留到解码 spec
- `CircuitParameters` / `SurfaceCodeErasureSampler` 对新 builder 的接入

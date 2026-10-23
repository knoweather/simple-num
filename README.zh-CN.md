# simple-num

纯 Go 的 **N 维数组**库，API 对标 NumPy —— **零 cgo、零第三方依赖（只用标准库）**。
凡是语义允许的地方都与 NumPy **逐位一致**，reference 由现场运行的 NumPy（2.5.4）生成，
同一份代码还能编译成 WebAssembly。

<p>
<a href="https://pkg.go.dev/github.com/knoweather/simple-num"><img src="https://pkg.go.dev/badge/github.com/knoweather/simple-num.svg" alt="Go Reference"></a>
<a href="go.mod"><img src="https://img.shields.io/badge/go-1.27%2B-00ADD8?logo=go" alt="Go 1.27+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
</p>

[English](./README.md) | **简体中文**

```go
a, _ := num.FromSlice([]float64{1, 2, 3, 4, 5, 6})
m, _ := a.Reshape([]int{2, 3})

b, _ := num.Transpose(m)             // 3×2 视图，不拷贝
_ = b.AsContiguous()                 // 需要独立内存时显式落连续

s, _ := num.Sum(m, []int{0}, false)  // [5, 7, 9]

// npy 互操作：NumPy 写出的文件本库能读，本库写的文件 NumPy 也能读
_ = num.SaveNPY("a.npy", m)
back, _ := num.LoadNPY("a.npy")
```

- **NumPy 语义，Go 用法**：广播、dtype 提升（NEP 50 规则）、视图与拷贝的区分、
  `axis`/`keepdims`、NaN 传播、整型溢出回绕 —— 全部按**实测的** NumPy 行为对齐，不靠猜。
- **该逐位就逐位**：归约走 pairwise 求和（同输入顺序下 `Sum` 与 `np.sum` 末位一致）、
  整型回绕、`-0.0` 与 NaN 位型都保留、`sign(-0.0)` 给 `+0`（与 NumPy 同）。
- **两个方向的 I/O 都验过**：`.npy` / `.npz` / 裸字节 / 文本。读端支持 v1/v2/v3 头；
  写端产出的头 NumPy 能直接解析（用现场 Python 会话加载本库写的文件验证过）。
- **不只是算术**：`linalg`（matmul/einsum/solve/inv/det/eig/svd）、
  `random`（逐位复刻 MT19937 `RandomState`）、`fft`（radix-2 + Bluestein）、
  掩码数组 `ma`、排序统计、padding、卷积、集合运算、`polyfit`/`unwrap`，以及 WASM 桥接。
- **可直接上 WASM**：`wasm/` 产出 `simple-num.wasm` 与 Node/JS API —— 不需要服务端、
  不需要 C 工具链（`CGO_ENABLED=0 go build ./...` 通过）。

> [!NOTE]
> 🚀 **初始版本，金丝雀测试中。** 遇到问题欢迎开
> [issue](https://github.com/knoweather/simple-num/issues)！

## 能力

| 能力 | 状态 |
|---|---|
| dtype：`bool`、`int8…int64`、`uint8…uint64`、`float16/32/64`、`complex64/128`、定宽 `S`/`U`、`datetime64` | ✅ |
| 视图：`Reshape`/`Transpose`/`Squeeze`/`ExpandDims`/`SwapAxes`/`BroadcastTo`/`Sub`（零拷贝）、显式 `AsContiguous` | ✅ |
| 索引：基本切片、花式索引、布尔掩码、`Take`/`TakeAlongAxis`/`Put`/`PutMask`/`Place` | ✅ |
| 逐元素运算（含广播 + NEP 50 提升）：四则、比较、逻辑、位运算、取整、超越函数（`tanh`/`cbrt`/`log2`/`expm1`/`log1p`/`sinc` 等） | ✅ |
| 归约（任意 `axis` + `keepdims`）：`Sum`/`Prod`/`Mean`/`Var`/`Std`/`Min`/`Max`/`AllTrue`/`Any`/`Median` 以及 `Nan*` 全家 | ✅ |
| 排序与统计：`Sort`/`Argsort`/`Median`/`Percentile`（**NumPy 全部 13 种 method**）/`Quantile`/`NanPercentile`/`Diff`/`Unique`（含 `axis`）/`SearchSorted`/`Digitize`/`Histogram`/`Histogram2D`/`Bincount(weights)` | ✅ |
| 形状操作：`Atleast1D/2D/3D`、`Flip`/`Roll`/`Repeat`/`Tile`、`Split`/`ArraySplit`/`SplitAt`/`H·V·DSplit`、`Append`/`Insert`/`Delete`、`Stack`/`VStack`/`HStack`/`ColumnStack`/`DStack`/`Concatenate` | ✅ |
| 线性代数：`Dot`/`MatMul`/`Inner`/`Outer`/`Einsum`/`Tensordot`、`Det`/`Inv`/`Solve`（部分选主元 LU）、`Norm`（向量 0/1/2/±inf；矩阵 fro/1/2/-2/inf）、`Eigvalsh`/`Eigh`（循环 Jacobi）、`Svd`/`Svdvals`（单边 Jacobi） | ✅ |
| 浮点辅助：`IsClose`/`AllClose`、`Invert`、`Round(decimals)`、`Clip`、`Where`、`Select`、`Piecewise`、`IsNan`/`IsInf`/`IsFinite`/`IsReal` | ✅ |
| 集合运算：`IsIn`/`Intersect1D`/`Union1D`/`Setdiff1D`/`Setxor1D`（NaN 不参与匹配，与 NumPy 同） | ✅ |
| 插值与信号：`Interp`、`MeshGrid`、`Gradient`、`Convolve`/`Correlate`（full/same/valid）、`Trapz`、`Polyval`、`Polyfit`、`Unwrap` | ✅ |
| `random`：逐位复刻 NumPy legacy `RandomState`/MT19937 —— `Seed`/`RandomSample`/`StandardNormal`/`Uniform`/`Normal`/`RandInt`/`Permutation`/`Shuffle` | ✅ |
| `fft`：`FFT`/`IFFT`/`RFFT`/`IRFFT`/`FFTFreq` —— 2 的幂走 radix-2，**任意长度走 Bluestein**（O(n log n)） | ✅ |
| `ma`（掩码数组）：数据 + 掩码、`Filled`/`Compressed`/`Count`、掩码 `Sum`/`Mean`/`Min`/`Max`/`Median`/`Var`/`Std`、谓词 `MaskedEqual`/`Less`/`Greater`/`Inside`/`Outside` | ✅ |
| I/O：`SaveNPY`/`LoadNPY`/`SaveNPYBytes`/`LoadNPYBytes`、`SaveZ`/`SaveZCompressed`/`LoadZ`（npz）、`FromBuffer`（零拷贝）、`FromFile`/`ToFile`、`SaveTxt`/`LoadTxt` | ✅ |
| ufunc 协议：`out=` 走 `BinaryOut`/`UnaryOut`，`where=` 走 `BinaryWhere`/`UnaryWhere`，另有 `Reduce`/`Accumulate`/`UfuncOuter` | ✅ |
| WASM 桥接（`wasm/`）：数组即 `{dtype, shape, data}`，`op`/`unary`/`reduce`/`linalg`/`cumulative` 分发器，npy 往返 | ✅ |

## 性能

在一台 Windows/amd64 机器上测得，`float64`、单进程。测法：每个用例**迭代到累计
≥400 ms 再除以迭代次数** —— 这台机器上单次计时量化太粗，不可信。

| 用例 | 耗时 | 备注 |
|---|---|---|
| `Add` 1e6 连续 | 1.21 ms | ≈21 GB/s |
| `Add` 1e6 标量广播 | 1.19 ms | 标量快速路径 |
| `Sum` 1e6（pairwise） | 1.15 ms | |
| `Take` 1e6（1e6 个下标） | 8.41 ms | gather |
| `Cast` f8→f4 1e6 | 0.55 ms | 类型化批量内核 |
| `CastUnsafe` f8→i4 1e6 | 0.94 ms | |
| `LoadNPY` 1e6 f8 | 1.89 ms | |
| `Arange` 1e6 | 3.09 ms | |
| `LinSpace` 1e6 | 1.08 ms | |
| `AsContiguous`（1000×1000 转置视图） | 5.54 ms | |
| `MatMul` 512³ | 6.0 ms | 行切分并行（≈44 GFLOP/s） |
| `Einsum` `ij,jk->ik` 512³ | 6.3 ms | 收缩图，与 `MatMul` 持平 |
| `FFT` n=1024 | 0.055 ms | radix-2 |
| `FFT` n=4000 | 0.77 ms | Bluestein（此前朴素 DFT 要 256 ms） |

几个值得说明的点：

- `MatMul` 是纯 Go、无 SIMD。缓存分块与内层展开**都试过并实测更慢**
  （6.99 / 6.32 ms vs 6.01 ms）——Go 编译器不会把这些循环向量化，
  4 行展开 + 行并行已经是这条路的平台期。
- `Einsum` 原来是对标签组合做完全枚举（512³ 要 4522 ms）。现在建贪心收缩树、
  最终落到 `MatMul`：**约 700×** 提速，且在 64/128/256 与 `MatMul` 逐位一致。
- `FFT` 非 2 的幂长度原来是朴素 O(n²) DFT；引入 Bluestein 后 n=4000 从 256 ms
  降到 0.77 ms（**约 330×**），与 `np.fft.fft` 的相对误差约 1e-14。

## 安装

```bash
go get github.com/knoweather/simple-num
```

只用标准库 —— `CGO_ENABLED=0 go build ./...` 可在 windows/amd64、linux/amd64、
darwin/arm64、windows/arm64 上通过（交叉编译属于发布门槛的一部分）。

## WASM

```sh
sh wasm/build.sh             # Linux / macOS / Git Bash → wasm/simple-num.wasm 与 wasm_exec.js
pwsh wasm/build.ps1          # Windows PowerShell
node wasm/smoke/smoke.mjs    # 构建 + 加载 + 跑一遍 JS API 断言
```

桥接层挂出 `globalThis.simpleNum`，数组对象以 `{dtype, shape, data}` 跨边界，
并提供 `op` / `unary` / `reduce` / `linalg` / `cumulative` 分发器与 npy 进出。
它是 Go API 的**精心挑选子集** —— [`wasm/README.zh-CN.md`](wasm/README.zh-CN.md)
写明当前暴露了什么、还计划补什么（npz、percentile、完整 `Eigh`/`Svd`、`Einsum`、
掩码数组、FFT、random 等）。

## 验证方式

这里没有"看起来对"——每一条行为都钉在真实 NumPy 上。

做法很直接：每个测试用例都在 Python 里用 NumPy 2.5.4 跑同一运算，把它的输出字节和
输入一起存进 `xcheck/reference/`（`.npy` 文件 + 一份 JSON 清单）；Go 测试再拿本库结果
与这些**预先存下来的字节**比对。对天然做不到逐位相同的运算（FFT、LU 这类末位会有差异），
用例会声明一个相对容差。

- **与 NumPy 的参考数据对拍**（`xcheck/`）：由 `xcheck/gen_reference.py` 产出
  **477 案 / 484 子测试**，覆盖整型/浮点/复数、广播、±0/NaN/Inf、整型溢出、
  按轴归约、gather/掩码索引、多输出算子与整个函数面。两种比较区：逐位
  （完全相等，含 NaN 位型）与相对容差（FFT、LU、`sinc`）。
- **未注册即失败**：manifest 里的算子若没有对应实现，测试直接失败，而不是悄悄算通过。
- **横向审计**：只读算子不修改入参；转置/切片视图与连续副本结果一致（含写回视图）；
  并行 `MatMul` 在不同 `GOMAXPROCS` 下逐位稳定。
- **覆盖率**：85.3%（门槛 85%）。
- **I/O 双向互操作**：NumPy 能加载本库写的文件，本库也能加载 NumPy 写的文件。

## 已知限制

- 不做结构化/记录型 dtype，不做 `timedelta64` 运算，不做 `nditer`；
  `datetime64` 只支持构造/索引/比较。
- `linalg` 只覆盖稠密实值：没有复数解法、没有 `lstsq`、没有一般（非对称）`eig`、
  没有 `qr`/`cholesky`。
- `Polyfit` 用法方程求解，会平方条件数 —— 低阶够用，高阶或 `x` 跨度很大时不行。
- `Einsum` 不支持省略号形式（`...`）；同一操作数内重复标签（取对角）会回退到枚举实现
  （结果正确，复杂度 ∏(各维长度)）。
- `Convolve`/`Correlate` 只支持 1-D，复杂度 O(n·m)，没有 FFT 快速路径。
- `ma` 覆盖构造、掩码归约、四则与谓词 —— 不含 `ma.corrcoef`
  （NumPy 本身在该场景会给出超过 ±1 的值，不值得复刻）。
- 文本 I/O（`SaveTxt`/`LoadTxt`）支持 1-D/2-D、分隔符与 `skiprows`；
  不支持 `usecols`、不支持逐列 dtype。

## 目录结构

```
array.go, dtype.go, view.go, broadcast.go, ufunc.go   核心：内存模型、dtype、视图、分发
impl_*.go                                             按领域拆分的算子内核
xcheck/                                               NumPy 生成的参考数据 + 对拍框架
wasm/                                                 WebAssembly 桥接与 JS 冒烟测试
```

## 与 NumPy 的功能对照

图例：✅ 已支持 · ⚠️ 部分支持（见备注）· 🚧 尚未支持，计划中。

| 领域 | 已支持 | 尚未支持（计划） |
|---|---|---|
| dtype | `bool`、`int8…int64`、`uint8…uint64`、`float16/32/64`、`complex64/128`、定宽 `S`/`U`、`datetime64` | `float128` 🚧、`timedelta64` 🚧、结构化/记录 dtype 🚧、`object` 🚧 |
| 构造 | `FromSlice`、`Zeros`、`Ones`、`Empty`、`Full`、`Eye`、`Identity`、`Arange`、`LinSpace`、`MeshGrid`、`Diag`、`Diagflat` | `LogSpace` 🚧、`GeomSpace` 🚧、`FromIter` 🚧、`FromString` 🚧、`Vander` 🚧、`Tri` 🚧 |
| 形状与视图 | `Reshape`、`Ravel`、`Transpose`、`Squeeze`、`ExpandDims`、`SwapAxes`、`BroadcastTo`、`Sub`（能零拷贝的都零拷贝） | `Moveaxis` 🚧、`Resize` 🚧、`BroadcastArrays` 🚧 |
| 拼装与切分 | `Concatenate`、`Stack`/`VStack`/`HStack`/`ColumnStack`/`DStack`、`Split`/`ArraySplit`/`SplitAt`/`H·V·DSplit`、`Append`/`Insert`/`Delete`、`Flip`/`FlipUD`/`FlipLR`、`Roll`、`Repeat`、`Tile`、`Pad`（10 种模式，含 `stat_length`）、`Atleast1D/2D/3D` | `Rot90` 🚧、`Triu`/`Tril` 的下标辅助 🚧 |
| 索引 | 基本切片、花式索引、布尔掩码、`Take`、`TakeAlongAxis`、`Put`、`PutAlongAxis`、`PutMask`、`Place`、`FillDiagonal`、`Compress`、`Nonzero`、`Argwhere`、`Flatnonzero`、`CountNonzero` | `Choose` 🚧、`TrilIndices` 🚧、`Unravel` 🚧 |
| 算术 | `Add`/`Subtract`/`Multiply`/`Div`/`FloorDivide`/`Remainder`/`Power`、`Negative`/`Positive`/`Abs`/`Fabs`/`Square`/`Reciprocal`、`Maximum`/`Minimum`、`Clip`、`Round`/`Around`、`Where`/`Select`/`Piecewise` | 位运算 `&`/`\|`/`^`/`<<`/`>>` 🚧（目前只有 `Invert`） |
| 比较与逻辑 | `Equal`/`NotEqual`/`Less`/`LessEqual`/`Greater`/`GreaterEqual`、`LogicalAnd/Or/Xor/Not`、`AllTrue`/`Any`、`IsNan`/`IsInf`/`IsFinite`/`IsReal`、`IsClose`/`AllClose` | — |
| 超越函数 | `Exp`/`Log`/`Log2`/`Log10`/`Expm1`/`Log1p`/`Sqrt`/`Cbrt`/`Sin`/`Cos`/`Tan`/`Arcsin`/`Arccos`/`Arctan`/`Arctan2`/`Sinh`/`Cosh`/`Tanh`/`Arcsinh`/`Arccosh`/`Arctanh`/`Sinc`、`Deg2Rad`/`Rad2Deg`、`Sign`、`Floor`/`Ceil`/`Trunc`/`Rint` | `Exp2` 🚧 |
| 归约 | `Sum`/`Prod`/`Mean`/`Var`/`Std`/`Min`/`Max`/`ArgMin`/`ArgMax`/`Median`/`Cumsum`/`Cumprod`，`Nan*` 全家，任意 `axis` + `keepdims` | `Average`（加权）🚧、`PtP` 🚧、`NanProd`/`NanCumsum` 🚧 |
| 统计 | `Percentile`（13 种 method 全支持）、`Quantile`、`Median`、`Histogram`、`Histogram2D`、`Bincount(weights)`、`Cov`、`CorrCoef`、`Diff` | 掩码数据的 `CorrCoef` 🚧 |
| 排序与集合 | `Sort`、`Argsort`、`SearchSorted`、`Digitize`、`Unique`（含 `axis`）、`UniqueIdx`（返回下标/计数）、`IsIn`/`Intersect1D`/`Union1D`/`Setdiff1D`/`Setxor1D` | `Partition`/`Argpartition` 🚧 |
| 线性代数 | `Dot`、`MatMul`、`Inner`、`Outer`、`Einsum`、`Tensordot`、`Det`、`Inv`、`Solve`、`Norm`、`Eigvalsh`、`Eigh`、`Svd`、`Svdvals` | 一般 `Eig`/`Eigvals`（非对称）🚧、`QR` 🚧、`Cholesky` 🚧、`Lstsq` 🚧、`Pinv` 🚧、`Slogdet` 🚧、`Kron` 🚧、复数求解 🚧 |
| 随机数 | 逐位复刻 legacy `RandomState`/MT19937：`Seed`、`RandomSample`、`StandardNormal`、`Uniform`、`Normal`、`RandInt`、`Permutation`、`Shuffle` | `Generator`/PCG64 系列 🚧、`Choice`/`Beta`/`Gamma`/`Poisson` 等 🚧 |
| FFT | `FFT`/`IFFT`/`RFFT`/`IRFFT`/`FFTFreq`（radix-2 + Bluestein，任意长度） | N 维变换 🚧、`FFTShift`/`FFT2` 🚧 |
| 掩码数组 | `MaskedArray`（数据+掩码）、`Filled`/`Compressed`/`Count`、掩码 `Sum`/`Mean`/`Min`/`Max`/`Median`/`Var`/`Std`、`MaskedAdd/Subtract/Multiply/Divide`、`MaskedInvalid`/`MaskedWhere`、8 个谓词构造器 | `ma.corrcoef` 🚧（见限制说明）、更多 `masked_*` 构造器 ⚠️ |
| I/O | `SaveNPY`/`LoadNPY`（含 bytes 版本）、`SaveZ`/`SaveZCompressed`/`LoadZ`、`FromBuffer`（零拷贝）、`FromFile`/`ToFile`、`SaveTxt`/`LoadTxt` | `Memmap` 🚧、`Genfromtxt` 🚧、`usecols`/逐列 dtype 🚧 |
| ufunc 协议 | `BinaryOut`/`UnaryOut`（`out=`）、`BinaryWhere`/`UnaryWhere`（`where=`）、`Reduce`、`Accumulate`、`UfuncOuter` | `at`/`buffers`/`signature` 等关键字 🚧 |
| 字符串与时间 | 构造、索引、比较、`datetime64` 单位处理 | 字符串 ufunc 🚧、`timedelta64` 运算 🚧 |
| 迭代与广播对象 | — | `nditer` 🚧、`vectorize`/`frompyfunc` 🚧、`broadcast_arrays` 🚧 |
| 错误控制 | fail-loud + 哨兵错误（`ErrShape`、`ErrDtype`、`ErrValue`、`ErrIndex`、`ErrZeroDiv`） | `errstate`/`seterr` 🚧 |

## 并发、内存与资源

- **包级状态只在 `init()` 里构建一次，之后只读。** dtype 提升表、ufunc 注册表与内核映射都在
  `init()` 完成，没有惰性初始化、也不需要 `sync.Once`；因此对包的并发调用是安全的。
- **数组本身不做内部同步**（与 NumPy 一致）。不同 goroutine 用不同数组随意；对**同一个数组**
  （或其别名视图）的并发**写**由调用方负责。并发只读已有测试
  （`TestConcurrentReadSameArray`：8 个 goroutine 反复读同一数组，结果一致且内容未被改动）。
- **`MatMul` 的并行是收敛的**：按行切分，每个 goroutine 只写自己那几行，无需加锁；
  不同 `GOMAXPROCS` 下结果逐位相同（`TestMatMulParallelStability`）。
- **文件句柄在所有路径上都会关闭**（open 成功后立刻 `defer Close`）。
  `TestNoFileHandleLeak` 跑了 **20000 轮** `.npy`/`.npz`/文本的读（含定期写），
  远超 Windows 句柄上限；最后还删掉刚写的文件，证明没有残留句柄占用。
- **视图会持有底层缓冲**：一个 1 GB 数组的切片视图会让这 1 GB 一直可达；
  只需要一小块时用 `AsContiguous`/`Copy`。`FromBuffer` 也是零拷贝，
  字节切片的生命周期由调用方负责。
- **不做内存池**：每个结果都是按结果形状精确分配的新内存，没有会在背后增长、
  让使用者意外的缓存。需要复用缓冲时显式用 `out=` 系列（`BinaryOut`/`UnaryOut`）。

## 文档

| 文档 | 内容 |
|---|---|
| [`wasm/README.md`](wasm/README.md) / [中文](wasm/README.zh-CN.md) | WASM 入口：构建、JS API、已暴露与计划暴露的对比、冒烟测试 |

## 许可

MIT —— 见 [LICENSE](LICENSE)。语义均来自对 NumPy 行为的实测观察，并用 Go 重新实现；
未引入 NumPy 源码。

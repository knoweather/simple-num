# simple-num

Pure-Go **n-dimensional array** library with a NumPy-compatible API — zero cgo,
no external dependencies, nothing but the standard library. Values are bit-exact
against NumPy where semantics allow, every behaviour is checked against a live
NumPy 2.5.4 session, and the same code builds to WebAssembly.

<p>
<a href="https://pkg.go.dev/github.com/knoweather/simple-num"><img src="https://pkg.go.dev/badge/github.com/knoweather/simple-num.svg" alt="Go Reference"></a>
<a href="go.mod"><img src="https://img.shields.io/badge/go-1.27%2B-00ADD8?logo=go" alt="Go 1.27+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
</p>

**English** | [简体中文](./README.zh-CN.md)

```go
a, _ := num.FromSlice([]float64{1, 2, 3, 4, 5, 6})
m, _ := a.Reshape([]int{2, 3})

b, _ := num.Transpose(m)             // 3×2 view, no copy
_ = b.AsContiguous()                 // explicit copy when you need one

s, _ := num.Sum(m, []int{0}, false)  // [5, 7, 9]

// npy interop: the file NumPy writes is the file this reads, and vice versa
_ = num.SaveNPY("a.npy", m)
back, _ := num.LoadNPY("a.npy")
```

- **NumPy semantics, Go ergonomics**: broadcasting, dtype promotion (NEP 50
  rules), views vs copies, `axis`/`keepdims`, NaN propagation, integer overflow
  wrap-around — all matched against measured NumPy behaviour, not guessed.
- **Bit-exact where it matters**: reductions use pairwise summation (so `Sum`
  matches `np.sum` to the last bit for the same input order), integer arithmetic
  wraps, `-0.0` and NaN payloads survive, `sign(-0.0)` is `+0` like NumPy.
- **Both I/O directions verified**: `.npy` / `.npz` / raw bytes / text. The
  reader accepts v1/v2/v3 headers; the writer emits headers NumPy parses without
  complaint (checked by loading our files in a live Python session).
- **Useful beyond arithmetic**: `linalg` (matmul/einsum/solve/inv/det/eig/svd),
  `random` (bit-exact MT19937 `RandomState`), `fft` (radix-2 + Bluestein),
  masked arrays (`ma`), sorting/statistics, padding, convolution, set ops,
  `polyfit`/`unwrap`, and a WASM bridge.
- **WASM ready**: `wasm/` builds `simple-num.wasm` plus a Node/JS API — no
  server, no C toolchain (`CGO_ENABLED=0 go build ./...` passes).

> [!NOTE]
> 🚀 **Initial release — canary testing in progress.** If you hit any problem,
> please open an [issue](https://github.com/knoweather/simple-num/issues)!

## Features

| Capability | Status |
|---|---|
| dtypes: `bool`, `int8…int64`, `uint8…uint64`, `float16/32/64`, `complex64/128`, fixed-width `S`/`U`, `datetime64` | ✅ |
| Views: `Reshape` / `Transpose` / `Squeeze` / `ExpandDims` / `SwapAxes` / `BroadcastTo` / `Sub` (zero-copy), explicit `AsContiguous` | ✅ |
| Indexing: basic slices, fancy indexing, boolean masks, `Take` / `TakeAlongAxis` / `Put` / `PutMask` / `Place` | ✅ |
| Element-wise ops with broadcasting + NEP 50 promotion: arithmetic, comparison, logical, bitwise, rounding, transcendentals (`tanh`/`cbrt`/`log2`/`expm1`/`log1p`/`sinc`, …) | ✅ |
| Reductions over any `axis` with `keepdims`: `Sum`/`Prod`/`Mean`/`Var`/`Std`/`Min`/`Max`/`AllTrue`/`Any`/`Median` plus the `Nan*` family | ✅ |
| Sorting/statistics: `Sort`/`Argsort`/`Median`/`Percentile` (**all 13 NumPy methods**)/`Quantile`/`NanPercentile`/`Diff`/`Unique` (+`axis`)/`SearchSorted`/`Digitize`/`Histogram`/`Histogram2D`/`Bincount(weights)` | ✅ |
| Shape manipulation: `Atleast1D/2D/3D`, `Flip`/`Roll`/`Repeat`/`Tile`, `Split`/`ArraySplit`/`SplitAt`/`H·V·DSplit`, `Append`/`Insert`/`Delete`, `Stack`/`VStack`/`HStack`/`ColumnStack`/`DStack`/`Concatenate` | ✅ |
| Linear algebra: `Dot`/`MatMul`/`Inner`/`Outer`/`Einsum`/`Tensordot`, `Det`/`Inv`/`Solve` (LU with partial pivoting), `Norm` (vector 0/1/2/±inf; matrix fro/1/2/-2/inf), `Eigvalsh`/`Eigh` (cyclic Jacobi), `Svd`/`Svdvals` (one-sided Jacobi) | ✅ |
| Floating-point helpers: `IsClose`/`AllClose`, `Invert`, `Round(decimals)`, `Clip`, `Where`, `Select`, `Piecewise`, `IsNan`/`IsInf`/`IsFinite`/`IsReal` | ✅ |
| Set ops: `IsIn`/`Intersect1D`/`Union1D`/`Setdiff1D`/`Setxor1D` (NaN never matches, like NumPy) | ✅ |
| Interpolation/signal: `Interp`, `MeshGrid`, `Gradient`, `Convolve`/`Correlate` (full/same/valid), `Trapz`, `Polyval`, `Polyfit`, `Unwrap` | ✅ |
| `random`: bit-exact NumPy legacy `RandomState`/MT19937 — `Seed`/`RandomSample`/`StandardNormal`/`Uniform`/`Normal`/`RandInt`/`Permutation`/`Shuffle` | ✅ |
| `fft`: `FFT`/`IFFT`/`RFFT`/`IRFFT`/`FFTFreq` — radix-2 for powers of two, **Bluestein** for any other length (O(n log n)) | ✅ |
| `ma` (masked arrays): data + mask, `Filled`/`Compressed`/`Count`, masked `Sum`/`Mean`/`Min`/`Max`/`Median`/`Var`/`Std`, predicates `MaskedEqual`/`Less`/`Greater`/`Inside`/`Outside` | ✅ |
| I/O: `SaveNPY`/`LoadNPY`/`SaveNPYBytes`/`LoadNPYBytes`, `SaveZ`/`SaveZCompressed`/`LoadZ` (npz), `FromBuffer` (zero-copy), `FromFile`/`ToFile`, `SaveTxt`/`LoadTxt` | ✅ |
| ufunc protocol: `out=` via `BinaryOut`/`UnaryOut`, `where=` via `BinaryWhere`/`UnaryWhere`, `Reduce`/`Accumulate`/`UfuncOuter` | ✅ |
| WASM bridge (`wasm/`): arrays as `{dtype, shape, data}`, `op`/`unary`/`reduce`/`linalg`/`cumulative` dispatchers, npy round-trip | ✅ |

## Performance

Measured on one Windows/amd64 machine, `float64`, single process. Method: each
case is **iterated until ≥400 ms of total time, then divided by the iteration
count** — single-shot timing on this platform is quantised too coarsely to trust.

| Case | Time | Note |
|---|---|---|
| `Add` 1e6 contiguous | 1.21 ms | ~21 GB/s |
| `Add` 1e6 with scalar broadcast | 1.19 ms | fast path |
| `Sum` 1e6 (pairwise) | 1.15 ms | |
| `Take` 1e6 from 1e6 indices | 8.41 ms | gather |
| `Cast` f8→f4 1e6 | 0.55 ms | typed bulk kernel |
| `CastUnsafe` f8→i4 1e6 | 0.94 ms | |
| `LoadNPY` 1e6 f8 | 1.89 ms | |
| `Arange` 1e6 | 3.09 ms | |
| `LinSpace` 1e6 | 1.08 ms | |
| `AsContiguous` on a transposed 1000×1000 | 5.54 ms | |
| `MatMul` 512³ | 6.0 ms | row-parallel (≈44 GFLOP/s) |
| `Einsum` `ij,jk->ik` 512³ | 6.3 ms | contraction graph, on par with `MatMul` |
| `FFT` n=1024 | 0.055 ms | radix-2 |
| `FFT` n=4000 | 0.77 ms | Bluestein (256 ms with the previous naive DFT) |

Notes on the numbers that matter:

- `MatMul` is pure Go without SIMD. Cache blocking and inner-loop unrolling were
  both tried and measured **slower** (6.99 / 6.32 ms vs 6.01 ms); the Go compiler
  does not vectorise these loops, so the four-row unroll plus row parallelism is
  where this plateaus.
- `Einsum` used to enumerate every label combination (4522 ms at 512³). It now
  builds a greedy contraction tree and lands on `MatMul`: **~700×** faster and
  bit-identical to `MatMul` at 64/128/256.
- `FFT` for non-power-of-two lengths used a naive O(n²) DFT; Bluestein brings
  n=4000 from 256 ms to 0.77 ms (**~330×**), with ~1e-14 relative agreement
  against `np.fft.fft`.

## Install

```bash
go get github.com/knoweather/simple-num
```

Standard library only — `CGO_ENABLED=0 go build ./...` works on windows/amd64,
linux/amd64, darwin/arm64 and windows/arm64 (cross-compilation is part of the
release gate).

## WASM

```sh
sh wasm/build.sh             # Linux / macOS / Git Bash  → wasm/simple-num.wasm + wasm_exec.js
pwsh wasm/build.ps1          # Windows PowerShell
node wasm/smoke/smoke.mjs    # build, load, run the JS API assertions
```

The bridge publishes `globalThis.simpleNum` with array objects crossing the
boundary as `{dtype, shape, data}` and dispatchers for `op` / `unary` / `reduce` /
`linalg` / `cumulative` plus npy in/out. It is a **curated subset** of the Go API —
[`wasm/README.md`](wasm/README.md) lists exactly what is exposed today and what is
planned (npz, percentile, full `Eigh`/`Svd`, `Einsum`, masked arrays, FFT, random…).

## Verification

Nothing here is "looks right" — every behaviour is pinned against a live NumPy.

The idea is simple: for each test case the same operation runs in Python with
NumPy 2.5.4, and its output bytes plus the inputs are saved into
`xcheck/reference/` (`.npy` files + a JSON manifest). The Go test then compares
our result against those saved bytes. Where an operation is not bit-exact by
nature (an FFT or an LU solve can differ in the last bits), the case declares a
relative tolerance instead.

- **NumPy reference cross-check** (`xcheck/`): **496 cases / 503 subtests**
  produced by `xcheck/gen_reference.py`, covering integers/floats/complex,
  broadcasting, ±0/NaN/Inf, integer overflow, axis reductions, gather/mask
  indexing, multi-output ops and the whole function surface. Two comparison
  regions: bitwise (exactly equal, NaN payloads included) and relative tolerance
  (FFT, LU, `sinc`).

- **Unregistered-op guard**: a manifest entry whose op has no registered
  implementation fails the test instead of silently counting as a pass.
- **Cross-cutting audits**: read-only ops never mutate their inputs;
  transposed/sliced views produce identical results to contiguous copies,
  including writes back into views; parallel `MatMul` is bit-stable across
  `GOMAXPROCS` values.
- **Coverage**: 85.3% of statements (gate: 85%).
- **I/O interop** is asserted in both directions: NumPy loads files this library
  writes, and this library loads files NumPy writes.

## Known limitations

- No structured/record dtypes, no `timedelta64` arithmetic, no `nditer`;
  `datetime64` is limited to construct/index/compare.
- `linalg` is dense real-valued only: no complex solvers, no `lstsq`, no general
  (non-symmetric) `eig`, no `qr`/`cholesky`.
- `Polyfit` solves normal equations, so it squares the condition number — fine
  for low degrees, not for large ones or wide `x` ranges.
- `Einsum` does not support the ellipsis form (`...`); repeated labels inside one
  operand (diagonal) fall back to the enumeration path (correct, O(∏ dims)).
- `Convolve`/`Correlate` are 1-D only and O(n·m); no FFT-based fast path.
- `ma` covers construction, masked reductions, arithmetic and predicates — not
  `ma.corrcoef` (NumPy's own result there can exceed ±1, which is not a
  behaviour worth reproducing).
- Text I/O (`SaveTxt`/`LoadTxt`) handles 1-D/2-D, a delimiter and `skiprows`;
  no `usecols`, no per-column dtypes.

## Project layout

```
array.go, dtype.go, view.go, broadcast.go, ufunc.go   core: memory, dtype, views, dispatch
impl_*.go                                             operator kernels by domain
xcheck/                                               NumPy-generated reference data + harness
wasm/                                                 WebAssembly bridge and JS smoke test
```

## NumPy API coverage

Legend: ✅ available · ⚠️ partial (see the note) · 🚧 not yet, planned.

| Area | Supported | Not yet (planned) |
|---|---|---|
| dtypes | `bool`, `int8…int64`, `uint8…uint64`, `float16/32/64`, `complex64/128`, fixed-width `S`/`U`, `datetime64` | `float128` 🚧, `timedelta64` 🚧, structured/record dtypes 🚧, `object` 🚧 |
| creation | `FromSlice`, `Zeros`, `Ones`, `Empty`, `Full`, `Eye`, `Identity`, `Arange`, `LinSpace`, `LogSpace`, `GeomSpace`, `MeshGrid`, `Diag`, `Diagflat` | `FromIter` 🚧, `FromString` 🚧, `Vander` 🚧, `Tri` 🚧 |
| shape & views | `Reshape`, `Ravel`, `Transpose`, `Squeeze`, `ExpandDims`, `SwapAxes`, `BroadcastTo`, `Sub` (all zero-copy where possible) | `Moveaxis` 🚧, `Resize` 🚧, `BroadcastArrays` 🚧 |
| manipulation | `Concatenate`, `Stack`/`VStack`/`HStack`/`ColumnStack`/`DStack`, `Split`/`ArraySplit`/`SplitAt`/`H·V·DSplit`, `Append`/`Insert`/`Delete`, `Flip`/`FlipUD`/`FlipLR`, `Roll`, `Rot90`, `Repeat`, `Tile`, `Pad` (10 modes incl. `stat_length`), `Atleast1D/2D/3D` | `Triu`/`Tril` indices helpers 🚧 |
| indexing | basic slices, fancy indexing, boolean masks, `Take`, `TakeAlongAxis`, `Put`, `PutAlongAxis`, `PutMask`, `Place`, `FillDiagonal`, `Compress`, `Nonzero`, `Argwhere`, `Flatnonzero`, `CountNonzero` | `Choose` 🚧, `TrilIndices` 🚧, `Unravel` 🚧 |
| arithmetic | `Add`/`Subtract`/`Multiply`/`Div`/`FloorDivide`/`Remainder`/`Power`, `Negative`/`Positive`/`Abs`/`Fabs`/`Square`/`Reciprocal`, `Maximum`/`Minimum`, `Clip`, `Round`/`Around`, `Where`/`Select`/`Piecewise` | — |
| comparison & logic | `Equal`/`NotEqual`/`Less`/`LessEqual`/`Greater`/`GreaterEqual`, `LogicalAnd/Or/Xor/Not`, `AllTrue`/`Any`, `IsNan`/`IsInf`/`IsFinite`/`IsReal`, `IsClose`/`AllClose` | — |
| bitwise & shifts | `BitwiseAnd`/`BitwiseOr`/`BitwiseXor`/`LeftShift`/`RightShift`/`Invert` (integers and bool; invalid shift counts yield 0 like NumPy) | — |
| transcendental | `Exp`/`Log`/`Log2`/`Log10`/`Expm1`/`Log1p`/`Sqrt`/`Cbrt`/`Sin`/`Cos`/`Tan`/`Arcsin`/`Arccos`/`Arctan`/`Arctan2`/`Sinh`/`Cosh`/`Tanh`/`Arcsinh`/`Arccosh`/`Arctanh`/`Sinc`, `Deg2Rad`/`Rad2Deg`, `Sign`, `Floor`/`Ceil`/`Trunc`/`Rint` | `Exp2` 🚧 |
| reductions | `Sum`/`Prod`/`Mean`/`Var`/`Std`/`Min`/`Max`/`ArgMin`/`ArgMax`/`Median`/`Cumsum`/`Cumprod`, `Average` (weights), `PtP`, all `Nan*` variants (incl. `NanProd`/`NanCumsum`), any `axis` + `keepdims` | — |
| statistics | `Percentile` (all 13 methods), `Quantile`, `Median`, `Histogram`, `Histogram2D`, `Bincount(weights)`, `Cov`, `CorrCoef`, `Diff` | `CorrCoef` of masked data 🚧 |
| sorting & sets | `Sort`, `Argsort`, `SearchSorted`, `Digitize`, `Unique` (+`axis`), `UniqueIdx` (returns index/counts), `IsIn`/`Intersect1D`/`Union1D`/`Setdiff1D`/`Setxor1D` | `Partition`/`Argpartition` 🚧 |
| linalg | `Dot`, `MatMul`, `Inner`, `Outer`, `Einsum`, `Tensordot`, `Det`, `Inv`, `Solve`, `Norm`, `Eigvalsh`, `Eigh`, `Svd`, `Svdvals`, `Slogdet`, `Kron`, `QR`, `Cholesky`, `Pinv`, `Lstsq` (returns x/residuals/rank/s) | general `Eig`/`Eigvals` (non-symmetric) 🚧, complex solvers 🚧 |
| random | bit-exact legacy `RandomState`/MT19937: `Seed`, `RandomSample`, `StandardNormal`, `Uniform`, `Normal`, `RandInt`, `Permutation`, `Shuffle` | `Generator`/PCG64 family 🚧, `Choice`/`Beta`/`Gamma`/`Poisson` etc. 🚧 |
| fft | `FFT`/`IFFT`/`RFFT`/`IRFFT`/`FFTFreq` (radix-2 + Bluestein, any length) | N-D transforms 🚧, `FFTShift`/`FFT2` 🚧 |
| masked arrays | `MaskedArray` (data+mask), `Filled`/`Compressed`/`Count`, masked `Sum`/`Mean`/`Min`/`Max`/`Median`/`Var`/`Std`, `MaskedAdd/Subtract/Multiply/Divide`, `MaskedInvalid`/`MaskedWhere`, predicates | `ma.corrcoef` 🚧 (see limitations), `masked_equal`-style constructors beyond the 8 provided ⚠️ |
| I/O | `SaveNPY`/`LoadNPY` (+ bytes), `SaveZ`/`SaveZCompressed`/`LoadZ`, `FromBuffer` (zero-copy), `FromFile`/`ToFile`, `SaveTxt`/`LoadTxt` | `Memmap` 🚧, `Genfromtxt` 🚧, `usecols`/per-column dtypes 🚧 |
| ufunc protocol | `BinaryOut`/`UnaryOut` (`out=`), `BinaryWhere`/`UnaryWhere` (`where=`), `Reduce`, `Accumulate`, `UfuncOuter` | `at`/`buffers`/`signature` keywords 🚧 |
| string & datetime | construction, indexing, comparison, `datetime64` unit handling | string ufuncs 🚧, `timedelta64` arithmetic 🚧 |
| iteration & broadcast objects | — | `nditer` 🚧, `vectorize`/`frompyfunc` 🚧, `broadcast_arrays` 🚧 |
| error control | fail-loud errors with sentinels (`ErrShape`, `ErrDtype`, `ErrValue`, `ErrIndex`, `ErrZeroDiv`) | `errstate`/`seterr` 🚧 |

## Concurrency, memory and resources

- **Package-level state is initialised once and read-only afterwards.** The dtype
  promotion table, ufunc registries and kernel maps are built in `init()`; there
  is no lazy initialisation and no `sync.Once` needed. Concurrent calls into the
  package are therefore safe.
- **Arrays are not internally synchronised** — exactly like NumPy. Distinct
  arrays can be used from different goroutines freely; concurrent *writes* to the
  same array (or to aliasing views of it) are the caller's responsibility.
  Read-only concurrent access to one array is tested (`TestConcurrentReadSameArray`).
- **`MatMul` parallelism is contained**: rows are split across goroutines and each
  worker writes only its own rows, so no locking is needed; results are
  bit-identical across `GOMAXPROCS` values (`TestMatMulParallelStability`).
- **File handles are closed on every path** (`defer` immediately after a
  successful open). `TestNoFileHandleLeak` performs 20 000 load/save rounds across
  `.npy`/`.npz`/text — well past the Windows handle limit — and additionally
  deletes a file it just wrote to prove no handle is still held.
- **Views keep their backing buffer alive.** A view of a 1 GB array keeps that
  1 GB reachable; copy with `AsContiguous`/`Copy` if you only need a slice.
  `FromBuffer` does not copy either, so the caller owns the byte slice lifetime.
- **Memory is not pooled.** Every result is a fresh allocation sized exactly to
  the result shape; there are no hidden caches whose growth could surprise you.
  Reuse buffers explicitly with the `out=` helpers (`BinaryOut`/`UnaryOut`).

## Documentation

| Doc | Contents |
|---|---|
| [`wasm/README.md`](wasm/README.md) | WASM entry: build, JS API, what is exposed vs planned, smoke test |

## Versioning

Releases use annotated tags of the form `v0.<minor>.<patch>`. Each release
increments only the last digit (`v0.1.0` → `v0.1.1` → … → `v0.1.9`); when that
digit would pass 9 the counter carries into the minor component
(`v0.1.9` → `v0.2.0`).

## License

MIT — see [LICENSE](LICENSE). Semantics were derived from observed NumPy
behaviour and re-implemented in Go; no NumPy source is vendored.

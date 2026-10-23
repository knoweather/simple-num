# wasm — WebAssembly bindings

Pure-Go array core (`wasm/api.go`) plus a thin `js/wasm` glue layer
(`wasm/wasm.go`) that publishes a global `simpleNum` in the browser or in Node:
build an ndarray → run operators/reductions → read values or round-trip npy.
Everything stays in memory — no file paths, no cgo, no data leaves the page
except the bytes you pass in.

**English** | [简体中文](./README.zh-CN.md)

## Build

Any platform (POSIX shell, incl. Git Bash on Windows):

```sh
sh wasm/build.sh
```

Windows PowerShell:

```powershell
powershell -NoProfile -File wasm/build.ps1
```

Portable one-liner if you prefer doing it by hand (any OS):

```sh
GOOS=js GOARCH=wasm CGO_ENABLED=0 go build -trimpath -o wasm/simple-num.wasm ./wasm/main
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" wasm/wasm_exec.js   # older toolchains: misc/wasm
```

Both outputs — `wasm/simple-num.wasm` and `wasm/wasm_exec.js` — are **not
committed** (`.gitignore` covers `wasm/*.wasm` and `wasm/wasm_exec.js`):
`wasm_exec.js` is tied to the Go toolchain version and the `.wasm` is a build
artifact.

`go build ./...` on a host stays clean: `wasm/main/main_stub.go`
(`//go:build !js || !wasm`) only prints a usage hint, while the real entry point
is `wasm/main/main.go` (`//go:build js && wasm`). The core API does not import
`syscall/js`, so `go test ./wasm/` runs natively (see `wasm/api_test.go`).

## Node smoke test

```sh
node wasm/smoke/smoke.mjs
```

The script builds into a temp directory itself, loads `wasm_exec.js`,
instantiates the module and runs assertions over construction, broadcasting,
reductions, unary ops, comparison, cast, npy round-trip, the linalg/cumulative
dispatchers and the error path. Success prints `[smoke] all passed`.

## Browser

```html
<script src="wasm_exec.js"></script>
<script>
  const go = new Go();
  WebAssembly.instantiateStreaming(fetch("simple-num.wasm"), go.importObject)
    .then((r) => go.run(r.instance));   // main stays resident; callbacks are js.Func driven
</script>
```

## JS API

After `go.run()`, use `globalThis.simpleNum`.

**Array objects** cross the Go ⇄ JS boundary as:

```js
{ dtype: "f8", shape: [2, 3], data: Uint8Array }   // data = little-endian, C-order bytes
```

`data` accepts a `Uint8Array`, an `ArrayBuffer`, any TypedArray (its byte view) or
a `number[]`.

| Method | Arguments | Returns |
| --- | --- | --- |
| `version()` | — | version string (e.g. `"0.1.0"`) |
| `zeros(shape, dtype)` | `number[]` + dtype string | array object |
| `arange(start, stop, step, dtype)` | three numbers + dtype string | array object |
| `fromBytes(dtype, shape, data)` | dtype string + `number[]` + bytes | array object (byte count must equal product × itemsize) |
| `toBytes(arr)` | array object | array object (made contiguous) |
| `values(arr)` | array object | JS array (row-major; see value mapping) |
| `op(name, a, b)` | operator name + two arrays | array object (broadcasting/promotion handled by the engine) |
| `unary(name, a)` | operator name + array | array object |
| `reduce(name, a, axis, keepdims)` | name + array + `null`\|`number`\|`number[]` + bool | array object (`axis=null` reduces everything → 0-d) |
| `linalg(name, a, b?)` | binary: `dot`/`matmul`/`inner`/`outer`/`solve`; unary: `det`/`inv`/`eigvalsh`/`svdvals`/`norm_fro`/`norm_2` | array object (`det`/`norm_*` are 0-d) |
| `cumulative(name, a, axis?)` | `cumsum`/`cumprod`; omitting `axis` means **flatten** | array object |
| `cast(arr, dtype)` | array + dtype string | array object (same-kind rules) |
| `reshape(arr, shape)` | array + `number[]` | array object (zero-copy when possible) |
| `item(arr, ...idx)` | array + indices | number / string / `[re, im]` |
| `loadNPY(bytes)` | npy bytes | array object |
| `saveNPY(arr)` | array object | `Uint8Array` (v1 header, 64-byte aligned) |
| `lastError()` | — | last error string; cleared on the next successful call |

**Errors**: every method that returns an array/bytes returns **null** on failure
and records the message in `lastError()`; query helpers (`version`, `values`,
`item`, `lastError`) validate their arguments instead. An unknown operator name
returns null with `lastError()` like `wasm: 未知算子 "nope"`.

**dtype vocabulary** (NumPy descr short forms): `b`, `i1..i8`, `u1..u8`,
`f2/f4/f8`, `c8/c16`, `M8[unit]` (e.g. `M8[D]`, `M8[ns]`), `S<n>`, `U<n>`.
The output `dtype` field uses the same vocabulary.

**Operator names** equal the library's ufunc names — binary
(`add subtract multiply divide floor_divide remainder power maximum minimum
arctan2 logical_and logical_or logical_xor less less_equal greater
greater_equal equal not_equal`), unary (`absolute/abs exp log log2 log10 expm1
log1p sqrt cbrt sin cos tan asin acos atan sinh cosh tanh arcsinh arccosh
arctanh sinc sign negative positive square reciprocal degrees radians real imag
floor ceil trunc rint conj isnan isinf isfinite logical_not`) and reductions
(`sum prod mean var std min max argmin argmax median any all` plus the `nan*`
family). The authoritative lists live in `wasm/api.go`.

**Value mapping**: `values`/`item` convert bool → `boolean`, integers and floats
→ `number` (f16/f32 widened to f64; int64/uint64 above 2^53 lose precision —
use `toBytes` when you need exact bits), complex → `[re, im]`, S/U → `string`,
M8 → `number` (int64 count).

```js
const a = simpleNum.fromBytes("f8", [2, 3], new Uint8Array(new Float64Array([0,1,2,3,4,5]).buffer));
const b = simpleNum.fromBytes("f8", [3],   new Uint8Array(new Float64Array([10,20,30]).buffer));
const c = simpleNum.op("add", a, b);        // broadcast → [2,3]: 10,21,32 / 13,24,35
const s = simpleNum.reduce("sum", c, null, false); // 0-d scalar
console.log(simpleNum.values(s)[0]);        // 135
const npy = simpleNum.saveNPY(c);           // Uint8Array, ready to download as .npy
```

## Not exposed yet (planned)

The bridge is a curated subset of the Go API; these are intentionally not
available from JS yet and are the next candidates:

- **I/O**: `npz` (`SaveZ`/`LoadZ`), text (`SaveTxt`/`LoadTxt`), `FromFile`
- **statistics**: `Percentile`/`Quantile`/`Median` beyond `median`, `Histogram`,
  `Histogram2D`, `Bincount`, `Cov`, `CorrCoef`
- **linalg**: `Eigh`/`Svd` full decompositions (only `eigvalsh`/`svdvals` today),
  `Einsum`, `Tensordot`
- **manipulation**: `Pad`, `Sort`/`Argsort`, `Unique`, `Repeat`/`Tile`,
  `Split`/`Stack`, `Where`/`Select`
- **domains**: masked arrays (`ma`), `fft`, `random`, `polyfit`/`unwrap`

Adding one is mechanical: a dispatcher entry in `wasm/api.go` plus a `ns.Set`
binding in `wasm/wasm.go`.

## Notes

- Pure Go, builds with `CGO_ENABLED=0`; the bridge performs no file I/O — pass
  bytes in and get bytes out.
- `js.Func` callbacks are driven by the Go runtime event loop: `main` parks in
  `select{}` and the instance stays resident. A wasm module instance is
  **single-threaded** — one call at a time, same as any `js/wasm` program.
- The dispatchers share the same operator entry points as the `xcheck/` reference
  suite (503 cases), so the wasm path is covered by the same semantics.

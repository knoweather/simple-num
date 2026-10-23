# wasm —— WebAssembly 绑定

纯 Go 的数组核心(`wasm/api.go`)加一层 `js/wasm` 胶水(`wasm/wasm.go`),在浏览器或
Node 里挂出全局 `simpleNum`:构造 ndarray → 算子/归约 → 取值/npy 往返,数据始终在
内存里。没有文件路径、没有 cgo,数据不出浏览器(除了你把字节传进来)。

[English](./README.md) | **简体中文**

## 构建

任意平台(POSIX shell,Windows 上可用 Git Bash):

```sh
sh wasm/build.sh
```

Windows PowerShell:

```powershell
powershell -NoProfile -File wasm/build.ps1
```

愿意手动做也很简单(任意系统):

```sh
GOOS=js GOARCH=wasm CGO_ENABLED=0 go build -trimpath -o wasm/simple-num.wasm ./wasm/main
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" wasm/wasm_exec.js   # 旧工具链在 misc/wasm
```

两个产物 —— `wasm/simple-num.wasm` 与 `wasm/wasm_exec.js` —— **都不入库**
(`.gitignore` 已覆盖):`wasm_exec.js` 与 Go 工具链版本强绑定,`.wasm` 是构建输出。

宿主机上 `go build ./...` 依然干净:`wasm/main/main_stub.go`(`//go:build !js || !wasm`)
只打印用法提示,真正的入口在 `wasm/main/main.go`(`//go:build js && wasm`)。
核心 API 不 `import syscall/js`,所以 `go test ./wasm/` 在宿主机直接可跑
(见 `wasm/api_test.go`)。

## Node 冒烟

```sh
node wasm/smoke/smoke.mjs
```

脚本自行 `go build` 到临时目录,载入 `wasm_exec.js`,实例化后跑断言:构造、广播、
归约、一元、比较、cast、npy 往返、`linalg`/`cumulative` 分发器与错误路径。
全部通过会输出 `[smoke] all passed`。

## 浏览器

```html
<script src="wasm_exec.js"></script>
<script>
  const go = new Go();
  WebAssembly.instantiateStreaming(fetch("simple-num.wasm"), go.importObject)
    .then((r) => go.run(r.instance));   // main 常驻,回调由 js.Func 驱动
</script>
```

## JS API

`go.run()` 之后可用 `globalThis.simpleNum`。

**数组对象**在 Go 与 JS 之间统一为:

```js
{ dtype: "f8", shape: [2, 3], data: Uint8Array }   // data = 小端、C 序连续字节
```

`data` 接受 `Uint8Array`、`ArrayBuffer`、任意 TypedArray(按其字节视图)或 `number[]`。

| 方法 | 参数 | 返回 |
| --- | --- | --- |
| `version()` | — | 版本串(如 `"0.1.0"`) |
| `zeros(shape, dtype)` | `number[]` + dtype 串 | 数组对象 |
| `arange(start, stop, step, dtype)` | 三个 number + dtype 串 | 数组对象 |
| `fromBytes(dtype, shape, data)` | dtype 串 + `number[]` + 字节 | 数组对象(字节数须 = 乘积 × itemsize) |
| `toBytes(arr)` | 数组对象 | 数组对象(强制连续化) |
| `values(arr)` | 数组对象 | JS Array(行主序;见「取值映射」) |
| `op(name, a, b)` | 算子名 + 两数组 | 数组对象(广播/提升由引擎处理) |
| `unary(name, a)` | 算子名 + 数组 | 数组对象 |
| `reduce(name, a, axis, keepdims)` | 名 + 数组 + `null`\|`number`\|`number[]` + bool | 数组对象(`axis=null` 全归约 → 0 维) |
| `linalg(name, a, b?)` | 二元:`dot`/`matmul`/`inner`/`outer`/`solve`;一元:`det`/`inv`/`eigvalsh`/`svdvals`/`norm_fro`/`norm_2` | 数组对象(`det`/`norm_*` 是 0 维) |
| `cumulative(name, a, axis?)` | `cumsum`/`cumprod`;省略 `axis` 表示**展平** | 数组对象 |
| `cast(arr, dtype)` | 数组 + dtype 串 | 数组对象(same_kind 口径) |
| `reshape(arr, shape)` | 数组 + `number[]` | 数组对象(能零拷贝就零拷贝) |
| `item(arr, ...idx)` | 数组 + 若干下标 | number / string / `[re, im]` |
| `loadNPY(bytes)` | npy 字节 | 数组对象 |
| `saveNPY(arr)` | 数组对象 | `Uint8Array`(v1 头,64 字节对齐) |
| `lastError()` | — | 最近一次错误串;下次成功调用后清空 |

**错误约定**:所有返回数组/字节的方法失败时返回 **null**,错误信息进 `lastError()`;
查询类(`version`/`values`/`item`/`lastError`)靠参数校验兜底。未知算子名返回 null,
`lastError()` 形如 `wasm: 未知算子 "nope"`。

**dtype 词汇**(numpy descr 短写):`b`、`i1..i8`、`u1..u8`、`f2/f4/f8`、`c8/c16`、
`M8[unit]`(如 `M8[D]`、`M8[ns]`)、`S<n>`、`U<n>`。输出侧 `dtype` 字段用同一词汇。

**算子名**与库里的 ufunc 名一致 —— 二元(`add subtract multiply divide
floor_divide remainder power maximum minimum arctan2 logical_and logical_or
logical_xor less less_equal greater greater_equal equal not_equal`)、一元
(`absolute/abs exp log log2 log10 expm1 log1p sqrt cbrt sin cos tan asin acos
atan sinh cosh tanh arcsinh arccosh arctanh sinc sign negative positive square
reciprocal degrees radians real imag floor ceil trunc rint conj isnan isinf
isfinite logical_not`)、归约(`sum prod mean var std min max argmin argmax median
any all` 加 `nan*` 系列)。权威清单以 `wasm/api.go` 为准。

**取值映射**:`values`/`item` 把 bool → `boolean`,整型与浮点 → `number`
(f16/f32 升到 f64 表示;int64/uint64 超过 2^53 会有 JS number 精度损失,
需要精确位型请走 `toBytes`),复数 → `[re, im]`,S/U → `string`,M8 → `number`(int64 计数)。

```js
const a = simpleNum.fromBytes("f8", [2, 3], new Uint8Array(new Float64Array([0,1,2,3,4,5]).buffer));
const b = simpleNum.fromBytes("f8", [3],   new Uint8Array(new Float64Array([10,20,30]).buffer));
const c = simpleNum.op("add", a, b);        // 广播 → [2,3]:10,21,32 / 13,24,35
const s = simpleNum.reduce("sum", c, null, false); // 0 维标量
console.log(simpleNum.values(s)[0]);        // 135
const npy = simpleNum.saveNPY(c);           // Uint8Array,可直接下载成 .npy
```

## 尚未暴露(计划中)

这套桥接是 Go API 的**子集**,以下暂不可从 JS 调用,也是后续的候选:

- **I/O**:`npz`(`SaveZ`/`LoadZ`)、文本(`SaveTxt`/`LoadTxt`)、`FromFile`
- **统计**:`Percentile`/`Quantile`(目前只有 `median`)、`Histogram`、`Histogram2D`、
  `Bincount`、`Cov`、`CorrCoef`
- **线代**:完整分解 `Eigh`/`Svd`(目前只有 `eigvalsh`/`svdvals`)、`Einsum`、`Tensordot`
- **形状/操作**:`Pad`、`Sort`/`Argsort`、`Unique`、`Repeat`/`Tile`、`Split`/`Stack`、
  `Where`/`Select`
- **领域模块**:掩码数组 `ma`、`fft`、`random`、`polyfit`/`unwrap`

加一个很机械:在 `wasm/api.go` 加一条分发项,再在 `wasm/wasm.go` 加一个 `ns.Set` 绑定。

## 说明

- 纯 Go,`CGO_ENABLED=0` 可构建;桥接层不做文件 IO —— 字节进去、字节出来。
- `js.Func` 回调由 Go runtime 事件循环驱动:`main` 停在 `select{}` 常驻。
  wasm 实例是**单线程**的,同一时刻只能跑一个调用(所有 `js/wasm` 程序都如此)。
- 分发器与 `xcheck/` 里 496 个参考用例共用同一套算子入口，所以 wasm 路径的语义
  与主库一致。

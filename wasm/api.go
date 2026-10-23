// Package wasm —— simple-num 的 WASM 核心 API(纯 Go 部分)。
//
// 本文件不 import syscall/js:核心逻辑可被宿主直接测试(go test./wasm/)。
// JS 胶水层在 wasm/wasm.go(//go:build js && wasm),入口 wasm/main;
// 范式参照 simple-sat 的 wasm/api.go —— 核心可测、胶水薄、lastError 语义同款。
// 本文件提供:dtype 词汇(numpy descr 短写,双向)、字节搬运、逐元素取值、
// 按 numpy 名的算子分发、错误记录(宿主回读 LastErr)。
package wasm

import (
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
	"sync"

	num "github.com/knoweather/simple-num"
)

// Version 是桥接层版本。构建脚本用 -ldflags -X 注入 git describe 的结果;
// 直接 go build 时回落到下面这个默认值(应随发版更新)。
var Version = "0.1.5"

// —— lastError(sat 同款语义:失败即记录,宿主可回读)——

var (
	errMu   sync.Mutex
	lastErr error
)

// NoteErr 记录最近一次错误(nil 忽略)。
func NoteErr(err error) {
	if err == nil {
		return
	}
	errMu.Lock()
	lastErr = err
	errMu.Unlock()
}

// ClearErr 清空错误记录。
func ClearErr() {
	errMu.Lock()
	lastErr = nil
	errMu.Unlock()
}

// LastErr 最近一次错误文本;无记录时为空串。
func LastErr() string {
	errMu.Lock()
	defer errMu.Unlock()
	if lastErr == nil {
		return ""
	}
	return lastErr.Error()
}

// note 记录并原样返回 err(分发入口用)。
func note(err error) error {
	NoteErr(err)
	return err
}

// —— dtype 词汇(numpy descr 短写,双向)——

// shortDtype 定宽短写 → Dtype(数值与 bool;S/U/M8 带参数,单独解析)。
var shortDtype = map[string]num.Dtype{
	"b":  num.DBool,
	"i1": num.DInt8, "i2": num.DInt16, "i4": num.DInt32, "i8": num.DInt64,
	"u1": num.DUint8, "u2": num.DUint16, "u4": num.DUint32, "u8": num.DUint64,
	"f2": num.DFloat16, "f4": num.DFloat32, "f8": num.DFloat64,
	"c8": num.DComplex64, "c16": num.DComplex128,
}

// ParseDtype numpy descr 短写 → Dtype:
// "b"(也 "?")、"i1".."i8"、"u1".."u8"、"f2/f4/f8"、"c8/c16"、"M8" 或 "M8[unit]"、
// "S<n>"、"U<n>"(n≥0;S0/U0 合法,宽度 0 由库内部按 1 处理)。无法识别 → "wasm: " 前缀错误。
func ParseDtype(s string) (num.Dtype, error) {
	if s == "" {
		return num.Dtype{}, fmt.Errorf("wasm: 空 dtype")
	}
	if dt, ok := shortDtype[s]; ok {
		return dt, nil
	}
	if s == "?" {
		return num.DBool, nil
	}
	if strings.HasPrefix(s, "M8") {
		rest := s[2:]
		if rest == "" {
			return num.Dtype{Kind: num.Datetime64, Itemsize: 8}, nil // 无方括号:unit 空串
		}
		if len(rest) < 3 || rest[0] != '[' || rest[len(rest)-1] != ']' {
			return num.Dtype{}, fmt.Errorf("wasm: dtype %q:M8 单位须写作 M8[unit]", s)
		}
		unit := rest[1 : len(rest)-1]
		switch unit {
		case "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns":
			return num.DTime(unit), nil
		}
		return num.Dtype{}, fmt.Errorf("wasm: dtype %q:datetime64 单位 %q 未支持", s, unit)
	}
	if s[0] == 'S' || s[0] == 'U' {
		n, err := strconv.Atoi(s[1:])
		if err != nil || n < 0 {
			return num.Dtype{}, fmt.Errorf("wasm: dtype %q:宽度须为非负整数", s)
		}
		if s[0] == 'S' {
			return num.DBytes(n), nil
		}
		return num.DUnicode(n), nil
	}
	return num.Dtype{}, fmt.Errorf("wasm: dtype %q 无法识别", s)
}

// DtypeString Dtype → numpy descr 短写(ParseDtype 的逆;S0/U0 原样回写)。
func DtypeString(dt num.Dtype) string {
	switch dt.Kind {
	case num.Bool:
		return "b"
	case num.Datetime64:
		if dt.Unit == "" {
			return "M8"
		}
		return "M8[" + dt.Unit + "]"
	case num.Bytes:
		return "S" + strconv.Itoa(dt.Itemsize)
	case num.Unicode:
		return "U" + strconv.Itoa(dt.Itemsize/4)
	case num.Int8, num.Int16, num.Int32, num.Int64:
		return "i" + strconv.Itoa(dt.Itemsize)
	case num.Uint8, num.Uint16, num.Uint32, num.Uint64:
		return "u" + strconv.Itoa(dt.Itemsize)
	case num.Float16, num.Float32, num.Float64:
		return "f" + strconv.Itoa(dt.Itemsize)
	case num.Complex64, num.Complex128:
		return "c" + strconv.Itoa(dt.Itemsize)
	}
	return dt.String()
}

// —— 字节搬运 ——

// prod 形状元素数(0 维 = 1)。
func prod(sh []int) int {
	n := 1
	for _, d := range sh {
		n *= d
	}
	return n
}

// FromBytes dtype 短写 + 形状 + 字节 → Array(字节数必须恰为 prod(shape)*itemsize)。
func FromBytes(dtype string, shape []int, data []byte) (*num.Array, error) {
	dt, err := ParseDtype(dtype)
	if err != nil {
		return nil, note(err)
	}
	need := prod(shape) * dt.Itemsize
	if len(data) != need {
		return nil, note(fmt.Errorf("wasm: FromBytes %s 期望 %d 字节,实际 %d 字节", dtype, need, len(data)))
	}
	a, err := num.FromBytes(data, dt, shape)
	if err != nil {
		return nil, note(err)
	}
	return a, nil
}

// Bytes 取数组数据区字节(剥 npy 头:魔数 6 + 版本 2 + 头长 2/4,数据区在 10/12 + 头长)。
// SaveNPYBytes 保证小端 C 序连续;返回新副本,不与 npy 缓冲共享。
func Bytes(a *num.Array) ([]byte, error) {
	raw, err := num.SaveNPYBytes(a)
	if err != nil {
		return nil, err
	}
	if len(raw) < 10 || string(raw[:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("wasm: Bytes npy 魔数不符")
	}
	var hoff, hlen int
	switch raw[6] {
	case 1:
		hlen = int(binary.LittleEndian.Uint16(raw[8:10]))
		hoff = 10
	case 2, 3:
		if len(raw) < 12 {
			return nil, fmt.Errorf("wasm: Bytes npy v%d 头截断", raw[6])
		}
		hlen = int(binary.LittleEndian.Uint32(raw[8:12]))
		hoff = 12
	default:
		return nil, fmt.Errorf("wasm: Bytes npy 版本 %d 未支持", raw[6])
	}
	if hoff+hlen > len(raw) {
		return nil, fmt.Errorf("wasm: Bytes npy 头长 %d 超界", hlen)
	}
	out := make([]byte, len(raw)-(hoff+hlen))
	copy(out, raw[hoff+hlen:])
	return out, nil
}

// Values 按 C 序(行主序)逐元素取值;0 维即 1 个值。
func Values(a *num.Array) ([]any, error) {
	sh := a.Shape()
	out := make([]any, a.Size())
	idx := make([]int, len(sh))
	for i := range out {
		v, err := a.Item(idx...)
		if err != nil {
			return nil, err
		}
		out[i] = v
		for d := len(sh) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < sh[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out, nil
}

// —— 算子分发(键 = numpy 名)——

type binFunc func(x, y *num.Array) (*num.Array, error)
type unaryFunc func(x *num.Array) (*num.Array, error)
type reduceFunc func(a *num.Array, axis []int, keepdims bool) (*num.Array, error)

var (
	binOps    map[string]binFunc
	unaryOps  map[string]unaryFunc
	reduceOps map[string]reduceFunc
)

func init() {
	binOps = map[string]binFunc{
		"add": num.Add, "subtract": num.Subtract, "multiply": num.Multiply,
		"divide": num.Div, "floor_divide": num.FloorDivide, "remainder": num.Remainder,
		"power": num.Power, "less": num.Less, "less_equal": num.LessEqual,
		"greater": num.Greater, "greater_equal": num.GreaterEqual,
		"equal": num.Equal, "not_equal": num.NotEqual,
	}
	binOps = map[string]binFunc{
		"add": num.Add, "subtract": num.Subtract, "multiply": num.Multiply,
		"divide": num.Div, "floor_divide": num.FloorDivide, "remainder": num.Remainder,
		"power": num.Power, "less": num.Less, "less_equal": num.LessEqual,
		"greater": num.Greater, "greater_equal": num.GreaterEqual,
		"equal": num.Equal, "not_equal": num.NotEqual,
		// Day16 追加:极值/角度/逻辑
		"maximum": num.Maximum, "minimum": num.Minimum, "arctan2": num.Arctan2,
		"logical_and": num.LogicalAnd, "logical_or": num.LogicalOr,
		"logical_xor": num.LogicalXor,
		// 位运算(整型/bool;浮点拒绝)
		"bitwise_and": num.BitwiseAnd, "bitwise_or": num.BitwiseOr,
		"bitwise_xor": num.BitwiseXor, "left_shift": num.LeftShift,
		"right_shift": num.RightShift,
	}
	unaryOps = map[string]unaryFunc{
		"abs": num.Abs, "exp": num.Exp, "log": num.Log, "log10": num.Log10,
		"sqrt": num.Sqrt, "sin": num.Sin, "cos": num.Cos, "tan": num.Tan,
		"asin": num.Asin, "acos": num.Acos, "atan": num.Atan,
		"floor": num.Floor, "ceil": num.Ceil, "trunc": num.Trunc,
		"rint": num.Rint, "conj": num.Conj,
		// Day16 追加:双曲/符号/角度/谓词/取整
		"tanh": num.Tanh, "sinh": num.Sinh, "cosh": num.Cosh, "sinc": num.Sinc,
		"sign": num.Sign, "fabs": num.Abs, "deg2rad": num.Deg2Rad, "rad2deg": num.Rad2Deg,
		"isnan": num.IsNan, "isinf": num.IsInf, "isfinite": num.IsFinite,
		"logical_not": num.LogicalNot, "round1": func(x *num.Array) (*num.Array, error) {
			return num.Round(x, 0)
		},
		// Day16 追加:negative / 幂与对数族 / 反双曲 / 复数分量
		"negative": num.Negative, "positive": num.Positive, "square": num.Square,
		"reciprocal": num.Reciprocal, "log2": num.Log2, "expm1": num.Expm1,
		"log1p": num.Log1p, "cbrt": num.Cbrt, "arcsinh": num.Arcsinh,
		"arccosh": num.Arccosh, "arctanh": num.Arctanh,
		"degrees": num.Degrees, "radians": num.Radians,
		"real": num.Real, "imag": num.Imag,
		"absolute": num.Absolute, "invert": num.Invert,
	}
	reduceOps = map[string]reduceFunc{
		"sum": num.Sum, "mean": num.Mean, "var": num.Var, "std": num.Std,
		"min": num.Min, "max": num.Max, "argmin": num.ArgMin, "argmax": num.ArgMax,
		"any": num.Any, "all": num.AllTrue,
		// Day16 追加:prod 与 NaN 族(签名与普通归约一致)
		"prod": num.Prod, "median": num.Median,
		"nansum": num.NanSum, "nanmean": num.NanMean, "nanvar": num.NanVar,
		"nanstd": num.NanStd, "nanmax": num.NanMax, "nanmin": num.NanMin,
		"nanmedian": num.NanMedian,
	}
}

// Op 二元算子分发。
func Op(name string, x, y *num.Array) (*num.Array, error) {
	f, ok := binOps[name]
	if !ok {
		return nil, note(fmt.Errorf("wasm: 未知算子 %q", name))
	}
	r, err := f(x, y)
	if err != nil {
		return nil, note(err)
	}
	return r, nil
}

// Unary 一元算子分发。
func Unary(name string, x *num.Array) (*num.Array, error) {
	f, ok := unaryOps[name]
	if !ok {
		return nil, note(fmt.Errorf("wasm: 未知算子 %q", name))
	}
	r, err := f(x)
	if err != nil {
		return nil, note(err)
	}
	return r, nil
}

// Reduce 归约分发(axis nil = 全归约)。
func Reduce(name string, a *num.Array, axis []int, keepdims bool) (*num.Array, error) {
	f, ok := reduceOps[name]
	if !ok {
		return nil, note(fmt.Errorf("wasm: 未知算子 %q", name))
	}
	r, err := f(a, axis, keepdims)
	if err != nil {
		return nil, note(err)
	}
	return r, nil
}

// —— 线代 / 前缀累积 分发(Step 29:wasm 与核心库对齐)——

var (
	linalgBinOps   map[string]binFunc
	linalgUnaryOps map[string]unaryFunc
	cumOps         map[string]func(*num.Array, []int) (*num.Array, error)
)

func init() {
	linalgBinOps = map[string]binFunc{
		"dot": num.Dot, "matmul": num.MatMul, "inner": num.Inner, "outer": num.Outer,
		"solve": num.Solve,
	}
	linalgUnaryOps = map[string]unaryFunc{
		"inv": num.Inv, "eigvalsh": num.Eigvalsh, "svdvals": num.Svdvals,
		"cholesky": num.Cholesky, "pinv": num.Pinv,
		"det": func(a *num.Array) (*num.Array, error) {
			v, err := num.Det(a)
			if err != nil {
				return nil, err
			}
			return scalarArr0(v)
		},
		"norm_fro": func(a *num.Array) (*num.Array, error) {
			v, err := num.Norm(a, "fro")
			if err != nil {
				return nil, err
			}
			return scalarArr0(v)
		},
		"norm_2": func(a *num.Array) (*num.Array, error) {
			v, err := num.Norm(a, "2")
			if err != nil {
				return nil, err
			}
			return scalarArr0(v)
		},
	}
	cumOps = map[string]func(*num.Array, []int) (*num.Array, error){
		"cumsum": num.Cumsum, "cumprod": num.Cumprod,
	}
}

// scalarArr0 把标量包成 0-d 数组(JS 侧仍按数组返回,便于统一处理)。
func scalarArr0(v float64) (*num.Array, error) {
	a, err := num.New([]int{}, num.DFloat64)
	if err != nil {
		return nil, err
	}
	if err := a.SetItem([]int{}, v); err != nil {
		return nil, err
	}
	return a, nil
}

// Linalg 线代分发:二元算子用 b;一元/标量算子忽略 b。
func Linalg(name string, a, b *num.Array) (*num.Array, error) {
	if f, ok := linalgBinOps[name]; ok {
		r, err := f(a, b)
		if err != nil {
			return nil, note(err)
		}
		return r, nil
	}
	if f, ok := linalgUnaryOps[name]; ok {
		r, err := f(a)
		if err != nil {
			return nil, note(err)
		}
		return r, nil
	}
	return nil, note(fmt.Errorf("wasm: 未知线代算子 %q", name))
}

// Cumulative 前缀累积分发(cumsum / cumprod;axis 为 nil 表示展平)。
func Cumulative(name string, a *num.Array, axis []int) (*num.Array, error) {
	f, ok := cumOps[name]
	if !ok {
		return nil, note(fmt.Errorf("wasm: 未知累积算子 %q", name))
	}
	r, err := f(a, axis)
	if err != nil {
		return nil, note(err)
	}
	return r, nil
}

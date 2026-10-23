package num

// T9:比较 6 算子(B 拥有)。
//
// 语义基准:numpy 2.5.4 实测( §6):输出恒 bool;NaN 参与有序比较
// 恒 False、equal 恒 False、not_equal 恒 True;±0.0 相等;inf 正常参与比较;
// bool 支持有序比较(False < True)。Go 原生比较运算符的 IEEE 语义与上述
// 逐位一致,故数值内核零特判。
//
// 分发路径( 已解, 落地):引擎对 ForceOut==Bool 的算子按输入 kind(per-idt)
// 查内核(ufunc.go dispatchBinary),registerCompareTyped 也改存输入 kind 键。本文件
// 因此从"适配层自建分发"平移到注册式:6 算子 × 14 kind 挂进 binaryReg;cmpDispatch
// 只保留 promote 表覆盖不到的两条特判(S/U 与 Datetime64),其余直接转交 dispatchBinary。
// 内核函数(cmpRun/cmpEqNe/cmpRunBool)保留,由注册闭包复用。

import (
	"encoding/binary"
	"fmt"
)

// OrdKinds 有序型别(整型 + 浮点;bool 不可排序,Go 泛型约束所限)。
type OrdKinds interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// EqKinds 可等型别(OrdKinds + bool)。
type EqKinds interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// cmpRun 数值型 6 比较(原生运算符;NaN/±0/inf 语义与 numpy 逐位一致,)。
func cmpRun[T OrdKinds](op string, dst []bool, a, b []T) {
	switch op {
	case "less":
		for i := range dst {
			dst[i] = a[i] < b[i]
		}
	case "less_equal":
		for i := range dst {
			dst[i] = a[i] <= b[i]
		}
	case "greater":
		for i := range dst {
			dst[i] = a[i] > b[i]
		}
	case "greater_equal":
		for i := range dst {
			dst[i] = a[i] >= b[i]
		}
	case "equal":
		for i := range dst {
			dst[i] = a[i] == b[i]
		}
	case "not_equal":
		for i := range dst {
			dst[i] = a[i] != b[i]
		}
	}
}

// cmpEqNe 复数专用:仅 eq/ne(Go 的 complex == 为分量比较,NaN → false,numpy 同)。
func cmpEqNe[T comparable](op string, dst []bool, a, b []T) {
	if op == "equal" {
		for i := range dst {
			dst[i] = a[i] == b[i]
		}
		return
	}
	for i := range dst {
		dst[i] = a[i] != b[i]
	}
}

// cmpRunBool bool 型(numpy:False < True;eq/ne 原生)。
func cmpRunBool(op string, dst, a, b []bool) {
	switch op {
	case "less":
		for i := range dst {
			dst[i] = !a[i] && b[i]
		}
	case "less_equal":
		for i := range dst {
			dst[i] = !a[i] || b[i]
		}
	case "greater":
		for i := range dst {
			dst[i] = a[i] && !b[i]
		}
	case "greater_equal":
		for i := range dst {
			dst[i] = a[i] || !b[i]
		}
	case "equal":
		for i := range dst {
			dst[i] = a[i] == b[i]
		}
	case "not_equal":
		for i := range dst {
			dst[i] = a[i] != b[i]
		}
	}
}

// —— 内核工厂( 平移到注册式):dst = bool 窗,a/b = 输入 kind 窗 ——

// cmpKernel 有序型内核工厂(原生运算符;与 cmpRun 同口径)。
func cmpKernel[T OrdKinds](op string) BinaryKernel {
	return func(dst, a, b []byte) error {
		cmpRun(op, win[bool](dst), win[T](a), win[T](b))
		return nil
	}
}

// cmpKernelBool bool 窗内核(复用 cmpRunBool)。
func cmpKernelBool(op string) BinaryKernel {
	return func(dst, a, b []byte) error {
		cmpRunBool(op, win[bool](dst), win[bool](a), win[bool](b))
		return nil
	}
}

// cmpKernelF16 f16 窗内核:加宽到 f32 后比较(numpy HALF 比较同;加宽精确无损)。
func cmpKernelF16(op string) BinaryKernel {
	return func(dst, a, b []byte) error {
		af := make([]float32, len(a)/2)
		bf := make([]float32, len(b)/2)
		for i := range af {
			af[i] = f16ToF32(binary.LittleEndian.Uint16(a[i*2:]))
			bf[i] = f16ToF32(binary.LittleEndian.Uint16(b[i*2:]))
		}
		cmpRun(op, win[bool](dst), af, bf)
		return nil
	}
}

// cmpKernelComplex 复数窗内核:仅 eq/ne;有序比较在内核内报错(numpy 同拒,文案逐字保留)。
func cmpKernelComplex[T comparable](op string) BinaryKernel {
	return func(dst, a, b []byte) error {
		if op != "equal" && op != "not_equal" {
			return fmt.Errorf("num: 复数不支持有序比较(numpy 同拒): %w", ErrDtype)
		}
		cmpEqNe[T](op, win[bool](dst), win[T](a), win[T](b))
		return nil
	}
}

// cmpOps 比较算子名(numpy ufunc 名)。
var cmpOps = []string{"less", "less_equal", "greater", "greater_equal", "equal", "not_equal"}

// 注册( 平移到注册式):6 算子 × 14 kind(= promote 表全集:Bool + i1..i8 +
// u1..u8 + f2/f4/f8 + c8/c16),ForceOut=Bool,查找键 = 输入 kind。等价于按 kind
// 逐个 registerCompareTyped(f16 无 Go 类型、复数有序需报错,故走 registerBinarySpec)。
func init() {
	for _, op := range cmpOps {
		registerBinarySpec(op, BinarySpec{ForceOut: Bool, Kernels: map[Kind]BinaryKernel{
			Bool:       cmpKernelBool(op),
			Int8:       cmpKernel[int8](op),
			Int16:      cmpKernel[int16](op),
			Int32:      cmpKernel[int32](op),
			Int64:      cmpKernel[int64](op),
			Uint8:      cmpKernel[uint8](op),
			Uint16:     cmpKernel[uint16](op),
			Uint32:     cmpKernel[uint32](op),
			Uint64:     cmpKernel[uint64](op),
			Float16:    cmpKernelF16(op),
			Float32:    cmpKernel[float32](op),
			Float64:    cmpKernel[float64](op),
			Complex64:  cmpKernelComplex[complex64](op),
			Complex128: cmpKernelComplex[complex128](op),
		}})
	}
}

// cmpDispatch 比较分发( 已解):S/U 与 Datetime64 不在 promote 表,单独走;
// 其余转交引擎(per-idt 查找 + 广播归一 + ForceOut=Bool)。
func cmpDispatch(op string, x, y *Array) (*Array, error) {
	// 定宽字符串 S/U 比较单独走(promote 表无 S/U;混合 S×U → ErrDtype)。
	if isStrKind(x.Dtype().Kind) || isStrKind(y.Dtype().Kind) {
		return strCompare(op, x, y)
	}
	// datetime64 比较单独走(NaT 口径 + 单位归一;promote 表无 Datetime64)。
	if x.Dtype().Kind == Datetime64 || y.Dtype().Kind == Datetime64 {
		return dtCompare(op, x, y)
	}
	return dispatchBinary(op, x, y)
}

// —— 适配入口(算子名 = numpy ufunc 名,PascalCase)——

// Less x < y。
func Less(x, y *Array) (*Array, error) { return cmpDispatch("less", x, y) }

// LessEqual x <= y。
func LessEqual(x, y *Array) (*Array, error) { return cmpDispatch("less_equal", x, y) }

// Greater x > y。
func Greater(x, y *Array) (*Array, error) { return cmpDispatch("greater", x, y) }

// GreaterEqual x >= y。
func GreaterEqual(x, y *Array) (*Array, error) {
	return cmpDispatch("greater_equal", x, y)
}

// Equal x == y(NaN 参与恒 False,numpy 同)。
func Equal(x, y *Array) (*Array, error) { return cmpDispatch("equal", x, y) }

// NotEqual x != y(NaN 参与恒 True,numpy 同)。
func NotEqual(x, y *Array) (*Array, error) {
	return cmpDispatch("not_equal", x, y)
}

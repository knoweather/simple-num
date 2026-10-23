package num

// :超越 11 算子(B 拥有):exp/log/log10/sqrt/abs/sin/cos/tan/asin/acos/atan。
//
// 语义基准(numpy 2.5.4 实测,):
// - 输出 dtype 由 numpy ufunc 类型解析决定:整型/bool 输入按"最小可无损容纳"
// 升 float —— b/i8/u8 → f16,i16/u16 → f32,i32/u32/i64/u64 → f64;
// f16/f32/f64 输入保 dtype。
// - f16 计算 = numpy HALF 循环策略:f16 → f32 → 计算 → f16;
// f32 计算 = f32 → f64 → math 函数 → f32(numpy 亦为 double 计算后回绕)。
// - 边角:sqrt(负)→NaN、sqrt(-0)→-0、log(0)→-Inf、log(负)→NaN、
// exp 溢出→±Inf/0、sin/cos/tan(±Inf)→NaN、asin/acos(|x|>1)→NaN、
// atan(±Inf)→±π/2;Go math 与 numpy 逐点一致(rel 门槛覆盖 1ulp,)。
// - abs 保 dtype(bool→bool、整型→整型含 MinInt 回绕、浮点→浮点,NaN→NaN)。
// 整型→浮点解析在入口适配层做( 同思路):引擎 Spec 的 Fallback 是单链,
// 表达不了"按输入 kind 选不同浮点",故预 Cast 后再走 dispatchUnary。

import (
	"encoding/binary"
	"fmt"
	"math"
)

// trigF 浮点一元内核(f64 计算,T 回绕;f32 与 numpy"double 计算后转 f32"同构)。
func trigF[T FloatKinds](dst, src []T, f func(float64) float64) {
	for i := range dst {
		dst[i] = T(f(float64(src[i])))
	}
}

// f16Kernel f16 一元内核(numpy HALF 策略:f16→f32→f64 计算→f32→f16)。
func f16Kernel(f func(float64) float64) UnaryKernel {
	return func(dst, src []byte) error {
		n := len(src) / 2
		for i := 0; i < n; i++ {
			h := binary.LittleEndian.Uint16(src[i*2:])
			v := f(float64(f16ToF32(h)))
			binary.LittleEndian.PutUint16(dst[i*2:], f32ToF16(float32(v)))
		}
		return nil
	}
}

// registerTrig 注册一个浮点超越算子:f32/f64 typed 内核 + f16 特例内核。
func registerTrig(op string, f func(float64) float64) {
	registerUnaryTyped[float32](op, func(dst, src []float32) { trigF(dst, src, f) })
	registerUnaryTyped[float64](op, func(dst, src []float64) { trigF(dst, src, f) })
	registerUnarySpec(op, UnarySpec{Kernels: map[Kind]UnaryKernel{Float16: f16Kernel(f)}})
}

// —— abs(保 dtype)——

func absInt[T Ints](dst, src []T) {
	for i := range dst {
		if src[i] < 0 {
			dst[i] = -src[i] // MinInt 回绕(numpy absolute(int8 -128) = -128 同)
		} else {
			dst[i] = src[i]
		}
	}
}

// fabs C fabs 口径:清符号位,保留 NaN payload(numpy absolute 实测:
// 0xFFF8…0123 → 0x7FF8…0123;Go math.Abs 会把 NaN 规范成 0x…001,不能用)。
func fabs(x float64) float64 {
	return math.Float64frombits(math.Float64bits(x) &^ (1 << 63))
}

func absF[T FloatKinds](dst, src []T) {
	if npF32[T]() {
		for i := range dst {
			dst[i] = T(math.Float32frombits(math.Float32bits(float32(src[i])) &^ 0x80000000))
		}
		return
	}
	for i := range dst {
		dst[i] = T(fabs(float64(src[i])))
	}
}

func absBool(dst, src []bool) {
	copy(dst, src) // numpy absolute(bool) → bool(恒等)
}

func init() {
	registerTrig("exp", math.Exp)
	registerTrig("log", math.Log)
	registerTrig("log10", math.Log10)
	registerTrig("sqrt", math.Sqrt)
	registerTrig("sin", math.Sin)
	registerTrig("cos", math.Cos)
	registerTrig("tan", math.Tan)
	registerTrig("asin", math.Asin)
	registerTrig("acos", math.Acos)
	registerTrig("atan", math.Atan)

	// abs:全数值 dtype 保 dtype 注册;f16 走 HALF 策略。
	registerUnaryTyped[bool]("absolute", absBool)
	registerUnaryTyped[int8]("absolute", absInt[int8])
	registerUnaryTyped[int16]("absolute", absInt[int16])
	registerUnaryTyped[int32]("absolute", absInt[int32])
	registerUnaryTyped[int64]("absolute", absInt[int64])
	registerUnaryTyped[uint8]("absolute", absInt[uint8])
	registerUnaryTyped[uint16]("absolute", absInt[uint16])
	registerUnaryTyped[uint32]("absolute", absInt[uint32])
	registerUnaryTyped[uint64]("absolute", absInt[uint64])
	registerUnaryTyped[float32]("absolute", absF[float32])
	registerUnaryTyped[float64]("absolute", absF[float64])
	registerUnarySpec("absolute", UnarySpec{Kernels: map[Kind]UnaryKernel{
		Float16: f16Kernel(fabs),
	}})
}

// —— 入口适配 ——

// trigFloatFor numpy 超越算子的整型→浮点类型解析:b/i8/u8 → f16,
// i16/u16 → f32,i32/u32/i64/u64 → f64;非整型返回 ok=false(原样)。
func trigFloatFor(k Kind) (Dtype, bool) {
	switch k {
	case Bool, Int8, Uint8:
		return DFloat16, true
	case Int16, Uint16:
		return DFloat32, true
	case Int32, Uint32, Int64, Uint64:
		return DFloat64, true
	}
	return Dtype{}, false
}

// trigDispatch 超越算子分发:整型/bool 输入按 numpy 解析预 Cast,再走引擎。
func trigDispatch(op string, x *Array) (*Array, error) {
	if dt, ok := trigFloatFor(x.Dtype().Kind); ok {
		var err error
		if x, err = Cast(x, dt); err != nil {
			return nil, fmt.Errorf("num: %s: %w", op, err)
		}
	}
	return dispatchUnary(op, x)
}

// Exp e**x。
func Exp(x *Array) (*Array, error) { return trigDispatch("exp", x) }

// Log ln(x)。
func Log(x *Array) (*Array, error) { return trigDispatch("log", x) }

// Log10 log10(x)。
func Log10(x *Array) (*Array, error) { return trigDispatch("log10", x) }

// Sqrt 平方根(负 → NaN)。
func Sqrt(x *Array) (*Array, error) { return trigDispatch("sqrt", x) }

// Sin 正弦。
func Sin(x *Array) (*Array, error) { return trigDispatch("sin", x) }

// Cos 余弦。
func Cos(x *Array) (*Array, error) { return trigDispatch("cos", x) }

// Tan 正切。
func Tan(x *Array) (*Array, error) { return trigDispatch("tan", x) }

// Asin 反正弦(|x|>1 → NaN)。
func Asin(x *Array) (*Array, error) { return trigDispatch("asin", x) }

// Acos 反余弦(|x|>1 → NaN)。
func Acos(x *Array) (*Array, error) { return trigDispatch("acos", x) }

// Atan 反正切。
func Atan(x *Array) (*Array, error) { return trigDispatch("atan", x) }

// Abs 绝对值(保 dtype;整型 MinInt 回绕,numpy 同;复数 → 模,c64→f32/c128→f64)。
func Abs(x *Array) (*Array, error) {
	switch x.Dtype().Kind {
	case Complex64, Complex128:
		return complexAbs(x)
	}
	return dispatchUnary("absolute", x)
}

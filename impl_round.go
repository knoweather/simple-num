package num

// :取整 4 算子(B 拥有):floor/ceil/trunc/rint。
//
// 语义基准(numpy 2.5.4 实测,):
// - floor/ceil/trunc:整型/bool 输入是**恒等**(numpy 有整型循环,保 dtype);
// 浮点保 dtype;f16 走 HALF 策略(f16→f32→计算→f16)。
// - rint:银行家舍入(round-half-to-even):0.5→0、1.5→2、2.5→2、4.5→4、
// -2.5→-2(Go math.RoundToEven 同);整型/bool 输入按"最小可无损容纳"
// 升 float(b/i8/u8→f16,i16/u16→f32,其余→f64)—— 与超越算子同一张表,
// 复用 trigFloatFor(impl_trig.go)预 Cast。
// - 边角:floor/ceil/trunc/rint 对 ±0/±Inf/NaN 保值
// (floor(-0.0)=-0.0、ceil(-0.5)=-0.0、trunc(-0.7)=-0.0、rint(-0.0)=-0.0)。

import "math"

// IntOrBool 整型 + bool(floor/ceil/trunc 的恒等内核约束)。
type IntOrBool interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// roundF 浮点取整内核(f64 计算,T 回绕;f32 与 numpy 同构)。
func roundF[T FloatKinds](dst, src []T, f func(float64) float64) {
	for i := range dst {
		dst[i] = T(f(float64(src[i])))
	}
}

// regRoundIdentity floor/ceil/trunc 的整型/bool 恒等内核。
func regRoundIdentity[T IntOrBool]() {
	id := func(dst, src []T) { copy(dst, src) }
	registerUnaryTyped[T]("floor", id)
	registerUnaryTyped[T]("ceil", id)
	registerUnaryTyped[T]("trunc", id)
}

// regRoundFloat 一个取整算子的浮点三路内核(f16 HALF / f32 / f64)。
func regRoundFloat(op string, f func(float64) float64) {
	registerUnaryTyped[float32](op, func(dst, src []float32) { roundF(dst, src, f) })
	registerUnaryTyped[float64](op, func(dst, src []float64) { roundF(dst, src, f) })
	registerUnarySpec(op, UnarySpec{Kernels: map[Kind]UnaryKernel{Float16: f16Kernel(f)}})
}

func init() {
	regRoundFloat("floor", math.Floor)
	regRoundFloat("ceil", math.Ceil)
	regRoundFloat("trunc", math.Trunc)
	regRoundFloat("rint", math.RoundToEven)

	for _, reg := range []func(){
		func() { regRoundIdentity[bool]() }, func() { regRoundIdentity[int8]() },
		func() { regRoundIdentity[int16]() }, func() { regRoundIdentity[int32]() },
		func() { regRoundIdentity[int64]() }, func() { regRoundIdentity[uint8]() },
		func() { regRoundIdentity[uint16]() }, func() { regRoundIdentity[uint32]() },
		func() { regRoundIdentity[uint64]() },
	} {
		reg()
	}
}

// —— 入口适配 ——

// rintDispatch rint 分发:整型/bool 按 numpy 类型解析预 Cast(/§7 同表)。
func rintDispatch(x *Array) (*Array, error) {
	if dt, ok := trigFloatFor(x.Dtype().Kind); ok {
		var err error
		if x, err = Cast(x, dt); err != nil {
			return nil, err
		}
	}
	return dispatchUnary("rint", x)
}

// Floor 向下取整(整型/bool 恒等)。
func Floor(x *Array) (*Array, error) { return dispatchUnary("floor", x) }

// Ceil 向上取整(整型/bool 恒等)。
func Ceil(x *Array) (*Array, error) { return dispatchUnary("ceil", x) }

// Trunc 向零取整(整型/bool 恒等)。
func Trunc(x *Array) (*Array, error) { return dispatchUnary("trunc", x) }

// Rint 四舍六入五成双(银行家舍入 round-half-to-even;整型/bool 输入升 float,
// numpy 同)。
func Rint(x *Array) (*Array, error) { return rintDispatch(x) }

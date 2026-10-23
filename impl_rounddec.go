package num

// T11b:Round(a, decimals) = numpy np.round(B 拥有)。
//
// 语义基准(numpy 2.5.4 实测,逐位照 np.round 公式):
// - decimals == 0 等价 Rint(银行家舍入 half-to-even);
// - decimals != 0:乘 10**decimals(与元素同精度的标量)→ Rint → 除同一标量。
// 标量精度随元素走:f64 = double、f32 = float32、f16 升 f32 算完回 f16
// (numpy HALF 策略);乘与除都在元素精度内,故 f32 的 2.675×100 = 267.5
// → 268 → 2.68,而 f64 的 2.675×100 = 267.49999999999997 → 267 → 2.67。
// - 整型/bool:恒等拷贝(保 dtype,numpy 对整型输入原样返回);复数按实部/
// 虚部各自取整;±0/±Inf/NaN 保值(含 -0 符号);任意形状含 0-d;
// 不支持 dtype(datetime/S/U)→ ErrDtype。
//
// decimals 是调用参数而非 ufunc 规格的一部分,故内核表独立于 unaryReg:
// 内核仍按 Kind 键、走 win[T] typed 窗(与 UnaryKernel 同一引擎前提)。

import (
	"encoding/binary"
	"fmt"
	"math"
)

// roundDecKernel 带 10**decimals 标量的一元内核(窗:连续、8 对齐、等长)。
type roundDecKernel func(dst, src []byte, pow10 float64) error

// roundDecReg Round 的按 Kind 内核表。
var roundDecReg = map[Kind]roundDecKernel{}

// roundDecF 浮点内核:乘 pow10(T 精度)→ RoundToEven → 除 pow10(T 精度)。
// RoundToEven 在 T 值的精确 f64 像上执行:|T 值| < 2^24 时整值结果可无损回 T,
// 更大者本身已是整值(无舍入),故与 numpy 的本精度 rint 逐位一致。
func roundDecF[T FloatKinds](dst, src []T, pow10 T) {
	for i := range dst {
		dst[i] = T(math.RoundToEven(float64(src[i]*pow10))) / pow10
	}
}

// roundDecC64/C128 复数内核:实部与虚部各自按本精度取整(numpy 同)。
func roundDecC64(dst, src []complex64, pow10 float32) {
	for i := range dst {
		re := float32(math.RoundToEven(float64(real(src[i])*pow10))) / pow10
		im := float32(math.RoundToEven(float64(imag(src[i])*pow10))) / pow10
		dst[i] = complex(re, im)
	}
}

func roundDecC128(dst, src []complex128, pow10 float64) {
	for i := range dst {
		re := math.RoundToEven(real(src[i])*pow10) / pow10
		im := math.RoundToEven(imag(src[i])*pow10) / pow10
		dst[i] = complex(re, im)
	}
}

// roundDecF16 f16 内核(numpy HALF 策略:f16→f32→计算→f16;pow10 取 f32 精度)。
func roundDecF16(dst, src []byte, pow10 float32) error {
	n := len(src) / 2
	for i := 0; i < n; i++ {
		v := f16ToF32(binary.LittleEndian.Uint16(src[i*2:])) * pow10
		r := float32(math.RoundToEven(float64(v))) / pow10
		binary.LittleEndian.PutUint16(dst[i*2:], f32ToF16(r))
	}
	return nil
}

// regRoundDecTyped typed 注册(out/in 同 T);pow10 按 T 精度取用。
func regRoundDecTyped[T FloatKinds](fn func(dst, src []T, pow10 T)) {
	k := kindOfVal[T]()
	if k == 0 {
		panic("num: regRoundDecTyped unsupported T") // 内部使用,程序员错误
	}
	roundDecReg[k] = func(dst, src []byte, pow10 float64) error {
		fn(win[T](dst), win[T](src), T(pow10))
		return nil
	}
}

// regRoundDecIdentity 整型/bool 恒等内核(numpy 对整型输入原样返回,保 dtype)。
func regRoundDecIdentity[T IntOrBool]() {
	k := kindOfVal[T]()
	if k == 0 {
		panic("num: regRoundDecIdentity unsupported T") // 内部使用,程序员错误
	}
	roundDecReg[k] = func(dst, src []byte, pow10 float64) error {
		copy(win[T](dst), win[T](src))
		return nil
	}
}

func init() {
	regRoundDecTyped[float32](roundDecF[float32])
	regRoundDecTyped[float64](roundDecF[float64])

	roundDecReg[Float16] = func(dst, src []byte, pow10 float64) error {
		roundDecF16(dst, src, float32(pow10))
		return nil
	}
	roundDecReg[Complex64] = func(dst, src []byte, pow10 float64) error {
		roundDecC64(win[complex64](dst), win[complex64](src), float32(pow10))
		return nil
	}
	roundDecReg[Complex128] = func(dst, src []byte, pow10 float64) error {
		roundDecC128(win[complex128](dst), win[complex128](src), pow10)
		return nil
	}

	for _, reg := range []func(){
		func() { regRoundDecIdentity[bool]() }, func() { regRoundDecIdentity[int8]() },
		func() { regRoundDecIdentity[int16]() }, func() { regRoundDecIdentity[int32]() },
		func() { regRoundDecIdentity[int64]() }, func() { regRoundDecIdentity[uint8]() },
		func() { regRoundDecIdentity[uint16]() }, func() { regRoundDecIdentity[uint32]() },
		func() { regRoundDecIdentity[uint64]() },
	} {
		reg()
	}
}

// Round numpy np.round(a, decimals):decimals == 0 等价 Rint;否则按
// "乘 10**decimals(与元素同精度标量)→ 银行家舍入 → 除同一标量"。
// 整型/bool 恒等保 dtype,复数按分量取整,±0/±Inf/NaN 保值;
// 不支持 dtype → ErrDtype。
func Round(a *Array, decimals int) (*Array, error) {
	fn := roundDecReg[a.Dtype().Kind]
	if fn == nil {
		return nil, fmt.Errorf("num: round 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	pow10 := math.Pow(10, float64(decimals))
	out, err := New(a.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	src := a.AsContiguous()
	if err := fn(engineWindow(out), engineWindow(src), pow10); err != nil {
		return nil, err
	}
	return out, nil
}

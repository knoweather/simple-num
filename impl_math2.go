package num

import (
	"fmt"
	"math"
)

// Deg2Rad ≡ np.deg2rad:x * (π/180)。
// 常量口径用 numpy 同式 `x * (NPY_PI/180)`(已用 1e8 / 12345.678 / 0.1 三点位型探针锁定;
// 注意不是 (x*π)/180 —— 那种写法在大值上差一位)。整型/bool 输入按 numpy 解析提升
// (i8/u8→f16,i16/u16→f32,i32 以上→f64)。
func Deg2Rad(x *Array) (*Array, error) { return trigDispatch("deg2rad", x) }

// Rad2Deg ≡ np.rad2deg:x * (180/π)(同为 numpy 同式)。
func Rad2Deg(x *Array) (*Array, error) { return trigDispatch("rad2deg", x) }

// Fabs ≡ np.fabs:浮点绝对值。整型/bool 输入按 numpy 解析提升为浮点
// (np.fabs(int8) → float16),复数无定义(走 ErrDtype)。
func Fabs(x *Array) (*Array, error) { return trigDispatch("fabs", x) }

// Sign ≡ np.sign:[-1, 0, 1];NaN 传播;整型保持原 dtype;
// 复数按 numpy 口径 x/|x|(零保持 0+0i);bool 无定义(numpy 同样报错)→ ErrDtype。
func Sign(x *Array) (*Array, error) { return dispatchUnary("sign", x) }

// Arctan2 ≡ np.arctan2(y, x):四象限反正切,atan2(y, x)。
// 整型/bool 操作数各自按 numpy 解析提升(见 Deg2Rad),再按 promote 决定结果 dtype。
func Arctan2(y, x *Array) (*Array, error) {
	yy, xx := y, x
	var err error
	if dt, ok := trigFloatFor(yy.Dtype().Kind); ok {
		if yy, err = Cast(yy, dt); err != nil {
			return nil, fmt.Errorf("num: arctan2: y 提升失败: %w", err)
		}
	}
	if dt, ok := trigFloatFor(xx.Dtype().Kind); ok {
		if xx, err = Cast(xx, dt); err != nil {
			return nil, fmt.Errorf("num: arctan2: x 提升失败: %w", err)
		}
	}
	return dispatchBinary("arctan2", yy, xx)
}

// signSignedInt 有符号整型 sign(0 → 0,正 → 1,负 → -1)。
func signSignedInt[T int8 | int16 | int32 | int64](dst, src []T) {
	for i, v := range src {
		switch {
		case v > 0:
			dst[i] = 1
		case v < 0:
			dst[i] = -1
		default:
			dst[i] = 0
		}
	}
}

// signUnsignedInt 无符号整型 sign(0 → 0,否则 1)。
func signUnsignedInt[T uint8 | uint16 | uint32 | uint64](dst, src []T) {
	for i, v := range src {
		if v > 0 {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// signF 浮点 sign(NaN 传播:返回原值保 payload;零值一律给 **+0**,
// 与 numpy 的 (x>0)-(x<0) 口径一致 —— 注意 -0.0 的结果是 +0 而非 -0)。
func signF[T float32 | float64](dst, src []T) {
	for i, v := range src {
		switch {
		case v != v:
			dst[i] = v
		case v > 0:
			dst[i] = 1
		case v < 0:
			dst[i] = -1
		default:
			dst[i] = 0
		}
	}
}

// signC 复数 sign = x/|x|(零向量保持 0)。
func signC[T complex64 | complex128](dst, src []T) {
	for i, v := range src {
		z := complex128(v)
		re, im := real(z), imag(z)
		m := math.Hypot(re, im)
		if m == 0 {
			dst[i] = 0
			continue
		}
		dst[i] = T(complex(re/m, im/m))
	}
}

// arctan2F 浮点二元内核。
func arctan2F[T float32 | float64](dst, a, b []T) {
	for i := range dst {
		dst[i] = T(math.Atan2(float64(a[i]), float64(b[i])))
	}
}

func init() {
	registerTrig("deg2rad", func(v float64) float64 { return v * (math.Pi / 180) })
	registerTrig("rad2deg", func(v float64) float64 { return v * (180 / math.Pi) })
	registerTrig("fabs", math.Abs)

	// sign:整型保型
	registerUnaryTyped[int8]("sign", signSignedInt[int8])
	registerUnaryTyped[int16]("sign", signSignedInt[int16])
	registerUnaryTyped[int32]("sign", signSignedInt[int32])
	registerUnaryTyped[int64]("sign", signSignedInt[int64])
	registerUnaryTyped[uint8]("sign", signUnsignedInt[uint8])
	registerUnaryTyped[uint16]("sign", signUnsignedInt[uint16])
	registerUnaryTyped[uint32]("sign", signUnsignedInt[uint32])
	registerUnaryTyped[uint64]("sign", signUnsignedInt[uint64])
	registerUnaryTyped[float32]("sign", signF[float32])
	registerUnaryTyped[float64]("sign", signF[float64])
	registerUnaryTyped[complex64]("sign", signC[complex64])
	registerUnaryTyped[complex128]("sign", signC[complex128])
	// sign:f16 走既有 f16Kernel(以 float64 语义计算再回写半精度)
	registerUnarySpec("sign", UnarySpec{Kernels: map[Kind]UnaryKernel{
		Float16: f16Kernel(func(v float64) float64 {
			switch {
			case v != v:
				return v
			case v > 0:
				return 1
			case v < 0:
				return -1
			default:
				return 0 // 零值一律 +0(numpy 同)
			}
		}),
	}})

	// arctan2:二元浮点内核 + f16
	registerBinaryTyped[float32]("arctan2", arctan2F[float32])
	registerBinaryTyped[float64]("arctan2", arctan2F[float64])
	registerBinarySpec("arctan2", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(arctan2F[float32]),
	}})
}

package num

// :复数算子(B 拥有):c64/c128 四则、abs、conj。
//
// 语义基准(numpy 2.5.4 实测,):
// - add/sub/mul 与 numpy **逐位一致**(Go 原生 complex 运算实测同;mul 为
// naive 公式 (ac-bd, ad+bc));
// - div 必须按 numpy 的缩放算法实现(loops.c.src @TYPE@_divide):Go 内建 `/`
// 在 (-3.5+0.5i)/(0.5+2i) 的实部差 1ulp;缩放算法逐位对齐(rel 门槛兜底);
// 除零:c==0&&d==0 → (a/|c|, b/|d|)(1+1j)/0 → (inf,inf);0/0 → (NaN,NaN);
// - abs:c64 → f32、c128 → f64(numpy npy_cabs = hypot(实, 虚));
// - conj:虚部取反(精确);
// - cmp:仅 eq/ne(有序比较 numpy 抛 TypeError → ErrDtype);NaN 参与 eq 恒 False;
// - 提升按 promote 表(c64+i8 → c64、c64+f8 → c128)。

import (
	"fmt"
	"math"
)

// complexKinds c64/c128 约束。
type complexKinds interface{ complex64 | complex128 }

func addC[T complexKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subC[T complexKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulC[T complexKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

// absT 分量绝对值(仅用于除法的 |c|/|d|;+0 归一,NaN 透传)。
func absT[T FloatKinds](x T) T {
	if x < 0 {
		return -x
	}
	if x == 0 {
		return 0
	}
	return x
}

// cdiv 复刻 numpy 复数除法缩放算法(@TYPE@_divide):
//
//	|c| >= |d|:rat = d/c; scl = 1/(c + d*rat); re = (a + b*rat)*scl; im = (b - a*rat)*scl
//	否则 :rat = c/d; scl = 1/(d + c*rat); re = (a*rat + b)*scl; im = (b*rat - a)*scl
//	c==0&&d==0 → (a/|c|, b/|d|)(IEEE:±Inf/NaN)
func cdiv[T FloatKinds](ar, ai, br, bi T) (T, T) {
	if br == 0 && bi == 0 {
		return ar / absT(br), ai / absT(bi)
	}
	abr, abi := absT(br), absT(bi)
	if abr >= abi {
		rat := bi / br
		scl := 1 / (br + bi*rat)
		return (ar + ai*rat) * scl, (ai - ar*rat) * scl
	}
	rat := br / bi
	scl := 1 / (bi + br*rat)
	return (ar*rat + ai) * scl, (ai*rat - ar) * scl
}

func divC64(dst, a, b []complex64) {
	for i := range dst {
		re, im := cdiv[float32](real(a[i]), imag(a[i]), real(b[i]), imag(b[i]))
		dst[i] = complex(re, im)
	}
}

func divC128(dst, a, b []complex128) {
	for i := range dst {
		re, im := cdiv[float64](real(a[i]), imag(a[i]), real(b[i]), imag(b[i]))
		dst[i] = complex(re, im)
	}
}

// conjC 虚部取反(-(-0) = +0,numpy 同)。Go 泛型不支持 real/imag 内建,
// 经 complex128 中转(两个候选型别均可无损转换)。
func conjC[T complexKinds](dst, src []T) {
	for i := range dst {
		z := complex128(src[i])
		dst[i] = T(complex(real(z), -imag(z)))
	}
}

// complexAbs c64 → f32、c128 → f64(numpy npy_cabs = hypot)。
func complexAbs(a *Array) (*Array, error) {
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Complex64:
		out, err := New(a.Shape(), DFloat32)
		if err != nil {
			return nil, err
		}
		if out.Size() == 0 {
			return out, nil
		}
		s, d := win[complex64](engineWindow(src)), win[float32](engineWindow(out))
		for i, z := range s {
			d[i] = float32(math.Hypot(float64(real(z)), float64(imag(z))))
		}
		return out, nil
	case Complex128:
		out, err := New(a.Shape(), DFloat64)
		if err != nil {
			return nil, err
		}
		if out.Size() == 0 {
			return out, nil
		}
		s, d := win[complex128](engineWindow(src)), win[float64](engineWindow(out))
		for i, z := range s {
			d[i] = math.Hypot(real(z), imag(z))
		}
		return out, nil
	}
	return nil, fmt.Errorf("num: abs 无 %s 内核: %w", a.Dtype(), ErrDtype)
}

func init() {
	registerBinaryTyped[complex64]("add", addC[complex64])
	registerBinaryTyped[complex64]("subtract", subC[complex64])
	registerBinaryTyped[complex64]("multiply", mulC[complex64])
	registerBinaryTyped[complex64]("divide", divC64)
	registerBinaryTyped[complex128]("add", addC[complex128])
	registerBinaryTyped[complex128]("subtract", subC[complex128])
	registerBinaryTyped[complex128]("multiply", mulC[complex128])
	registerBinaryTyped[complex128]("divide", divC128)

	registerUnaryTyped[complex64]("conj", conjC[complex64])
	registerUnaryTyped[complex128]("conj", conjC[complex128])
}

// Conj 共轭(虚部取反)。
func Conj(a *Array) (*Array, error) { return dispatchUnary("conj", a) }

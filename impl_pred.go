package num

// 谓词与逻辑系(B 拥有):isnan / isinf / isfinite(IEEE 判定)与
// logical_and / logical_or / logical_xor / logical_not(真值逻辑)。
//
// 语义基准:numpy 2.5.4 实测:
// - 三个 IEEE 谓词输出恒 bool、形状同输入;整型与 bool 输入无 ufunc 循环
// (numpy TypeError),本库以 ErrDtype 同口径拒绝;S/U/M8 同样 ErrDtype。
// - 复数按分量判定:isnan「任一分量 NaN」、isinf「任一分量 Inf」、
// isfinite「两分量皆有限」。f16 经 f16ToF32 升格后判定(升格无损,位型完整)。
// - logical_* 接受 bool 与任意数值 dtype,非零即真(NaN 为真、复数任一分量非零);
// 二元按 numpy 类型解析先 promote 再取真值,输出恒 bool;不可广播 → ErrShape。
//
// 复用:真值判定走 impl_where.go 的 truthyMask,广播与归一走 broadcastShapes /
// normalizeOperand。谓词与逻辑的输出 dtype 恒 bool(≠ 输入 dtype),不符合 ufunc 注册表
// 的"同 dtype 内核"形态,故本文件自带类型化循环,不经注册表。

import (
	"fmt"
	"math"
)

// predJudge 逐元素判定:src 为输入 dtype 的窗,dst 为逐元素 1 字节 bool 窗。
type predJudge func(k Kind, dst []bool, src []byte)

// predJudgeable 谓词可判定的 dtype 面(浮点 + 复数;其余 numpy 无循环)。
func predJudgeable(k Kind) bool {
	switch k {
	case Float16, Float32, Float64, Complex64, Complex128:
		return true
	}
	return false
}

// predUnary 谓词一元入口:归一(非连续视图经 transformCopy)后逐元素判定,
// 输出 bool 数组、形状同输入。
func predUnary(a *Array, op string, judge predJudge) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: %s: 操作数为 nil: %w", op, ErrDtype)
	}
	k := a.Dtype().Kind
	if !predJudgeable(k) {
		return nil, fmt.Errorf("num: ufunc %q 无 %s 循环: %w", op, a.Dtype(), ErrDtype)
	}
	out, err := New(a.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	ax, err := normalizeOperand(a, out.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	judge(k, win[bool](engineWindow(out)), engineWindow(ax))
	return out, nil
}

// IsNan ≡ np.isnan:逐元素 NaN 判定(复数任一分量)。整型/bool/S/U/M8 → ErrDtype。
func IsNan(a *Array) (*Array, error) { return predUnary(a, "isnan", judgeIsNan) }

// IsInf ≡ np.isinf:逐元素 Inf 判定(复数任一分量)。
func IsInf(a *Array) (*Array, error) { return predUnary(a, "isinf", judgeIsInf) }

// IsFinite ≡ np.isfinite:逐元素有限判定(复数须两分量皆有限)。
func IsFinite(a *Array) (*Array, error) { return predUnary(a, "isfinite", judgeIsFinite) }

func judgeIsNan(k Kind, dst []bool, src []byte) {
	switch k {
	case Float16:
		for i, h := range win[uint16](src) {
			dst[i] = math.IsNaN(float64(f16ToF32(h)))
		}
	case Float32:
		for i, v := range win[float32](src) {
			dst[i] = math.IsNaN(float64(v))
		}
	case Float64:
		for i, v := range win[float64](src) {
			dst[i] = math.IsNaN(v)
		}
	case Complex64:
		for i, v := range win[complex64](src) {
			dst[i] = math.IsNaN(float64(real(v))) || math.IsNaN(float64(imag(v)))
		}
	case Complex128:
		for i, v := range win[complex128](src) {
			dst[i] = math.IsNaN(real(v)) || math.IsNaN(imag(v))
		}
	}
}

func judgeIsInf(k Kind, dst []bool, src []byte) {
	switch k {
	case Float16:
		for i, h := range win[uint16](src) {
			dst[i] = math.IsInf(float64(f16ToF32(h)), 0)
		}
	case Float32:
		for i, v := range win[float32](src) {
			dst[i] = math.IsInf(float64(v), 0)
		}
	case Float64:
		for i, v := range win[float64](src) {
			dst[i] = math.IsInf(v, 0)
		}
	case Complex64:
		for i, v := range win[complex64](src) {
			dst[i] = math.IsInf(float64(real(v)), 0) || math.IsInf(float64(imag(v)), 0)
		}
	case Complex128:
		for i, v := range win[complex128](src) {
			dst[i] = math.IsInf(real(v), 0) || math.IsInf(imag(v), 0)
		}
	}
}

func judgeIsFinite(k Kind, dst []bool, src []byte) {
	switch k {
	case Float16:
		for i, h := range win[uint16](src) {
			f := float64(f16ToF32(h))
			dst[i] = !math.IsNaN(f) && !math.IsInf(f, 0)
		}
	case Float32:
		for i, v := range win[float32](src) {
			f := float64(v)
			dst[i] = !math.IsNaN(f) && !math.IsInf(f, 0)
		}
	case Float64:
		for i, v := range win[float64](src) {
			dst[i] = !math.IsNaN(v) && !math.IsInf(v, 0)
		}
	case Complex64:
		for i, v := range win[complex64](src) {
			r, im := float64(real(v)), float64(imag(v))
			dst[i] = !math.IsNaN(r) && !math.IsInf(r, 0) && !math.IsNaN(im) && !math.IsInf(im, 0)
		}
	case Complex128:
		for i, v := range win[complex128](src) {
			r, im := real(v), imag(v)
			dst[i] = !math.IsNaN(r) && !math.IsInf(r, 0) && !math.IsNaN(im) && !math.IsInf(im, 0)
		}
	}
}

// truthyMaskOf 归一(广播 + cast)到 dt 后取真值掩码;S/U/M8 由 truthyMask 拒(ErrDtype)。
func truthyMaskOf(a *Array, outSh []int, dt Dtype) ([]bool, error) {
	ax, err := normalizeOperand(a, outSh, dt)
	if err != nil {
		return nil, err
	}
	return truthyMask(ax)
}

// logicalBinary 二元逻辑:numpy 右对齐广播 → promote → 归一 → 逐元素真值合成 → bool。
// 先 promote 再取真值(numpy 类型解析口径),故 i4×f8 在 f8 上判非零。
func logicalBinary(op string, x, y *Array, combine func(a, b bool) bool) (*Array, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("num: %s: 操作数为 nil: %w", op, ErrDtype)
	}
	outSh, err := broadcastShapes(x.Shape(), y.Shape())
	if err != nil {
		return nil, fmt.Errorf("num: %s: %w", op, err)
	}
	pdt, err := promoteTypes(x.Dtype(), y.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: %s: %w", op, err)
	}
	mx, err := truthyMaskOf(x, outSh, pdt)
	if err != nil {
		return nil, err
	}
	my, err := truthyMaskOf(y, outSh, pdt)
	if err != nil {
		return nil, err
	}
	out, err := New(outSh, DBool)
	if err != nil {
		return nil, err
	}
	n := out.Size()
	if n == 0 {
		return out, nil
	}
	d := win[bool](engineWindow(out))
	for i := 0; i < n; i++ {
		d[i] = combine(mx[i], my[i])
	}
	return out, nil
}

// LogicalAnd ≡ np.logical_and:两操作数广播后逐元素"非零即真"求与,输出恒 bool。
func LogicalAnd(x, y *Array) (*Array, error) {
	return logicalBinary("logical_and", x, y, func(a, b bool) bool { return a && b })
}

// LogicalOr ≡ np.logical_or:同口径求或。
func LogicalOr(x, y *Array) (*Array, error) {
	return logicalBinary("logical_or", x, y, func(a, b bool) bool { return a || b })
}

// LogicalXor ≡ np.logical_xor:同口径求异或。
func LogicalXor(x, y *Array) (*Array, error) {
	return logicalBinary("logical_xor", x, y, func(a, b bool) bool { return a != b })
}

// LogicalNot ≡ np.logical_not:一元取反真值,输出同形状 bool。
func LogicalNot(x *Array) (*Array, error) {
	if x == nil {
		return nil, fmt.Errorf("num: logical_not: 操作数为 nil: %w", ErrDtype)
	}
	m, err := truthyMaskOf(x, x.Shape(), x.Dtype())
	if err != nil {
		return nil, err
	}
	out, err := New(x.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	n := out.Size()
	if n == 0 {
		return out, nil
	}
	d := win[bool](engineWindow(out))
	for i := 0; i < n; i++ {
		d[i] = !m[i]
	}
	return out, nil
}

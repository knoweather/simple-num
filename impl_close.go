package num

import (
	"fmt"
	"math"
)

// IsClose ≡ np.isclose(a, b, rtol, atol):逐元素判"足够接近"。
//
// numpy 的判定式(逐字):
//
//	a == b → True (同号 Inf 相等也走这里)
//	isinf(a) || isinf(b) → False
//	|a-b| <= atol + rtol*|b| → True
//
// 故 NaN 与任何值(含 NaN)都是 False;两个同号 Inf 是 True。整型/bool 输入按
// numpy 解析提升为浮点比较(f32 参与时保持 f32 精度)。
func IsClose(a, b *Array, rtol, atol float64) (*Array, error) {
	x, y, dt, err := closeOperands(a, b)
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
	switch dt.Kind {
	case Float32:
		closeInto[float32](win[uint8](engineWindow(out)), win[float32](engineWindow(x)),
			win[float32](engineWindow(y)), float32(rtol), float32(atol))
	case Float64:
		closeInto[float64](win[uint8](engineWindow(out)), win[float64](engineWindow(x)),
			win[float64](engineWindow(y)), rtol, atol)
	default:
		return nil, fmt.Errorf("num: isclose 无 %s 内核: %w", dt, ErrDtype)
	}
	return out, nil
}

// AllClose ≡ np.allclose:IsClose 结果全为真(空数组视为真,与 numpy 同)。
func AllClose(a, b *Array, rtol, atol float64) (bool, error) {
	m, err := IsClose(a, b, rtol, atol)
	if err != nil {
		return false, err
	}
	w := win[uint8](engineWindow(m.AsContiguous()))
	for _, v := range w {
		if v == 0 {
			return false, nil
		}
	}
	return true, nil
}

// closeInto 按 numpy 判定式逐元素写 bool(1 字节)。
func closeInto[T float32 | float64](dst []uint8, a, b []T, rtol, atol T) {
	for i := range a {
		av, bv := a[i], b[i]
		switch {
		case av == bv: // 含同号 Inf
			dst[i] = 1
		case isInfT(av) || isInfT(bv):
			dst[i] = 0
		default:
			d := av - bv
			if d < 0 {
				d = -d
			}
			lim := atol + rtol*absT(bv)
			if d <= lim {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
	}
}

func isInfT[T float32 | float64](v T) bool {
	return v == T(math.Inf(1)) || v == T(math.Inf(-1))
}

// closeOperands 广播 + 提升到浮点(numpy 的 isclose 走 ufunc 解析)。
func closeOperands(a, b *Array) (*Array, *Array, Dtype, error) {
	sh, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, nil, Dtype{}, fmt.Errorf("num: isclose: 形状不可广播: %w", err)
	}
	pa, pb := a.Dtype(), b.Dtype()
	// 非浮点先按 trigFloatFor 的 numpy 解析口径提升
	if dt, ok := trigFloatFor(pa.Kind); ok {
		pa = dt
	}
	if dt, ok := trigFloatFor(pb.Kind); ok {
		pb = dt
	}
	dt, err := promoteTypes(pa, pb)
	if err != nil {
		return nil, nil, Dtype{}, fmt.Errorf("num: isclose: dtype 不可提升: %w", err)
	}
	if dt.Kind != Float32 && dt.Kind != Float64 {
		return nil, nil, Dtype{}, fmt.Errorf("num: isclose 无 %s 内核: %w", dt, ErrDtype)
	}
	x, err := normalizeOperand(a, sh, dt)
	if err != nil {
		return nil, nil, Dtype{}, err
	}
	y, err := normalizeOperand(b, sh, dt)
	if err != nil {
		return nil, nil, Dtype{}, err
	}
	return x, y, dt, nil
}

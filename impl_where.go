package num

import "fmt"

// Where ≡ np.where(cond, x, y):三操作数广播选择 —— cond 为真取 x,否则取 y。
//
// - 三个操作数按 numpy 右对齐广播到公共形状(cond 与 x 先广播,x 结果再与 y 广播);
// - cond 非 bool 时按 != 0 取真(NaN 为真、复数任一分量非零为真,与 numpy
// truthiness 同口径);S/U/M8 无真值语义 → ErrDtype;
// - 结果 dtype = promote(x, y)(cond 不参与 promote,与 numpy 同);
// - 选择是**按元素整块搬运**,与元素语义无关,故 f16 / S / U / c128 同样适用。
//
// 单参形态 np.where(cond)(等价非零索引元组)见 Nonzero,本函数不做重载。
func Where(cond, x, y *Array) (*Array, error) {
	if cond == nil || x == nil || y == nil {
		return nil, fmt.Errorf("num: where: 操作数为 nil: %w", ErrDtype)
	}
	sh, err := broadcastShapes(cond.Shape(), x.Shape())
	if err != nil {
		return nil, fmt.Errorf("num: where: cond 与 x 不可广播: %w", err)
	}
	if sh, err = broadcastShapes(sh, y.Shape()); err != nil {
		return nil, fmt.Errorf("num: where: y 不可广播到 %v: %w", sh, err)
	}
	odt, err := promoteTypes(x.Dtype(), y.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: where: x/y dtype 不可提升: %w", err)
	}
	nc, err := normalizeOperand(cond, sh, cond.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: where: cond 归一失败: %w", err)
	}
	nx, err := normalizeOperand(x, sh, odt)
	if err != nil {
		return nil, fmt.Errorf("num: where: x 归一失败: %w", err)
	}
	ny, err := normalizeOperand(y, sh, odt)
	if err != nil {
		return nil, fmt.Errorf("num: where: y 归一失败: %w", err)
	}
	mask, err := truthyMask(nc)
	if err != nil {
		return nil, err
	}
	out, err := New(sh, odt)
	if err != nil {
		return nil, err
	}
	n := out.Size()
	if n == 0 {
		return out, nil
	}
	if err := selectByMask(engineWindow(out), engineWindow(nx), engineWindow(ny), mask, odt.Itemsize); err != nil {
		return nil, err
	}
	return out, nil
}

// truthyMask 把数组压成逐元素真值掩码(numpy truthiness:非 bool 按 != 0;
// NaN != 0 为真;复数任一分量非零为真)。
func truthyMask(a *Array) ([]bool, error) {
	n := a.Size()
	m := make([]bool, n)
	if n == 0 {
		return m, nil
	}
	b := engineWindow(a)
	switch a.Dtype().Kind {
	case Bool:
		for i, v := range win[uint8](b) {
			m[i] = v != 0
		}
	case Int8:
		for i, v := range win[int8](b) {
			m[i] = v != 0
		}
	case Int16:
		for i, v := range win[int16](b) {
			m[i] = v != 0
		}
	case Int32:
		for i, v := range win[int32](b) {
			m[i] = v != 0
		}
	case Int64:
		for i, v := range win[int64](b) {
			m[i] = v != 0
		}
	case Uint8:
		for i, v := range win[uint8](b) {
			m[i] = v != 0
		}
	case Uint16:
		for i, v := range win[uint16](b) {
			m[i] = v != 0
		}
	case Uint32:
		for i, v := range win[uint32](b) {
			m[i] = v != 0
		}
	case Uint64:
		for i, v := range win[uint64](b) {
			m[i] = v != 0
		}
	case Float16:
		for i, v := range win[uint16](b) {
			m[i] = f16ToF32(v) != 0
		}
	case Float32:
		for i, v := range win[float32](b) {
			m[i] = v != 0
		}
	case Float64:
		for i, v := range win[float64](b) {
			m[i] = v != 0
		}
	case Complex64:
		for i, v := range win[complex64](b) {
			m[i] = real(v) != 0 || imag(v) != 0
		}
	case Complex128:
		for i, v := range win[complex128](b) {
			m[i] = real(v) != 0 || imag(v) != 0
		}
	default:
		return nil, fmt.Errorf("num: 该 dtype 无真值语义 %s: %w", a.Dtype(), ErrDtype)
	}
	return m, nil
}

// selectByMask 按掩码把 xb/yb 的元素整块搬到 dst(元素宽 es 字节),与 dtype 无关。
func selectByMask(dst, xb, yb []byte, mask []bool, es int) error {
	n := len(mask)
	if es <= 0 {
		return fmt.Errorf("num: where: 元素宽度非法 %d: %w", es, ErrDtype)
	}
	need := n * es
	if len(dst) < need || len(xb) < need || len(yb) < need {
		return fmt.Errorf("num: where: 缓冲长度不足(need %d, got %d/%d/%d): %w",
			need, len(dst), len(xb), len(yb), ErrShape)
	}
	switch es {
	case 1:
		d, a, b := dst[:n], xb[:n], yb[:n]
		for i := 0; i < n; i++ {
			if mask[i] {
				d[i] = a[i]
			} else {
				d[i] = b[i]
			}
		}
	case 2:
		d, a, b := win[uint16](dst), win[uint16](xb), win[uint16](yb)
		for i := 0; i < n; i++ {
			if mask[i] {
				d[i] = a[i]
			} else {
				d[i] = b[i]
			}
		}
	case 4:
		d, a, b := win[uint32](dst), win[uint32](xb), win[uint32](yb)
		for i := 0; i < n; i++ {
			if mask[i] {
				d[i] = a[i]
			} else {
				d[i] = b[i]
			}
		}
	case 8:
		d, a, b := win[uint64](dst), win[uint64](xb), win[uint64](yb)
		for i := 0; i < n; i++ {
			if mask[i] {
				d[i] = a[i]
			} else {
				d[i] = b[i]
			}
		}
	case 16:
		for i := 0; i < n; i++ {
			lo := i * es
			if mask[i] {
				copy(dst[lo:lo+es], xb[lo:lo+es])
			} else {
				copy(dst[lo:lo+es], yb[lo:lo+es])
			}
		}
	default:
		for i := 0; i < n; i++ {
			lo := i * es
			if mask[i] {
				copy(dst[lo:lo+es], xb[lo:lo+es])
			} else {
				copy(dst[lo:lo+es], yb[lo:lo+es])
			}
		}
	}
	return nil
}

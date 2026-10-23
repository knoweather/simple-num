package num

import "fmt"

// dotComplex 复数版 Dot:1-D 内积与 2-D 矩阵乘与 numpy 一致;
// 更高维的复数 dot(numpy 对最后两轴做不同的收缩)暂不支持,明确报错而不是给错结果。
func dotComplex(a, b *Array) (*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: dot: 入参为 nil: %w", ErrDtype)
	}
	if a.NDim() <= 2 && b.NDim() <= 2 {
		if a.NDim() == 1 && b.NDim() == 1 {
			if a.Size() != b.Size() {
				return nil, fmt.Errorf("num: dot: 长度 %d ≠ %d: %w", a.Size(), b.Size(), ErrShape)
			}
			af, err := transformCopy(a.AsContiguous(), a.Shape(), DComplex128)
			if err != nil {
				return nil, err
			}
			bf, err := transformCopy(b.AsContiguous(), b.Shape(), DComplex128)
			if err != nil {
				return nil, err
			}
			aw := win[complex128](engineWindow(af))
			bw := win[complex128](engineWindow(bf))
			var s complex128
			for i := range aw {
				s += aw[i] * bw[i]
			}
			out, err := New([]int{}, DComplex128)
			if err != nil {
				return nil, err
			}
			if err := out.SetItem([]int{}, s); err != nil {
				return nil, err
			}
			return out, nil
		}
		return matmulComplex(a, b)
	}
	return nil, fmt.Errorf("num: dot: 复数仅支持 1-D/2-D(得 %d/%d 维): %w",
		a.NDim(), b.NDim(), ErrShape)
}

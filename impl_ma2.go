package num

import (
	"fmt"
	"math"
	"sort"
)

// —— 掩码数组长尾(Step 20):masked_invalid / masked_where / Median·Var·Std ——

// MaskedInvalid ≡ np.ma.masked_invalid:把 NaN 与 ±Inf 标为无效。
func MaskedInvalid(a *Array) (*MaskedArray, error) {
	mask, err := New(a.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	mw := win[uint8](engineWindow(mask))
	switch a.Dtype().Kind {
	case Float64:
		for i, v := range win[float64](engineWindow(a.AsContiguous())) {
			if v != v || math.IsInf(v, 0) {
				mw[i] = 1
			}
		}
	case Float32:
		for i, v := range win[float32](engineWindow(a.AsContiguous())) {
			f := float64(v)
			if f != f || math.IsInf(f, 0) {
				mw[i] = 1
			}
		}
	default:
		// 整型/文本等没有 NaN/Inf 概念:掩码全 false(numpy 对非浮点同样不标记)
	}
	return NewMasked(a, mask)
}

// MaskedWhere ≡ np.ma.masked_where(cond, a):条件为真处标为无效。
func MaskedWhere(cond, a *Array) (*MaskedArray, error) {
	nc, err := normalizeOperand(cond, a.Shape(), cond.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_where: cond 归一失败: %w", err)
	}
	bm, err := truthyMask(nc)
	if err != nil {
		return nil, err
	}
	mask, err := New(a.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	mw := win[uint8](engineWindow(mask))
	for i, v := range bm {
		if v {
			mw[i] = 1
		}
	}
	return NewMasked(a, mask)
}

// lineReduceMasked 掩码归约的公共骨架:逐"线"收集未掩码值交给 fn。
// fn 返回 (值, 是否整条线都被掩码)。
func (m *MaskedArray) lineReduceMasked(axis []int, keepdims bool, outDT Dtype,
	fn func(vals []float64) (float64, bool),
) (*MaskedArray, error) {
	// 数据转 float64(掩码归约只对数值有意义)
	data, err := transformCopy(m.data.AsContiguous(), m.data.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	mask := m.mask.AsContiguous()
	if len(axis) == 0 {
		dw := win[float64](engineWindow(data))
		bm := win[uint8](engineWindow(mask))
		vals := make([]float64, 0, len(dw))
		for i, v := range dw {
			if bm[i] == 0 {
				vals = append(vals, v)
			}
		}
		val, masked := fn(vals)
		var sh []int
		if keepdims {
			sh = make([]int, m.data.NDim())
			for i := range sh {
				sh[i] = 1
			}
		}
		out, err := New(sh, outDT)
		if err != nil {
			return nil, err
		}
		if err := setScalarFloat(out, val); err != nil {
			return nil, err
		}
		om, err := New(out.Shape(), DBool)
		if err != nil {
			return nil, err
		}
		if masked {
			win[uint8](engineWindow(om))[0] = 1
		}
		return &MaskedArray{data: out, mask: om}, nil
	}
	d, err := normalizeAxis(m.data, axis[0])
	if err != nil {
		return nil, fmt.Errorf("num: ma: %w", err)
	}
	rows, mm, n, err := axisRows(data, d)
	if err != nil {
		return nil, err
	}
	mrows, _, _, err := axisRows(mask, d)
	if err != nil {
		return nil, err
	}
	outSh := make([]int, m.data.NDim()-1)
	k := 0
	for i := 0; i < m.data.NDim(); i++ {
		if i == d {
			continue
		}
		outSh[k] = m.data.Shape()[i]
		k++
	}
	if keepdims {
		outSh = append([]int(nil), m.data.Shape()...)
		outSh[d] = 1
	}
	out, err := New(outSh, outDT)
	if err != nil {
		return nil, err
	}
	om, err := New(outSh, DBool)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, fmt.Errorf("num: ma: 归约轴长度为 0: %w", ErrShape)
	}
	if mm == 0 {
		return &MaskedArray{data: out, mask: om}, nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	mw := win[uint8](engineWindow(mrows.AsContiguous()))
	ow := win[float64](engineWindow(out))
	omb := win[uint8](engineWindow(om))
	vals := make([]float64, 0, n)
	for r := 0; r < mm; r++ {
		vals = vals[:0]
		for i := 0; i < n; i++ {
			if mw[r*n+i] == 0 {
				vals = append(vals, win[float64](sb[(r*n+i)*es : (r*n+i+1)*es])[0])
			}
		}
		v, masked := fn(vals)
		ow[r] = v
		if masked {
			omb[r] = 1
		}
	}
	return &MaskedArray{data: out, mask: om}, nil
}

// Median ≡ ma.median:跳过被掩码元素的中位数。
func (m *MaskedArray) Median(axis []int, keepdims bool) (*MaskedArray, error) {
	return m.lineReduceMasked(axis, keepdims, DFloat64, func(vals []float64) (float64, bool) {
		if len(vals) == 0 {
			return math.NaN(), true
		}
		sort.Float64s(vals)
		n := len(vals)
		if n%2 == 1 {
			return vals[n/2], false
		}
		return (vals[n/2-1] + vals[n/2]) / 2, false
	})
}

// Var ≡ ma.var(ddof=0):跳过被掩码元素的方差。
func (m *MaskedArray) Var(axis []int, keepdims bool) (*MaskedArray, error) {
	return m.lineReduceMasked(axis, keepdims, DFloat64, func(vals []float64) (float64, bool) {
		if len(vals) == 0 {
			return math.NaN(), true
		}
		mu := meanOf(vals)
		s := 0.0
		for _, v := range vals {
			d := v - mu
			s += d * d
		}
		return s / float64(len(vals)), false
	})
}

// Std ≡ ma.std(ddof=0):跳过被掩码元素的标准差。
func (m *MaskedArray) Std(axis []int, keepdims bool) (*MaskedArray, error) {
	return m.lineReduceMasked(axis, keepdims, DFloat64, func(vals []float64) (float64, bool) {
		if len(vals) == 0 {
			return math.NaN(), true
		}
		mu := meanOf(vals)
		s := 0.0
		for _, v := range vals {
			d := v - mu
			s += d * d
		}
		return math.Sqrt(s / float64(len(vals))), false
	})
}

// meanOf 简单均值(内部用;逐元素求和,与 numpy 的 ma 口径在中小规模一致)。
func meanOf(vals []float64) float64 {
	s := 0.0
	for _, v := range vals {
		s += v
	}
	return s / float64(len(vals))
}

package num

import (
	"fmt"
	"math"
	"sort"
)

// —— quantile 族(Step 17):quantile / nanpercentile / nanquantile ——
//
// np.quantile 与 np.percentile 是同一内核的孪生接口(q ∈ [0,1] vs [0,100]);
// np.nanpercentile / np.nanquantile 在每条"线"上先剔除 NaN 再计算
// (全 NaN 的线返回 NaN,与 numpy 的 warning + NaN 一致)。

// Quantile ≡ np.quantile(a, q, method)。q ∈ [0,1]。
func Quantile(a *Array, q float64, axis []int, keepdims bool) (*Array, error) {
	return QuantileMethod(a, q, "linear", axis, keepdims)
}

// QuantileMethod ≡ np.quantile(a, q, method=...)。
func QuantileMethod(a *Array, q float64, method string, axis []int, keepdims bool) (*Array, error) {
	if q < 0 || q > 1 || math.IsNaN(q) {
		return nil, fmt.Errorf("num: quantile: q=%v 越界(0..1): %w", q, ErrValue)
	}
	return PercentileMethod(a, q*100, method, axis, keepdims)
}

// QuantileMany ≡ np.quantile(a, [q1,...]):q 为数组。
func QuantileMany(a *Array, qs []float64, axis []int, keepdims bool) (*Array, error) {
	pq := make([]float64, len(qs))
	for i, q := range qs {
		if q < 0 || q > 1 || math.IsNaN(q) {
			return nil, fmt.Errorf("num: quantile: q=%v 越界(0..1): %w", q, ErrValue)
		}
		pq[i] = q * 100
	}
	return PercentileMany(a, pq, axis, keepdims)
}

// NanPercentile ≡ np.nanpercentile(a, q, method):每条线先剔除 NaN 再算分位数。
func NanPercentile(a *Array, q float64, axis []int, keepdims bool) (*Array, error) {
	return nanQuantileCore(a, q, "linear", axis, keepdims, true)
}

// NanQuantile ≡ np.nanquantile(a, q, method)。q ∈ [0,1]。
func NanQuantile(a *Array, q float64, axis []int, keepdims bool) (*Array, error) {
	if q < 0 || q > 1 || math.IsNaN(q) {
		return nil, fmt.Errorf("num: nanquantile: q=%v 越界(0..1): %w", q, ErrValue)
	}
	return nanQuantileCore(a, q*100, "linear", axis, keepdims, true)
}

// nanQuantileCore nan* 的公共实现(q 用百分数口径,与 percentile 一致)。
func nanQuantileCore(a *Array, q float64, method string, axis []int, keepdims bool, skipNaN bool) (*Array, error) {
	if q < 0 || q > 100 || math.IsNaN(q) {
		return nil, fmt.Errorf("num: nanpercentile: q=%v 越界(0..100): %w", q, ErrValue)
	}
	_ = skipNaN
	if len(axis) == 0 {
		v, err := quantileOfArray(a, q, method, true)
		if err != nil {
			return nil, err
		}
		outDT := DFloat64
		if a.Dtype().Kind == Float32 {
			outDT = DFloat32
		}
		var sh []int
		if keepdims {
			sh = make([]int, a.NDim())
			for i := range sh {
				sh[i] = 1
			}
		}
		out, err := New(sh, outDT)
		if err != nil {
			return nil, err
		}
		return out, setScalarFloat(out, v)
	}
	d, err := normalizeAxis(a, axis[0])
	if err != nil {
		return nil, fmt.Errorf("num: nanpercentile: %w", err)
	}
	return quantileAxisWalk(a, d, keepdims, func(line *Array) (float64, error) {
		return quantileOfArray(line, q, method, true)
	})
}

// quantileAxisWalk 对 a 沿轴 d 的每条"线"求值,结果按 numpy 的归约形状返回。
func quantileAxisWalk(a *Array, d int, keepdims bool, fn func(line *Array) (float64, error)) (*Array, error) {
	rows, m, n, err := axisRows(a, d)
	if err != nil {
		return nil, err
	}
	outSh := make([]int, a.NDim()-1)
	k := 0
	for i := 0; i < a.NDim(); i++ {
		if i == d {
			continue
		}
		outSh[k] = a.Shape()[i]
		k++
	}
	if keepdims {
		outSh = append([]int(nil), a.Shape()...)
		outSh[d] = 1
	}
	outDT := DFloat64
	if a.Dtype().Kind == Float32 {
		outDT = DFloat32
	}
	out, err := New(outSh, outDT)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, fmt.Errorf("num: quantile: 归约轴长度为 0,分位数无定义(numpy 亦报错): %w", ErrShape)
	}
	if m == 0 {
		return out, nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	vals := make([]float64, m)
	for r := 0; r < m; r++ {
		sub := &Array{buf: sb[r*n*es : (r+1)*n*es], shape: []int{n},
			st: cStrides([]int{n}, es), dtype: rows.Dtype()}
		v, err := fn(sub)
		if err != nil {
			return nil, err
		}
		vals[r] = v
	}
	return out, setFlatFloats(out, vals)
}

// quantileOfArray 对整数组求分位数;skipNaN=true 时先剔除 NaN。
func quantileOfArray(a *Array, q float64, method string, skipNaN bool) (float64, error) {
	if !skipNaN {
		sorted, err := sortedCopy(a.AsContiguous().Flatten())
		if err != nil {
			return 0, err
		}
		return quantileMethodOf(sorted, q, method)
	}
	f, err := transformCopy(a.AsContiguous().Flatten(), []int{a.Size()}, DFloat64)
	if err != nil {
		return 0, err
	}
	w := win[float64](engineWindow(f))
	vals := make([]float64, 0, len(w))
	for _, v := range w {
		if v == v {
			vals = append(vals, v)
		}
	}
	if len(vals) == 0 {
		return math.NaN(), nil // 全 NaN(numpy 同样返回 NaN)
	}
	sort.Float64s(vals)
	tmp, err := FromSlice(vals)
	if err != nil {
		return 0, err
	}
	return quantileMethodOf(tmp, q, method)
}

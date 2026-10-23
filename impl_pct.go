package num

import (
	"fmt"
	"math"
)

// —— 分位数方法扩展(Step 16):percentile 的 method 与 q 数组 ——
//
// numpy 的 9 种 method 归约为"虚拟下标 + 是否插值"两类。虚拟下标公式(设 n 个元素、
// t = q/100):
//
//	linear (n-1)·t
//	lower/higher/midpoint/nearest (n-1)·t 再按 floor/ceil/round 取离散点
//	inverted_cdf n·t (取 ceil-1,不插值)
//	averaged_inverted_cdf n·t (整数时取相邻两点均值)
//	closest_observation n·t - 0.5 (取 floor,不插值)
//	interpolated_inverted_cdf n·t - 0.5
//	hazen n·t - 0.5
//	weibull (n+1)·t - 1
//	median_unbiased (n + 1/3)·t + 1/3 - 1
//	normal_unbiased (n + 1/4)·t + 3/8 - 1
//
// 连续方法统一做 a[floor] + γ·(a[floor+1] - a[floor]),越界钳到两端。

// quantileMethodOf 计算单个 q 的分位数(已排序的 1-D 输入)。
func quantileMethodOf(sorted *Array, q float64, method string) (float64, error) {
	n := sorted.Size()
	if n == 0 {
		return math.NaN(), nil
	}
	flat := sorted.AsContiguous().Flatten()
	get := func(i int) (float64, error) {
		if i < 0 {
			i = 0
		}
		if i >= n {
			i = n - 1
		}
		return numToFloat(flat, i)
	}
	t := q / 100
	nf := float64(n)
	// 离散方法
	switch method {
	case "lower", "higher", "midpoint", "nearest":
		virt := (nf - 1) * t
		var idx int
		switch method {
		case "lower":
			idx = int(math.Floor(virt))
		case "higher":
			idx = int(math.Ceil(virt))
		case "nearest":
			idx = int(math.RoundToEven(virt)) // numpy 的 nearest 是"银行家舍入"(half-to-even)
		case "midpoint":
			lo := int(math.Floor(virt))
			hi := int(math.Ceil(virt))
			a, err := get(lo)
			if err != nil {
				return 0, err
			}
			b, err := get(hi)
			if err != nil {
				return 0, err
			}
			return (a + b) / 2, nil
		}
		return get(idx)
	case "inverted_cdf":
		j := int(math.Ceil(nf*t)) - 1
		return get(j)
	case "averaged_inverted_cdf":
		virt := nf * t
		j := int(math.Ceil(virt)) - 1
		if j >= n-1 {
			return get(n - 1)
		}
		if virt == math.Floor(virt) && j >= 0 {
			a, err := get(j)
			if err != nil {
				return 0, err
			}
			b, err := get(j + 1)
			if err != nil {
				return 0, err
			}
			return (a + b) / 2, nil
		}
		return get(j)
	case "closest_observation":
		// numpy:index = n·t - 1 - 0.5;离散规则为"取最近偶数序统计量"
		// (gamma==0 且 floor(index) 为奇数时取前一个)。
		idx := nf*t - 1 - 0.5
		prev := math.Floor(idx)
		gamma := idx - prev
		res := prev + 1
		if gamma == 0 && math.Mod(prev, 2) != 0 {
			res = prev
		}
		if res < 0 {
			res = 0
		}
		return get(int(res))
	}
	// 连续方法
	var virt float64
	switch method {
	case "", "linear":
		virt = (nf - 1) * t
	case "interpolated_inverted_cdf":
		// numpy:_compute_virtual_index(n, q, 0, 1) = n·t - 1
		virt = nf*t - 1
	case "hazen":
		virt = nf*t - 0.5
	case "weibull":
		virt = (nf+1)*t - 1
	case "median_unbiased":
		virt = (nf+1.0/3.0)*t + 1.0/3.0 - 1
	case "normal_unbiased":
		virt = (nf+1.0/4.0)*t + 3.0/8.0 - 1
	default:
		return 0, fmt.Errorf("num: percentile: 未知 method %q: %w", method, ErrValue)
	}
	if virt <= 0 {
		return get(0)
	}
	if virt >= nf-1 {
		return get(n - 1)
	}
	j := int(math.Floor(virt))
	gamma := virt - float64(j)
	a, err := get(j)
	if err != nil {
		return 0, err
	}
	if gamma == 0 {
		return a, nil
	}
	b, err := get(j + 1)
	if err != nil {
		return 0, err
	}
	return a + (b-a)*gamma, nil
}

// PercentileMethod ≡ np.percentile(a, q, method=...):单个 q + 指定 method。
func PercentileMethod(a *Array, q float64, method string, axis []int, keepdims bool) (*Array, error) {
	if q < 0 || q > 100 || math.IsNaN(q) {
		return nil, fmt.Errorf("num: percentile: q=%v 越界(0..100): %w", q, ErrValue)
	}
	if len(axis) == 0 {
		sorted, err := sortedCopy(a.AsContiguous().Flatten())
		if err != nil {
			return nil, err
		}
		v, err := quantileMethodOf(sorted, q, method)
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
		return nil, fmt.Errorf("num: percentile: %w", err)
	}
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
		return nil, fmt.Errorf("num: percentile: 归约轴长度为 0,分位数无定义(numpy 亦报错): %w", ErrShape)
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
		sorted, err := sortedCopy(sub)
		if err != nil {
			return nil, err
		}
		v, err := quantileMethodOf(sorted, q, method)
		if err != nil {
			return nil, err
		}
		vals[r] = v
	}
	return out, setFlatFloats(out, vals)
}

// sortedCopy 返回 1-D 数组的排序副本。
func sortedCopy(a *Array) (*Array, error) {
	c, err := transformCopy(a, a.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	if err := sortFlat(c); err != nil {
		return nil, err
	}
	return c, nil
}

// PercentileMany ≡ np.percentile(a, [q1,q2,...]):q 为数组,linear 方法。
//
// 输出形状:axis 省略 → (len(qs),);给 axis → (len(qs),) + 归约后形状(numpy 把 q 放最前)。
func PercentileMany(a *Array, qs []float64, axis []int, keepdims bool) (*Array, error) {
	for _, q := range qs {
		if q < 0 || q > 100 || math.IsNaN(q) {
			return nil, fmt.Errorf("num: percentile: q=%v 越界(0..100): %w", q, ErrValue)
		}
	}
	nq := len(qs)
	if nq == 0 {
		// 空 q 列表 → 空数组(numpy 返回 shape (0,))
		return New([]int{0}, DFloat64)
	}
	// 先按单 q 算出"归约后"的每个 q 的结果形状
	base := a
	if len(axis) > 0 {
		d, err := normalizeAxis(a, axis[0])
		if err != nil {
			return nil, fmt.Errorf("num: percentile: %w", err)
		}
		_ = d
	}
	// 逐个 q 计算,再堆到最前
	parts := make([]*Array, nq)
	for i, q := range qs {
		p, err := PercentileMethod(a, q, "linear", axis, keepdims)
		if err != nil {
			return nil, err
		}
		parts[i] = p
	}
	_ = base
	if nq == 1 {
		return parts[0], nil
	}
	// 结果形状:(nq,) + parts[0].Shape
	sh := append([]int{nq}, parts[0].Shape()...)
	out, err := New(sh, parts[0].Dtype())
	if err != nil {
		return nil, err
	}
	// 逐 q 写入
	for i, p := range parts {
		idx := make([]Indexer, len(sh))
		idx[0] = Slice(i, i+1)
		for k := 1; k < len(sh); k++ {
			idx[k] = All()
		}
		src, err := p.Reshape(sh[1:])
		if err != nil {
			return nil, err
		}
		if len(sh) == 1 {
			v, err := p.Item()
			if err != nil {
				return nil, err
			}
			if err := out.SetItem([]int{i}, v); err != nil {
				return nil, err
			}
			continue
		}
		if err := out.SetSub(src, idx...); err != nil {
			return nil, err
		}
	}
	return out, nil
}

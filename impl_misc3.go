package num

import (
	"fmt"
	"math"
)

// —— 补充工具(Step 34 后半):rot90 / average / ptp / logspace / geomspace /
//    nanprod / nancumsum / kron / slogdet ——

// Rot90 ≡ np.rot90(a, k, axes):把 ax1/ax2 两轴确定的平面逆时针旋转 k 次。
// numpy 默认 axes=(0,1);k 可为负(等价于 k mod 4)。
func Rot90(a *Array, k int, ax1, ax2 int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: rot90: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: rot90: 至少 2 维(得 %d): %w", a.NDim(), ErrShape)
	}
	d1, err := normalizeAxis(a, ax1)
	if err != nil {
		return nil, fmt.Errorf("num: rot90: %w", err)
	}
	d2, err := normalizeAxis(a, ax2)
	if err != nil {
		return nil, fmt.Errorf("num: rot90: %w", err)
	}
	if d1 == d2 {
		return nil, fmt.Errorf("num: rot90: 两轴相同(%d): %w", d1, ErrShape)
	}
	steps := ((k % 4) + 4) % 4
	cur := a
	for i := 0; i < steps; i++ {
		// 一次 90°:交换两轴(转置)后沿第一个轴翻转
		perm := make([]int, a.NDim())
		for j := range perm {
			perm[j] = j
		}
		perm[d1], perm[d2] = perm[d2], perm[d1]
		tr, err := cur.Transpose(perm...)
		if err != nil {
			return nil, err
		}
		fl, err := Flip(tr.AsContiguous(), []int{d1})
		if err != nil {
			return nil, err
		}
		cur = fl
	}
	if steps == 0 {
		return transformCopy(a, a.Shape(), a.Dtype())
	}
	return cur, nil
}

// Average ≡ np.average(a, axis, weights):
// weights 为 nil 时等于 Mean;1-D 权重按轴加权;与 a 同形的权重逐元素加权。
func Average(a *Array, axis []int, weights *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: average: 数组为 nil: %w", ErrDtype)
	}
	if weights == nil {
		return Mean(a, axis, false)
	}
	var d int
	if len(axis) > 0 {
		var err error
		if d, err = normalizeAxis(a, axis[0]); err != nil {
			return nil, fmt.Errorf("num: average: %w", err)
		}
	}
	w := weights
	if !shapeEqual(weights.Shape(), a.Shape()) {
		// 1-D 权重:长度须等于该轴长度,补 1 维后沿轴广播
		if weights.NDim() != 1 || weights.Size() != a.Shape()[d] {
			return nil, fmt.Errorf("num: average: 权重形状 %v 与 %v(轴 %d)不匹配: %w",
				weights.Shape(), a.Shape(), d, ErrShape)
		}
		wsh := make([]int, a.NDim())
		for i := range wsh {
			wsh[i] = 1
		}
		wsh[d] = weights.Size()
		r, err := weights.Reshape(wsh)
		if err != nil {
			return nil, err
		}
		w = r
	}
	prod, err := Multiply(a, w)
	if err != nil {
		return nil, err
	}
	num, err := Sum(prod, axis, false)
	if err != nil {
		return nil, err
	}
	den, err := Sum(w, axis, false)
	if err != nil {
		return nil, err
	}
	return Div(num, den)
}

// PtP ≡ np.ptp(peak-to-peak):max - min。
func PtP(a *Array, axis []int, keepdims bool) (*Array, error) {
	hi, err := Max(a, axis, keepdims)
	if err != nil {
		return nil, err
	}
	lo, err := Min(a, axis, keepdims)
	if err != nil {
		return nil, err
	}
	return Subtract(hi, lo)
}

// LogSpace ≡ np.logspace(start, stop, num, base):base**linspace(start, stop, num)。
func LogSpace(start, stop float64, num int, base float64) (*Array, error) {
	if num < 0 {
		return nil, fmt.Errorf("num: logspace: num 须非负(得 %d): %w", num, ErrValue)
	}
	if base <= 0 {
		base = 10
	}
	lin, err := LinSpace(start, stop, num, true)
	if err != nil {
		return nil, err
	}
	f, err := transformCopy(lin, lin.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	for i := range w {
		w[i] = math.Pow(base, w[i])
	}
	return f, nil
}

// GeomSpace ≡ np.geomspace(start, stop, num):等比数列(端点须同号且非零)。
func GeomSpace(start, stop float64, num int) (*Array, error) {
	if num < 0 {
		return nil, fmt.Errorf("num: geomspace: num 须非负(得 %d): %w", num, ErrValue)
	}
	if start == 0 || stop == 0 {
		return nil, fmt.Errorf("num: geomspace: 端点不能为 0: %w", ErrValue)
	}
	if (start < 0) != (stop < 0) {
		return nil, fmt.Errorf("num: geomspace: 端点须同号(%v, %v): %w", start, stop, ErrValue)
	}
	sign := 1.0
	if start < 0 {
		sign = -1
		start, stop = -start, -stop
	}
	lin, err := LinSpace(math.Log(start), math.Log(stop), num, true)
	if err != nil {
		return nil, err
	}
	f, err := transformCopy(lin, lin.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	for i := range w {
		w[i] = sign * math.Exp(w[i])
	}
	return f, nil
}

// NanProd ≡ np.nanprod:忽略 NaN 的连乘;**全 NaN 的线返回 1**(numpy 同)。
func NanProd(a *Array, axis []int, keepdims bool) (*Array, error) {
	return nanReduceF64(a, axis, keepdims, func(vals []float64) float64 {
		p := 1.0
		for _, v := range vals {
			p *= v
		}
		return p
	})
}

// nanReduceF64 跳过 NaN 的逐线归约骨架(全 NaN 的线交给 fn 的空切片语义)。
func nanReduceF64(a *Array, axis []int, keepdims bool, fn func(vals []float64) float64) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: 数组为 nil: %w", ErrDtype)
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	pick := func(row []float64) []float64 {
		vals := make([]float64, 0, len(row))
		for _, v := range row {
			if v == v {
				vals = append(vals, v)
			}
		}
		return vals
	}
	if len(axis) == 0 {
		v := fn(pick(win[float64](engineWindow(f))))
		var sh []int
		if keepdims {
			sh = make([]int, a.NDim())
			for i := range sh {
				sh[i] = 1
			}
		}
		out, err := New(sh, DFloat64)
		if err != nil {
			return nil, err
		}
		return out, setScalarFloat(out, v)
	}
	d, err := normalizeAxis(a, axis[0])
	if err != nil {
		return nil, fmt.Errorf("num: nan 归约: %w", err)
	}
	rows, m, n, err := axisRows(f, d)
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
	out, err := New(outSh, DFloat64)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, fmt.Errorf("num: nan 归约: 归约轴长度为 0: %w", ErrShape)
	}
	if m == 0 {
		return out, nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	ow := win[float64](engineWindow(out))
	row := make([]float64, n)
	for r := 0; r < m; r++ {
		for i := 0; i < n; i++ {
			row[i] = win[float64](sb[(r*n+i)*es : (r*n+i+1)*es])[0]
		}
		ow[r] = fn(pick(row))
	}
	return out, nil
}

// NanCumsum ≡ np.nancumsum:把 NaN 当 0 的前缀和。
func NanCumsum(a *Array, axis []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: nancumsum: 数组为 nil: %w", ErrDtype)
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	for i := range w {
		if w[i] != w[i] {
			w[i] = 0
		}
	}
	return Cumsum(f, axis)
}

// Kron ≡ np.kron:Kronecker 积(1-D×1-D → 1-D;其余按 2-D 处理)。
func Kron(a, b *Array) (*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: kron: 入参为 nil: %w", ErrDtype)
	}
	if a.NDim() > 2 || b.NDim() > 2 {
		return nil, fmt.Errorf("num: kron: 只支持 1-D/2-D(得 %d/%d 维): %w",
			a.NDim(), b.NDim(), ErrShape)
	}
	bothVec := a.NDim() == 1 && b.NDim() == 1
	am, bm := a, b
	if a.NDim() == 1 {
		r, err := a.Reshape([]int{1, a.Size()})
		if err != nil {
			return nil, err
		}
		am = r
	}
	if b.NDim() == 1 {
		r, err := b.Reshape([]int{1, b.Size()})
		if err != nil {
			return nil, err
		}
		bm = r
	}
	ar, ac := am.Shape()[0], am.Shape()[1]
	br, bc := bm.Shape()[0], bm.Shape()[1]
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, err
	}
	af, err := transformCopy(am, am.Shape(), dt)
	if err != nil {
		return nil, err
	}
	bf, err := transformCopy(bm, bm.Shape(), dt)
	if err != nil {
		return nil, err
	}
	outSh := []int{ar * br, ac * bc}
	if bothVec {
		outSh = []int{ar * br * ac * bc}
	}
	out, err := New(outSh, dt)
	if err != nil {
		return nil, err
	}
	switch dt.Kind {
	case Float64:
		kronTyped[float64](win[float64](engineWindow(af)), win[float64](engineWindow(bf)), win[float64](engineWindow(out)), ar, ac, br, bc)
	case Float32:
		kronTyped[float32](win[float32](engineWindow(af)), win[float32](engineWindow(bf)), win[float32](engineWindow(out)), ar, ac, br, bc)
	case Int8:
		kronTyped[int8](win[int8](engineWindow(af)), win[int8](engineWindow(bf)), win[int8](engineWindow(out)), ar, ac, br, bc)
	case Int16:
		kronTyped[int16](win[int16](engineWindow(af)), win[int16](engineWindow(bf)), win[int16](engineWindow(out)), ar, ac, br, bc)
	case Int32:
		kronTyped[int32](win[int32](engineWindow(af)), win[int32](engineWindow(bf)), win[int32](engineWindow(out)), ar, ac, br, bc)
	case Int64:
		kronTyped[int64](win[int64](engineWindow(af)), win[int64](engineWindow(bf)), win[int64](engineWindow(out)), ar, ac, br, bc)
	case Uint8:
		kronTyped[uint8](win[uint8](engineWindow(af)), win[uint8](engineWindow(bf)), win[uint8](engineWindow(out)), ar, ac, br, bc)
	case Uint16:
		kronTyped[uint16](win[uint16](engineWindow(af)), win[uint16](engineWindow(bf)), win[uint16](engineWindow(out)), ar, ac, br, bc)
	case Uint32:
		kronTyped[uint32](win[uint32](engineWindow(af)), win[uint32](engineWindow(bf)), win[uint32](engineWindow(out)), ar, ac, br, bc)
	case Uint64:
		kronTyped[uint64](win[uint64](engineWindow(af)), win[uint64](engineWindow(bf)), win[uint64](engineWindow(out)), ar, ac, br, bc)
	default:
		return nil, fmt.Errorf("num: kron: 不支持 %s: %w", dt, ErrDtype)
	}
	return out, nil
}

// kronTyped 计算 Kronecker 积的块扩展。
func kronTyped[T castableNum](ab, bb, ob []T, ar, ac, br, bc int) {
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			av := ab[i*ac+j]
			for k := 0; k < br; k++ {
				for l := 0; l < bc; l++ {
					ob[(i*br+k)*(ac*bc)+(j*bc+l)] = av * bb[k*bc+l]
				}
			}
		}
	}
}

// Slogdet ≡ np.linalg.slogdet:返回 (符号, log|det|);奇异矩阵为 (0, -Inf)。
func Slogdet(a *Array) (float64, float64, error) {
	n, flat, err := squareF64(a, "slogdet")
	if err != nil {
		return 0, 0, err
	}
	if n == 0 {
		return 1, 0, nil
	}
	lu, _, sign, ok := luDecompose(flat, n)
	if !ok {
		return 0, math.Inf(-1), nil
	}
	logAbs := 0.0
	s := sign
	for i := 0; i < n; i++ {
		d := lu[i*n+i]
		if d < 0 {
			s = -s
			d = -d
		}
		logAbs += math.Log(d)
	}
	return s, logAbs, nil
}

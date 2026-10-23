package num

import (
	"fmt"
	"math"
	"sort"
)

// —— 集合运算与数值小工具(Step 26):isin / intersect1d / union1d / setdiff1d /
// setxor1d / trapz / polyval ——
//
// 集合运算按 numpy 口径统一为:输入先拉平、取**排序去重**结果;
// **NaN 不参与匹配**(numpy:np.isin([nan],[nan]) → False,而 intersect1d 会把它丢掉),
// 这里显式剔除 NaN 以保持一致。非数值 dtype 报错(numpy 支持但本库不承诺)。

// setF64 把数组拉平成 float64 并剔除 NaN(numpy 的集合运算不认为 NaN 相等)。
func setF64(a *Array, op string) ([]float64, error) {
	if a == nil {
		return nil, fmt.Errorf("num: %s: 数组为 nil: %w", op, ErrDtype)
	}
	switch a.Dtype().Kind {
	case Float16, Float32, Float64, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Bool:
	default:
		return nil, fmt.Errorf("num: %s: 仅支持数值 dtype(得 %s): %w", op, a.Dtype(), ErrDtype)
	}
	flat := a.AsContiguous().Flatten()
	f, err := transformCopy(flat, flat.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	out := make([]float64, 0, len(w))
	for _, v := range w {
		if v == v { // 剔除 NaN
			out = append(out, v)
		}
	}
	return out, nil
}

// sortedUniqueF64 排序去重。
func sortedUniqueF64(v []float64) []float64 {
	if len(v) == 0 {
		return v
	}
	sort.Float64s(v)
	out := v[:1]
	for _, x := range v[1:] {
		if x != out[len(out)-1] {
			out = append(out, x)
		}
	}
	return out
}

// IsIn ≡ np.isin(element, test_elements):逐元素判断是否属于测试集合(输出同形 bool)。
func IsIn(element, test *Array) (*Array, error) {
	if element == nil || test == nil {
		return nil, fmt.Errorf("num: isin: 入参为 nil: %w", ErrDtype)
	}
	tv, err := setF64(test, "isin")
	if err != nil {
		return nil, err
	}
	set := make(map[float64]struct{}, len(tv))
	for _, v := range tv {
		set[v] = struct{}{}
	}
	eflat := element.AsContiguous().Flatten()
	ef, err := transformCopy(eflat, eflat.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	ew := win[float64](engineWindow(ef))
	out, err := New(element.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	ow := win[uint8](engineWindow(out))
	for i, v := range ew {
		if v != v { // NaN 永不匹配
			continue
		}
		if _, ok := set[v]; ok {
			ow[i] = 1
		}
	}
	return out, nil
}

// Intersect1D ≡ np.intersect1d:公共元素的排序去重结果。
func Intersect1D(a, b *Array) (*Array, error) {
	av, err := setF64(a, "intersect1d")
	if err != nil {
		return nil, err
	}
	bv, err := setF64(b, "intersect1d")
	if err != nil {
		return nil, err
	}
	set := make(map[float64]struct{}, len(bv))
	for _, v := range bv {
		set[v] = struct{}{}
	}
	both := make([]float64, 0, len(av))
	for _, v := range av {
		if _, ok := set[v]; ok {
			both = append(both, v)
		}
	}
	return f64ToArray(sortedUniqueF64(both), a.Dtype())
}

// Union1D ≡ np.union1d:并集的排序去重结果。
func Union1D(a, b *Array) (*Array, error) {
	av, err := setF64(a, "union1d")
	if err != nil {
		return nil, err
	}
	bv, err := setF64(b, "union1d")
	if err != nil {
		return nil, err
	}
	all := append(append([]float64{}, av...), bv...)
	return f64ToArray(sortedUniqueF64(all), a.Dtype())
}

// Setdiff1D ≡ np.setdiff1d:a 中有而 b 中没有的(排序去重)。
func Setdiff1D(a, b *Array) (*Array, error) {
	av, err := setF64(a, "setdiff1d")
	if err != nil {
		return nil, err
	}
	bv, err := setF64(b, "setdiff1d")
	if err != nil {
		return nil, err
	}
	set := make(map[float64]struct{}, len(bv))
	for _, v := range bv {
		set[v] = struct{}{}
	}
	only := make([]float64, 0, len(av))
	for _, v := range av {
		if _, ok := set[v]; !ok {
			only = append(only, v)
		}
	}
	return f64ToArray(sortedUniqueF64(only), a.Dtype())
}

// Setxor1D ≡ np.setxor1d:只出现在其中一个集合里的元素(排序去重)。
func Setxor1D(a, b *Array) (*Array, error) {
	av, err := setF64(a, "setxor1d")
	if err != nil {
		return nil, err
	}
	bv, err := setF64(b, "setxor1d")
	if err != nil {
		return nil, err
	}
	inA := make(map[float64]struct{}, len(av))
	for _, v := range av {
		inA[v] = struct{}{}
	}
	inB := make(map[float64]struct{}, len(bv))
	for _, v := range bv {
		inB[v] = struct{}{}
	}
	var out []float64
	for v := range inA {
		if _, ok := inB[v]; !ok {
			out = append(out, v)
		}
	}
	for v := range inB {
		if _, ok := inA[v]; !ok {
			out = append(out, v)
		}
	}
	return f64ToArray(sortedUniqueF64(out), a.Dtype())
}

// f64ToArray 把 float64 切片落成指定 dtype 的 1-D 数组。
func f64ToArray(vals []float64, dt Dtype) (*Array, error) {
	tmp, err := New([]int{len(vals)}, DFloat64)
	if err != nil {
		return nil, err
	}
	copy(win[float64](engineWindow(tmp)), vals)
	if dt == DFloat64 {
		return tmp, nil
	}
	return transformCopy(tmp, tmp.Shape(), dt)
}

// Trapz ≡ np.trapezoid(y, x=None, dx=1):梯形积分。
// x 为 nil 时按等距(步长 dx);否则用 x 的间距(要求与 y 同长)。
func Trapz(y *Array, x *Array, dx float64) (float64, error) {
	if y == nil {
		return 0, fmt.Errorf("num: trapz: y 为 nil: %w", ErrDtype)
	}
	flat, err := y.AsContiguous().Reshape([]int{y.Size()})
	if err != nil {
		return 0, err
	}
	yv, err := setF64KeepNaN(flat, "trapz")
	if err != nil {
		return 0, err
	}
	n := len(yv)
	if n == 0 {
		return 0, nil
	}
	if n == 1 {
		return 0, nil
	}
	if x == nil {
		if dx == 0 {
			return 0, fmt.Errorf("num: trapz: dx 不能为 0: %w", ErrValue)
		}
		s := 0.0
		for i := 0; i < n-1; i++ {
			s += (yv[i] + yv[i+1]) / 2 * dx
		}
		return s, nil
	}
	if x.Size() != n {
		return 0, fmt.Errorf("num: trapz: x 长度 %d ≠ y 长度 %d: %w", x.Size(), n, ErrShape)
	}
	xf, err := x.AsContiguous().Reshape([]int{x.Size()})
	if err != nil {
		return 0, err
	}
	xv, err := setF64KeepNaN(xf, "trapz")
	if err != nil {
		return 0, err
	}
	s := 0.0
	for i := 0; i < n-1; i++ {
		s += (yv[i] + yv[i+1]) / 2 * (xv[i+1] - xv[i])
	}
	return s, nil
}

// Trapezoid ≡ np.trapezoid(新名;与 Trapz 同义)。
func Trapezoid(y *Array, x *Array, dx float64) (float64, error) { return Trapz(y, x, dx) }

// setF64KeepNaN 拉平转 float64(**保留** NaN:积分需要按原值参与)。
func setF64KeepNaN(a *Array, op string) ([]float64, error) {
	switch a.Dtype().Kind {
	case Float16, Float32, Float64, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Bool:
	default:
		return nil, fmt.Errorf("num: %s: 仅支持数值 dtype(得 %s): %w", op, a.Dtype(), ErrDtype)
	}
	flat := a.AsContiguous().Flatten()
	f, err := transformCopy(flat, flat.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	out := make([]float64, len(w))
	copy(out, w)
	return out, nil
}

// Polyval ≡ np.polyval(p, x):按**降幂**系数求多项式值(Horner)。
func Polyval(p, x *Array) (*Array, error) {
	if p == nil || x == nil {
		return nil, fmt.Errorf("num: polyval: 入参为 nil: %w", ErrDtype)
	}
	if p.NDim() != 1 {
		return nil, fmt.Errorf("num: polyval: 系数须为 1-D(得 %d 维): %w", p.NDim(), ErrShape)
	}
	coef, err := setF64KeepNaN(p, "polyval")
	if err != nil {
		return nil, err
	}
	xflat := x.AsContiguous().Flatten()
	xf, err := transformCopy(xflat, xflat.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	xw := win[float64](engineWindow(xf))
	out, err := New(x.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	for i, xv := range xw {
		acc := 0.0
		for _, c := range coef {
			acc = acc*xv + c
		}
		ow[i] = acc
	}
	_ = math.NaN
	return out, nil
}

// Absolute ≡ np.absolute(即 np.abs 的别名)。
func Absolute(a *Array) (*Array, error) { return Abs(a) }

// Any ≡ np.any(即 AnyTrue 的别名)。
//
// 注意命名不对称:`All` 这个标识符在本库被**索引器** `All` 占用(切片用),
// 归约的"全真"请用 `AllTrue`;`Any` 没有冲突,故与 numpy 同名。
func Any(a *Array, axis []int, keepdims bool) (*Array, error) {
	return AnyTrue(a, axis, keepdims)
}

// Invert ≡ np.invert:整型/bool 按位取反(浮点在 numpy 里报错,这里同样报错)。
func Invert(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: invert: 数组为 nil: %w", ErrDtype)
	}
	switch a.Dtype().Kind {
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Bool:
	default:
		return nil, fmt.Errorf("num: invert: 不支持 %s(numpy 对浮点/复数同样报错): %w", a.Dtype(), ErrDtype)
	}
	src := a.AsContiguous()
	out, err := New(a.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	sb := engineWindow(src)
	db := engineWindow(out)
	es := a.Dtype().Itemsize
	if a.Dtype().Kind == Bool {
		// bool 是**逻辑取反**(1→0、0→1),不能按位取反再规整(0x01 → 0xFE)
		for i := 0; i < len(sb); i += es {
			if sb[i] != 0 {
				db[i] = 0
			} else {
				db[i] = 1
			}
		}
		return out, nil
	}
	for i := 0; i < len(sb); i += es {
		for k := 0; k < es; k++ {
			db[i+k] = ^sb[i+k]
		}
	}
	return out, nil
}

// Cross ≡ np.cross(a, b):3 维向量叉积,沿**最后一轴**,其它轴按广播。
// numpy 2.x 已不支持 2 维向量(这里同样只接受长度 3)。
func Cross(a, b *Array) (*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: cross: 入参为 nil: %w", ErrDtype)
	}
	if a.NDim() == 0 || b.NDim() == 0 {
		return nil, fmt.Errorf("num: cross: 不支持 0-d: %w", ErrShape)
	}
	if a.Shape()[a.NDim()-1] != 3 || b.Shape()[b.NDim()-1] != 3 {
		return nil, fmt.Errorf("num: cross: 最后一轴必须长度为 3(numpy 2.x 同): %w", ErrShape)
	}
	// 前导维度广播
	la := a.Shape()[:a.NDim()-1]
	lb := b.Shape()[:b.NDim()-1]
	lead, err := broadcastShapes(la, lb)
	if err != nil {
		return nil, fmt.Errorf("num: cross: 前导维不可广播: %w", err)
	}
	outShape := append(append([]int{}, lead...), 3)
	af, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	bf, err := transformCopy(b.AsContiguous(), b.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	out, err := New(outShape, DFloat64)
	if err != nil {
		return nil, err
	}
	nLead := prodInts(lead)
	for t := 0; t < nLead; t++ {
		ia, err := leadIndex(la, lead, t)
		if err != nil {
			return nil, err
		}
		ib, err := leadIndex(lb, lead, t)
		if err != nil {
			return nil, err
		}
		av, err := triple(af, ia)
		if err != nil {
			return nil, err
		}
		bv, err := triple(bf, ib)
		if err != nil {
			return nil, err
		}
		writeAt(out, t, []float64{
			av[1]*bv[2] - av[2]*bv[1],
			av[2]*bv[0] - av[0]*bv[2],
			av[0]*bv[1] - av[1]*bv[0],
		})
	}
	return out, nil
}

// leadIndex 把前导维的平坦下标映射到 src 的索引(长度不足的前导维视作广播)。
func leadIndex(srcShape, lead []int, flat int) ([]int, error) {
	out := make([]int, len(lead))
	rem := flat
	for i := len(lead) - 1; i >= 0; i-- {
		out[i] = rem % lead[i]
		rem /= lead[i]
	}
	// src 的前导维可能比 lead 短(左侧广播):取对应右侧对齐位置
	if len(srcShape) < len(lead) {
		out = out[len(lead)-len(srcShape):]
	}
	for i := range out {
		if srcShape[i] == 1 {
			out[i] = 0
		}
	}
	return out, nil
}

// triple 读取 src 在前导索引 idx 下的三个分量。
func triple(src *Array, idx []int) ([3]float64, error) {
	var out [3]float64
	for k := 0; k < 3; k++ {
		full := append(append([]int{}, idx...), k)
		v, err := src.Item(full...)
		if err != nil {
			return out, err
		}
		f, ok := v.(float64)
		if !ok {
			return out, fmt.Errorf("num: cross: 期望 float64 元素,得 %T: %w", v, ErrDtype)
		}
		out[k] = f
	}
	return out, nil
}

// writeAt 把三个分量写进 out 的第 t 组。
func writeAt(out *Array, t int, vals []float64) {
	w := win[float64](engineWindow(out))
	copy(w[t*3:(t+1)*3], vals)
}

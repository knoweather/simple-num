package num

import (
	"fmt"
	"math"
	"sort"
)

type sortedInts interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// —— 排序与统计(Step 6):sort / argsort / median / percentile / diff / unique ——
//
// 轴口径与 numpy 同:axis 省略(Nil)表示先拉平再算;单轴时沿该轴运算。
// 浮点排序把 **NaN 排在最后**(numpy 同),排序用稳定序(值相同时保持原序)。

// Sort ≡ np.sort(a, axis=None/单轴):返回排序后的新数组(默认浮点 NaN 置尾)。
func Sort(a *Array, axis []int) (*Array, error) {
	if len(axis) == 0 {
		flat := a.AsContiguous().Flatten()
		out, err := transformCopy(flat, flat.Shape(), flat.Dtype())
		if err != nil {
			return nil, err
		}
		if err := sortFlat(out); err != nil {
			return nil, err
		}
		return out, nil
	}
	rows, m, n, err := axisRows(a, axis[0])
	if err != nil {
		return nil, err
	}
	// 关键:末轴时 axisRows 的 AsContiguous 不拷贝、rows 直接共享调用方缓冲,
	// 必须先取独立副本再排序(numpy 的 sort 返回新数组,绝不改入参)。
	work, err := transformCopy(rows, rows.Shape(), rows.Dtype())
	if err != nil {
		return nil, err
	}
	if err := sortRowsInPlace(work, m, n); err != nil {
		return nil, err
	}
	return axisRowsBack(work, a, axis[0], m, n)
}

// sortRowsInPlace 对 (M×n) 连续数组**逐行**排序。
func sortRowsInPlace(rows *Array, m, n int) error {
	if m == 0 || n <= 1 {
		return nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	for r := 0; r < m; r++ {
		sub := &Array{
			buf:   sb[r*n*es : (r+1)*n*es],
			shape: []int{n},
			st:    cStrides([]int{n}, es),
			dtype: rows.Dtype(),
		}
		if err := sortFlat(sub); err != nil {
			return err
		}
	}
	return nil
}

// Argsort ≡ np.argsort(a, axis=None/单轴):返回 int64 下标数组(形状同输入)。
func Argsort(a *Array, axis []int) (*Array, error) {
	if len(axis) == 0 {
		flat := a.AsContiguous().Flatten()
		idx, err := argsortFlat(flat)
		if err != nil {
			return nil, err
		}
		return idx, nil
	}
	rows, m, n, err := axisRows(a, axis[0])
	if err != nil {
		return nil, err
	}
	idx, err := argsortRows(rows, m, n)
	if err != nil {
		return nil, err
	}
	// 下标数组与输入同形状:借助 axisRowsBack 的对称变换
	return axisRowsBack(idx, a, axis[0], m, n)
}

// Median ≡ np.median:中位数(偶数个取中间两数均值,输出浮点)。
func Median(a *Array, axis []int, keepdims bool) (*Array, error) {
	return quantileLike(a, axis, keepdims, 50, true)
}

// Percentile ≡ np.percentile(a, q):线性插值分位数,0 ≤ q ≤ 100。
func Percentile(a *Array, q float64, axis []int, keepdims bool) (*Array, error) {
	if q < 0 || q > 100 || math.IsNaN(q) {
		return nil, fmt.Errorf("num: percentile: q=%v 越界(0..100): %w", q, ErrValue)
	}
	return quantileLike(a, axis, keepdims, q, false)
}

// Diff ≡ np.diff(a, n, axis):沿轴 n 阶差分(n 默认 1)。
func Diff(a *Array, n int, axis []int) (*Array, error) {
	if n < 0 {
		return nil, fmt.Errorf("num: diff: n 不能为负: %w", ErrValue)
	}
	if n == 0 {
		return transformCopy(a, a.Shape(), a.Dtype())
	}
	ax := -1
	if len(axis) > 0 {
		d, err := normalizeAxis(a, axis[0])
		if err != nil {
			return nil, fmt.Errorf("num: diff: %w", err)
		}
		ax = d
	} else if a.NDim() > 0 {
		ax = a.NDim() - 1 // numpy 默认最后一轴
	}
	cur := a
	for k := 0; k < n; k++ {
		next, err := diffOnce(cur, ax)
		if err != nil {
			return nil, err
		}
		cur = next
	}
	return cur, nil
}

// Unique ≡ np.unique(a):拉平后排序去重(返回 1-D,与 numpy 默认一致)。
func Unique(a *Array) (*Array, error) {
	flat := a.AsContiguous().Flatten()
	sorted, err := transformCopy(flat, flat.Shape(), flat.Dtype())
	if err != nil {
		return nil, err
	}
	if err := sortFlat(sorted); err != nil {
		return nil, err
	}
	n := sorted.Size()
	keep := make([]int, 0, n)
	es := sorted.Dtype().Itemsize
	buf := engineWindow(sorted)
	for i := 0; i < n; i++ {
		if i == 0 || !sameElemBytes(buf[(i-1)*es:i*es], buf[i*es:(i+1)*es]) {
			keep = append(keep, i)
		}
	}
	out, err := New([]int{len(keep)}, sorted.Dtype())
	if err != nil {
		return nil, err
	}
	db := engineWindow(out)
	for k, i := range keep {
		copy(db[k*es:(k+1)*es], buf[i*es:(i+1)*es])
	}
	return out, nil
}

// sameElemBytes 按位比较一个元素(浮点按位 → NaN 与 -0.0 各自视为不同值,与 numpy
// 的 unique 对 NaN 的处理一致:numpy 会把多个 NaN 收敛成一个,这里按位区分,
// 属受控差异,已在文档注明)。
func sameElemBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// —— 内部工具 ——

// axisPerm 返回"把 axis 轴移到末尾"的置换,以及其逆置换。
func axisPerm(nd, d int) (perm, inv []int) {
	perm = make([]int, 0, nd)
	for i := 0; i < nd; i++ {
		if i != d {
			perm = append(perm, i)
		}
	}
	perm = append(perm, d)
	inv = make([]int, nd)
	for i, p := range perm {
		inv[p] = i
	}
	return perm, inv
}

// axisRows 把 a 的 axis 轴移到末尾并连续化,返回 (M×n 的 2-D 视图, M, n)。
func axisRows(a *Array, axis int) (*Array, int, int, error) {
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, 0, 0, err
	}
	perm, _ := axisPerm(a.NDim(), d)
	tr, err := a.Transpose(perm...)
	if err != nil {
		return nil, 0, 0, err
	}
	cont := tr.AsContiguous()
	n := a.Shape()[d]
	// 归约轴长度为 0 时不能做 cont.Size/n(会除零):直接算出"线数" m。
	m := 1
	for i, s := range a.Shape() {
		if i != d {
			m *= s
		}
	}
	if n == 0 {
		// Reshape 不接受 0 维,直接返回连续数组(m 为线数,n=0 由调用方处理)。
		return cont, m, 0, nil
	}
	if m == 0 {
		return cont, 0, n, nil
	}
	rows, err := cont.Reshape([]int{m, n})
	if err != nil {
		return nil, 0, 0, err
	}
	return rows, m, n, nil
}

// axisRowsBack 把 (M×n) 的行视图变回原形状:先按**置换后**的形状 reshape,
// 再用逆置换 transpose 回原轴序。
func axisRowsBack(rows *Array, ref *Array, axis int, m, n int) (*Array, error) {
	d, err := normalizeAxis(ref, axis)
	if err != nil {
		return nil, err
	}
	perm, inv := axisPerm(ref.NDim(), d)
	permShape := make([]int, len(perm))
	for i, p := range perm {
		permShape[i] = ref.Shape()[p]
	}
	tmp, err := rows.Reshape(permShape)
	if err != nil {
		return nil, err
	}
	return tmp.Transpose(inv...)
}

// sortFlat 就地把连续数组排序(NaN 置尾,稳定)。
func sortFlat(a *Array) error {
	n := a.Size()
	if n <= 1 {
		return nil
	}
	b := engineWindow(a)
	switch a.Dtype().Kind {
	case Bool:
		w := win[uint8](b)
		s := make([]uint8, n)
		copy(s, w)
		sort.SliceStable(s, func(i, j int) bool { return s[i] < s[j] })
		copy(w, s)
	case Int8:
		sortNum(win[int8](b))
	case Int16:
		sortNum(win[int16](b))
	case Int32:
		sortNum(win[int32](b))
	case Int64:
		sortNum(win[int64](b))
	case Uint8:
		sortNum(win[uint8](b))
	case Uint16:
		sortNum(win[uint16](b))
	case Uint32:
		sortNum(win[uint32](b))
	case Uint64:
		sortNum(win[uint64](b))
	case Float32:
		sortFloat(win[float32](b))
	case Float64:
		sortFloat(win[float64](b))
	case Float16:
		w := win[uint16](b)
		f := make([]float32, n)
		for i, v := range w {
			f[i] = f16ToF32(v)
		}
		sortFloat(f)
		for i, v := range f {
			w[i] = f32ToF16(v)
		}
	default:
		return fmt.Errorf("num: sort 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return nil
}

// sortNum 有序类型排序(泛型约束内的 < 可用)。
func sortNum[T sortedInts](w []T) {
	sort.SliceStable(w, func(i, j int) bool { return w[i] < w[j] })
}

// sortFloat 浮点排序:NaN 排在最后(numpy 同)。
func sortFloat[T float32 | float64](w []T) {
	sort.SliceStable(w, func(i, j int) bool {
		a, b := w[i], w[j]
		if a != a { // a 是 NaN:只有 b 非 NaN 时才"大于"
			return false
		}
		if b != b {
			return true
		}
		return a < b
	})
}

// argsortFlat 返回按值排序后的下标(int64,升序;NaN 置尾)。
func argsortFlat(a *Array) (*Array, error) {
	n := a.Size()
	out, err := New([]int{n}, DInt64)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return out, nil
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	b := engineWindow(a)
	switch a.Dtype().Kind {
	case Bool:
		w := win[uint8](b)
		sort.SliceStable(idx, func(i, j int) bool { return w[idx[i]] < w[idx[j]] })
	case Int8:
		argSortNum(win[int8](b), idx)
	case Int16:
		argSortNum(win[int16](b), idx)
	case Int32:
		argSortNum(win[int32](b), idx)
	case Int64:
		argSortNum(win[int64](b), idx)
	case Uint8:
		argSortNum(win[uint8](b), idx)
	case Uint16:
		argSortNum(win[uint16](b), idx)
	case Uint32:
		argSortNum(win[uint32](b), idx)
	case Uint64:
		argSortNum(win[uint64](b), idx)
	case Float32:
		argSortFloat(win[float32](b), idx)
	case Float64:
		argSortFloat(win[float64](b), idx)
	case Float16:
		w := win[uint16](b)
		f := make([]float32, n)
		for i, v := range w {
			f[i] = f16ToF32(v)
		}
		argSortFloat(f, idx)
	default:
		return nil, fmt.Errorf("num: argsort 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	dw := win[int64](engineWindow(out))
	for i, k := range idx {
		dw[i] = int64(k)
	}
	return out, nil
}

func argSortNum[T sortedInts](w []T, idx []int) {
	sort.SliceStable(idx, func(i, j int) bool { return w[idx[i]] < w[idx[j]] })
}

func argSortFloat[T float32 | float64](w []T, idx []int) {
	sort.SliceStable(idx, func(i, j int) bool {
		a, b := w[idx[i]], w[idx[j]]
		if a != a {
			return false
		}
		if b != b {
			return true
		}
		return a < b
	})
}

// argsortRows 对 (M×n) 每行分别 argsort,返回同形状 int64。
func argsortRows(rows *Array, m, n int) (*Array, error) {
	out, err := New([]int{m, n}, DInt64)
	if err != nil {
		return nil, err
	}
	if m == 0 || n == 0 {
		return out, nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	dw := win[int64](engineWindow(out))
	for r := 0; r < m; r++ {
		sub := &Array{buf: sb[r*n*es : (r+1)*n*es], shape: []int{n}, st: cStrides([]int{n}, es), dtype: rows.Dtype()}
		sorted, err := argsortFlat(sub)
		if err != nil {
			return nil, err
		}
		copy(dw[r*n:(r+1)*n], win[int64](engineWindow(sorted)))
	}
	return out, nil
}

// quantileLike 中位数/分位数共用内核:排序后线性插值(numpy 的 _lerp 口径)。
func quantileLike(a *Array, axis []int, keepdims bool, q float64, isMedian bool) (*Array, error) {
	// 输出 dtype:浮点(numpy:median/percentile 输出 float64,输入 f32 时保持 f32)
	outDT := DFloat64
	if a.Dtype().Kind == Float32 {
		outDT = DFloat32
	}
	_ = isMedian
	if len(axis) == 0 {
		flat := a.AsContiguous().Flatten()
		v, err := quantileOf(flat, q)
		if err != nil {
			return nil, err
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
		return nil, fmt.Errorf("num: quantile: %w", err)
	}
	rows, m, n, err := axisRows(a, d)
	if err != nil {
		return nil, err
	}
	sh := make([]int, a.NDim()-1)
	k := 0
	for i := 0; i < a.NDim(); i++ {
		if i == d {
			continue
		}
		sh[k] = a.Shape()[i]
		k++
	}
	outSh := sh
	if keepdims {
		outSh = append([]int(nil), a.Shape()...)
		outSh[d] = 1
	}
	out, err := New(outSh, outDT)
	if err != nil {
		return nil, err
	}
	if m == 0 || n == 0 {
		return out, nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	vals := make([]float64, m)
	for r := 0; r < m; r++ {
		sub := &Array{buf: sb[r*n*es : (r+1)*n*es], shape: []int{n}, st: cStrides([]int{n}, es), dtype: rows.Dtype()}
		v, err := quantileOf(sub, q)
		if err != nil {
			return nil, err
		}
		vals[r] = v
	}
	return out, setFlatFloats(out, vals)
}

// quantileOf 对 1-D 数组求 q 分位(numpy linear 插值)。
func quantileOf(a *Array, q float64) (float64, error) {
	n := a.Size()
	if n == 0 {
		return math.NaN(), nil
	}
	sorted, err := transformCopy(a, a.Shape(), a.Dtype())
	if err != nil {
		return 0, err
	}
	if err := sortFlat(sorted); err != nil {
		return 0, err
	}
	flat := sorted.Flatten()
	if q == 50 && n%2 == 0 {
		lo, err := numToFloat(flat, n/2-1)
		if err != nil {
			return 0, err
		}
		hi, err := numToFloat(flat, n/2)
		if err != nil {
			return 0, err
		}
		return (lo + hi) / 2, nil
	}
	virtual := (float64(n) - 1) * q / 100
	lo := int(math.Floor(virtual))
	hi := lo + 1
	gamma := virtual - float64(lo)
	a0, err := numToFloat(flat, lo)
	if err != nil {
		return 0, err
	}
	if hi >= n || gamma == 0 {
		return a0, nil
	}
	a1, err := numToFloat(flat, hi)
	if err != nil {
		return 0, err
	}
	// numpy: a0 + (a1-a0)*gamma(当 gamma==0 直接返回 a0)
	return a0 + (a1-a0)*gamma, nil
}

// numToFloat 取 1-D 数组某元素并转 float64。
func numToFloat(a *Array, i int) (float64, error) {
	v, err := a.Item(i)
	if err != nil {
		return 0, err
	}
	return toFloat64(v)
}

// toFloat64 把任意数值标量转 float64。
func toFloat64(v any) (float64, error) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	case int8:
		return float64(x), nil
	case int16:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case int64:
		return float64(x), nil
	case uint8:
		return float64(x), nil
	case uint16:
		return float64(x), nil
	case uint32:
		return float64(x), nil
	case uint64:
		return float64(x), nil
	case float32:
		return float64(x), nil
	case float64:
		return x, nil
	default:
		return 0, fmt.Errorf("num: 无法转 float64: %T: %w", v, ErrDtype)
	}
}

// setScalarFloat 写入单元素结果(支持 0-d 与 keepdims 的 (1,...,1) 形状)。
func setScalarFloat(out *Array, v float64) error {
	if out.Size() == 0 {
		return nil
	}
	es := out.Dtype().Itemsize
	buf := engineWindow(out)
	if len(buf) < es {
		return fmt.Errorf("num: 标量写入缓冲不足: %w", ErrShape)
	}
	return writeScalarToBytes(buf[:es], out.Dtype(), v)
}

// setFlatFloats 按平坦顺序写入浮点值(直接写引擎窗,out 必为连续)。
func setFlatFloats(out *Array, vals []float64) error {
	if out.Dtype().Kind == Float32 {
		w := win[float32](engineWindow(out))
		for i, v := range vals {
			w[i] = float32(v)
		}
		return nil
	}
	w := win[float64](engineWindow(out))
	copy(w, vals)
	return nil
}

func pickFloat(dt Dtype, v float64) any {
	if dt.Kind == Float32 {
		return float32(v)
	}
	return v
}

// diffOnce 沿轴做一次差分。
func diffOnce(a *Array, ax int) (*Array, error) {
	if a.NDim() == 0 || a.Size() == 0 {
		return a, nil
	}
	d, err := normalizeAxis(a, ax)
	if err != nil {
		return nil, err
	}
	sh := append([]int(nil), a.Shape()...)
	n := sh[d]
	if n < 2 {
		sh[d] = 0
		return New(sh, a.Dtype())
	}
	sh[d]--
	nd := a.NDim()
	hiIdx := make([]Indexer, nd)
	loIdx := make([]Indexer, nd)
	for i := 0; i < nd; i++ {
		if i == d {
			hiIdx[i] = Slice(1, n)
			loIdx[i] = Slice(0, n-1)
		} else {
			hiIdx[i] = All()
			loIdx[i] = All()
		}
	}
	hi, err := a.Sub(hiIdx...)
	if err != nil {
		return nil, err
	}
	lo, err := a.Sub(loIdx...)
	if err != nil {
		return nil, err
	}
	return Subtract(hi, lo)
}

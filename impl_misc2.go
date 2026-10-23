package num

import (
	"fmt"
	"math"
)

// —— 杂项算法(Step 8b):pad / histogram / select / putmask ——

// Pad ≡ np.pad(a, pad_width, mode, constant_values=cval)。
//
// widths 支持三种写法(与 numpy 一致):
// - 空/nil:所有轴前后各补 0(等价拷贝);
// - 长度 2 的单组 {before, after}:作用于所有轴;
// - 每轴一组 {before, after}(长度 = ndim)。
//
// mode 支持 "constant"(默认)、"edge"、"reflect"、"symmetric"、"wrap"。
func Pad(a *Array, widths [][2]int, mode string, cval float64) (*Array, error) {
	if mode == "" {
		mode = "constant"
	}
	nd := a.NDim()
	if nd == 0 {
		return nil, fmt.Errorf("num: pad: 0-d 输入不支持: %w", ErrShape)
	}
	pw, err := normalizePadWidths(widths, nd)
	if err != nil {
		return nil, err
	}
	outSh := make([]int, nd)
	for d := 0; d < nd; d++ {
		outSh[d] = a.Shape()[d] + pw[d][0] + pw[d][1]
		if outSh[d] < 0 {
			return nil, fmt.Errorf("num: pad: 轴 %d 补边后长度为负: %w", d, ErrShape)
		}
	}
	out, err := New(outSh, a.Dtype())
	if err != nil {
		return nil, err
	}
	// constant 模式:整块填常量后只写内部区域;其余模式:逐元素索引映射。
	switch mode {
	case "constant":
		if err := fillConstant(out, cval); err != nil {
			return nil, err
		}
		return out, copyPadInterior(out, a, pw)
	case "edge", "reflect", "symmetric", "wrap":
		return remapCopy(a, outSh, func(idx []int) []int {
			src := make([]int, nd)
			for d := 0; d < nd; d++ {
				i := idx[d] - pw[d][0]
				n := a.Shape()[d]
				src[d] = padMapIndex(i, n, mode)
			}
			return src
		})
	default:
		return nil, fmt.Errorf("num: pad: 不支持的 mode %q: %w", mode, ErrValue)
	}
}

// normalizePadWidths 归一 pad 宽度写法。
func normalizePadWidths(widths [][2]int, nd int) ([][2]int, error) {
	pw := make([][2]int, nd)
	switch len(widths) {
	case 0:
		return pw, nil
	case 1:
		if widths[0][0] < 0 || widths[0][1] < 0 {
			return nil, fmt.Errorf("num: pad: 补边宽度不能为负: %w", ErrShape)
		}
		for d := 0; d < nd; d++ {
			pw[d] = widths[0]
		}
		return pw, nil
	}
	if len(widths) != nd {
		return nil, fmt.Errorf("num: pad: widths 长度 %d ≠ ndim %d: %w", len(widths), nd, ErrShape)
	}
	for d := 0; d < nd; d++ {
		if widths[d][0] < 0 || widths[d][1] < 0 {
			return nil, fmt.Errorf("num: pad: 轴 %d 补边宽度为负: %w", d, ErrShape)
		}
		pw[d] = widths[d]
	}
	return pw, nil
}

// fillConstant 按 cval 填满输出(用 0-d 转码保持 dtype 语义)。
func fillConstant(out *Array, cval float64) error {
	tmp, err := New([]int{}, DFloat64)
	if err != nil {
		return err
	}
	if err := tmp.SetItem([]int{}, cval); err != nil {
		return err
	}
	conv, err := transformCopy(tmp, []int{}, out.Dtype())
	if err != nil {
		return err
	}
	v, err := conv.Item()
	if err != nil {
		return err
	}
	es := out.Dtype().Itemsize
	buf := engineWindow(out)
	one, err := New([]int{1}, out.Dtype())
	if err != nil {
		return err
	}
	if err := one.SetItem([]int{0}, v); err != nil {
		return err
	}
	ob := engineWindow(one)
	for i := 0; i < out.Size(); i++ {
		copy(buf[i*es:(i+1)*es], ob[:es])
	}
	return nil
}

// copyPadInterior 把源数组内容搬到输出内部的对应位置。
func copyPadInterior(out, a *Array, pw [][2]int) error {
	nd := a.NDim()
	es := a.Dtype().Itemsize
	src := a.AsContiguous()
	sb := engineWindow(src)
	db := engineWindow(out)
	sh := a.Shape()
	outSh := out.Shape()
	// 输出内部区域的起止
	start := make([]int, nd)
	for d := 0; d < nd; d++ {
		start[d] = pw[d][0]
	}
	idx := make([]int, nd)
	// 源步长(元素)
	st := make([]int, nd)
	acc := 1
	for d := nd - 1; d >= 0; d-- {
		st[d] = acc
		acc *= sh[d]
	}
	// 输出步长
	ost := make([]int, nd)
	acc = 1
	for d := nd - 1; d >= 0; d-- {
		ost[d] = acc
		acc *= outSh[d]
	}
	size := a.Size()
	for fi := 0; fi < size; fi++ {
		rem := fi
		si := 0
		oi := 0
		for d := nd - 1; d >= 0; d-- {
			idx[d] = rem % sh[d]
			rem /= sh[d]
			si += idx[d] * st[d]
			oi += (idx[d] + start[d]) * ost[d]
		}
		copy(db[oi*es:(oi+1)*es], sb[si*es:(si+1)*es])
	}
	return nil
}

// padMapIndex 把(可能越界的)源下标按模式映射回 [0,n) 内。
func padMapIndex(i, n int, mode string) int {
	if n == 1 {
		return 0
	}
	switch mode {
	case "edge":
		if i < 0 {
			return 0
		}
		if i >= n {
			return n - 1
		}
		return i
	case "reflect": // 不重复边缘:... 1 0 | 0 1 2 | 2 1...
		period := 2 * (n - 1)
		m := ((i % period) + period) % period
		if m >= n {
			m = period - m
		}
		return m
	case "symmetric": // 重复边缘:... 1 0 | 0 1 2 | 2 1...
		period := 2 * n
		m := ((i % period) + period) % period
		if m >= n {
			m = period - 1 - m
		}
		return m
	case "wrap":
		m := ((i % n) + n) % n
		return m
	}
	return i
}

// Histogram ≡ np.histogram(a, bins, range):返回 (hist int64, edges float64)。
//
// bins 为整数时在 [lo, hi] 上等分(bins+1 个边界);numpy 的口径是右开左闭,
// 恰好等于 lo 的落进第一箱、恰好等于 hi 的落进最后一箱。
func Histogram(a *Array, bins int, lo, hi float64) ([]*Array, error) {
	if bins <= 0 {
		return nil, fmt.Errorf("num: histogram: bins 必须为正,得 %d: %w", bins, ErrValue)
	}
	flat, err := transformCopy(a.AsContiguous().Flatten(), []int{a.Size()}, DFloat64)
	if err != nil {
		return nil, err
	}
	v := win[float64](engineWindow(flat))
	// 自动范围
	if math.IsNaN(lo) || math.IsNaN(hi) {
		l, h, ok := minMaxF64(v)
		if !ok { // 空数组:numpy 给 [0,1]
			l, h = 0, 1
		}
		lo, hi = l, h
	}
	if hi < lo {
		return nil, fmt.Errorf("num: histogram: hi %v < lo %v: %w", hi, lo, ErrValue)
	}
	edgesArr, err := New([]int{bins + 1}, DFloat64)
	if err != nil {
		return nil, err
	}
	ew := win[float64](engineWindow(edgesArr))
	step := (hi - lo) / float64(bins)
	for i := 0; i <= bins; i++ {
		ew[i] = lo + float64(i)*step
	}
	ew[bins] = hi // 末边界精确置 hi(numpy 同)
	hist, err := New([]int{bins}, DInt64)
	if err != nil {
		return nil, err
	}
	hw := win[int64](engineWindow(hist))
	if hi == lo {
		// 退化:全部落第 0 箱(numpy 对常数数组的行为)
		hw[0] = int64(len(v))
		return []*Array{hist, edgesArr}, nil
	}
	sc := float64(bins) / (hi - lo)
	for _, x := range v {
		if x != x || x < lo || x > hi {
			continue
		}
		b := int((x - lo) * sc)
		if b >= bins {
			b = bins - 1 // 恰为 hi
		}
		if b < 0 {
			b = 0
		}
		hw[b]++
	}
	return []*Array{hist, edgesArr}, nil
}

// minMaxF64 返回切片的最小/最大值(忽略 NaN);空切片 ok=false。
func minMaxF64(v []float64) (float64, float64, bool) {
	lo, hi := math.Inf(1), math.Inf(-1)
	found := false
	for _, x := range v {
		if x != x {
			continue
		}
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
		found = true
	}
	return lo, hi, found
}

// Select ≡ np.select(condlist, choicelist, default):首个为真的条件胜出。
//
// conds 与 choices 等长;全部按 numpy 右对齐广播;结果 dtype = promote(choices, default)。
func Select(conds, choices []*Array, def *Array) (*Array, error) {
	if len(conds) != len(choices) || len(conds) == 0 {
		return nil, fmt.Errorf("num: select: conds 与 choices 需等长且非空: %w", ErrShape)
	}
	// 广播形状:conds 与 choices、default 全部参与
	sh := conds[0].Shape()
	var err error
	for _, a := range conds[1:] {
		if sh, err = broadcastShapes(sh, a.Shape()); err != nil {
			return nil, fmt.Errorf("num: select: cond 形状不可广播: %w", err)
		}
	}
	for _, a := range choices {
		if sh, err = broadcastShapes(sh, a.Shape()); err != nil {
			return nil, fmt.Errorf("num: select: choice 形状不可广播: %w", err)
		}
	}
	if def != nil {
		if sh, err = broadcastShapes(sh, def.Shape()); err != nil {
			return nil, fmt.Errorf("num: select: default 形状不可广播: %w", err)
		}
	}
	dt := choices[0].Dtype()
	for _, a := range choices[1:] {
		if dt, err = promoteTypes(dt, a.Dtype()); err != nil {
			return nil, fmt.Errorf("num: select: choice dtype 不可提升: %w", err)
		}
	}
	var defN *Array
	if def != nil {
		if dt, err = promoteTypes(dt, def.Dtype()); err != nil {
			return nil, fmt.Errorf("num: select: default dtype 不可提升: %w", err)
		}
		if defN, err = normalizeOperand(def, sh, dt); err != nil {
			return nil, err
		}
	}
	out, err := New(sh, dt)
	if err != nil {
		return nil, err
	}
	// 先填 default(或零)
	if defN != nil {
		es := dt.Itemsize
		copy(engineWindow(out), engineWindow(defN)[:out.Size()*es])
	}
	chosen := make([]bool, out.Size())
	for k := range conds {
		mask, err := truthyMask2(conds[k], sh)
		if err != nil {
			return nil, err
		}
		ch, err := normalizeOperand(choices[k], sh, dt)
		if err != nil {
			return nil, err
		}
		if err := selectInto(out, ch, mask, chosen); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// truthyMask2 广播 + 真值化。
func truthyMask2(a *Array, sh []int) ([]bool, error) {
	na, err := normalizeOperand(a, sh, a.Dtype())
	if err != nil {
		return nil, err
	}
	return truthyMask(na)
}

// selectInto 把 ch 中 mask 为真且尚未被选中的元素写进 out。
func selectInto(out, ch *Array, mask, chosen []bool) error {
	es := out.Dtype().Itemsize
	db := engineWindow(out)
	sb := engineWindow(ch)
	for i, m := range mask {
		if m && !chosen[i] {
			chosen[i] = true
			copy(db[i*es:(i+1)*es], sb[i*es:(i+1)*es])
		}
	}
	return nil
}

// PutMask ≡ np.putmask(a, mask, values):把 mask 为真的位置改成 values(循环取用)。
func PutMask(a, mask, values *Array) error {
	sh := a.Shape()
	nm, err := normalizeOperand(mask, sh, mask.Dtype())
	if err != nil {
		return fmt.Errorf("num: putmask: mask 归一失败: %w", err)
	}
	m, err := truthyMask(nm)
	if err != nil {
		return err
	}
	if values.Size() == 0 {
		return nil
	}
	nv, err := normalizeOperand(values.AsContiguous().Flatten(), []int{values.Size()}, a.Dtype())
	if err != nil {
		return fmt.Errorf("num: putmask: values 归一失败: %w", err)
	}
	es := a.Dtype().Itemsize
	// 取用规则与 numpy 一致:**按下标** i % len(values)(不是按被命中次数)。
	if a.fastWindow() == nil {
		// 非连续视图:逐元素 SetItem 回退(保证写回真实内存,不落到临时副本)
		flat := nv.Flatten()
		nvLen := values.Size()
		for i, t := range m {
			if !t {
				continue
			}
			v, err := flat.Item(i % nvLen)
			if err != nil {
				return err
			}
			idx := flatToIdx(a.Shape(), i)
			if err := a.SetItem(idx, v); err != nil {
				return err
			}
		}
		return nil
	}
	db := engineWindow(a)
	vb := engineWindow(nv)
	nvLen := values.Size()
	for i, t := range m {
		if !t {
			continue
		}
		j := i % nvLen
		copy(db[i*es:(i+1)*es], vb[j*es:(j+1)*es])
	}
	return nil
}

// flatToIdx 把平坦下标拆成多维下标。
func flatToIdx(sh []int, flat int) []int {
	idx := make([]int, len(sh))
	rem := flat
	for d := len(sh) - 1; d >= 0; d-- {
		if sh[d] == 0 {
			idx[d] = 0
			continue
		}
		idx[d] = rem % sh[d]
		rem /= sh[d]
	}
	return idx
}

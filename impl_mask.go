package num

// :掩码系(B 拥有):compress / nonzero / count_nonzero / bincount。
//
// 语义基准(numpy 2.5.4 实测,):
// - Compress(平铺,≡ np.compress(axis=None)):条件须 1-D;非 bool 按 != 0
// 判真;条件长于数据时只作用于前 min(len(cond), size) 个;输出 1-D。
// - CompressAxis:沿轴压缩;条件长于轴长 → ErrIndex(numpy IndexError);
// 短则只作用于前 len(cond) 个位置;输出形状 = 源形状(轴维 → 保留数)。
// - Nonzero:返回每维一个 i64 索引数组(numpy 返回 tuple);NaN 视为非零;
// 0-d → ErrShape(numpy ValueError);非数值 dtype → ErrDtype。
// - CountNonzero:非零计数(NaN 计入)。
// - Bincount:x 须整型 1-D(浮点/ bool → ErrDtype,numpy 同拒);负值 →
// ErrIndex(numpy ValueError;四哨兵无 ValueError,取最接近的 ErrIndex);
// 输出 i64,长度 = max(max(x)+1, minlength);空输入 → zeros(minlength)。

import "fmt"

// truthyItem 标量真值(非 bool 按 != 0;NaN != 0 为真,numpy 同)。
func truthyItem(v any) (bool, error) {
	switch x := v.(type) {
	case bool:
		return x, nil
	case int:
		return x != 0, nil
	case int8:
		return x != 0, nil
	case int16:
		return x != 0, nil
	case int32:
		return x != 0, nil
	case int64:
		return x != 0, nil
	case uint:
		return x != 0, nil
	case uint8:
		return x != 0, nil
	case uint16:
		return x != 0, nil
	case uint32:
		return x != 0, nil
	case uint64:
		return x != 0, nil
	case float32:
		return x != 0, nil
	case float64:
		return x != 0, nil
	}
	return false, fmt.Errorf("num: 真值判断不支持 %T: %w", v, ErrDtype)
}

// keepPositions 条件数组的真值位置(0..m-1),m 已按调用方截断。
func keepPositions(cond *Array, m int) ([]int, error) {
	keep := make([]int, 0, m)
	for i := 0; i < m; i++ {
		v, err := cond.Item(i)
		if err != nil {
			return nil, err
		}
		t, err := truthyItem(v)
		if err != nil {
			return nil, err
		}
		if t {
			keep = append(keep, i)
		}
	}
	return keep, nil
}

// Compress 平铺压缩(≡ np.compress(axis=None))。
func Compress(a, cond *Array) (*Array, error) {
	if cond.NDim() != 1 {
		return nil, fmt.Errorf("num: Compress 条件须 1-D,得 %v: %w", cond.Shape(), ErrShape)
	}
	m := cond.Size()
	if m > a.Size() {
		m = a.Size()
	}
	keep, err := keepPositions(cond, m)
	if err != nil {
		return nil, err
	}
	out, err := New([]int{len(keep)}, a.Dtype())
	if err != nil {
		return nil, err
	}
	if len(keep) == 0 {
		return out, nil
	}
	src := a.AsContiguous()
	it := a.Dtype().Itemsize
	sw, dw := engineWindow(src), engineWindow(out)
	for j, i := range keep {
		copy(dw[j*it:(j+1)*it], sw[i*it:(i+1)*it])
	}
	return out, nil
}

// CompressAxis 沿轴压缩。
func CompressAxis(a, cond *Array, axis int) (*Array, error) {
	nd := a.NDim()
	if nd == 0 {
		return nil, fmt.Errorf("num: CompressAxis 源为 0-d: %w", ErrShape)
	}
	if cond.NDim() != 1 {
		return nil, fmt.Errorf("num: CompressAxis 条件须 1-D,得 %v: %w", cond.Shape(), ErrShape)
	}
	ax := axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return nil, fmt.Errorf("num: CompressAxis axis %d 越界(ndim=%d): %w", axis, nd, ErrIndex)
	}
	aSh := a.Shape()
	if cond.Size() > aSh[ax] {
		return nil, fmt.Errorf("num: CompressAxis 条件长 %d > 轴 %d 长 %d: %w", cond.Size(), ax, aSh[ax], ErrIndex)
	}
	keep, err := keepPositions(cond, cond.Size())
	if err != nil {
		return nil, err
	}
	outSh := append([]int{}, aSh...)
	outSh[ax] = len(keep)
	out, err := New(outSh, a.Dtype())
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	src := a.AsContiguous()
	it := a.Dtype().Itemsize
	st := cStrides(aSh, it)
	sw, dw := engineWindow(src), engineWindow(out)
	pos := make([]int, nd)
	for k := 0; k < out.Size(); k++ {
		off := 0
		for d := 0; d < nd; d++ {
			ix := pos[d]
			if d == ax {
				ix = keep[pos[d]]
			}
			off += ix * st[d]
		}
		copy(dw[k*it:(k+1)*it], sw[off:off+it])
		for d := nd - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < outSh[d] {
				break
			}
			pos[d] = 0
		}
	}
	return out, nil
}

// nonzeroFlat 收集非零元素的 C 序扁平下标。
func nonzeroFlat(a *Array) ([]int, error) {
	nd := a.NDim()
	if nd == 0 {
		return nil, fmt.Errorf("num: Nonzero 不接受 0-d(numpy 同): %w", ErrShape)
	}
	sh := a.Shape()
	pos := make([]int, nd)
	idx := make([]int, 0, 8)
	for k := 0; k < a.Size(); k++ {
		v, err := a.Item(pos...)
		if err != nil {
			return nil, err
		}
		t, err := truthyItem(v)
		if err != nil {
			return nil, err
		}
		if t {
			idx = append(idx, k)
		}
		for d := nd - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < sh[d] {
				break
			}
			pos[d] = 0
		}
	}
	return idx, nil
}

// Nonzero 返回每维一个 i64 索引数组(≡ np.nonzero 的 tuple 展开)。
func Nonzero(a *Array) ([]*Array, error) {
	idx, err := nonzeroFlat(a)
	if err != nil {
		return nil, err
	}
	nd := a.NDim()
	sh := a.Shape()
	outs := make([]*Array, nd)
	for d := 0; d < nd; d++ {
		o, err := New([]int{len(idx)}, DInt64)
		if err != nil {
			return nil, err
		}
		outs[d] = o
	}
	if len(idx) == 0 {
		return outs, nil
	}
	ws := make([][]int64, nd)
	for d := range outs {
		ws[d] = win[int64](engineWindow(outs[d]))
	}
	for j, k := range idx {
		kk := k
		for d := nd - 1; d >= 0; d-- {
			ws[d][j] = int64(kk % sh[d])
			kk /= sh[d]
		}
	}
	return outs, nil
}

// CountNonzero 非零计数(NaN 计入)。
func CountNonzero(a *Array) (int, error) {
	if a.NDim() == 0 {
		v, err := a.Item()
		if err != nil {
			return 0, err
		}
		t, err := truthyItem(v)
		if err != nil {
			return 0, err
		}
		if t {
			return 1, nil
		}
		return 0, nil
	}
	idx, err := nonzeroFlat(a)
	if err != nil {
		return 0, err
	}
	return len(idx), nil
}

// Bincount 非负整型计数(≡ np.bincount(x, minlength=…),weights 暂不支持)。
func Bincount(x *Array, minlength int) (*Array, error) {
	if minlength < 0 {
		return nil, fmt.Errorf("num: Bincount minlength %d 为负: %w", minlength, ErrShape)
	}
	if x.NDim() != 1 {
		return nil, fmt.Errorf("num: Bincount 输入须 1-D,得 %v: %w", x.Shape(), ErrShape)
	}
	switch x.Dtype().Kind {
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
	default:
		return nil, fmt.Errorf("num: Bincount 输入须整型,得 %s(numpy 同拒): %w", x.Dtype(), ErrDtype)
	}
	vals, err := idxInts(x)
	if err != nil {
		return nil, err
	}
	maxv := -1
	for _, v := range vals {
		if v < 0 {
			return nil, fmt.Errorf("num: Bincount 含负值 %d(numpy 同拒): %w", v, ErrIndex)
		}
		if int(v) > maxv {
			maxv = int(v)
		}
	}
	L := maxv + 1
	if minlength > L {
		L = minlength
	}
	out, err := New([]int{L}, DInt64)
	if err != nil {
		return nil, err
	}
	if L == 0 {
		return out, nil
	}
	w := win[int64](engineWindow(out))
	for _, v := range vals {
		w[v]++
	}
	return out, nil
}

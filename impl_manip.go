package num

import "fmt"

// —— 数组拼装/切分族(Step 24):array_split / h·v·dsplit / append·insert·delete /
// column_stack·dstack / put_along_axis / copy·copyto ——
//
// axis 传 -1 表示 numpy 的 axis=None(展平语义),仅 Append/Insert/Delete 使用。

// ArraySplit ≡ np.array_split(a, sections, axis):**不要求等分**,
// 前 n%sections 段各多一个元素(numpy 同)。
func ArraySplit(a *Array, sections, axis int) ([]*Array, error) {
	if sections <= 0 {
		return nil, fmt.Errorf("num: array_split: sections 须为正(得 %d): %w", sections, ErrValue)
	}
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, fmt.Errorf("num: array_split: %w", err)
	}
	n := a.Shape()[d]
	base := n / sections
	extra := n % sections
	out := make([]*Array, 0, sections)
	start := 0
	for i := 0; i < sections; i++ {
		size := base
		if i < extra {
			size++
		}
		part, err := sliceAlong(a, d, start, start+size)
		if err != nil {
			return nil, err
		}
		out = append(out, part)
		start += size
	}
	return out, nil
}

// SplitAt ≡ np.split(a, [i1, i2,...], axis):按索引点切分。
func SplitAt(a *Array, indices []int, axis int) ([]*Array, error) {
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, fmt.Errorf("num: split: %w", err)
	}
	n := a.Shape()[d]
	out := make([]*Array, 0, len(indices)+1)
	prev := 0
	for _, idx := range indices {
		if idx < 0 {
			idx += n
		}
		if idx < prev || idx > n {
			return nil, fmt.Errorf("num: split: 索引 %d 越界(0..%d,须递增): %w", idx, n, ErrIndex)
		}
		part, err := sliceAlong(a, d, prev, idx)
		if err != nil {
			return nil, err
		}
		out = append(out, part)
		prev = idx
	}
	last, err := sliceAlong(a, d, prev, n)
	if err != nil {
		return nil, err
	}
	return append(out, last), nil
}

// HSplit ≡ np.hsplit:2 维及以上沿轴 1,1 维沿轴 0(numpy 同)。
func HSplit(a *Array, sections int) ([]*Array, error) {
	ax := 1
	if a.NDim() == 1 {
		ax = 0
	}
	return ArraySplit(a, sections, ax)
}

// VSplit ≡ np.vsplit:沿轴 0 切分(要求至少 2 维)。
func VSplit(a *Array, sections int) ([]*Array, error) {
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: vsplit: 需要至少 2 维(得 %d): %w", a.NDim(), ErrShape)
	}
	return ArraySplit(a, sections, 0)
}

// DSplit ≡ np.dsplit:沿轴 2 切分(要求至少 3 维)。
func DSplit(a *Array, sections int) ([]*Array, error) {
	if a.NDim() < 3 {
		return nil, fmt.Errorf("num: dsplit: 需要至少 3 维(得 %d): %w", a.NDim(), ErrShape)
	}
	return ArraySplit(a, sections, 2)
}

// sliceAlong 取 a 沿轴 d 的 [lo,hi) 视图。
func sliceAlong(a *Array, d, lo, hi int) (*Array, error) {
	idx := make([]Indexer, a.NDim())
	for i := range idx {
		idx[i] = All()
	}
	idx[d] = Slice(lo, hi)
	return a.Sub(idx...)
}

// asValueArray 把 values(标量或 *Array)归一成 a 的同 dtype 数组。
func asValueArray(values any, dt Dtype) (*Array, error) {
	switch v := values.(type) {
	case *Array:
		if v.Dtype() == dt {
			return v, nil
		}
		return transformCopy(v.AsContiguous(), v.Shape(), dt)
	default:
		sc, err := New([]int{}, dt)
		if err != nil {
			return nil, err
		}
		if err := sc.SetItem([]int{}, v); err != nil {
			return nil, fmt.Errorf("num: 值 %v 无法写入 %s: %w", v, dt, err)
		}
		return sc, nil
	}
}

// Append ≡ np.append(a, values, axis):axis=-1 表示展平后拼接。
func Append(a *Array, values any, axis int) (*Array, error) {
	v, err := asValueArray(values, a.Dtype())
	if err != nil {
		return nil, err
	}
	if axis < 0 {
		// 展平拼接
		return Concatenate(0, a.AsContiguous().Flatten(), v.AsContiguous().Flatten())
	}
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, fmt.Errorf("num: append: %w", err)
	}
	// 标量 → 沿该轴复制成一"片"
	if v.NDim() == 0 {
		bsh := append([]int{}, a.Shape()...)
		bsh[d] = 1
		v, err = broadcastScalarTo(v, bsh)
		if err != nil {
			return nil, err
		}
	} else if v.NDim() != a.NDim() {
		return nil, fmt.Errorf("num: append: 维数不匹配(%d vs %d): %w", v.NDim(), a.NDim(), ErrShape)
	}
	return Concatenate(d, a, v)
}

// Insert ≡ np.insert(a, index, values, axis):axis=-1 表示插入到展平序列。
//
// index 允许负数(numpy 口径:负索引从尾部算,超出则钳到末尾)。
func Insert(a *Array, index int, values any, axis int) (*Array, error) {
	v, err := asValueArray(values, a.Dtype())
	if err != nil {
		return nil, err
	}
	if axis < 0 {
		flat := a.AsContiguous().Flatten()
		n := flat.Size()
		if index < 0 {
			index += n
		}
		if index < 0 {
			index = 0
		}
		if index > n {
			index = n
		}
		head, err := sliceAlong(flat, 0, 0, index)
		if err != nil {
			return nil, err
		}
		tail, err := sliceAlong(flat, 0, index, n)
		if err != nil {
			return nil, err
		}
		return Concatenate(0, head, v.AsContiguous().Flatten(), tail)
	}
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, fmt.Errorf("num: insert: %w", err)
	}
	n := a.Shape()[d]
	if index < 0 {
		index += n
	}
	if index < 0 {
		index = 0
	}
	if index > n {
		index = n
	}
	// 标量 → 一片;维数不足 → 补前导 1
	if v.NDim() == 0 {
		bsh := append([]int{}, a.Shape()...)
		bsh[d] = 1
		if v, err = broadcastScalarTo(v, bsh); err != nil {
			return nil, err
		}
	} else if v.NDim() < a.NDim() {
		sh := append([]int{}, a.Shape()...)
		copy(sh[len(sh)-v.NDim():], v.Shape())
		sh[d] = 1
		if v, err = v.Reshape(sh); err != nil {
			return nil, err
		}
	}
	head, err := sliceAlong(a, d, 0, index)
	if err != nil {
		return nil, err
	}
	tail, err := sliceAlong(a, d, index, n)
	if err != nil {
		return nil, err
	}
	return Concatenate(d, head, v, tail)
}

// Delete ≡ np.delete(a, index, axis):axis=-1 表示从展平序列删除。
func Delete(a *Array, index int, axis int) (*Array, error) {
	if axis < 0 {
		flat := a.AsContiguous().Flatten()
		n := flat.Size()
		if index < 0 {
			index += n
		}
		if index < 0 || index >= n {
			return nil, fmt.Errorf("num: delete: 索引 %d 越界(0..%d): %w", index, n-1, ErrIndex)
		}
		head, err := sliceAlong(flat, 0, 0, index)
		if err != nil {
			return nil, err
		}
		tail, err := sliceAlong(flat, 0, index+1, n)
		if err != nil {
			return nil, err
		}
		return Concatenate(0, head, tail)
	}
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, fmt.Errorf("num: delete: %w", err)
	}
	n := a.Shape()[d]
	if index < 0 {
		index += n
	}
	if index < 0 || index >= n {
		return nil, fmt.Errorf("num: delete: 索引 %d 越界(0..%d): %w", index, n-1, ErrIndex)
	}
	head, err := sliceAlong(a, d, 0, index)
	if err != nil {
		return nil, err
	}
	tail, err := sliceAlong(a, d, index+1, n)
	if err != nil {
		return nil, err
	}
	if head.Shape()[d] == 0 {
		return tail, nil
	}
	if tail.Shape()[d] == 0 {
		return head, nil
	}
	return Concatenate(d, head, tail)
}

// ColumnStack ≡ np.column_stack:1-D 序列变列;2-D 直接沿轴 1 拼接。
func ColumnStack(as ...*Array) (*Array, error) {
	if len(as) == 0 {
		return nil, fmt.Errorf("num: column_stack: 至少一个数组: %w", ErrShape)
	}
	parts := make([]*Array, len(as))
	for i, a := range as {
		switch a.NDim() {
		case 1:
			r, err := a.Reshape([]int{a.Size(), 1})
			if err != nil {
				return nil, err
			}
			parts[i] = r
		default:
			if a.NDim() != 2 {
				return nil, fmt.Errorf("num: column_stack: 需要 1-D/2-D(得 %d 维): %w", a.NDim(), ErrShape)
			}
			parts[i] = a
		}
	}
	return Concatenate(1, parts...)
}

// DStack ≡ np.dstack:1-D → (1,n,1)、2-D → (m,n,1),再沿轴 2 拼接。
func DStack(as ...*Array) (*Array, error) {
	if len(as) == 0 {
		return nil, fmt.Errorf("num: dstack: 至少一个数组: %w", ErrShape)
	}
	parts := make([]*Array, len(as))
	for i, a := range as {
		switch a.NDim() {
		case 1:
			r, err := a.Reshape([]int{1, a.Size(), 1})
			if err != nil {
				return nil, err
			}
			parts[i] = r
		case 2:
			r, err := a.Reshape([]int{a.Shape()[0], a.Shape()[1], 1})
			if err != nil {
				return nil, err
			}
			parts[i] = r
		case 3:
			parts[i] = a
		default:
			return nil, fmt.Errorf("num: dstack: 需要 1-D/2-D/3-D(得 %d 维): %w", a.NDim(), ErrShape)
		}
	}
	return Concatenate(2, parts...)
}

// PutAlongAxis ≡ np.put_along_axis(a, idx, values, axis):按 idx 沿轴写入 values(原地)。
func PutAlongAxis(a, idx, vals *Array, axis int) error {
	if a == nil || idx == nil || vals == nil {
		return fmt.Errorf("num: put_along_axis: 入参为 nil: %w", ErrDtype)
	}
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return fmt.Errorf("num: put_along_axis: %w", err)
	}
	ish := idx.Shape()
	if len(ish) != a.NDim() {
		return fmt.Errorf("num: put_along_axis: idx 维数 %d ≠ a 的 %d: %w", len(ish), a.NDim(), ErrShape)
	}
	for i := 0; i < a.NDim(); i++ {
		if i == d {
			continue
		}
		if ish[i] != a.Shape()[i] {
			return fmt.Errorf("num: put_along_axis: idx 在第 %d 轴长度 %d ≠ %d: %w",
				i, ish[i], a.Shape()[i], ErrShape)
		}
	}
	// 遍历 idx 的所有位置
	flatIdx, err := idxInts(idx.AsContiguous().Flatten())
	if err != nil {
		return fmt.Errorf("num: put_along_axis: idx 需为整型: %w", err)
	}
	// vals 应与 idx 同形(或单值);统一按 float64 读取后由 SetItem 落回目标 dtype
	vf, err := transformCopy(vals.AsContiguous().Flatten(), []int{vals.Size()}, DFloat64)
	if err != nil {
		return fmt.Errorf("num: put_along_axis: values 需为数值: %w", err)
	}
	flatVals := win[float64](engineWindow(vf))
	if len(flatVals) != 0 && len(flatVals) != len(flatIdx) {
		if len(flatVals) == 1 {
			v0 := flatVals[0]
			flatVals = make([]float64, len(flatIdx))
			for i := range flatVals {
				flatVals[i] = v0
			}
		} else {
			return fmt.Errorf("num: put_along_axis: values 长度 %d 与 idx 长度 %d 不符: %w",
				len(flatVals), len(flatIdx), ErrShape)
		}
	}
	pos := make([]int, len(ish))
	idxLen := a.Shape()[d]
	for k := range flatIdx {
		rem := k
		for i := len(ish) - 1; i >= 0; i-- {
			pos[i] = rem % ish[i]
			rem /= ish[i]
		}
		t := int(flatIdx[k])
		if t < 0 {
			t += idxLen
		}
		if t < 0 || t >= idxLen {
			return fmt.Errorf("num: put_along_axis: 目标下标 %d 越界(%d): %w", flatIdx[k], idxLen, ErrIndex)
		}
		pos[d] = t
		if err := a.SetItem(pos, flatVals[k]); err != nil {
			return err
		}
	}
	return nil
}

// Copy 返回独立副本(等价 a.Clone;便于与 numpy 的 np.copy 对照)。
func Copy(a *Array) *Array {
	if a == nil {
		return nil
	}
	c, err := transformCopy(a, a.Shape(), a.Dtype())
	if err != nil {
		return nil
	}
	return c
}

// CopyTo ≡ np.copyto(dst, src):把 src 写进 dst(原地,形状需一致)。
func CopyTo(dst, src *Array) error {
	if dst == nil || src == nil {
		return fmt.Errorf("num: copyto: 入参为 nil: %w", ErrDtype)
	}
	if !shapeEqual(dst.Shape(), src.Shape()) {
		return fmt.Errorf("num: copyto: 形状 %v ≠ %v: %w", dst.Shape(), src.Shape(), ErrShape)
	}
	s := src
	if dst.Dtype() != src.Dtype() {
		c, err := transformCopy(src.AsContiguous(), src.Shape(), dst.Dtype())
		if err != nil {
			return err
		}
		s = c
	}
	copy(engineWindow(dst), engineWindow(s.AsContiguous()))
	return nil
}

// broadcastScalarTo 把 0-d 值铺成指定形状。
func broadcastScalarTo(v *Array, sh []int) (*Array, error) {
	out, err := New(sh, v.Dtype())
	if err != nil {
		return nil, err
	}
	val, err := v.Item()
	if err != nil {
		return nil, err
	}
	es := v.Dtype().Itemsize
	sc, err := New([]int{}, v.Dtype())
	if err != nil {
		return nil, err
	}
	if err := sc.SetItem([]int{}, val); err != nil {
		return nil, err
	}
	src := engineWindow(sc)
	buf := engineWindow(out)
	for i := 0; i < len(buf); i += es {
		copy(buf[i:i+es], src)
	}
	return out, nil
}

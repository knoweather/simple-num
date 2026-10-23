package num

import "fmt"

// —— 形状操纵与拼接 ——
//
// 视图优先(numpy 同):Squeeze/ExpandDims/SwapAxes/Split 恒零拷贝;
// Flatten 恒拷贝(C 序,与 numpy flatten 一致:即使源是 F 序也按 C 序展平);
// Concatenate/Stack/VStack/HStack 需要新缓冲(逐块经 SetSub 写入,dtype 按
// promoteTypes 提升 —— numpy concatenate 亦提升)。

// Flatten C 序展平(恒拷贝)。
func (a *Array) Flatten() *Array {
	out, err := New([]int{a.Size()}, a.dtype)
	if err != nil {
		return nil // dtype 已校验;防御分支不可达
	}
	if w := a.fastWindow(); w != nil {
		copy(out.buf, w)
		return out
	}
	copyStrided(out.buf, a.buf, a.shape, a.st, a.off, a.dtype.Itemsize)
	return out
}

// Squeeze 去掉长度 1 的轴;无参去掉全部长度 1 轴(轴长≠1 → ErrShape,numpy 同)。
func (a *Array) Squeeze(axes ...int) (*Array, error) {
	n := a.NDim()
	drop := make([]bool, n)
	if len(axes) == 0 {
		for i, d := range a.shape {
			drop[i] = d == 1
		}
	} else {
		for _, ax := range axes {
			k := ax
			if k < 0 {
				k += n
			}
			if k < 0 || k >= n {
				return nil, fmt.Errorf("num: Squeeze 轴 %d 越界(ndim %d): %w", ax, n, ErrIndex)
			}
			if drop[k] {
				return nil, fmt.Errorf("num: Squeeze 轴 %d 重复: %w", ax, ErrShape)
			}
			if a.shape[k] != 1 {
				return nil, fmt.Errorf("num: Squeeze 轴 %d 长度 %d ≠ 1: %w", ax, a.shape[k], ErrShape)
			}
			drop[k] = true
		}
	}
	sh := make([]int, 0, n)
	st := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if drop[i] {
			continue
		}
		sh = append(sh, a.shape[i])
		st = append(st, a.st[i])
	}
	return &Array{buf: a.buf, shape: sh, st: st, off: a.off, dtype: a.dtype}, nil
}

// ExpandDims 在 axis 处插入长度 1 的轴(支持负轴;合法区间 [-(n+1), n])。
func (a *Array) ExpandDims(axis int) (*Array, error) {
	n := a.NDim()
	k := axis
	if k < 0 {
		k += n + 1
	}
	if k < 0 || k > n {
		return nil, fmt.Errorf("num: ExpandDims 轴 %d 越界(ndim %d): %w", axis, n, ErrIndex)
	}
	sh := make([]int, 0, n+1)
	st := make([]int, 0, n+1)
	sh = append(sh, a.shape[:k]...)
	st = append(st, a.st[:k]...)
	sh = append(sh, 1)
	st = append(st, 0)
	sh = append(sh, a.shape[k:]...)
	st = append(st, a.st[k:]...)
	return &Array{buf: a.buf, shape: sh, st: st, off: a.off, dtype: a.dtype}, nil
}

// SwapAxes 交换两轴(支持负轴;零拷贝视图)。
func (a *Array) SwapAxes(i, j int) (*Array, error) {
	n := a.NDim()
	x, y := i, j
	if x < 0 {
		x += n
	}
	if y < 0 {
		y += n
	}
	if x < 0 || x >= n || y < 0 || y >= n {
		return nil, fmt.Errorf("num: SwapAxes (%d,%d) 越界(ndim %d): %w", i, j, n, ErrIndex)
	}
	sh := append([]int(nil), a.shape...)
	st := append([]int(nil), a.st...)
	sh[x], sh[y] = sh[y], sh[x]
	st[x], st[y] = st[y], st[x]
	return &Array{buf: a.buf, shape: sh, st: st, off: a.off, dtype: a.dtype}, nil
}

// Concatenate 沿 axis(支持负轴)拼接;除拼接轴外形状必须一致(numpy 同),
// dtype 按 promoteTypes 提升。
func Concatenate(axis int, as ...*Array) (*Array, error) {
	if len(as) == 0 {
		return nil, fmt.Errorf("num: Concatenate 空入参: %w", ErrShape)
	}
	nd := as[0].NDim()
	ax := axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return nil, fmt.Errorf("num: Concatenate 轴 %d 越界(ndim %d): %w", axis, nd, ErrIndex)
	}
	dt := as[0].dtype
	sh := append([]int(nil), as[0].shape...)
	for _, x := range as[1:] {
		if x.NDim() != nd {
			return nil, fmt.Errorf("num: Concatenate 维度 %d ≠ %d: %w", x.NDim(), nd, ErrShape)
		}
		for d := 0; d < nd; d++ {
			if d == ax {
				continue
			}
			if x.shape[d] != sh[d] {
				return nil, fmt.Errorf("num: Concatenate 轴 %d 外形状 %v ≠ %v: %w", d, x.shape, sh, ErrShape)
			}
		}
		sh[ax] += x.shape[ax]
		var err error
		if dt, err = promoteTypes(dt, x.dtype); err != nil {
			return nil, err
		}
	}
	out, err := New(sh, dt)
	if err != nil {
		return nil, err
	}
	off := 0
	for _, x := range as {
		idx := make([]Indexer, nd)
		for d := 0; d < nd; d++ {
			if d == ax {
				idx[d] = Slice(off, off+x.shape[ax])
			} else {
				idx[d] = All()
			}
		}
		if err := out.SetSub(x, idx...); err != nil {
			return nil, err
		}
		off += x.shape[ax]
	}
	return out, nil
}

// Stack 沿新轴 axis(支持负轴)堆叠;所有入参形状必须一致(numpy 同)。
func Stack(axis int, as ...*Array) (*Array, error) {
	if len(as) == 0 {
		return nil, fmt.Errorf("num: Stack 空入参: %w", ErrShape)
	}
	sh0 := as[0].shape
	for _, x := range as[1:] {
		if !shapeEqual(x.shape, sh0) {
			return nil, fmt.Errorf("num: Stack 形状 %v ≠ %v: %w", x.shape, sh0, ErrShape)
		}
	}
	k, err := normAxis(axis, len(sh0)+1)
	if err != nil {
		return nil, err
	}
	dt := as[0].dtype
	for _, x := range as[1:] {
		if dt, err = promoteTypes(dt, x.dtype); err != nil {
			return nil, err
		}
	}
	sh := make([]int, 0, len(sh0)+1)
	sh = append(sh, sh0[:k]...)
	sh = append(sh, len(as))
	sh = append(sh, sh0[k:]...)
	out, err := New(sh, dt)
	if err != nil {
		return nil, err
	}
	for i, x := range as {
		// 新轴上用整点(而非 1 长切片):视图形状与 x 完全一致 ——
		// numpy 赋值不允许 (n,) 广播进 (n,1)(实测 ValueError),故不能留 1 长轴。
		idx := make([]Indexer, len(sh))
		for d := 0; d < len(sh); d++ {
			if d == k {
				idx[d] = Idx(i)
			} else {
				idx[d] = All()
			}
		}
		if err := out.SetSub(x, idx...); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// VStack 沿轴 0 拼接;1 维入参先升成行向量(atleast_2d)。
func VStack(as ...*Array) (*Array, error) {
	if len(as) == 0 {
		return nil, fmt.Errorf("num: VStack 空入参: %w", ErrShape)
	}
	up := make([]*Array, len(as))
	for i, a := range as {
		x, err := atLeast2D(a)
		if err != nil {
			return nil, err
		}
		up[i] = x
	}
	return Concatenate(0, up...)
}

// HStack 1 维/0 维入参沿轴 0 拼接,否则沿轴 1(numpy 同)。
func HStack(as ...*Array) (*Array, error) {
	if len(as) == 0 {
		return nil, fmt.Errorf("num: HStack 空入参: %w", ErrShape)
	}
	flat := true
	for _, a := range as {
		if a.NDim() > 1 {
			flat = false
			break
		}
	}
	if !flat {
		return Concatenate(1, as...)
	}
	up := make([]*Array, len(as))
	for i, a := range as {
		x, err := atLeast1D(a)
		if err != nil {
			return nil, err
		}
		up[i] = x
	}
	return Concatenate(0, up...)
}

// Split 沿 axis 等分 sections 份(不能等分 → ErrShape,numpy split 同;返回视图)。
// 注:只支持 numpy 的"份数"形式,不支持"切点下标数组"形式(v0 出界)。
func Split(a *Array, sections int, axis int) ([]*Array, error) {
	if sections <= 0 {
		return nil, fmt.Errorf("num: Split 份数 %d ≤ 0: %w", sections, ErrShape)
	}
	ax, err := normAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	n := a.shape[ax]
	if n%sections != 0 {
		return nil, fmt.Errorf("num: Split 轴长 %d 不能被 %d 等分: %w", n, sections, ErrShape)
	}
	each := n / sections
	out := make([]*Array, 0, sections)
	for k := 0; k < sections; k++ {
		idx := make([]Indexer, a.NDim())
		for d := range idx {
			if d == ax {
				idx[d] = Slice(k*each, (k+1)*each)
			} else {
				idx[d] = All()
			}
		}
		v, err := a.Sub(idx...)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// normAxis 负轴归一 + 界校验(合法区间 [-n, n-1])。
func normAxis(axis, n int) (int, error) {
	k := axis
	if k < 0 {
		k += n
	}
	if k < 0 || k >= n {
		return 0, fmt.Errorf("num: 轴 %d 越界(ndim %d): %w", axis, n, ErrIndex)
	}
	return k, nil
}

// atLeast1D 0 维 → (1,);其余原样。
func atLeast1D(a *Array) (*Array, error) {
	if a.NDim() == 0 {
		return a.Reshape([]int{1})
	}
	return a, nil
}

// atLeast2D 0 维 → (1,1);1 维 (n,) → (1,n);其余原样(numpy 同)。
func atLeast2D(a *Array) (*Array, error) {
	switch a.NDim() {
	case 0:
		return a.Reshape([]int{1, 1})
	case 1:
		return a.Reshape([]int{1, a.shape[0]})
	}
	return a, nil
}

package num

import "fmt"

// —— 形状/构造族(numpy 高频面):*_like、atleast_*、flip 族、roll、repeat、tile ——
//
// 说明:本库不支持负步长视图,故 flip 族与 roll/repeat/tile 一律**返回新数组**
// (numpy 的 flip 是视图,值语义相同;此处是受控差异,已在文档注明)。

// ZerosLike ≡ np.zeros_like:同形状同 dtype 的全零。
func ZerosLike(a *Array) (*Array, error) { return Zeros(a.Shape(), a.Dtype()) }

// OnesLike ≡ np.ones_like:同形状同 dtype 的全一。
func OnesLike(a *Array) (*Array, error) { return Ones(a.Shape(), a.Dtype()) }

// EmptyLike ≡ np.empty_like:同形状同 dtype。本库 New 会零填(numpy 的 empty 是
// 未初始化内存,内容不确定);这里给确定性的零,属受控差异。
func EmptyLike(a *Array) (*Array, error) { return New(a.Shape(), a.Dtype()) }

// FullLike ≡ np.full_like:同形状同 dtype,填 v。
func FullLike(a *Array, v any) (*Array, error) { return Full(a.Shape(), v, a.Dtype()) }

// Atleast1D ≡ np.atleast_1d:0-d → (1,),其余原样(不拷贝)。
func Atleast1D(a *Array) (*Array, error) {
	if a.NDim() == 0 {
		return a.Reshape([]int{1})
	}
	return a, nil
}

// Atleast2D ≡ np.atleast_2d:0-d → (1,1);1-D → (1,n);其余原样。
func Atleast2D(a *Array) (*Array, error) {
	switch a.NDim() {
	case 0:
		return a.Reshape([]int{1, 1})
	case 1:
		return a.Reshape([]int{1, a.Shape()[0]})
	}
	return a, nil
}

// Atleast3D ≡ np.atleast_3d:0-d → (1,1,1);1-D → (1,n,1);2-D → (m,n,1);其余原样。
// (注意 1-D 的 numpy 口径是 (1,n,1),不是 (1,1,n)。)
func Atleast3D(a *Array) (*Array, error) {
	switch a.NDim() {
	case 0:
		return a.Reshape([]int{1, 1, 1})
	case 1:
		return a.Reshape([]int{1, a.Shape()[0], 1})
	case 2:
		return a.Reshape([]int{a.Shape()[0], a.Shape()[1], 1})
	}
	return a, nil
}

// Flip ≡ np.flip:沿给定轴反转;axis 为空表示所有轴。
func Flip(a *Array, axis []int) (*Array, error) {
	nd := a.NDim()
	fl := make([]bool, nd)
	if len(axis) == 0 {
		for i := range fl {
			fl[i] = true
		}
	} else {
		for _, ax := range axis {
			d, err := normalizeAxis(a, ax)
			if err != nil {
				return nil, fmt.Errorf("num: flip: %w", err)
			}
			fl[d] = true
		}
	}
	return remapCopy(a, a.Shape(), func(idx []int) []int {
		src := make([]int, len(idx))
		copy(src, idx)
		for d := range src {
			if fl[d] {
				src[d] = a.Shape()[d] - 1 - src[d]
			}
		}
		return src
	})
}

// FlipUD ≡ np.flipud:沿轴 0 反转。
func FlipUD(a *Array) (*Array, error) { return Flip(a, []int{0}) }

// FlipLR ≡ np.fliplr:沿轴 1 反转(要求 ndim >= 2,numpy 同)。
func FlipLR(a *Array) (*Array, error) {
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: fliplr 需要 ndim >= 2,得 %d: %w", a.NDim(), ErrShape)
	}
	return Flip(a, []int{1})
}

// Roll ≡ np.roll:沿给定轴循环移位;axis 为空表示先拉平再移、最后恢复原形状。
func Roll(a *Array, shift int, axis []int) (*Array, error) {
	if len(axis) == 0 {
		flat := a.AsContiguous().Flatten()
		n := flat.Size()
		if n == 0 {
			return flat, nil
		}
		s := shift % n
		if s < 0 {
			s += n
		}
		out, err := remapCopy(flat, []int{n}, func(idx []int) []int {
			return []int{(idx[0] + n - s) % n}
		})
		if err != nil {
			return nil, err
		}
		return out.Reshape(a.Shape())
	}
	if len(axis) != 1 {
		return nil, fmt.Errorf("num: roll 目前只支持单轴: %w", ErrShape)
	}
	d, err := normalizeAxis(a, axis[0])
	if err != nil {
		return nil, fmt.Errorf("num: roll: %w", err)
	}
	n := a.Shape()[d]
	if n == 0 {
		return a.AsContiguous(), nil
	}
	s := shift % n
	if s < 0 {
		s += n
	}
	return remapCopy(a, a.Shape(), func(idx []int) []int {
		src := make([]int, len(idx))
		copy(src, idx)
		src[d] = (idx[d] + n - s) % n
		return src
	})
}

// Repeat ≡ np.repeat(a, repeats, axis=None):每个元素重复 repeats 次;
// axis 为空表示先拉平(与 numpy 同)。
func Repeat(a *Array, repeats int, axis []int) (*Array, error) {
	if repeats < 0 {
		return nil, fmt.Errorf("num: repeat: repeats 不能为负 %d: %w", repeats, ErrShape)
	}
	if len(axis) == 0 {
		flat := a.AsContiguous().Flatten()
		n := flat.Size()
		out, err := remapCopy(flat, []int{n * repeats}, func(idx []int) []int {
			return []int{idx[0] / repeats}
		})
		if err != nil {
			return nil, err
		}
		return out, nil
	}
	d, err := normalizeAxis(a, axis[0])
	if err != nil {
		return nil, fmt.Errorf("num: repeat: %w", err)
	}
	sh := append([]int(nil), a.Shape()...)
	sh[d] *= repeats
	return remapCopy(a, sh, func(idx []int) []int {
		src := make([]int, len(idx))
		copy(src, idx)
		src[d] = idx[d] / repeats
		return src
	})
}

// Tile ≡ np.tile:按 reps 各维重复次数铺贴(numpy:reps 短则前置 1,
// 长则给形状前置 1)。
func Tile(a *Array, reps []int) (*Array, error) {
	nd := a.NDim()
	m := len(reps)
	if m < nd { // reps 前置补 1
		r2 := make([]int, nd)
		copy(r2[nd-m:], reps)
		for i := 0; i < nd-m; i++ {
			r2[i] = 1
		}
		reps = r2
		m = nd
	}
	// 形状前置补 1 到 m 维
	sh := make([]int, m)
	for i := 0; i < m; i++ {
		sh[i] = 1
	}
	copy(sh[m-nd:], a.Shape())
	for i := range reps {
		if reps[i] < 0 {
			return nil, fmt.Errorf("num: tile: reps[%d] 不能为负: %w", i, ErrShape)
		}
		sh[i] *= reps[i]
	}
	return remapCopy(a, sh, func(idx []int) []int {
		src := make([]int, nd)
		for i := 0; i < nd; i++ {
			d := a.Shape()[i]
			if d == 0 {
				src[i] = 0
				continue
			}
			src[i] = idx[m-nd+i] % d
		}
		return src
	})
}

// normalizeAxis 把负轴归一(squeeze 等既有代码里也有类似逻辑,此处独立实现避免耦合)。
func normalizeAxis(a *Array, ax int) (int, error) {
	nd := a.NDim()
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return 0, fmt.Errorf("轴 %d 越界(ndim=%d): %w", ax, nd, ErrIndex)
	}
	return ax, nil
}

// remapCopy 按"输出下标 → 源下标"映射做一次整块元素拷贝(与 dtype 无关)。
func remapCopy(a *Array, outShape []int, mapIdx func(idx []int) []int) (*Array, error) {
	src := a.AsContiguous()
	out, err := New(outShape, a.Dtype())
	if err != nil {
		return nil, err
	}
	size := out.Size()
	if size == 0 {
		return out, nil
	}
	es := a.Dtype().Itemsize
	sb, db := engineWindow(src), engineWindow(out)
	srcSh := src.Shape()
	nsd := len(srcSh)
	st := make([]int, nsd) // 源(连续)各维元素步长
	acc := 1
	for d := nsd - 1; d >= 0; d-- {
		st[d] = acc
		acc *= srcSh[d]
	}
	idx := make([]int, len(outShape))
	for fi := 0; fi < size; fi++ {
		rem := fi
		for d := len(outShape) - 1; d >= 0; d-- {
			if outShape[d] == 0 {
				idx[d] = 0
				continue
			}
			idx[d] = rem % outShape[d]
			rem /= outShape[d]
		}
		mapped := mapIdx(idx)
		si := 0
		for d := 0; d < nsd && d < len(mapped); d++ {
			si += mapped[d] * st[d]
		}
		copy(db[fi*es:(fi+1)*es], sb[si*es:(si+1)*es])
	}
	return out, nil
}

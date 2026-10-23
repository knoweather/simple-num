package num

import "fmt"

// —— 杂项算法(Step 8 第一批):searchsorted / digitize / interp / meshgrid / gradient ——
//
// 口径与 numpy 一致:
// - searchsorted:对已排序 1-D 数组做二分;right=false → lower_bound(取 >= 的首个),
// right=true → upper_bound(取 > 的首个);结果 int64、形状同 v。
// - digitize:等价 searchsorted(bins, x, side = right ? 'left' : 'right')。
// - interp:分段线性插值,xp 需递增;越界取两端 fp 值(numpy 的 left/right 默认行为);
// 结果 float64、形状同 x。
// - meshgrid:按 'ij' 或 'xy' 索引序生成坐标网格(返回降维的坐标数组)。
// - gradient:中心差分(边界用一阶前/后差分);结果 float64,每个轴一个数组。

// SearchSorted ≡ np.searchsorted(a, v, side)。
func SearchSorted(a, v *Array, right bool) (*Array, error) {
	if a.NDim() != 1 {
		return nil, fmt.Errorf("num: searchsorted: a 必须是 1-D,得 %v: %w", a.Shape(), ErrShape)
	}
	dt, err := promoteTypes(a.Dtype(), v.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: searchsorted: dtype 不可提升: %w", err)
	}
	ac, err := transformCopy(a, a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	vc, err := transformCopy(v, v.Shape(), dt)
	if err != nil {
		return nil, err
	}
	out, err := New(v.Shape(), DInt64)
	if err != nil {
		return nil, err
	}
	n := a.Size()
	if out.Size() == 0 {
		return out, nil
	}
	if n == 0 {
		return out, nil // 空 a:全部得 0(numpy 同)
	}
	if !searchSortedDispatch(dt, ac, vc, out, right) {
		return nil, fmt.Errorf("num: searchsorted 无 %s 内核: %w", dt, ErrDtype)
	}
	return out, nil
}

func searchSortedDispatch(dt Dtype, a, v, out *Array, right bool) bool {
	switch dt.Kind {
	case Float64:
		searchTyped(win[float64](engineWindow(a)), win[float64](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Float32:
		searchTyped(win[float32](engineWindow(a)), win[float32](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Int8:
		searchTyped(win[int8](engineWindow(a)), win[int8](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Int16:
		searchTyped(win[int16](engineWindow(a)), win[int16](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Int32:
		searchTyped(win[int32](engineWindow(a)), win[int32](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Int64:
		searchTyped(win[int64](engineWindow(a)), win[int64](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Uint8:
		searchTyped(win[uint8](engineWindow(a)), win[uint8](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Uint16:
		searchTyped(win[uint16](engineWindow(a)), win[uint16](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Uint32:
		searchTyped(win[uint32](engineWindow(a)), win[uint32](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	case Uint64:
		searchTyped(win[uint64](engineWindow(a)), win[uint64](engineWindow(v)),
			win[int64](engineWindow(out)), right)
	default:
		return false
	}
	return true
}

// searchTyped 二分:right=false → 首个 a[i] >= v;right=true → 首个 a[i] > v。
func searchTyped[T castableNum](a, v []T, out []int64, right bool) {
	n := len(a)
	for j, x := range v {
		lo, hi := 0, n
		for lo < hi {
			mid := int(uint(lo+hi) >> 1)
			if a[mid] < x || (right && a[mid] == x) {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		out[j] = int64(lo)
	}
}

// Digitize ≡ np.digitize(x, bins, right):right=false 取 bins[i-1] <= x < bins[i]。
func Digitize(x, bins *Array, right bool) (*Array, error) {
	// numpy:归位索引 = searchsorted(bins, x, side = right ? 'left' : 'right')
	return SearchSorted(bins, x, !right)
}

// Interp ≡ np.interp(x, xp, fp):分段线性插值(越界取端点值)。
func Interp(x, xp, fp *Array) (*Array, error) {
	if xp.NDim() != 1 || fp.NDim() != 1 {
		return nil, fmt.Errorf("num: interp: xp/fp 必须是 1-D: %w", ErrShape)
	}
	if xp.Size() != fp.Size() {
		return nil, fmt.Errorf("num: interp: xp 长度 %d ≠ fp 长度 %d: %w", xp.Size(), fp.Size(), ErrShape)
	}
	if xp.Size() == 0 {
		return nil, fmt.Errorf("num: interp: xp 不能为空: %w", ErrShape)
	}
	xf, err := transformCopy(x.AsContiguous().Flatten(), []int{x.Size()}, DFloat64)
	if err != nil {
		return nil, err
	}
	xpf, err := transformCopy(xp, xp.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	fpf, err := transformCopy(fp, fp.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	out, err := New(x.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	xw := win[float64](engineWindow(xf))
	pw := win[float64](engineWindow(xpf))
	fw := win[float64](engineWindow(fpf))
	ow := win[float64](engineWindow(out))
	n := len(pw)
	for i, v := range xw {
		switch {
		case v != v: // NaN → NaN(numpy 同)
			ow[i] = v
		case v <= pw[0]:
			ow[i] = fw[0]
		case v >= pw[n-1]:
			ow[i] = fw[n-1]
		default:
			// 找 i 使 pw[i] <= v < pw[i+1]
			lo, hi := 0, n-1
			for lo < hi {
				mid := int(uint(lo+hi) >> 1)
				if pw[mid] <= v {
					lo = mid + 1
				} else {
					hi = mid
				}
			}
			j := lo - 1
			if j >= n-1 {
				j = n - 2
			}
			if pw[j+1] == pw[j] { // 重复 xp:取左值(numpy 亦按 slope=0 处理)
				ow[i] = fw[j]
				continue
			}
			slope := (fw[j+1] - fw[j]) / (pw[j+1] - pw[j])
			ow[i] = slope*(v-pw[j]) + fw[j]
		}
	}
	return out, nil
}

// MeshGrid ≡ np.meshgrid(xs..., indexing):返回每个输入对应的坐标网格。
// xy=true 时按 'xy' 序(前两维交换),否则 'ij'。
func MeshGrid(xs []*Array, xy bool) ([]*Array, error) {
	k := len(xs)
	if k == 0 {
		return nil, fmt.Errorf("num: meshgrid: 至少需要一个输入: %w", ErrShape)
	}
	dims := make([]int, k)
	for i, a := range xs {
		if a.NDim() != 1 {
			// numpy 对非 1-D 输入先拉平
			xs[i] = a.AsContiguous().Flatten()
		}
		dims[i] = xs[i].Size()
	}
	if xy && k >= 2 {
		dims[0], dims[1] = dims[1], dims[0]
	}
	out := make([]*Array, k)
	for i := 0; i < k; i++ {
		// 该数组沿哪一维变化
		axisPos := i
		if xy && k >= 2 {
			switch i {
			case 0:
				axisPos = 1
			case 1:
				axisPos = 0
			}
		}
		sh := make([]int, k)
		for d := range sh {
			sh[d] = 1
		}
		sh[axisPos] = xs[i].Size()
		view, err := xs[i].Reshape(sh)
		if err != nil {
			return nil, err
		}
		bc, err := view.BroadcastTo(dims)
		if err != nil {
			return nil, err
		}
		// 物化成独立数组(numpy 的 meshgrid 返回可写副本,不共享内存)
		m, err := transformCopy(bc, dims, xs[i].Dtype())
		if err != nil {
			return nil, err
		}
		out[i] = m
	}
	return out, nil
}

// Gradient ≡ np.gradient(f, axis):中心差分(边界一阶),结果 float64,每轴一个数组。
// axis 为空表示对所有轴各求一次(numpy 同)。
func Gradient(f *Array, axis []int) ([]*Array, error) {
	if f.NDim() == 0 {
		return nil, fmt.Errorf("num: gradient: 0-d 输入无梯度: %w", ErrShape)
	}
	axes := axis
	if len(axes) == 0 {
		axes = make([]int, f.NDim())
		for i := range axes {
			axes[i] = i
		}
	}
	out := make([]*Array, len(axes))
	for k, ax := range axes {
		g, err := gradAxis(f, ax)
		if err != nil {
			return nil, err
		}
		out[k] = g
	}
	return out, nil
}

// gradAxis 单轴梯度(float64)。
func gradAxis(f *Array, ax int) (*Array, error) {
	d, err := normalizeAxis(f, ax)
	if err != nil {
		return nil, fmt.Errorf("num: gradient: %w", err)
	}
	ff, err := transformCopy(f, f.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	rows, m, n, err := axisRows(ff, d)
	if err != nil {
		return nil, err
	}
	work, err := transformCopy(rows, rows.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	if m > 0 && n > 1 {
		w := win[float64](engineWindow(work))
		for r := 0; r < m; r++ {
			row := w[r*n : (r+1)*n]
			tmp := make([]float64, n)
			switch n {
			case 2:
				tmp[0] = row[1] - row[0]
				tmp[1] = row[1] - row[0]
			default:
				tmp[0] = row[1] - row[0]
				tmp[n-1] = row[n-1] - row[n-2]
				for i := 1; i < n-1; i++ {
					tmp[i] = (row[i+1] - row[i-1]) / 2
				}
			}
			copy(row, tmp)
		}
	}
	return axisRowsBack(work, ff, d, m, n)
}

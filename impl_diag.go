package num

import (
	"fmt"
	"math"
)

// —— 对角线/三角/迹 与 协方差族(Step 12)——

// Diag ≡ np.diag(v, k):
// 1-D 输入 → 构造 (n+|k|) 方阵,把 v 放在第 k 条对角线上;
// 2-D 输入 → 抽取第 k 条对角线为 1-D。
func Diag(v *Array, k int) (*Array, error) {
	switch v.NDim() {
	case 1:
		n := v.Size()
		sz := n + absOfInt(k)
		out, err := New([]int{sz, sz}, v.Dtype())
		if err != nil {
			return nil, err
		}
		flat := v.AsContiguous().Flatten()
		es := v.Dtype().Itemsize
		db := engineWindow(out)
		for i := 0; i < n; i++ {
			r, cc := i, i
			if k >= 0 {
				cc = i + k
			} else {
				r = i - k
			}
			src := engineWindow(flat)[i*es : (i+1)*es]
			copy(db[(r*sz+cc)*es:(r*sz+cc+1)*es], src)
		}
		return out, nil
	case 2:
		rows, cols := v.Shape()[0], v.Shape()[1]
		// 第 k 条对角线上的元素个数
		var cnt int
		if k >= 0 {
			cnt = minInt(rows, cols-k)
		} else {
			cnt = minInt(rows+k, cols)
		}
		if cnt < 0 {
			cnt = 0
		}
		out, err := New([]int{cnt}, v.Dtype())
		if err != nil {
			return nil, err
		}
		es := v.Dtype().Itemsize
		sb := engineWindow(v.AsContiguous())
		db := engineWindow(out)
		for i := 0; i < cnt; i++ {
			r, cc := i, i
			if k >= 0 {
				cc = i + k
			} else {
				r = i - k
			}
			copy(db[i*es:(i+1)*es], sb[(r*cols+cc)*es:(r*cols+cc+1)*es])
		}
		return out, nil
	default:
		return nil, fmt.Errorf("num: diag: 只支持 1-D/2-D(得 %d 维): %w", v.NDim(), ErrShape)
	}
}

// Diagflat ≡ np.diagflat(v, k):先拉平再构造对角阵。
func Diagflat(v *Array, k int) (*Array, error) {
	return Diag(v.AsContiguous().Flatten(), k)
}

// Trace ≡ np.trace(a, offset):对角线元素之和(结果 0-d,保持输入 dtype)。
func Trace(a *Array, offset int) (*Array, error) {
	if a.NDim() != 2 {
		return nil, fmt.Errorf("num: trace: 只支持 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	dg, err := Diag(a, offset)
	if err != nil {
		return nil, err
	}
	// 按 dtype 累加(整型保持精确)
	dw := engineWindow(dg.AsContiguous())
	var sum any
	switch a.Dtype().Kind {
	case Float64:
		s := 0.0
		for _, v := range win[float64](dw) {
			s += v
		}
		sum = s
	case Float32:
		s := float32(0)
		for _, v := range win[float32](dw) {
			s += v
		}
		sum = s
	case Int8:
		s := int8(0)
		for _, v := range win[int8](dw) {
			s += v
		}
		sum = s
	case Int16:
		s := int16(0)
		for _, v := range win[int16](dw) {
			s += v
		}
		sum = s
	case Int32:
		s := int32(0)
		for _, v := range win[int32](dw) {
			s += v
		}
		sum = s
	case Int64:
		s := int64(0)
		for _, v := range win[int64](dw) {
			s += v
		}
		sum = s
	case Uint8:
		s := uint8(0)
		for _, v := range win[uint8](dw) {
			s += v
		}
		sum = s
	case Uint16:
		s := uint16(0)
		for _, v := range win[uint16](dw) {
			s += v
		}
		sum = s
	case Uint32:
		s := uint32(0)
		for _, v := range win[uint32](dw) {
			s += v
		}
		sum = s
	case Uint64:
		s := uint64(0)
		for _, v := range win[uint64](dw) {
			s += v
		}
		sum = s
	case Bool:
		s := uint8(0)
		for _, v := range win[uint8](dw) {
			if v != 0 {
				s++
			}
		}
		sum = s != 0
	default:
		return nil, fmt.Errorf("num: trace 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	out, err := New([]int{}, a.Dtype())
	if err != nil {
		return nil, err
	}
	if err := out.SetItem([]int{}, sum); err != nil {
		return nil, err
	}
	return out, nil
}

// Triu ≡ np.triu(m, k):保留主对角线(含)以上,其余置 0(在最后两轴上操作)。
func Triu(m *Array, k int) (*Array, error) { return triLike(m, k, true) }

// Tril ≡ np.tril(m, k):保留主对角线(含)以下,其余置 0。
func Tril(m *Array, k int) (*Array, error) { return triLike(m, k, false) }

func triLike(m *Array, k int, upper bool) (*Array, error) {
	if m.NDim() < 2 {
		return nil, fmt.Errorf("num: triu/tril: 至少 2 维(得 %d): %w", m.NDim(), ErrShape)
	}
	out, err := transformCopy(m, m.Shape(), m.Dtype())
	if err != nil {
		return nil, err
	}
	sh := m.Shape()
	rows, cols := sh[len(sh)-2], sh[len(sh)-1]
	outer := out.Size() / (rows * cols)
	es := m.Dtype().Itemsize
	db := engineWindow(out)
	for o := 0; o < outer; o++ {
		base := o * rows * cols
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				keep := false
				if upper {
					keep = j-i >= k
				} else {
					keep = j-i <= k
				}
				if keep {
					continue
				}
				pos := (base + i*cols + j) * es
				for b := pos; b < pos+es; b++ {
					db[b] = 0
				}
			}
		}
	}
	return out, nil
}

// FillDiagonal ≡ np.fill_diagonal(a, v):原地把对角线写成 v(标量;2-D)。
func FillDiagonal(a *Array, v any) error {
	if a.NDim() != 2 {
		return fmt.Errorf("num: fill_diagonal: 只支持 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	rows, cols := a.Shape()[0], a.Shape()[1]
	n := minInt(rows, cols)
	for i := 0; i < n; i++ {
		if err := a.SetItem([]int{i, i}, v); err != nil {
			return err
		}
	}
	return nil
}

// Cov ≡ np.cov(a, rowvar, ddof):协方差(默认 rowvar=true、ddof=N-1,与 numpy 相同)。
//
// 1-D 输入 → 0-d 标量(方差);2-D 输入 → (nvar × nvar) 矩阵。
// ddof < 0 表示按 numpy 默认取 N-1。
func Cov(a *Array, rowvar bool, ddof int) (*Array, error) {
	x, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	nvar, nobs, obs, err := covLayout(x, rowvar)
	if err != nil {
		return nil, err
	}
	if ddof < 0 {
		ddof = 1
	}
	den := float64(nobs - ddof)
	if den <= 0 {
		return nil, fmt.Errorf("num: cov: 观测量 %d 不足以减 ddof=%d: %w", nobs, ddof, ErrValue)
	}
	// 中心化:每行一个变量(在独立副本上做,避免改到调用方数据)
	obsArr, err := transformCopy(obs, obs.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(obsArr))
	for i := 0; i < nvar; i++ {
		row := ow[i*nobs : (i+1)*nobs]
		s := 0.0
		for _, v := range row {
			s += v
		}
		mean := s / float64(nobs)
		for j := range row {
			row[j] -= mean
		}
	}
	obsT, err := obsArr.Transpose()
	if err != nil {
		return nil, err
	}
	prod, err := MatMul(obsArr, obsT.AsContiguous())
	if err != nil {
		return nil, err
	}
	pw := win[float64](engineWindow(prod))
	for i := range pw {
		pw[i] /= den
	}
	if a.NDim() == 1 {
		// 1-D:返回 0-d 标量
		out, err := New([]int{}, DFloat64)
		if err != nil {
			return nil, err
		}
		if err := out.SetItem([]int{}, pw[0]); err != nil {
			return nil, err
		}
		return out, nil
	}
	return prod, nil
}

// CorrCoef ≡ np.corrcoef(a, rowvar):相关系数矩阵(结果裁剪到 [-1,1],numpy 同)。
func CorrCoef(a *Array, rowvar bool) (*Array, error) {
	c, err := Cov(a, rowvar, -1)
	if err != nil {
		return nil, err
	}
	if c.NDim() == 0 {
		// 单变量:与自身的相关系数恒为 1
		out, err := New([]int{}, DFloat64)
		if err != nil {
			return nil, err
		}
		if err := out.SetItem([]int{}, 1.0); err != nil {
			return nil, err
		}
		return out, nil
	}
	n := c.Shape()[0]
	cw := win[float64](engineWindow(c))
	out, err := New([]int{n, n}, DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			di := cw[i*n+i]
			dj := cw[j*n+j]
			if di <= 0 || dj <= 0 {
				ow[i*n+j] = math.NaN()
				continue
			}
			v := cw[i*n+j] / math.Sqrt(di*dj)
			if v > 1 {
				v = 1
			} else if v < -1 {
				v = -1
			}
			ow[i*n+j] = v
		}
	}
	return out, nil
}

// covLayout 归一输入布局:返回 (变量数, 观测量, 变量×观测 的连续数组)。
func covLayout(x *Array, rowvar bool) (int, int, *Array, error) {
	switch x.NDim() {
	case 1:
		obs, err := x.Reshape([]int{1, x.Size()})
		if err != nil {
			return 0, 0, nil, err
		}
		return 1, x.Size(), obs, nil
	case 2:
		rows, cols := x.Shape()[0], x.Shape()[1]
		if rowvar {
			return rows, cols, x, nil
		}
		// 列是变量:转置
		tr, err := x.Transpose()
		if err != nil {
			return 0, 0, nil, err
		}
		return cols, rows, tr.AsContiguous(), nil
	default:
		return 0, 0, nil, fmt.Errorf("num: cov: 只支持 1-D/2-D(得 %d 维): %w", x.NDim(), ErrShape)
	}
}

func absOfInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

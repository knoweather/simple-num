package num

import (
	"fmt"
	"math"
)

// —— 统计面补齐(Step 15):bincount(weights) / histogram2d ——

// BincountWeights ≡ np.bincount(x, weights, minlength):
// 无 weights 时等价 Bincount(int64 计数);给 weights 时返回 **float64 加权和**。
func BincountWeights(x, weights *Array, minlength int) (*Array, error) {
	if weights == nil {
		return Bincount(x, minlength)
	}
	if minlength < 0 {
		return nil, fmt.Errorf("num: bincount(minlength=%d) 为负: %w", minlength, ErrShape)
	}
	if x.NDim() != 1 || weights.NDim() != 1 {
		return nil, fmt.Errorf("num: bincount: 输入须均为 1-D: %w", ErrShape)
	}
	if x.Size() != weights.Size() {
		return nil, fmt.Errorf("num: bincount: x 长度 %d ≠ weights 长度 %d: %w",
			x.Size(), weights.Size(), ErrShape)
	}
	switch x.Dtype().Kind {
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
	default:
		return nil, fmt.Errorf("num: bincount 输入须整型,得 %s(numpy 同拒): %w", x.Dtype(), ErrDtype)
	}
	vals, err := idxInts(x)
	if err != nil {
		return nil, err
	}
	wf, err := transformCopy(weights.AsContiguous().Flatten(), []int{weights.Size()}, DFloat64)
	if err != nil {
		return nil, err
	}
	maxv := -1
	for _, v := range vals {
		if v < 0 {
			return nil, fmt.Errorf("num: bincount 含负值 %d(numpy 同拒): %w", v, ErrIndex)
		}
		if int(v) > maxv {
			maxv = int(v)
		}
	}
	L := maxv + 1
	if minlength > L {
		L = minlength
	}
	out, err := New([]int{L}, DFloat64)
	if err != nil {
		return nil, err
	}
	if L == 0 {
		return out, nil
	}
	ow := win[float64](engineWindow(out))
	ww := win[float64](engineWindow(wf))
	for i, v := range vals {
		ow[v] += ww[i]
	}
	return out, nil
}

// Histogram2D ≡ np.histogram2d(x, y, bins, range):返回 (H, xedges, yedges)。
//
// H 形状为 (nx, ny)(x 分箱在前)、float64;range 传 NaN 表示按数据自动取。
func Histogram2D(x, y *Array, nx, ny int, xlo, xhi, ylo, yhi float64) ([]*Array, error) {
	if nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("num: histogram2d: bins 须为正(得 %d,%d): %w", nx, ny, ErrValue)
	}
	if x.NDim() != 1 || y.NDim() != 1 {
		return nil, fmt.Errorf("num: histogram2d: x/y 须为 1-D: %w", ErrShape)
	}
	if x.Size() != y.Size() {
		return nil, fmt.Errorf("num: histogram2d: x/y 长度不等(%d vs %d): %w", x.Size(), y.Size(), ErrShape)
	}
	xf, err := transformCopy(x.AsContiguous(), x.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	yf, err := transformCopy(y.AsContiguous(), y.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	xv := win[float64](engineWindow(xf))
	yv := win[float64](engineWindow(yf))
	autoRange := func(v []float64, lo, hi float64) (float64, float64) {
		if !math.IsNaN(lo) && !math.IsNaN(hi) {
			return lo, hi
		}
		l, h, ok := minMaxF64(v)
		if !ok {
			return 0, 1
		}
		return l, h
	}
	xlo, xhi = autoRange(xv, xlo, xhi)
	ylo, yhi = autoRange(yv, ylo, yhi)
	if xhi < xlo || yhi < ylo {
		return nil, fmt.Errorf("num: histogram2d: hi < lo(%.3g,%.3g / %.3g,%.3g): %w",
			xlo, xhi, ylo, yhi, ErrValue)
	}
	mkEdges := func(lo, hi float64, n int) (*Array, error) {
		e, err := New([]int{n + 1}, DFloat64)
		if err != nil {
			return nil, err
		}
		w := win[float64](engineWindow(e))
		step := (hi - lo) / float64(n)
		for i := 0; i <= n; i++ {
			w[i] = lo + float64(i)*step
		}
		w[n] = hi
		return e, nil
	}
	xedges, err := mkEdges(xlo, xhi, nx)
	if err != nil {
		return nil, err
	}
	yedges, err := mkEdges(ylo, yhi, ny)
	if err != nil {
		return nil, err
	}
	H, err := New([]int{nx, ny}, DFloat64)
	if err != nil {
		return nil, err
	}
	hw := win[float64](engineWindow(H))
	bin := func(v, lo, hi float64, n int) (int, bool) {
		if v != v || v < lo || v > hi {
			return 0, false
		}
		b := int((v - lo) * float64(n) / (hi - lo))
		if b >= n {
			b = n - 1
		}
		if b < 0 {
			b = 0
		}
		return b, true
	}
	for i := range xv {
		bx, ok := bin(xv[i], xlo, xhi, nx)
		if !ok {
			continue
		}
		by, ok := bin(yv[i], ylo, yhi, ny)
		if !ok {
			continue
		}
		hw[bx*ny+by]++
	}
	return []*Array{H, xedges, yedges}, nil
}

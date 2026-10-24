package num

import (
	"fmt"
	"math"
	"math/cmplx"
)

// —— 本轮补全(Step 39):hfft/ihfft + matrix_rank/cond/multi_dot/vdot ——

// HFFT ≡ np.fft.hfft:输入是 Hermitian 半谱(复数),输出实信号。
// n ≤ 0 时取默认长度 2*(m-1),与 numpy 一致。
func HFFT(a *Array, n int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: hfft: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 1 {
		return nil, fmt.Errorf("num: hfft: 0-d 输入不支持: %w", ErrShape)
	}
	if n <= 0 {
		n = 2 * (a.Shape()[a.NDim()-1] - 1)
	}
	if n < 1 {
		return nil, fmt.Errorf("num: hfft: 输出长度 %d 非法: %w", n, ErrValue)
	}
	// numpy 的关系:hfft(a, n) = n · irfft(conj(a' ), n),
	// 其中 a' 是补齐/截断到 n/2+1 的半谱
	c, err := Conj(a)
	if err != nil {
		return nil, err
	}
	half := n/2 + 1
	if got := a.Shape()[a.NDim()-1]; got != half {
		c, err = resizeLastAxis(c, half)
		if err != nil {
			return nil, err
		}
	}
	r, err := IRFFT(c, n)
	if err != nil {
		return nil, err
	}
	scale, err := FromSlice([]float64{float64(n)})
	if err != nil {
		return nil, err
	}
	return Multiply(r, scale)
}

// IHFFT ≡ np.fft.ihfft:输入实信号,输出 Hermitian 半谱(共轭的 rfft / n)。
// n ≤ 0 时取输入长度;n 更大时补零、更小时截断(与 numpy 同)。
func IHFFT(a *Array, n int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: ihfft: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 1 {
		return nil, fmt.Errorf("num: ihfft: 0-d 输入不支持: %w", ErrShape)
	}
	last := a.Shape()[a.NDim()-1]
	if n <= 0 {
		n = last
	}
	if n < 1 {
		return nil, fmt.Errorf("num: ihfft: 长度 %d 非法: %w", n, ErrValue)
	}
	src := a
	if n != last {
		var err error
		src, err = resizeLastAxis(a, n)
		if err != nil {
			return nil, err
		}
	}
	r, err := RFFT(src)
	if err != nil {
		return nil, err
	}
	cj, err := Conj(r)
	if err != nil {
		return nil, err
	}
	den, err := FromSlice([]float64{float64(n)})
	if err != nil {
		return nil, err
	}
	return Div(cj, den)
}

// MatrixRank ≡ np.linalg.matrix_rank:按 SVD 统计"显著"奇异值个数。
// 阈值取 max(m,n)·eps·smax(与 numpy 的默认 tol 同口径)。
func MatrixRank(a *Array) (int, error) {
	if a == nil {
		return 0, fmt.Errorf("num: matrix_rank: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() == 0 {
		return 0, fmt.Errorf("num: matrix_rank: 0-d 输入不支持: %w", ErrShape)
	}
	// 1-D 视为 1×n
	m := a
	if a.NDim() == 1 {
		r, err := a.Reshape([]int{1, a.Size()})
		if err != nil {
			return 0, err
		}
		m = r
	}
	s, err := Svdvals(m)
	if err != nil {
		return 0, err
	}
	sv := win[float64](engineWindow(s.AsContiguous()))
	if len(sv) == 0 {
		return 0, nil
	}
	rows := a.Shape()[0]
	cols := a.Size() / rows
	tol := float64(maxInt(rows, cols)) * 2.220446049250313e-16 * sv[0]
	rank := 0
	for _, v := range sv {
		if v > tol {
			rank++
		}
	}
	return rank, nil
}

// Cond ≡ np.linalg.cond:矩阵的条件数。
//
// ord 取值:"2"(默认,= smax/smin)、"1"、"inf"、"fro"、"-2"(= smin/smax)。
// 奇异矩阵的 "2"/"-2" 返回 +Inf / 0 由自然语义决定。
func Cond(a *Array, ord string) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("num: cond: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return 0, fmt.Errorf("num: cond: 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	switch ord {
	case "", "2", "-2":
		s, err := Svdvals(a)
		if err != nil {
			return 0, err
		}
		sv := win[float64](engineWindow(s.AsContiguous()))
		if len(sv) == 0 {
			return 0, fmt.Errorf("num: cond: 空矩阵: %w", ErrShape)
		}
		smax, smin := sv[0], sv[len(sv)-1]
		if ord == "-2" {
			if smax == 0 {
				return 0, nil
			}
			return smin / smax, nil
		}
		if smin == 0 {
			return math.Inf(1), nil
		}
		return smax / smin, nil
	case "1", "inf", "fro":
		na, err := Norm(a, ord)
		if err != nil {
			return 0, err
		}
		inv, err := Inv(a)
		if err != nil {
			return 0, err
		}
		ni, err := Norm(inv, ord)
		if err != nil {
			return 0, err
		}
		return na * ni, nil
	default:
		return 0, fmt.Errorf("num: cond: 不支持的 ord %q(支持 2/1/inf/fro/-2): %w", ord, ErrValue)
	}
}

// MultiDot ≡ np.linalg.multi_dot:链式矩阵乘,等价于从右到左两两相乘。
// (numpy 会选最优结合顺序以省浮点运算,这里按从左到右结合 —— 结果相同。)
func MultiDot(arrays ...*Array) (*Array, error) {
	if len(arrays) < 2 {
		return nil, fmt.Errorf("num: multi_dot: 至少 2 个数组(得 %d): %w", len(arrays), ErrShape)
	}
	cur, err := MatMul(arrays[0], arrays[1])
	if err != nil {
		return nil, err
	}
	for _, a := range arrays[2:] {
		cur, err = MatMul(cur, a)
		if err != nil {
			return nil, err
		}
	}
	return cur, nil
}

// Vdot ≡ np.vdot:展平后的内积,第一个参数取共轭。
func Vdot(a, b *Array) (*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: vdot: 入参为 nil: %w", ErrDtype)
	}
	if a.Size() != b.Size() {
		return nil, fmt.Errorf("num: vdot: 元素数 %d ≠ %d: %w", a.Size(), b.Size(), ErrShape)
	}
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, err
	}
	af, err := transformCopy(a.AsContiguous(), a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	bf, err := transformCopy(b.AsContiguous(), b.Shape(), dt)
	if err != nil {
		return nil, err
	}
	if isCplxKind(dt.Kind) {
		aw := win[complex128](engineWindow(af))
		bw := win[complex128](engineWindow(bf))
		var s complex128
		for i := range aw {
			s += cmplx.Conj(aw[i]) * bw[i]
		}
		out, err := New([]int{}, DComplex128)
		if err != nil {
			return nil, err
		}
		return out, out.SetItem([]int{}, s)
	}
	fw, err := transformCopy(af, af.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	gw, err := transformCopy(bf, bf.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	x := win[float64](engineWindow(fw))
	y := win[float64](engineWindow(gw))
	sum := 0.0
	for i := range x {
		sum += x[i] * y[i]
	}
	out, err := New([]int{}, DFloat64)
	if err != nil {
		return nil, err
	}
	return out, out.SetItem([]int{}, sum)
}

// resizeLastAxis 把数组末轴调整到 m(超出截断、不足补零),dtype 不变。
func resizeLastAxis(a *Array, m int) (*Array, error) {
	if a.NDim() < 1 {
		return nil, fmt.Errorf("num: 末轴调整: 需要至少 1 维: %w", ErrShape)
	}
	last := a.Shape()[a.NDim()-1]
	if last == m {
		return a, nil
	}
	if m < 0 {
		return nil, fmt.Errorf("num: 末轴调整: 目标长度 %d 非法: %w", m, ErrValue)
	}
	shape := append([]int(nil), a.Shape()...)
	shape[len(shape)-1] = m
	out, err := New(shape, a.Dtype())
	if err != nil {
		return nil, err
	}
	es := a.Dtype().Itemsize
	ac := a.AsContiguous()
	ab := engineWindow(ac)
	ob := engineWindow(out)
	rows := prodShape(a.Shape()[:len(a.Shape())-1])
	keep := last
	if m < keep {
		keep = m
	}
	for r := 0; r < rows; r++ {
		copy(ob[r*m*es:(r*m+keep)*es], ab[r*last*es:(r*last+keep)*es])
	}
	return out, nil
}

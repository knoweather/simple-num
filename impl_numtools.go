package num

import (
	"fmt"
	"math"
)

// —— 数值小工具收尾(Step 28):polyfit / unwrap ——

// Polyfit ≡ np.polyfit(x, y, deg):最小二乘多项式拟合,返回**降幂**系数(长度 deg+1)。
//
// 用**法方程** (VᵀV)·c = Vᵀy 加 LU 求解;`deg` 较小时足够稳定,
// 但法方程会平方条件数 —— 高次或 x 跨度很大时应改用正交多项式拟合(未实现)。
func Polyfit(x, y *Array, deg int) (*Array, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("num: polyfit: 入参为 nil: %w", ErrDtype)
	}
	if x.NDim() != 1 || y.NDim() != 1 {
		return nil, fmt.Errorf("num: polyfit: x/y 须为 1-D(得 %d/%d 维): %w", x.NDim(), y.NDim(), ErrShape)
	}
	if x.Size() != y.Size() {
		return nil, fmt.Errorf("num: polyfit: x 长度 %d ≠ y 长度 %d: %w", x.Size(), y.Size(), ErrShape)
	}
	if deg < 0 {
		return nil, fmt.Errorf("num: polyfit: deg 须非负(得 %d): %w", deg, ErrValue)
	}
	n := x.Size()
	if n < deg+1 {
		return nil, fmt.Errorf("num: polyfit: 点数 %d 少于待定系数 %d: %w", n, deg+1, ErrShape)
	}
	xf, err := x.AsContiguous().Flatten().Reshape([]int{n})
	if err != nil {
		return nil, err
	}
	yf, err := y.AsContiguous().Flatten().Reshape([]int{n})
	if err != nil {
		return nil, err
	}
	xv, err := setF64KeepNaN(xf, "polyfit")
	if err != nil {
		return nil, err
	}
	yv, err := setF64KeepNaN(yf, "polyfit")
	if err != nil {
		return nil, err
	}
	// 范德蒙德矩阵(降幂):V[i][j] = x_i^(deg-j)
	p := deg + 1
	v := make([]float64, n*p)
	for i := 0; i < n; i++ {
		acc := 1.0
		for j := p - 1; j >= 0; j-- {
			v[i*p+j] = acc
			acc *= xv[i]
		}
	}
	// A = VᵀV(p×p), rhs = Vᵀy
	a := make([]float64, p*p)
	rhs := make([]float64, p)
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			s := 0.0
			for k := 0; k < n; k++ {
				s += v[k*p+i] * v[k*p+j]
			}
			a[i*p+j] = s
		}
		s := 0.0
		for k := 0; k < n; k++ {
			s += v[k*p+i] * yv[k]
		}
		rhs[i] = s
	}
	lu, piv, _, ok := luDecompose(a, p)
	if !ok {
		return nil, fmt.Errorf("num: polyfit: 法方程奇异(数据退化或点太少): %w", ErrValue)
	}
	c := luSolve(lu, piv, p, rhs)
	out, err := New([]int{p}, DFloat64)
	if err != nil {
		return nil, err
	}
	copy(win[float64](engineWindow(out)), c)
	return out, nil
}

// Unwrap ≡ np.unwrap(p, discont, period, axis=-1):
// 当相邻差的绝对值超过 discont 时,按 period 的整数倍补偿,使相位连续。
//
// period <= 0 取 2π;discont <= 0 取 period/2(numpy 的默认口径)。
func Unwrap(p *Array, discont, period float64) (*Array, error) {
	if p == nil {
		return nil, fmt.Errorf("num: unwrap: 数组为 nil: %w", ErrDtype)
	}
	if p.NDim() == 0 {
		return nil, fmt.Errorf("num: unwrap: 0-d 输入不支持: %w", ErrShape)
	}
	if period <= 0 {
		period = 2 * math.Pi
	}
	if discont <= 0 {
		discont = period / 2
	}
	flat := p.AsContiguous()
	nf, err := transformCopy(flat, flat.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	d := nf.NDim() - 1
	rows, m, n, err := axisRows(nf, d)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, fmt.Errorf("num: unwrap: 最后一轴长度为 0: %w", ErrShape)
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	out, err := New(nf.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	line := make([]float64, n)
	for r := 0; r < m; r++ {
		for i := 0; i < n; i++ {
			line[i] = win[float64](sb[(r*n+i)*es : (r*n+i+1)*es])[0]
		}
		// 注意:差值必须取**原始序列**的相邻差(numpy 用 diff(p) 计算补偿量),
		// 若拿已补偿的值再求差,补偿会被重复累计。
		orig := make([]float64, n)
		copy(orig, line)
		acc := 0.0
		for i := 1; i < n; i++ {
			delta := orig[i] - orig[i-1]
			if math.Abs(delta) > discont {
				acc -= period * math.Round(delta/period)
			}
			line[i] = orig[i] + acc
		}
		for i := 0; i < n; i++ {
			ow[r*n+i] = line[i]
		}
	}
	return out, nil
}

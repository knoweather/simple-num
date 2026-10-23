package num

import (
	"fmt"
	"math"
)

// —— 线性代数核心(Step 25):LU 分解支撑的 det / inv / solve / norm ——

// luDecompose 对 n×n 矩阵做**部分选主元**的 LU 分解,原地返回 LU 与行交换信息。
// 返回 (lu 平坦数组, 排列 piv, 行列式符号 sign, 是否可逆)。
func luDecompose(m []float64, n int) ([]float64, []int, float64, bool) {
	lu := make([]float64, len(m))
	copy(lu, m)
	piv := make([]int, n)
	for i := range piv {
		piv[i] = i
	}
	sign := 1.0
	ok := true
	for k := 0; k < n; k++ {
		// 选主元
		p := k
		best := math.Abs(lu[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := math.Abs(lu[i*n+k]); v > best {
				best, p = v, i
			}
		}
		if best == 0 {
			ok = false
			continue
		}
		if p != k {
			for j := 0; j < n; j++ {
				lu[k*n+j], lu[p*n+j] = lu[p*n+j], lu[k*n+j]
			}
			piv[k], piv[p] = piv[p], piv[k]
			sign = -sign
		}
		pivot := lu[k*n+k]
		for i := k + 1; i < n; i++ {
			f := lu[i*n+k] / pivot
			lu[i*n+k] = f
			for j := k + 1; j < n; j++ {
				lu[i*n+j] -= f * lu[k*n+j]
			}
		}
	}
	return lu, piv, sign, ok
}

// Det ≡ np.linalg.det:方阵行列式。
func Det(a *Array) (float64, error) {
	n, flat, err := squareF64(a, "det")
	if err != nil {
		return 0, err
	}
	if n == 0 {
		return 1, nil
	}
	lu, _, sign, ok := luDecompose(flat, n)
	if !ok {
		return 0, nil // 奇异
	}
	d := sign
	for i := 0; i < n; i++ {
		d *= lu[i*n+i]
	}
	return d, nil
}

// Inv ≡ np.linalg.inv:方阵求逆。
func Inv(a *Array) (*Array, error) {
	n, flat, err := squareF64(a, "inv")
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return New([]int{0, 0}, DFloat64)
	}
	lu, piv, _, ok := luDecompose(flat, n)
	if !ok {
		return nil, fmt.Errorf("num: inv: 矩阵奇异(不可逆): %w", ErrValue)
	}
	out, err := New([]int{n, n}, DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	e := make([]float64, n)
	for col := 0; col < n; col++ {
		for i := range e {
			e[i] = 0
		}
		e[col] = 1
		x := luSolve(lu, piv, n, e)
		for i := 0; i < n; i++ {
			ow[i*n+col] = x[i]
		}
	}
	return out, nil
}

// Solve ≡ np.linalg.solve(a, b):解 ax = b。b 可为 (n,) 或 (n,k)。
func Solve(a, b *Array) (*Array, error) {
	n, flat, err := squareF64(a, "solve")
	if err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("num: solve: b 为 nil: %w", ErrDtype)
	}
	if b.NDim() < 1 || b.NDim() > 2 {
		return nil, fmt.Errorf("num: solve: b 须为 1-D/2-D(得 %d 维): %w", b.NDim(), ErrShape)
	}
	if b.Shape()[0] != n {
		return nil, fmt.Errorf("num: solve: a 的阶 %d ≠ b 的首维 %d: %w", n, b.Shape()[0], ErrShape)
	}
	k := 1
	if b.NDim() == 2 {
		k = b.Shape()[1]
	}
	bf, err := transformCopy(b.AsContiguous(), b.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	bw := win[float64](engineWindow(bf))
	lu, piv, _, ok := luDecompose(flat, n)
	if !ok {
		return nil, fmt.Errorf("num: solve: 矩阵奇异: %w", ErrValue)
	}
	out, err := New(b.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	col := make([]float64, n)
	for j := 0; j < k; j++ {
		for i := 0; i < n; i++ {
			col[i] = bw[i*k+j]
		}
		x := luSolve(lu, piv, n, col)
		for i := 0; i < n; i++ {
			ow[i*k+j] = x[i]
		}
	}
	return out, nil
}

// luSolve 用 LU 分解解一组(前代 + 回代)。
func luSolve(lu []float64, piv []int, n int, b []float64) []float64 {
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := b[piv[i]]
		for j := 0; j < i; j++ {
			s -= lu[i*n+j] * y[j]
		}
		y[i] = s
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := y[i]
		for j := i + 1; j < n; j++ {
			s -= lu[i*n+j] * x[j]
		}
		x[i] = s / lu[i*n+i]
	}
	return x
}

// squareF64 校验方阵并取出 float64 平坦数据。
func squareF64(a *Array, op string) (int, []float64, error) {
	if a == nil {
		return 0, nil, fmt.Errorf("num: %s: 数组为 nil: %w", op, ErrDtype)
	}
	if a.NDim() != 2 {
		return 0, nil, fmt.Errorf("num: %s: 需要 2-D 方阵(得 %d 维): %w", op, a.NDim(), ErrShape)
	}
	r, c := a.Shape()[0], a.Shape()[1]
	if r != c {
		return 0, nil, fmt.Errorf("num: %s: 需要方阵(得 %dx%d): %w", op, r, c, ErrShape)
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return 0, nil, err
	}
	return r, win[float64](engineWindow(f)), nil
}

// Norm ≡ np.linalg.norm:向量范数(ord "0"/"1"/"2"/"inf"/"-inf")
// 与矩阵范数(ord "fro"/"1"/"inf")。
func Norm(a *Array, ord string) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("num: norm: 数组为 nil: %w", ErrDtype)
	}
	switch a.NDim() {
	case 1:
		f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
		if err != nil {
			return 0, err
		}
		w := win[float64](engineWindow(f))
		switch ord {
		case "1":
			s := 0.0
			for _, v := range w {
				s += math.Abs(v)
			}
			return s, nil
		case "2", "":
			s := 0.0
			for _, v := range w {
				s += v * v
			}
			return math.Sqrt(s), nil
		case "inf":
			m := 0.0
			for _, v := range w {
				if av := math.Abs(v); av > m {
					m = av
				}
			}
			return m, nil
		case "-inf":
			if len(w) == 0 {
				return 0, nil
			}
			m := math.Abs(w[0])
			for _, v := range w[1:] {
				if av := math.Abs(v); av < m {
					m = av
				}
			}
			return m, nil
		case "0":
			c := 0.0
			for _, v := range w {
				if v != 0 {
					c++
				}
			}
			return c, nil
		default:
			return 0, fmt.Errorf("num: norm: 向量不支持 ord=%q: %w", ord, ErrValue)
		}
	case 2:
		r, c := a.Shape()[0], a.Shape()[1]
		f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
		if err != nil {
			return 0, err
		}
		w := win[float64](engineWindow(f))
		switch ord {
		case "fro", "F", "":
			s := 0.0
			for _, v := range w {
				s += v * v
			}
			return math.Sqrt(s), nil
		case "1":
			// 列和绝对值的最大者
			m := 0.0
			for j := 0; j < c; j++ {
				s := 0.0
				for i := 0; i < r; i++ {
					s += math.Abs(w[i*c+j])
				}
				if s > m {
					m = s
				}
			}
			return m, nil
		case "inf":
			m := 0.0
			for i := 0; i < r; i++ {
				s := 0.0
				for j := 0; j < c; j++ {
					s += math.Abs(w[i*c+j])
				}
				if s > m {
					m = s
				}
			}
			return m, nil
		case "2", "-2":
			// 矩阵 2-范数 = 最大奇异值;-2 范数 = 最小奇异值(需要 SVD)
			s, err := Svdvals(a)
			if err != nil {
				return 0, err
			}
			if s.Size() == 0 {
				return 0, nil
			}
			if ord == "2" {
				v, _ := s.Item(0)
				return v.(float64), nil
			}
			v, _ := s.Item(s.Size() - 1)
			return v.(float64), nil
		default:
			return 0, fmt.Errorf("num: norm: 矩阵不支持 ord=%q: %w", ord, ErrValue)
		}
	default:
		return 0, fmt.Errorf("num: norm: 只支持 1-D/2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
}

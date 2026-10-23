package num

import (
	"fmt"
	"math"
	"sort"
)

// —— 特征分解与奇异值分解(Step 27)——
//
// Eigvalsh / Eigh:对称矩阵用**循环 Jacobi 旋转**(数值稳定、实现简单),
// 只读下三角(与 numpy 的 UPLO='L' 口径一致),特征值**升序**返回。
// Svd / Svdvals:用**单边 Jacobi**(对列做正交化),数值精度优于 AᵀA 的路线,
// 奇异值**降序**返回,只给 thin SVD(U 为 m×k、Vt 为 k×n,k=min(m,n))。

// jacobiEigenSym 对对称矩阵做循环 Jacobi 分解(输入按行主序,内部会修改)。
// 返回升序特征值与特征向量(vecs 按行主序,v[k*n+i] 是第 i 个特征向量的第 k 个分量)。
func jacobiEigenSym(a []float64, n int) ([]float64, []float64) {
	vecs := make([]float64, n*n)
	for i := 0; i < n; i++ {
		vecs[i*n+i] = 1
	}
	if n == 1 {
		return []float64{a[0]}, vecs
	}
	const maxSweeps = 100
	for sweep := 0; sweep < maxSweeps; sweep++ {
		off := 0.0
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				off += a[p*n+q] * a[p*n+q]
			}
		}
		if off <= 1e-30 {
			break
		}
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p*n+q]
				if math.Abs(apq) < 1e-300 {
					continue
				}
				theta := (a[q*n+q] - a[p*n+p]) / (2 * apq)
				var t float64
				if theta == 0 {
					t = 1
				} else {
					t = 1 / (math.Abs(theta) + math.Sqrt(theta*theta+1))
					if theta < 0 {
						t = -t
					}
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				// A ← Jᵀ A J(只更新受影响的 p,q 两行两列)
				for k := 0; k < n; k++ {
					akp := a[k*n+p]
					akq := a[k*n+q]
					a[k*n+p] = c*akp - s*akq
					a[k*n+q] = s*akp + c*akq
				}
				for k := 0; k < n; k++ {
					apk := a[p*n+k]
					aqk := a[q*n+k]
					a[p*n+k] = c*apk - s*aqk
					a[q*n+k] = s*apk + c*aqk
				}
				for k := 0; k < n; k++ {
					vkp := vecs[k*n+p]
					vkq := vecs[k*n+q]
					vecs[k*n+p], vecs[k*n+q] = c*vkp-s*vkq, s*vkp+c*vkq
				}
			}
		}
	}
	vals := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = a[i*n+i]
	}
	// 升序排序(同时重排特征向量列)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(x, y int) bool { return vals[idx[x]] < vals[idx[y]] })
	sv := make([]float64, n)
	svv := make([]float64, n*n)
	for i, k := range idx {
		sv[i] = vals[k]
		for r := 0; r < n; r++ {
			svv[r*n+i] = vecs[r*n+k]
		}
	}
	return sv, svv
}

// symPart 校验方阵并取出**下三角**构造的对称矩阵(下三角 + 转置,
// 与 numpy eigh/eigvalsh 默认 UPLO='L' 一致)。
func symPart(a *Array, op string) (int, []float64, error) {
	n, flat, err := squareF64(a, op)
	if err != nil {
		return 0, nil, err
	}
	sym := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			v := flat[i*n+j] // 下三角
			sym[i*n+j] = v
			sym[j*n+i] = v
		}
	}
	return n, sym, nil
}

// Eigvalsh ≡ np.linalg.eigvalsh:对称矩阵特征值(升序)。
func Eigvalsh(a *Array) (*Array, error) {
	n, sym, err := symPart(a, "eigvalsh")
	if err != nil {
		return nil, err
	}
	vals, _ := jacobiEigenSym(sym, n)
	out, err := New([]int{n}, DFloat64)
	if err != nil {
		return nil, err
	}
	copy(win[float64](engineWindow(out)), vals)
	return out, nil
}

// Eigh ≡ np.linalg.eigh:返回 (特征值升序, 特征向量矩阵——**列**为特征向量)。
func Eigh(a *Array) ([]*Array, error) {
	n, sym, err := symPart(a, "eigh")
	if err != nil {
		return nil, err
	}
	vals, vecs := jacobiEigenSym(sym, n)
	valsOut, err := New([]int{n}, DFloat64)
	if err != nil {
		return nil, err
	}
	copy(win[float64](engineWindow(valsOut)), vals)
	vecOut, err := New([]int{n, n}, DFloat64)
	if err != nil {
		return nil, err
	}
	copy(win[float64](engineWindow(vecOut)), vecs)
	return []*Array{valsOut, vecOut}, nil
}

// Svdvals ≡ np.linalg.svd(..., compute_uv=False):奇异值(降序)。
func Svdvals(a *Array) (*Array, error) {
	_, s, _, err := svdCore(a)
	return s, err
}

// Svd ≡ np.linalg.svd(a, full_matrices=False):thin SVD,返回 (U, S, Vt)。
func Svd(a *Array) ([]*Array, error) {
	u, s, vt, err := svdCore(a)
	if err != nil {
		return nil, err
	}
	return []*Array{u, s, vt}, nil
}

// svdCore 单边 Jacobi SVD。A = U·diag(S)·Vt,thin 形式。
func svdCore(a *Array) (*Array, *Array, *Array, error) {
	if a == nil {
		return nil, nil, nil, fmt.Errorf("num: svd: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return nil, nil, nil, fmt.Errorf("num: svd: 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	m, n := a.Shape()[0], a.Shape()[1]
	if m == 0 || n == 0 {
		return nil, nil, nil, fmt.Errorf("num: svd: 空矩阵不支持: %w", ErrShape)
	}
	if m < n {
		// 转置后做,再交换 U/V
		tr, err := a.Transpose()
		if err != nil {
			return nil, nil, nil, err
		}
		u, s, vt, err := svdCore(tr)
		if err != nil {
			return nil, nil, nil, err
		}
		// A = (Aᵀ)ᵀ = (u s vt)ᵀ = vtᵀ · s · uᵀ
		uNew, err := vt.Transpose()
		if err != nil {
			return nil, nil, nil, err
		}
		vtNew, err := u.Transpose()
		if err != nil {
			return nil, nil, nil, err
		}
		return uNew, s, vtNew, nil
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, nil, nil, err
	}
	w := win[float64](engineWindow(f))
	// 按列取数据
	cols := make([][]float64, n)
	for j := 0; j < n; j++ {
		cols[j] = make([]float64, m)
		for i := 0; i < m; i++ {
			cols[j][i] = w[i*n+j]
		}
	}
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1
	}
	dot := func(x, y []float64) float64 {
		s := 0.0
		for i := range x {
			s += x[i] * y[i]
		}
		return s
	}
	const maxSweeps = 60
	for sweep := 0; sweep < maxSweeps; sweep++ {
		rotations := 0
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				alpha := dot(cols[p], cols[p])
				beta := dot(cols[q], cols[q])
				if alpha == 0 || beta == 0 {
					continue
				}
				gamma := dot(cols[p], cols[q])
				if math.Abs(gamma) <= 1e-15*math.Sqrt(alpha*beta) {
					continue
				}
				rotations++
				zeta := (beta - alpha) / (2 * gamma)
				var t float64
				if zeta == 0 {
					t = 1
				} else {
					t = 1 / (math.Abs(zeta) + math.Sqrt(1+zeta*zeta))
					if zeta < 0 {
						t = -t
					}
				}
				c := 1 / math.Sqrt(1+t*t)
				sn := c * t
				for k := 0; k < m; k++ {
					x := cols[p][k]
					y := cols[q][k]
					cols[p][k] = c*x - sn*y
					cols[q][k] = sn*x + c*y
				}
				for k := 0; k < n; k++ {
					x := v[k*n+p]
					y := v[k*n+q]
					v[k*n+p] = c*x - sn*y
					v[k*n+q] = sn*x + c*y
				}
			}
		}
		if rotations == 0 {
			break
		}
	}
	// 奇异值与排序(降序)
	sing := make([]float64, n)
	for j := 0; j < n; j++ {
		sing[j] = math.Sqrt(dot(cols[j], cols[j]))
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(x, y int) bool { return sing[idx[x]] > sing[idx[y]] })
	sOut, err := New([]int{n}, DFloat64)
	if err != nil {
		return nil, nil, nil, err
	}
	sw := win[float64](engineWindow(sOut))
	for i, k := range idx {
		sw[i] = sing[k]
	}
	// U:归一化后的列
	uOut, err := New([]int{m, n}, DFloat64)
	if err != nil {
		return nil, nil, nil, err
	}
	uw := win[float64](engineWindow(uOut))
	for i, k := range idx {
		sig := sing[k]
		for r := 0; r < m; r++ {
			if sig > 0 {
				uw[r*n+i] = cols[k][r] / sig
			} else {
				uw[r*n+i] = 0
			}
		}
	}
	// Vt = Vᵀ(按同样的列顺序)
	vtOut, err := New([]int{n, n}, DFloat64)
	if err != nil {
		return nil, nil, nil, err
	}
	vtw := win[float64](engineWindow(vtOut))
	for i, k := range idx {
		for r := 0; r < n; r++ {
			vtw[i*n+r] = v[r*n+k]
		}
	}
	return uOut, sOut, vtOut, nil
}

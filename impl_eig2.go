package num

import (
	"fmt"
	"math"
	"math/cmplx"
)

// —— 一般(非对称)特征值:Householder Hessenberg + 移位 QR + 反迭代求特征向量 ——
//
// 输出恒为 complex128(与 numpy 2.x 的 eig/eigvals 口径一致:实矩阵也返回复数)。
// 特征值顺序与 LAPACK 一致地"不作保证",调用方需要固定顺序时自行排序。

// groupNorm 返回一段复数的欧氏范数。
func groupNorm(v []complex128) float64 {
	s := 0.0
	for _, z := range v {
		s += real(z)*real(z) + imag(z)*imag(z)
	}
	return math.Sqrt(s)
}

// hessenbergInPlace 用 Householder 相似变换把 n×n 复矩阵化为上 Hessenberg。
func hessenbergInPlace(h []complex128, n int) {
	if n < 3 {
		return
	}
	v := make([]complex128, n)
	for k := 0; k < n-2; k++ {
		// 列 k 的 k+1..n-1 段
		for i := k + 1; i < n; i++ {
			v[i] = h[i*n+k]
		}
		norm := groupNorm(v[k+1:])
		if norm == 0 {
			continue
		}
		x0 := v[k+1]
		var phase complex128 = 1
		if x0 != 0 {
			phase = x0 / complex(cmplx.Abs(x0), 0)
		}
		alpha := -phase * complex(norm, 0)
		v[k+1] -= alpha
		vv := groupNorm(v[k+1:])
		if vv == 0 {
			continue
		}
		inv := 2 / (vv * vv)
		// 左乘 H ← (I - inv·v v*) H
		for c := k; c < n; c++ {
			var dot complex128
			for i := k + 1; i < n; i++ {
				dot += cmplx.Conj(v[i]) * h[i*n+c]
			}
			dot *= complex(inv, 0)
			for i := k + 1; i < n; i++ {
				h[i*n+c] -= dot * v[i]
			}
		}
		// 右乘 H ← H (I - inv·v v*)
		for r := 0; r < n; r++ {
			var dot complex128
			for i := k + 1; i < n; i++ {
				dot += h[r*n+i] * v[i]
			}
			dot *= complex(inv, 0)
			for i := k + 1; i < n; i++ {
				h[r*n+i] -= dot * cmplx.Conj(v[i])
			}
		}
	}
}

// qrEigvals 在 Hessenberg 矩阵上做移位 QR 迭代,返回全部特征值。
func qrEigvals(h []complex128, n int) []complex128 {
	eig := make([]complex128, n)
	if n == 0 {
		return eig
	}
	m := n
	iters := 0
	for m > 1 {
		// 找可收缩的次对角(从 m-2 往前)
		l := m - 1
		for l > 0 {
			hi := cmplx.Abs(h[(l-1)*n+l-1]) + cmplx.Abs(h[l*n+l])
			if hi == 0 {
				hi = 1
			}
			if cmplx.Abs(h[l*n+l-1]) <= 1e-15*hi {
				h[l*n+l-1] = 0
				break
			}
			l--
		}
		if l == m-1 {
			// 尾块已 1×1:直接取对角
			eig[m-1] = h[(m-1)*n+m-1]
			m--
			iters = 0
			continue
		}
		iters++
		if iters > 40 {
			// 异常移位:强行扰乱尾部对角,帮助脱离停滞
			perturb := complex(1e-3*(1+cmplx.Abs(h[(m-1)*n+m-2])), 1e-3)
			h[(m-1)*n+m-1] += perturb
			iters = 0
		}
		// Wilkinson 移位:尾 2×2 块里离 h[m-1][m-1] 更近的那个特征值
		mu := wilkinsonShift(h, n, m-1)
		// 对活动块 l..m-1 做一次带移位的 QR 步
		qrStepBlock(h, n, l, m-1, mu)
	}
	if n > 0 {
		eig[0] = h[0]
	}
	return eig
}

// wilkinsonShift 取尾 2×2 块中更接近右下角元素的特征值。
func wilkinsonShift(h []complex128, n, m int) complex128 {
	if m < 1 {
		return h[0]
	}
	d := h[m*n+m]
	if m == 0 {
		return d
	}
	a := h[(m-1)*n+m-1]
	b := h[(m-1)*n+m]
	c := h[m*n+m-1]
	tr := a + d
	det := a*d - b*c
	disc := cmplx.Sqrt(tr*tr - 4*det)
	l1 := (tr + disc) / 2
	l2 := (tr - disc) / 2
	if cmplx.Abs(l1-d) <= cmplx.Abs(l2-d) {
		return l1
	}
	return l2
}

// qrStepBlock 对 h 的 [lo..hi]×[lo..hi] 子块做一次显式移位的 QR 迭代:
// 分解 (块 - μI) = Q·R,再写回 R·Q + μI。
func qrStepBlock(h []complex128, n, lo, hi int, mu complex128) {
	size := hi - lo + 1
	if size < 2 {
		return
	}
	a := make([]complex128, size*size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			a[i*size+j] = h[(lo+i)*n+lo+j]
		}
		a[i*size+i] -= mu
	}
	q, r := qrComplex(a, size)
	// 新的块 = R·Q + μI
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			var s complex128
			for k := 0; k < size; k++ {
				s += r[i*size+k] * q[k*size+j]
			}
			if i == j {
				s += mu
			}
			h[(lo+i)*n+lo+j] = s
		}
	}
	// 结构性清零(保持 Hessenberg 形态,避免浮点残渣污染判据)
	for i := 2; i < size; i++ {
		for j := 0; j < i-1; j++ {
			h[(lo+i)*n+lo+j] = 0
		}
	}
}

// qrComplex 返回 Householder QR 分解(方阵,复矩阵)。
func qrComplex(a []complex128, n int) (q, r []complex128) {
	r = make([]complex128, n*n)
	copy(r, a)
	q = make([]complex128, n*n)
	for i := 0; i < n; i++ {
		q[i*n+i] = 1
	}
	v := make([]complex128, n)
	for k := 0; k < n-1; k++ {
		for i := k; i < n; i++ {
			v[i] = r[i*n+k]
		}
		norm := groupNorm(v[k:])
		if norm == 0 {
			continue
		}
		x0 := v[k]
		var phase complex128 = 1
		if x0 != 0 {
			phase = x0 / complex(cmplx.Abs(x0), 0)
		}
		alpha := -phase * complex(norm, 0)
		v[k] -= alpha
		vv := groupNorm(v[k:])
		if vv == 0 {
			continue
		}
		inv := 2 / (vv * vv)
		for c := k; c < n; c++ {
			var dot complex128
			for i := k; i < n; i++ {
				dot += cmplx.Conj(v[i]) * r[i*n+c]
			}
			dot *= complex(inv, 0)
			for i := k; i < n; i++ {
				r[i*n+c] -= dot * v[i]
			}
		}
		for row := 0; row < n; row++ {
			var dot complex128
			for i := k; i < n; i++ {
				dot += q[row*n+i] * v[i]
			}
			dot *= complex(inv, 0)
			for i := k; i < n; i++ {
				q[row*n+i] -= dot * cmplx.Conj(v[i])
			}
		}
	}
	return q, r
}

// Eigvals ≡ np.linalg.eigvals:一般方阵的特征值(complex128)。
func Eigvals(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: eigvals: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return nil, fmt.Errorf("num: eigvals: 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	n := a.Shape()[0]
	if a.Shape()[1] != n {
		return nil, fmt.Errorf("num: eigvals: 需要方阵(得 %v): %w", a.Shape(), ErrShape)
	}
	if n == 0 {
		return New([]int{0}, DComplex128)
	}
	h, err := toComplexFlat(a)
	if err != nil {
		return nil, err
	}
	hessenbergInPlace(h, n)
	w := qrEigvals(h, n)
	out, err := New([]int{n}, DComplex128)
	if err != nil {
		return nil, err
	}
	copy(win[complex128](engineWindow(out)), w)
	return out, nil
}

// Eig ≡ np.linalg.eig:返回 (特征值, 特征向量矩阵)。特征向量按列存放、单位长度。
func Eig(a *Array) ([]*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: eig: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return nil, fmt.Errorf("num: eig: 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	n := a.Shape()[0]
	if a.Shape()[1] != n {
		return nil, fmt.Errorf("num: eig: 需要方阵(得 %v): %w", a.Shape(), ErrShape)
	}
	w, err := Eigvals(a)
	if err != nil {
		return nil, err
	}
	wv := win[complex128](engineWindow(w.AsContiguous()))
	m, err := toComplexFlat(a)
	if err != nil {
		return nil, err
	}
	// 反迭代:每次解 (A - λI + δI) x = x,再归一
	scale := 0.0
	for _, z := range m {
		scale += cmplx.Abs(z)
	}
	if scale == 0 {
		scale = 1
	}
	delta := complex(1e-12*scale, 0)
	vec := make([]complex128, n)
	vout, err := New([]int{n, n}, DComplex128)
	if err != nil {
		return nil, err
	}
	vw := win[complex128](engineWindow(vout))
	for j := 0; j < n; j++ {
		mat := make([]complex128, n*n)
		copy(mat, m)
		for i := 0; i < n; i++ {
			mat[i*n+i] -= wv[j]
			mat[i*n+i] += delta
		}
		for i := range vec {
			vec[i] = 1
		}
		lu, piv, ok := luComplex(mat, n)
		if !ok {
			// 极端退化:退回对角近似,保证仍返回单位向量
			for i := range vec {
				vec[i] = 0
			}
			vec[j%n] = 1
		} else {
			for it := 0; it < 4; it++ {
				x := luSolveComplex(lu, piv, vec, n)
				nrm := groupNorm(x)
				if nrm == 0 {
					break
				}
				for i := range vec {
					vec[i] = x[i] / complex(nrm, 0)
				}
			}
		}
		// 单位化
		nrm := groupNorm(vec)
		if nrm > 0 {
			for i := range vec {
				vec[i] /= complex(nrm, 0)
			}
		}
		for i := 0; i < n; i++ {
			vw[i*n+j] = vec[i]
		}
	}
	return []*Array{w, vout}, nil
}

// toComplexFlat 把数组转成 complex128 的扁平拷贝。
func toComplexFlat(a *Array) ([]complex128, error) {
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DComplex128)
	if err != nil {
		return nil, fmt.Errorf("num: 特征值: 该 dtype 不支持(%s): %w", a.Dtype(), ErrDtype)
	}
	src := win[complex128](engineWindow(f))
	out := make([]complex128, len(src))
	copy(out, src)
	return out, nil
}

// luComplex 复矩阵的列主元 LU 分解(原地);奇异时返回 ok=false。
func luComplex(a []complex128, n int) (lu []complex128, piv []int, ok bool) {
	piv = make([]int, n)
	lu = a
	for i := 0; i < n; i++ {
		piv[i] = i
	}
	for k := 0; k < n; k++ {
		p := k
		best := cmplx.Abs(lu[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cmplx.Abs(lu[i*n+k]); v > best {
				best = v
				p = i
			}
		}
		if best == 0 {
			return nil, nil, false
		}
		if p != k {
			for j := 0; j < n; j++ {
				lu[k*n+j], lu[p*n+j] = lu[p*n+j], lu[k*n+j]
			}
			piv[k], piv[p] = piv[p], piv[k]
		}
		pivot := lu[k*n+k]
		for i := k + 1; i < n; i++ {
			if lu[i*n+k] == 0 {
				continue
			}
			factor := lu[i*n+k] / pivot
			lu[i*n+k] = factor
			for j := k + 1; j < n; j++ {
				lu[i*n+j] -= factor * lu[k*n+j]
			}
		}
	}
	return lu, piv, true
}

// luSolveComplex 解 LUx = b。
func luSolveComplex(lu []complex128, piv []int, b []complex128, n int) []complex128 {
	x := make([]complex128, n)
	for i := 0; i < n; i++ {
		x[i] = b[piv[i]]
	}
	for i := 1; i < n; i++ {
		var s complex128
		for j := 0; j < i; j++ {
			s += lu[i*n+j] * x[j]
		}
		x[i] -= s
	}
	for i := n - 1; i >= 0; i-- {
		var s complex128
		for j := i + 1; j < n; j++ {
			s += lu[i*n+j] * x[j]
		}
		x[i] = (x[i] - s) / lu[i*n+i]
	}
	return x
}

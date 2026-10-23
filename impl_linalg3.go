package num

import (
	"fmt"
	"math"
)

// —— 线代补全(Step 35):QR / Cholesky / Pinv / Lstsq ——

// Cholesky ≡ np.linalg.cholesky:对称正定矩阵的下三角分解 L(满足 L·Lᵀ = a)。
// 非正定(出现非正主元)时 fail-loud,与 numpy 的 LinAlgError 对应。
func Cholesky(a *Array) (*Array, error) {
	n, flat, err := squareF64(a, "cholesky")
	if err != nil {
		return nil, err
	}
	l := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := flat[i*n+j]
			for k := 0; k < j; k++ {
				sum -= l[i*n+k] * l[j*n+k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("num: cholesky: 矩阵非正定(第 %d 个主元 %v ≤ 0): %w", i, sum, ErrValue)
				}
				l[i*n+i] = math.Sqrt(sum)
			} else {
				l[i*n+j] = sum / l[j*n+j]
			}
		}
	}
	out, err := New([]int{n, n}, DFloat64)
	if err != nil {
		return nil, err
	}
	copy(win[float64](engineWindow(out)), l)
	return out, nil
}

// QR ≡ np.linalg.qr(mode="reduced"):返回 (Q, R),Q 为 m×k、R 为 k×n(k=min(m,n))。
// 用 Householder 反射实现,数值上比 Gram-Schmidt 稳。
func QR(a *Array) ([]*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: qr: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return nil, fmt.Errorf("num: qr: 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	m, n := a.Shape()[0], a.Shape()[1]
	if m == 0 || n == 0 {
		return nil, fmt.Errorf("num: qr: 空矩阵不支持: %w", ErrShape)
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	r := make([]float64, m*n)
	copy(r, win[float64](engineWindow(f)))
	q := make([]float64, m*m)
	for i := 0; i < m; i++ {
		q[i*m+i] = 1
	}
	k := m
	if n < k {
		k = n
	}
	for j := 0; j < k; j++ {
		// 取列 j 的第 j..m-1 段构造 Householder 向量
		var norm float64
		for i := j; i < m; i++ {
			norm += r[i*n+j] * r[i*n+j]
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			continue
		}
		alpha := -math.Copysign(norm, r[j*n+j])
		v := make([]float64, m-j)
		for i := j; i < m; i++ {
			v[i-j] = r[i*n+j]
		}
		v[0] -= alpha
		var vv float64
		for _, x := range v {
			vv += x * x
		}
		if vv == 0 {
			continue
		}
		// 作用于 R 的剩余列
		for c := j; c < n; c++ {
			var dot float64
			for i := j; i < m; i++ {
				dot += v[i-j] * r[i*n+c]
			}
			f := 2 * dot / vv
			for i := j; i < m; i++ {
				r[i*n+c] -= f * v[i-j]
			}
		}
		// 累积到 Q(Q ← Q·H)
		for row := 0; row < m; row++ {
			var dot float64
			for i := j; i < m; i++ {
				dot += q[row*m+i] * v[i-j]
			}
			f := 2 * dot / vv
			for i := j; i < m; i++ {
				q[row*m+i] -= f * v[i-j]
			}
		}
	}
	qOut, err := New([]int{m, k}, DFloat64)
	if err != nil {
		return nil, err
	}
	qw := win[float64](engineWindow(qOut))
	for i := 0; i < m; i++ {
		copy(qw[i*k:(i+1)*k], q[i*m:i*m+k])
	}
	rOut, err := New([]int{k, n}, DFloat64)
	if err != nil {
		return nil, err
	}
	rw := win[float64](engineWindow(rOut))
	for i := 0; i < k; i++ {
		copy(rw[i*n:(i+1)*n], r[i*n:(i+1)*n])
	}
	return []*Array{qOut, rOut}, nil
}

// Pinv ≡ np.linalg.pinv:Moore-Penrose 伪逆(经 SVD;小于 rcond 的奇异值按 0 处理)。
func Pinv(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: pinv: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return nil, fmt.Errorf("num: pinv: 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	m, n := a.Shape()[0], a.Shape()[1]
	u, s, vt, err := svdCore(a)
	if err != nil {
		return nil, err
	}
	k := s.Size()
	sv := win[float64](engineWindow(s.AsContiguous()))
	cut := 0.0
	if k > 0 && sv[0] > 0 {
		cut = float64(maxInt(m, n)) * sv[0] * 2.220446049250313e-16
	}
	out, err := New([]int{n, m}, DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	uw := win[float64](engineWindow(u.AsContiguous()))
	vtw := win[float64](engineWindow(vt.AsContiguous()))
	// pinv[i][j] = Σ_k vt[k][i] * (1/s_k) * u[j][k]
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			sum := 0.0
			for t := 0; t < k; t++ {
				if sv[t] <= cut || sv[t] == 0 {
					continue
				}
				sum += vtw[t*n+i] * uw[j*k+t] / sv[t]
			}
			ow[i*m+j] = sum
		}
	}
	return out, nil
}

// Lstsq ≡ np.linalg.lstsq(a, b):最小二乘解。返回 (x, residuals, rank, s)。
//
// SVD 口径(与 numpy 一致):rcond = max(m,n)·eps;rank = 大于阈值的奇异值个数;
// residuals 仅在 m>n 且满秩时给出残差平方和(否则为空数组)。
func Lstsq(a, b *Array) ([]*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: lstsq: 入参为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 {
		return nil, fmt.Errorf("num: lstsq: a 需要 2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	m, n := a.Shape()[0], a.Shape()[1]
	if b.NDim() < 1 || b.NDim() > 2 {
		return nil, fmt.Errorf("num: lstsq: b 需要 1-D/2-D(得 %d 维): %w", b.NDim(), ErrShape)
	}
	if b.Shape()[0] != m {
		return nil, fmt.Errorf("num: lstsq: a 的行数 %d ≠ b 的首维 %d: %w", m, b.Shape()[0], ErrShape)
	}
	u, s, vt, err := svdCore(a)
	if err != nil {
		return nil, err
	}
	k := s.Size()
	sv := win[float64](engineWindow(s.AsContiguous()))
	cut := 0.0
	if k > 0 && sv[0] > 0 {
		cut = float64(maxInt(m, n)) * sv[0] * 2.220446049250313e-16
	}
	rank := 0
	for t := 0; t < k; t++ {
		if sv[t] > cut {
			rank++
		}
	}
	// 右端: 1-D → (m,1)
	nrhs := 1
	if b.NDim() == 2 {
		nrhs = b.Shape()[1]
	}
	bf, err := transformCopy(b.AsContiguous(), b.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	bw := win[float64](engineWindow(bf))
	uw := win[float64](engineWindow(u.AsContiguous()))
	vtw := win[float64](engineWindow(vt.AsContiguous()))
	xOut, err := New([]int{n, nrhs}, DFloat64)
	if err != nil {
		return nil, err
	}
	xw := win[float64](engineWindow(xOut))
	// x = V · diag(1/s) · Uᵀ b
	for col := 0; col < nrhs; col++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for t := 0; t < k; t++ {
				if sv[t] <= cut || sv[t] == 0 {
					continue
				}
				// (Uᵀb)[t] = Σ_j u[j][t]·b[j][col]
				utb := 0.0
				for j := 0; j < m; j++ {
					utb += uw[j*k+t] * bw[j*nrhs+col]
				}
				sum += vtw[t*n+i] * utb / sv[t]
			}
			xw[i*nrhs+col] = sum
		}
	}
	var x *Array
	if b.NDim() == 1 {
		x, err = xOut.Reshape([]int{n})
	} else {
		x = xOut
	}
	if err != nil {
		return nil, err
	}
	// residuals:仅在 m>n 且满秩时非空
	res, err := New([]int{0}, DFloat64)
	if err != nil {
		return nil, err
	}
	if m > n && rank == n {
		sums := make([]float64, nrhs)
		for col := 0; col < nrhs; col++ {
			total := 0.0
			for j := 0; j < m; j++ {
				// (Ax)[j] = Σ_i a[j][i]·x[i][col]
				ax := 0.0
				for i := 0; i < n; i++ {
					av, err := a.Item(j, i)
					if err != nil {
						return nil, err
					}
					ax += numToFloatOrZero(av) * xw[i*nrhs+col]
				}
				d := bw[j*nrhs+col] - ax
				total += d * d
			}
			sums[col] = total
		}
		res, err = New([]int{nrhs}, DFloat64)
		if err != nil {
			return nil, err
		}
		copy(win[float64](engineWindow(res)), sums)
	}
	rankOut, err := New([]int{}, DInt64)
	if err != nil {
		return nil, err
	}
	if err := rankOut.SetItem([]int{}, int64(rank)); err != nil {
		return nil, err
	}
	return []*Array{x, res, rankOut, s}, nil
}

// numToFloatOrZero 把数值元素转 float64(内部辅助,失败给 0)。
func numToFloatOrZero(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int64:
		return float64(x)
	case int32:
		return float64(x)
	case int16:
		return float64(x)
	case int8:
		return float64(x)
	case uint64:
		return float64(x)
	case uint32:
		return float64(x)
	case uint16:
		return float64(x)
	case uint8:
		return float64(x)
	case bool:
		if x {
			return 1
		}
		return 0
	default:
		return 0
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

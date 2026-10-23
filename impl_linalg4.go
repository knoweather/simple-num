package num

import (
	"fmt"
	"math/cmplx"
)

// —— 本轮补全(Step 38):复数矩阵求解 / 实输入 N 维 FFT ——

// isCplxKind 判断 dtype 是否为复数。
func isCplxKind(k Kind) bool { return k == Complex64 || k == Complex128 }

// solveComplexPair 复数线性方程组:返回 x 使 a·x = b(1-D 或 2-D 右端)。
func solveComplexPair(a, b *Array) (*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: solve: 入参为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return nil, fmt.Errorf("num: solve: a 须为方阵(得 %v): %w", a.Shape(), ErrShape)
	}
	n := a.Shape()[0]
	if b.NDim() < 1 || b.NDim() > 2 {
		return nil, fmt.Errorf("num: solve: b 须为 1-D/2-D(得 %d 维): %w", b.NDim(), ErrShape)
	}
	if b.Shape()[0] != n {
		return nil, fmt.Errorf("num: solve: a 的阶 %d ≠ b 的首维 %d: %w", n, b.Shape()[0], ErrShape)
	}
	flat, err := toComplexFlat(a)
	if err != nil {
		return nil, err
	}
	bf, err := transformCopy(b.AsContiguous(), b.Shape(), DComplex128)
	if err != nil {
		return nil, err
	}
	bw := win[complex128](engineWindow(bf))
	k := 1
	if b.NDim() == 2 {
		k = b.Shape()[1]
	}
	lu, piv, ok := luComplex(flat, n)
	if !ok {
		return nil, fmt.Errorf("num: solve: 矩阵奇异: %w", ErrValue)
	}
	out, err := New(b.Shape(), DComplex128)
	if err != nil {
		return nil, err
	}
	ow := win[complex128](engineWindow(out))
	col := make([]complex128, n)
	for j := 0; j < k; j++ {
		for i := 0; i < n; i++ {
			col[i] = bw[i*k+j]
		}
		x := luSolveComplex(lu, piv, col, n)
		for i := 0; i < n; i++ {
			ow[i*k+j] = x[i]
		}
	}
	return out, nil
}

// invComplex 复数矩阵求逆。
func invComplex(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: inv: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return nil, fmt.Errorf("num: inv: 需要方阵(得 %v): %w", a.Shape(), ErrShape)
	}
	n := a.Shape()[0]
	if n == 0 {
		return New([]int{0, 0}, DComplex128)
	}
	flat, err := toComplexFlat(a)
	if err != nil {
		return nil, err
	}
	lu, piv, ok := luComplex(flat, n)
	if !ok {
		return nil, fmt.Errorf("num: inv: 矩阵奇异: %w", ErrValue)
	}
	out, err := New([]int{n, n}, DComplex128)
	if err != nil {
		return nil, err
	}
	ow := win[complex128](engineWindow(out))
	e := make([]complex128, n)
	x := make([]complex128, n)
	for j := 0; j < n; j++ {
		for i := range e {
			e[i] = 0
		}
		e[j] = 1
		copy(x, luSolveComplex(lu, piv, e, n))
		for i := 0; i < n; i++ {
			ow[i*n+j] = x[i]
		}
	}
	return out, nil
}

// RFFT2 ≡ np.fft.rfft2:实数组最后两轴的 FFT,只保留最后一轴的 0..n/2 频点。
func RFFT2(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: rfft2: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: rfft2: 至少 2 维(得 %d 维): %w", a.NDim(), ErrShape)
	}
	return RFFTN(a, nil)
}

// RFFTN ≡ np.fft.rfftn:对给定各轴做 FFT,最后一轴只保留非负频率(长度 n/2+1)。
// axes 为空表示全部轴。
func RFFTN(a *Array, axes []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: rfftn: 数组为 nil: %w", ErrDtype)
	}
	nd := a.NDim()
	if len(axes) == 0 {
		axes = make([]int, nd)
		for i := range axes {
			axes[i] = i
		}
	}
	last := axes[len(axes)-1]
	lastAxis, err := normalizeAxis(a, last)
	if err != nil {
		return nil, fmt.Errorf("num: rfftn: %w", err)
	}
	full, err := FFTN(a, axes)
	if err != nil {
		return nil, err
	}
	keep := a.Shape()[lastAxis]/2 + 1
	idx := make([]Indexer, nd)
	for i := range idx {
		idx[i] = All()
	}
	idx[lastAxis] = Slice(0, keep)
	return full.Sub(idx...)
}

// IRFFT2 ≡ np.fft.irfft2:把半谱还原成实数组;s 为期望形状(末维默认 2*(n-1))。
func IRFFT2(a *Array, s []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: irfft2: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: irfft2: 至少 2 维(得 %d 维): %w", a.NDim(), ErrShape)
	}
	return IRFFTN(a, s, nil)
}

// IRFFTN ≡ np.fft.irfftn:半谱的 N 维逆变换(结果为实数)。
func IRFFTN(a *Array, s []int, axes []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: irfftn: 数组为 nil: %w", ErrDtype)
	}
	nd := a.NDim()
	if len(axes) == 0 {
		axes = make([]int, nd)
		for i := range axes {
			axes[i] = i
		}
	}
	lastAxis, err := normalizeAxis(a, axes[len(axes)-1])
	if err != nil {
		return nil, fmt.Errorf("num: irfftn: %w", err)
	}
	half := a.Shape()[lastAxis]
	if half < 1 {
		return nil, fmt.Errorf("num: irfftn: 末轴长度须 ≥ 1: %w", ErrShape)
	}
	var want []int
	if len(s) > 0 {
		if len(s) != len(axes) {
			return nil, fmt.Errorf("num: irfftn: s 长度 %d 与 axes 长度 %d 不同: %w", len(s), len(axes), ErrShape)
		}
		want = append([]int(nil), s...)
	}
	// 末轴完整长度
	n := 2 * (half - 1)
	if len(want) > 0 {
		n = want[len(want)-1]
	}
	if n < 1 {
		return nil, fmt.Errorf("num: irfftn: 还原长度 %d 非法: %w", n, ErrShape)
	}
	full, err := expandHalfSpectrum(a, lastAxis, n)
	if err != nil {
		return nil, err
	}
	inv, err := IFFTN(full, axes)
	if err != nil {
		return nil, err
	}
	// 取实部并裁剪到期望形状
	shape := append([]int(nil), a.Shape()...)
	shape[lastAxis] = n
	if len(want) > 0 {
		for i, ax := range axes {
			v, err := normalizeAxis(a, ax)
			if err != nil {
				return nil, fmt.Errorf("num: irfftn: %w", err)
			}
			shape[v] = want[i]
		}
	}
	out, err := New(shape, DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	iw := win[complex128](engineWindow(inv.AsContiguous()))
	copy(ow, make([]float64, len(ow)))
	// inv 的形状可能因 s 裁剪而不同,按元素顺序拷贝实部(长度取小者)
	src := iw
	if len(src) < len(ow) {
		return nil, fmt.Errorf("num: irfftn: 还原结果 %d 元素 < 期望 %d: %w", len(src), len(ow), ErrShape)
	}
	for i := range ow {
		ow[i] = real(src[i])
	}
	return out, nil
}

// expandHalfSpectrum 把轴 axis 上的半谱(长度 half)补成 Hermitian 全谱(长度 n)。
func expandHalfSpectrum(a *Array, axis, n int) (*Array, error) {
	shape := append([]int(nil), a.Shape()...)
	shape[axis] = n
	out, err := New(shape, DComplex128)
	if err != nil {
		return nil, err
	}
	src, err := transformCopy(a.AsContiguous(), a.Shape(), DComplex128)
	if err != nil {
		return nil, err
	}
	in := win[complex128](engineWindow(src))
	ow := win[complex128](engineWindow(out))
	half := a.Shape()[axis]
	// 逐"线"复制:线 = 固定其余维、沿 axis
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= a.Shape()[i]
	}
	inner := 1
	for i := axis + 1; i < a.NDim(); i++ {
		inner *= a.Shape()[i]
	}
	for o := 0; o < outer; o++ {
		for j := 0; j < inner; j++ {
			for k := 0; k < half; k++ {
				ow[(o*n+k)*inner+j] = in[(o*half+k)*inner+j]
			}
			// Hermitian 镜像:full[n-k] = conj(full[k])
			for k := 1; k < half; k++ {
				idx := n - k
				if idx < half || idx < 0 || idx >= n {
					continue
				}
				ow[(o*n+idx)*inner+j] = cmplx.Conj(in[(o*half+k)*inner+j])
			}
			// 偶数长度的 Nyquist 频点必须是实数
			if n%2 == 0 && n/2 < half {
				ow[(o*n+n/2)*inner+j] = complex(real(in[(o*half+n/2)*inner+j]), 0)
			}
		}
	}
	return out, nil
}

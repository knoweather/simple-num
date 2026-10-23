package num

import (
	"fmt"
	"math"
)

// —— 本轮补全(Step 37):N 维 FFT、nan_to_num、diag_indices ——

// fftAlongAxis 把指定轴搬到末尾做一维 FFT 后再搬回(axis 为已归一的非负下标)。
func fftAlongAxis(a *Array, axis int, inverse bool) (*Array, error) {
	nd := a.NDim()
	if nd == 0 {
		return nil, fmt.Errorf("num: fftn: 0-d 输入不支持: %w", ErrShape)
	}
	if axis < 0 || axis >= nd {
		return nil, fmt.Errorf("num: fftn: 轴 %d 越界(共 %d 维): %w", axis, nd, ErrIndex)
	}
	if axis == nd-1 {
		return fftAxis(a, inverse)
	}
	perm := make([]int, 0, nd)
	inv := make([]int, nd)
	for i := 0; i < nd; i++ {
		if i != axis {
			perm = append(perm, i)
		}
	}
	perm = append(perm, axis)
	for pos, p := range perm {
		inv[p] = pos
	}
	t, err := a.Transpose(perm...)
	if err != nil {
		return nil, err
	}
	f, err := fftAxis(t, inverse)
	if err != nil {
		return nil, err
	}
	return f.Transpose(inv...)
}

// FFTN ≡ np.fft.fftn:对给定各轴依次做一维 FFT。axes 为空表示所有轴。
// 归一化与 numpy 一致:正向不缩放,逆向按各轴长度之积缩放。
func FFTN(a *Array, axes []int) (*Array, error) { return fftNAxes(a, axes, false) }

// IFFTN ≡ np.fft.ifftn:FFTN 的逆。
func IFFTN(a *Array, axes []int) (*Array, error) { return fftNAxes(a, axes, true) }

func fftNAxes(a *Array, axes []int, inverse bool) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: fftn: 数组为 nil: %w", ErrDtype)
	}
	nd := a.NDim()
	if len(axes) == 0 {
		axes = make([]int, nd)
		for i := range axes {
			axes[i] = i
		}
	}
	norm := make([]int, len(axes))
	seen := make(map[int]bool, len(axes))
	for i, ax := range axes {
		v, err := normalizeAxis(a, ax)
		if err != nil {
			return nil, fmt.Errorf("num: fftn: %w", err)
		}
		if seen[v] {
			return nil, fmt.Errorf("num: fftn: 轴 %d 重复: %w", v, ErrShape)
		}
		seen[v] = true
		norm[i] = v
	}
	cur := a
	for _, ax := range norm {
		next, err := fftAlongAxis(cur, ax, inverse)
		if err != nil {
			return nil, err
		}
		cur = next
	}
	return cur, nil
}

// FFT2 ≡ np.fft.fft2:对最后两个轴做二维 FFT。
func FFT2(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: fft2: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: fft2: 至少 2 维(得 %d 维): %w", a.NDim(), ErrShape)
	}
	nd := a.NDim()
	return FFTN(a, []int{nd - 2, nd - 1})
}

// IFFT2 ≡ np.fft.ifft2:二维逆变换。
func IFFT2(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: ifft2: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: ifft2: 至少 2 维(得 %d 维): %w", a.NDim(), ErrShape)
	}
	nd := a.NDim()
	return IFFTN(a, []int{nd - 2, nd - 1})
}

// NanToNumOptions 是 NanToNum 的替换规则;零值即 numpy 的默认口径
// (NaN→0,+Inf→MaxFloat64,-Inf→-MaxFloat64)。
type NanToNumOptions struct {
	Nan       float64
	PosInf    float64
	NegInf    float64
	HasNan    bool
	HasPosInf bool
	HasNegInf bool
}

// NanToNum ≡ np.nan_to_num:把 NaN 与 ±Inf 替换成有限值。
func NanToNum(a *Array, opts NanToNumOptions) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: nan_to_num: 数组为 nil: %w", ErrDtype)
	}
	switch a.Dtype().Kind {
	case Float16, Float32, Float64:
	default:
		return nil, fmt.Errorf("num: nan_to_num: 需要浮点类型(得 %s): %w", a.Dtype(), ErrDtype)
	}
	nanV := 0.0
	if opts.HasNan {
		nanV = opts.Nan
	}
	posV := math.MaxFloat64
	if opts.HasPosInf {
		posV = opts.PosInf
	}
	negV := -math.MaxFloat64
	if opts.HasNegInf {
		negV = opts.NegInf
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	switch a.Dtype() {
	case DFloat16:
		// f16 经 f32 中转,写回仍保持 f16
		w := win[uint16](engineWindow(f))
		for i, raw := range w {
			v := f16ToF32(raw)
			switch {
			case v != v:
				w[i] = f32ToF16(float32(nanV))
			case math.IsInf(float64(v), 1):
				w[i] = f32ToF16(float32(posV))
			case math.IsInf(float64(v), -1):
				w[i] = f32ToF16(float32(negV))
			}
		}
		return f, nil
	case DFloat32:
		w := win[float32](engineWindow(f))
		for i, v := range w {
			switch {
			case v != v:
				w[i] = float32(nanV)
			case math.IsInf(float64(v), 1):
				w[i] = float32(posV)
			case math.IsInf(float64(v), -1):
				w[i] = float32(negV)
			}
		}
		return f, nil
	}
	w := win[float64](engineWindow(f))
	for i, v := range w {
		switch {
		case v != v:
			w[i] = nanV
		case math.IsInf(v, 1):
			w[i] = posV
		case math.IsInf(v, -1):
			w[i] = negV
		}
	}
	return f, nil
}

// DiagIndices ≡ np.diag_indices(n, ndim):主对角线的各维下标(每维 0..n-1)。
func DiagIndices(n, ndim int) ([]*Array, error) {
	if n < 0 {
		return nil, fmt.Errorf("num: diag_indices: n 须非负(得 %d): %w", n, ErrValue)
	}
	if ndim <= 0 {
		return nil, fmt.Errorf("num: diag_indices: ndim 须为正(得 %d): %w", ndim, ErrValue)
	}
	idx := make([]int64, n)
	for i := range idx {
		idx[i] = int64(i)
	}
	out := make([]*Array, ndim)
	for d := 0; d < ndim; d++ {
		arr, err := FromSlice(idx)
		if err != nil {
			return nil, err
		}
		out[d] = arr
	}
	return out, nil
}

// DiagIndicesFrom ≡ np.diag_indices_from:方阵(各维等长)的对角线下标。
func DiagIndicesFrom(a *Array) ([]*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: diag_indices_from: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() < 2 {
		return nil, fmt.Errorf("num: diag_indices_from: 至少 2 维(得 %d 维): %w", a.NDim(), ErrShape)
	}
	n := a.Shape()[0]
	for i, d := range a.Shape() {
		if d != n {
			return nil, fmt.Errorf("num: diag_indices_from: 形状 %v 不是各维等长的方阵: %w", a.Shape(), ErrShape)
		}
		_ = i
	}
	return DiagIndices(n, a.NDim())
}

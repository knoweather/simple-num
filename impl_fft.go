package num

import (
	"fmt"
	"math"
)

// —— FFT 子集(Step 8d):fft / ifft / rfft / irfft / fftfreq ——
//
// 口径:沿**最后一轴**做变换(numpy 默认 axis=-1),输入实数或复数、输出 complex128。
// 算法:长度为 2 的幂时用迭代 radix-2 Cooley-Tukey,否则退化为 O(n²) 朴素 DFT
// (正确性优先;后续可换 Bluestein 提速)。
//
// 与 numpy 的关系:numpy 用 pocketfft,不同算法在浮点末位必然有差异,故本组函数
// 只保证**数值精度**(相对 1e-9 量级),不做逐位对拍 —— 这是与"逐位复刻"面相区别的
// 一类,文档与 reference 的 region 均为 rel。

// FFT ≡ np.fft.fft:沿最后一轴的离散傅里叶变换。
func FFT(a *Array) (*Array, error) { return fftAxis(a, false) }

// IFFT ≡ np.fft.ifft:逆变换(按 1/n 归一,numpy 同)。
func IFFT(a *Array) (*Array, error) { return fftAxis(a, true) }

// fftAxis 沿最后一轴逐行做 FFT/IFFT。
func fftAxis(a *Array, inverse bool) (*Array, error) {
	if a.NDim() == 0 {
		return nil, fmt.Errorf("num: fft: 0-d 输入不支持: %w", ErrShape)
	}
	sh := a.Shape()
	n := sh[len(sh)-1]
	out, err := New(sh, DComplex128)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return out, nil
	}
	rows := a.Size() / n
	buf := make([]complex128, n)
	for r := 0; r < rows; r++ {
		if err := readLine(a, r, n, buf); err != nil {
			return nil, err
		}
		if inverse {
			ifft1d(buf)
		} else {
			fft1d(buf)
		}
		writeLine(out, r, n, buf)
	}
	return out, nil
}

// readLine 把第 r 条(末轴)line 读成 complex128。
func readLine(a *Array, r, n int, dst []complex128) error {
	sh := a.Shape()
	// 把 r 拆成除末轴外的下标
	idx := make([]int, len(sh))
	rem := r
	for d := len(sh) - 2; d >= 0; d-- {
		if sh[d] == 0 {
			idx[d] = 0
			continue
		}
		idx[d] = rem % sh[d]
		rem /= sh[d]
	}
	idx[len(sh)-1] = 0
	c128 := a.Dtype().Kind == Complex128
	c64 := a.Dtype().Kind == Complex64
	cplx := c128 || c64
	for i := 0; i < n; i++ {
		idx[len(sh)-1] = i
		v, err := a.Item(idx...)
		if err != nil {
			return err
		}
		if cplx {
			c, _ := complexOf(v)
			dst[i] = c
			continue
		}
		f, err := toFloat64(v)
		if err != nil {
			return fmt.Errorf("num: fft: 元素 %v 无法转复数: %w", v, ErrDtype)
		}
		dst[i] = complex(f, 0)
	}
	return nil
}

// writeLine 写回一条 line。
func writeLine(out *Array, r, n int, src []complex128) {
	sh := out.Shape()
	idx := make([]int, len(sh))
	rem := r
	for d := len(sh) - 2; d >= 0; d-- {
		if sh[d] == 0 {
			idx[d] = 0
			continue
		}
		idx[d] = rem % sh[d]
		rem /= sh[d]
	}
	for i := 0; i < n; i++ {
		idx[len(sh)-1] = i
		_ = out.SetItem(idx, src[i])
	}
}

// fft1d 原地正变换。
func fft1d(x []complex128) {
	if len(x) <= 1 {
		return
	}
	fftForwardRaw(x)
}

// ifft1d 原地逆变换(1/n 归一)。
func ifft1d(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}
	fftInverseRaw(x)
	// numpy 的 ifft 归一为 1/n
	inv := complex(1/float64(n), 0)
	for i := range x {
		x[i] *= inv
	}
}

// fftForwardRaw 正变换(不归一):2 的幂走 radix-2;较大且非 2 的幂走 Bluestein;
// 很小的长度直接用 O(n²) 朴素 DFT(常数更小)。
func fftForwardRaw(x []complex128) {
	if isPow2(len(x)) {
		fftRadix2(x, false)
		return
	}
	if len(x) > fftBluesteinThreshold {
		bluestein(x, false)
		return
	}
	naiveDFT(x, false)
}

// fftInverseRaw 逆变换(不归一),分派规则同上。
func fftInverseRaw(x []complex128) {
	if isPow2(len(x)) {
		fftRadix2(x, true)
		return
	}
	if len(x) > fftBluesteinThreshold {
		bluestein(x, true)
		return
	}
	naiveDFT(x, true)
}

// fftBluesteinThreshold 低于此长度用朴素 DFT(常数更小),高于则走 Bluestein。
const fftBluesteinThreshold = 64

// bluestein ≡ Bluestein(chirp-z)算法:把任意长度 DFT 化为三次 2 的幂 FFT,
// 复杂度 O(n log n)。用于非 2 的幂长度(如 1000、4000、质数长度)。
//
// 依据恒等式 jk = (j² + k² - (j-k)²)/2:
//
//	X[k] = chirp[k] · Σ_j (x[j]·chirp[j]) · conj(chirp[j-k])
//
// 其中 chirp[t] = exp(sign·i·π·t²/n)。实现里用 (t² mod 2n) 计算相位,
// 避免 t² 很大时 cos/sin 的精度损失。
func bluestein(x []complex128, inverse bool) {
	n := len(x)
	if n <= 1 {
		return
	}
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	// chirp 相位:π·(t² mod 2n)/n
	phase := func(t int) float64 {
		t2 := (t * t) % (2 * n)
		return sign * math.Pi * float64(t2) / float64(n)
	}
	a := make([]complex128, m)
	b := make([]complex128, m)
	for k := 0; k < n; k++ {
		ang := phase(k)
		c := complex(math.Cos(ang), math.Sin(ang))
		a[k] = x[k] * c
		cc := complex(math.Cos(ang), -math.Sin(ang))
		b[k] = cc
		if k > 0 {
			b[m-k] = cc
		}
	}
	fftRadix2(a, false)
	fftRadix2(b, false)
	for i := 0; i < m; i++ {
		a[i] *= b[i]
	}
	fftRadix2(a, true) // 未归一
	inv := complex(1/float64(m), 0)
	for k := 0; k < n; k++ {
		ang := phase(k)
		c := complex(math.Cos(ang), math.Sin(ang))
		x[k] = a[k] * c * inv
	}
}

func isPow2(n int) bool { return n > 0 && n&(n-1) == 0 }

// fftRadix2 迭代基 2 Cooley-Tukey(inverse 时用共轭技巧,归一留给调用方)。
func fftRadix2(x []complex128, inverse bool) {
	n := len(x)
	// 位反转置换
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for length := 2; length <= n; length <<= 1 {
		ang := sign * 2 * math.Pi / float64(length)
		wlen := complex(math.Cos(ang), math.Sin(ang))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			half := length / 2
			for j := 0; j < half; j++ {
				u := x[i+j]
				v := x[i+j+half] * w
				x[i+j] = u + v
				x[i+j+half] = u - v
				w *= wlen
			}
		}
	}
}

// naiveDFT 任意长度 O(n²) DFT。
func naiveDFT(x []complex128, inverse bool) {
	n := len(x)
	out := make([]complex128, n)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for k := 0; k < n; k++ {
		var s complex128
		for t := 0; t < n; t++ {
			ang := sign * 2 * math.Pi * float64(k*t) / float64(n)
			s += x[t] * complex(math.Cos(ang), math.Sin(ang))
		}
		out[k] = s
	}
	copy(x, out)
}

// RFFT ≡ np.fft.rfft:实数输入的半谱,输出 n/2+1 个复数。
func RFFT(a *Array) (*Array, error) {
	if a.NDim() == 0 {
		return nil, fmt.Errorf("num: rfft: 0-d 输入不支持: %w", ErrShape)
	}
	sh := a.Shape()
	n := sh[len(sh)-1]
	outSh := append([]int{}, sh...)
	outSh[len(outSh)-1] = n/2 + 1
	out, err := New(outSh, DComplex128)
	if err != nil {
		return nil, err
	}
	rows := a.Size() / n
	buf := make([]complex128, n)
	for r := 0; r < rows; r++ {
		if err := readLine(a, r, n, buf); err != nil {
			return nil, err
		}
		fftForwardRaw(buf)
		half := n/2 + 1
		idx := make([]int, len(outSh))
		rem := r
		for d := len(outSh) - 2; d >= 0; d-- {
			idx[d] = rem % outSh[d]
			rem /= outSh[d]
		}
		for i := 0; i < half; i++ {
			idx[len(outSh)-1] = i
			_ = out.SetItem(idx, buf[i])
		}
	}
	return out, nil
}

// IRFFT ≡ np.fft.irfft:半谱还原实数序列(长度 n;n=0 表示按 2*(m-1) 推断)。
func IRFFT(a *Array, n int) (*Array, error) {
	if a.NDim() == 0 {
		return nil, fmt.Errorf("num: irfft: 0-d 输入不支持: %w", ErrShape)
	}
	sh := a.Shape()
	m := sh[len(sh)-1]
	if n <= 0 {
		n = 2 * (m - 1)
	}
	if m != n/2+1 {
		return nil, fmt.Errorf("num: irfft: 半谱长度 %d 与 n=%d 不匹配(应为 %d): %w", m, n, n/2+1, ErrShape)
	}
	outSh := append([]int{}, sh...)
	outSh[len(outSh)-1] = n
	out, err := New(outSh, DFloat64)
	if err != nil {
		return nil, err
	}
	rows := a.Size() / m
	buf := make([]complex128, n)
	for r := 0; r < rows; r++ {
		// 读半谱并补全共轭对称
		idx := make([]int, len(sh))
		rem := r
		for d := len(sh) - 2; d >= 0; d-- {
			idx[d] = rem % sh[d]
			rem /= sh[d]
		}
		for i := 0; i < m; i++ {
			idx[len(sh)-1] = i
			v, err := a.Item(idx...)
			if err != nil {
				return nil, err
			}
			c, _ := complexOf(v)
			buf[i] = c
		}
		for i := m; i < n; i++ {
			buf[i] = conjScalar(buf[n-i])
		}
		fftInverseRaw(buf)
		inv := 1 / float64(n)
		oidx := make([]int, len(outSh))
		rem = r
		for d := len(outSh) - 2; d >= 0; d-- {
			oidx[d] = rem % outSh[d]
			rem /= outSh[d]
		}
		for i := 0; i < n; i++ {
			oidx[len(outSh)-1] = i
			_ = out.SetItem(oidx, real(buf[i])*inv)
		}
	}
	return out, nil
}

func conjScalar(v complex128) complex128 { return complex(real(v), -imag(v)) }

// FFTFreq ≡ np.fft.fftfreq(n, d):频率分档(numpy 同式)。
func FFTFreq(n int, d float64) (*Array, error) {
	if n <= 0 {
		return nil, fmt.Errorf("num: fftfreq: n 必须为正,得 %d: %w", n, ErrValue)
	}
	if d == 0 {
		return nil, fmt.Errorf("num: fftfreq: d 不能为 0: %w", ErrValue)
	}
	out, err := New([]int{n}, DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(out))
	val := 1.0 / (float64(n) * d)
	N := (n-1)/2 + 1
	for i := 0; i < N; i++ {
		w[i] = float64(i) * val
	}
	for i := n / 2; i > 0; i-- {
		w[n-i] = -float64(i) * val
	}
	return out, nil
}

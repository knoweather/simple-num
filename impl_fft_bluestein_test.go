package num

import (
	"math"
	"math/cmplx"
	"testing"
)

// naiveDFTRef 朴素 DFT(测试基准,与实现解耦)。
func naiveDFTRef(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var s complex128
		for t := 0; t < n; t++ {
			ang := -2 * math.Pi * float64(k*t) / float64(n)
			s += x[t] * complex(math.Cos(ang), math.Sin(ang))
		}
		out[k] = s
	}
	return out
}

// Bluestein 路径:各非 2 的幂长度上与朴素 DFT 一致(相对误差 < 1e-11)。
func TestFFTBluesteinAccuracy(t *testing.T) {
	sizes := []int{65, 96, 100, 127, 129, 200, 255, 257, 300, 700, 999, 1000, 1200, 1500}
	for _, n := range sizes {
		vals := make([]float64, n)
		for i := range vals {
			vals[i] = math.Sin(float64(i)*0.7) + float64(i%5)*0.3
		}
		a := bArr(t, vals)
		got, err := FFT(a)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		ref := naiveDFTRef(func() []complex128 {
			x := make([]complex128, n)
			for i, v := range vals {
				x[i] = complex(v, 0)
			}
			return x
		}())
		// 度量:误差 / 谱的最大幅度(与 numpy 的 relative 口径一致;
		// 若按"各频点自身幅度"归一,幅度接近 0 的频点会放大舍入噪声)
		scale := 0.0
		for i := range ref {
			if m := cmplx.Abs(ref[i]); m > scale {
				scale = m
			}
		}
		if scale < 1 {
			scale = 1
		}
		worst := 0.0
		for i := 0; i < n; i++ {
			v, err := got.Item(i)
			if err != nil {
				t.Fatalf("n=%d Item(%d): %v", n, i, err)
			}
			g := v.(complex128)
			d := cmplx.Abs(g-ref[i]) / scale
			if d > worst {
				worst = d
			}
		}
		if worst > 1e-12 {
			t.Fatalf("n=%d 最大相对误差 %.3e 超阈值", n, worst)
		}
	}
}

// Bluestein 逆变换:FFT→IFFT 往返应还原原序列。
func TestFFTBluesteinRoundTrip(t *testing.T) {
	for _, n := range []int{100, 999, 1000, 1200} {
		vals := make([]float64, n)
		for i := range vals {
			vals[i] = math.Cos(float64(i)*0.31) * 2
		}
		a := bArr(t, vals)
		spec, err := FFT(a)
		if err != nil {
			t.Fatalf("n=%d FFT: %v", n, err)
		}
		back, err := IFFT(spec)
		if err != nil {
			t.Fatalf("n=%d IFFT: %v", n, err)
		}
		worst := 0.0
		for i := 0; i < n; i++ {
			v, _ := back.Item(i)
			c := v.(complex128)
			if d := math.Abs(real(c) - vals[i]); d > worst {
				worst = d
			}
			if imag(c) > 1e-9 {
				t.Fatalf("n=%d 往返虚部 %v 应接近 0", n, imag(c))
			}
		}
		if worst > 1e-9 {
			t.Fatalf("n=%d 往返最大绝对误差 %.3e", n, worst)
		}
	}
}

// RFFT/IRFFT 在非 2 的幂长度上的往返。
func TestFFTBluesteinRealFFT(t *testing.T) {
	for _, n := range []int{1000, 999, 1200} {
		vals := make([]float64, n)
		for i := range vals {
			vals[i] = math.Sin(float64(i) * 0.11)
		}
		a := bArr(t, vals)
		half, err := RFFT(a)
		if err != nil {
			t.Fatalf("n=%d RFFT: %v", n, err)
		}
		if !shapeEqual(half.Shape(), []int{n/2 + 1}) {
			t.Fatalf("n=%d RFFT 形状 %v", n, half.Shape())
		}
		back, err := IRFFT(half, n)
		if err != nil {
			t.Fatalf("n=%d IRFFT: %v", n, err)
		}
		worst := 0.0
		for i := 0; i < n; i++ {
			v, _ := back.Item(i)
			if d := math.Abs(numToFloatOr(t, v) - vals[i]); d > worst {
				worst = d
			}
		}
		if worst > 1e-9 {
			t.Fatalf("n=%d 实数往返最大误差 %.3e", n, worst)
		}
	}
}

// numToFloatOr 测试辅助:把 Item 值转 float64。
func numToFloatOr(t *testing.T, v any) float64 {
	t.Helper()
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	default:
		t.Fatalf("期望浮点值,得 %T", v)
		return 0
	}
}

package num

import (
	"math"
	"testing"
)

// wantCApprox 复数数组近似比较(FFT 算法不同,末位有差异,故用相对容差)。
func wantCApprox(t *testing.T, a *Array, tol float64, want ...complex128) {
	t.Helper()
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	flat := a.AsContiguous().Flatten()
	for i, w := range want {
		v, err := flat.Item(i)
		if err != nil {
			t.Fatal(err)
		}
		g, ok := complexOf(v)
		if !ok {
			t.Fatalf("[%d] 不是复数: %T", i, v)
		}
		scale := math.Hypot(real(w), imag(w))
		if scale < 1 {
			scale = 1
		}
		if math.Hypot(real(g)-real(w), imag(g)-imag(w)) > tol*scale {
			t.Fatalf("[%d] = %v, want %v(tol %g)", i, g, w, tol)
		}
	}
}

// FFT:2 的幂与任意长度。
func TestFFT(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	out, err := FFT(a)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DComplex128 {
		t.Fatalf("dtype %s, want complex128", out.Dtype())
	}
	wantCApprox(t, out, 1e-12, 10, complex(-2, 2), -2, complex(-2, -2))

	// 非 2 的幂(n=5,走朴素 DFT)
	b := bArr(t, []float64{1, 2, 3, 4, 5})
	o2, err := FFT(b)
	if err != nil {
		t.Fatal(err)
	}
	wantCApprox(t, o2, 1e-12,
		15,
		complex(-2.5, 3.4409548011779334),
		complex(-2.5, 0.8122992405822659),
		complex(-2.5, -0.8122992405822659),
		complex(-2.5, -3.4409548011779334))

	// 复数输入
	c := bArr(t, []complex128{1 + 1i, 2 - 1i})
	o3, err := FFT(c)
	if err != nil {
		t.Fatal(err)
	}
	wantCApprox(t, o3, 1e-12, 3, complex(-1, 2))

	// 2-D:沿最后一轴
	m := mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	o4, err := FFT(m)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o4.Shape(), []int{3, 2}) {
		t.Fatalf("形状 %v", o4.Shape())
	}
	wantCApprox(t, o4, 1e-12, 3, -1, 7, -1, 11, -1)
}

// IFFT:往返还原 + 1/n 归一。
func TestIFFTRoundTrip(t *testing.T) {
	x := bArr(t, []float64{1, 2, 3, 4, 5})
	f, err := FFT(x)
	if err != nil {
		t.Fatal(err)
	}
	back, err := IFFT(f)
	if err != nil {
		t.Fatal(err)
	}
	flat := back.AsContiguous().Flatten()
	for i, w := range []float64{1, 2, 3, 4, 5} {
		v, _ := flat.Item(i)
		c, _ := complexOf(v)
		if math.Abs(real(c)-w) > 1e-12 || math.Abs(imag(c)) > 1e-12 {
			t.Fatalf("往返 [%d] = %v, want %v", i, c, w)
		}
	}
}

// RFFT / IRFFT:半谱与还原。
func TestRFFTIRFFT(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	r, err := RFFT(a)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r.Shape(), []int{3}) {
		t.Fatalf("rfft 形状 %v, want [3]", r.Shape())
	}
	wantCApprox(t, r, 1e-12, 10, complex(-2, 2), -2)

	back, err := IRFFT(r, 4)
	if err != nil {
		t.Fatal(err)
	}
	if back.Dtype() != DFloat64 {
		t.Fatalf("irfft dtype %s, want float64", back.Dtype())
	}
	bWantF64(t, back, 1, 2, 3, 4)

	// n 省略时按 2*(m-1) 推断
	back2, err := IRFFT(r, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, back2, 1, 2, 3, 4)

	// 长度不匹配 → ErrShape
	if _, err := IRFFT(r, 8); err == nil {
		t.Fatal("半谱长度与 n 不匹配应报错")
	}
}

// FFTFreq:numpy 同式。
func TestFFTFreq(t *testing.T) {
	f, err := FFTFreq(8, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, f, 0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125)

	// 奇数 n
	f2, err := FFTFreq(5, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, f2, 0, 0.2, 0.4, -0.4, -0.2)

	if _, err := FFTFreq(0, 1); err == nil {
		t.Fatal("n=0 应报错")
	}
	if _, err := FFTFreq(4, 0); err == nil {
		t.Fatal("d=0 应报错")
	}
}

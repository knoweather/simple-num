package num

import (
	"math"
	"math/cmplx"
	"testing"
)

// HFFT / IHFFT:与 numpy 2.5.4 的数值对齐。
func TestHFFTIHFFT(t *testing.T) {
	r := bArr(t, []float64{1, 2, 3, 4})
	h, err := IHFFT(r, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(h.Shape(), []int{3}) {
		t.Fatalf("ihfft 形状 %v, want [3]", h.Shape())
	}
	// numpy:[(2.5-0j), (-0.5-0.5j), (-0.5-0j)]
	wantH := []complex128{2.5, complex(-0.5, -0.5), -0.5}
	gotH := complexValues(t, h)
	for i := range wantH {
		if cmplx.Abs(gotH[i]-wantH[i]) > 1e-12 {
			t.Fatalf("ihfft[%d] = %v, want %v", i, gotH[i], wantH[i])
		}
	}
	// hfft(ihfft(x)) == x
	back, err := HFFT(h, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(back.Shape(), []int{4}) {
		t.Fatalf("hfft 形状 %v, want [4]", back.Shape())
	}
	bWantF64(t, back, 1, 2, 3, 4)

	// 显式长度:n=6
	c, _ := FromSlice([]complex128{1 + 0i, -1 + 1i, 2 + 0i})
	h6, err := HFFT(c, 6)
	if err != nil {
		t.Fatal(err)
	}
	want6 := []float64{3, -0.267949, 1.732051, 7, -1.732051, -3.732051}
	got6 := win[float64](engineWindow(h6.AsContiguous()))
	for i := range want6 {
		if math.Abs(got6[i]-want6[i]) > 1e-5 {
			t.Fatalf("hfft n=6 [%d] = %v, want %v", i, got6[i], want6[i])
		}
	}
	// 默认长度 = 2*(m-1) = 4
	h4, err := HFFT(c, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, h4, 1, 1, 5, -3)
	// n=8
	h8, err := HFFT(c, 8)
	if err != nil {
		t.Fatal(err)
	}
	want8 := []float64{3, 1, -1, 3.828427, 7, 1, -5, -1.828427}
	got8 := win[float64](engineWindow(h8.AsContiguous()))
	for i := range want8 {
		if math.Abs(got8[i]-want8[i]) > 1e-5 {
			t.Fatalf("hfft n=8 [%d] = %v, want %v", i, got8[i], want8[i])
		}
	}
	// ihfft 指定更长的 n:补零
	hN, err := IHFFT(r, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(hN.Shape(), []int{4}) {
		t.Fatalf("ihfft n=6 形状 %v, want [4](n/2+1)", hN.Shape())
	}
	// 错误路径
	if _, err := HFFT(nil, 0); err == nil {
		t.Fatal("nil hfft 应报错")
	}
	if _, err := IHFFT(nil, 0); err == nil {
		t.Fatal("nil ihfft 应报错")
	}
	if _, err := HFFT(bArr(t, []float64{1}), -3); err == nil {
		t.Fatal("非法长度应报错")
	}
}

// MatrixRank / Cond:与 numpy 对齐。
func TestMatrixRankCond(t *testing.T) {
	sing := mustMat(t, []int{2, 2}, []float64{1, 2, 2, 4})
	if r, err := MatrixRank(sing); err != nil || r != 1 {
		t.Fatalf("奇异矩阵 rank = %d (%v), want 1", r, err)
	}
	full := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	if r, err := MatrixRank(full); err != nil || r != 2 {
		t.Fatalf("满秩矩阵 rank = %d (%v), want 2", r, err)
	}
	rect := mustMat(t, []int{3, 2}, []float64{1, 2, 2, 4, 3, 6})
	if r, err := MatrixRank(rect); err != nil || r != 1 {
		t.Fatalf("3×2 rank = %d (%v), want 1", r, err)
	}
	zero := mustNew(t, []int{2, 3}, DFloat64)
	if r, err := MatrixRank(zero); err != nil || r != 0 {
		t.Fatalf("零矩阵 rank = %d (%v), want 0", r, err)
	}
	// 1-D 视为行向量
	if r, err := MatrixRank(bArr(t, []float64{1, 2, 3})); err != nil || r != 1 {
		t.Fatalf("1-D rank = %d (%v), want 1", r, err)
	}
	if _, err := MatrixRank(nil); err == nil {
		t.Fatal("nil 应报错")
	}

	// cond
	if c, err := Cond(full, ""); err != nil || math.Abs(c-14.933034373659265) > 1e-9 {
		t.Fatalf("cond 2 = %v (%v), want 14.933034...", c, err)
	}
	if c, err := Cond(full, "1"); err != nil || math.Abs(c-21) > 1e-9 {
		t.Fatalf("cond 1 = %v (%v), want 21", c, err)
	}
	if c, err := Cond(full, "inf"); err != nil || math.Abs(c-21) > 1e-9 {
		t.Fatalf("cond inf = %v (%v), want 21", c, err)
	}
	// 奇异矩阵:2-范数条件数为 +Inf
	if c, err := Cond(sing, "2"); err != nil || !math.IsInf(c, 1) {
		t.Fatalf("奇异矩阵 cond 2 = %v (%v), want +Inf", c, err)
	}
	// -2 = smin/smax
	if c, err := Cond(full, "-2"); err != nil || math.Abs(c-1/14.933034373659265) > 1e-9 {
		t.Fatalf("cond -2 = %v (%v)", c, err)
	}
	if _, err := Cond(full, "3"); err == nil {
		t.Fatal("不支持的 ord 应报错")
	}
	if _, err := Cond(bArr(t, []float64{1}), ""); err == nil {
		t.Fatal("1-D cond 应报错")
	}
}

// MultiDot / Vdot。
func TestMultiDotVdot(t *testing.T) {
	a := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	b := mustMat(t, []int{3, 4}, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	c := mustMat(t, []int{4, 2}, []float64{0, 1, 2, 3, 4, 5, 6, 7})
	got, err := MultiDot(a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{324, 422, 1008, 1304}
	gw := win[float64](engineWindow(got.AsContiguous()))
	for i := range want {
		if math.Abs(gw[i]-want[i]) > 1e-9 {
			t.Fatalf("multi_dot[%d] = %v, want %v", i, gw[i], want[i])
		}
	}
	// 两个数组时应等于 MatMul
	two, err := MultiDot(a, b)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !ArrayEqual(two, ref) {
		t.Fatal("两数组 multi_dot 应等于 matmul")
	}
	if _, err := MultiDot(a); err == nil {
		t.Fatal("单数组应报错")
	}
	if _, err := MultiDot(); err == nil {
		t.Fatal("空参应报错")
	}

	// vdot 复数:(-2+1j)
	x, _ := FromSlice([]complex128{1 + 2i, 3 - 1i})
	y, _ := FromSlice([]complex128{2 - 1i, 0 + 2i})
	d, err := Vdot(x, y)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := d.Item()
	z, ok := v.(complex128)
	if !ok || cmplx.Abs(z-(-2+1i)) > 1e-12 {
		t.Fatalf("vdot = %v, want -2+1j", v)
	}
	// vdot 实数:11
	dr, err := Vdot(bArr(t, []float64{1, 2}), bArr(t, []float64{3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	vr, _ := dr.Item()
	if f, ok := vr.(float64); !ok || math.Abs(f-11) > 1e-12 {
		t.Fatalf("vdot 实数 = %v, want 11", vr)
	}
	// 长度不符
	if _, err := Vdot(bArr(t, []float64{1}), bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("长度不符应报错")
	}
	if _, err := Vdot(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}
}

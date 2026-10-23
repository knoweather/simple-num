package num

import (
	"math"
	"math/cmplx"
	"testing"
)

// 复数线性代数:与 numpy 2.5.4 的数值逐项比较。
func TestComplexSolveInv(t *testing.T) {
	a, err := FromSlice([]complex128{
		3 + 1i, 1 - 2i,
		1 + 0i, 4 + 3i,
	})
	if err != nil {
		t.Fatal(err)
	}
	am, err := a.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	b, err := FromSlice([]complex128{1 + 1i, 2 - 1i})
	if err != nil {
		t.Fatal(err)
	}
	x, err := Solve(am, b)
	if err != nil {
		t.Fatal(err)
	}
	if x.Dtype() != DComplex128 {
		t.Fatalf("solve 复数解 dtype = %s", x.Dtype())
	}
	want := []complex128{complex(0.650519, 0.280277), complex(0.062284, -0.366782)}
	got := complexValues(t, x)
	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > 1e-5 {
			t.Fatalf("solve[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// 残差 A·x = b
	res := 0.0
	for i := 0; i < 2; i++ {
		s := complex(0, 0)
		av := win[complex128](engineWindow(am.AsContiguous()))
		for k := 0; k < 2; k++ {
			s += av[i*2+k] * got[k]
		}
		bv := win[complex128](engineWindow(b.AsContiguous()))
		res += cmplx.Abs(s-bv[i]) * cmplx.Abs(s-bv[i])
	}
	if math.Sqrt(res) > 1e-10 {
		t.Fatalf("solve 残差 %.3e", math.Sqrt(res))
	}

	// 求逆
	inv, err := Inv(am)
	if err != nil {
		t.Fatal(err)
	}
	if inv.Dtype() != DComplex128 {
		t.Fatalf("inv dtype = %s", inv.Dtype())
	}
	iw := complexValues(t, inv)
	wantI := []complex128{
		complex(0.266436, -0.124567), complex(0.076125, 0.107266),
		complex(-0.027682, 0.051903), complex(0.134948, -0.128028),
	}
	for i := range wantI {
		if cmplx.Abs(iw[i]-wantI[i]) > 1e-5 {
			t.Fatalf("inv[%d] = %v, want %v", i, iw[i], wantI[i])
		}
	}
	// A·A⁻¹ = I
	prod, err := MatMul(am, inv)
	if err != nil {
		t.Fatal(err)
	}
	pw := complexValues(t, prod)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			wantIJ := complex(0, 0)
			if i == j {
				wantIJ = 1
			}
			if cmplx.Abs(pw[i*2+j]-wantIJ) > 1e-10 {
				t.Fatalf("A·A⁻¹ [%d,%d] = %v", i, j, pw[i*2+j])
			}
		}
	}
	// 2-D 右端(两列)
	b2, err := FromSlice([]complex128{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	b2m, err := b2.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	x2, err := Solve(am, b2m)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(x2.Shape(), []int{2, 2}) || x2.Dtype() != DComplex128 {
		t.Fatalf("2-D 右端结果 %v %s", x2.Shape(), x2.Dtype())
	}
	// 实数路径不受影响
	rx, err := Solve(mustMat(t, []int{2, 2}, []float64{2, 0, 0, 4}), bArr(t, []float64{2, 8}))
	if err != nil {
		t.Fatal(err)
	}
	if rx.Dtype() != DFloat64 {
		t.Fatalf("实数 solve dtype = %s", rx.Dtype())
	}
	bWantF64(t, rx, 1, 2)
	// 错误路径
	sing, _ := FromSlice([]complex128{1, 2, 2, 4})
	sm, _ := sing.Reshape([]int{2, 2})
	if _, err := Inv(sm); err == nil {
		t.Fatal("奇异复矩阵求逆应报错")
	}
	if _, err := Solve(sm, b); err == nil {
		t.Fatal("奇异复矩阵求解应报错")
	}
	if _, err := Inv(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
}

// RFFT2 / IRFFT2:与 numpy 的数值与形状对齐。
func TestRFFT2(t *testing.T) {
	a := mustMat(t, []int{3, 4}, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	r, err := RFFT2(a)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r.Shape(), []int{3, 3}) {
		t.Fatalf("rfft2 形状 %v, want [3 3]", r.Shape())
	}
	want := []complex128{
		78, complex(-6, 6), -6,
		complex(-24, 13.8564), 0, 0,
		complex(-24, -13.8564), 0, 0,
	}
	got := complexValues(t, r)
	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > 1e-3 {
			t.Fatalf("rfft2[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// 往返(偶数长度)
	back, err := IRFFT2(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(back.Shape(), []int{3, 4}) {
		t.Fatalf("irfft2 默认形状 %v, want [3 4]", back.Shape())
	}
	bw := win[float64](engineWindow(back.AsContiguous()))
	orig := win[float64](engineWindow(a.AsContiguous()))
	for i := range orig {
		if math.Abs(bw[i]-orig[i]) > 1e-9 {
			t.Fatalf("往返 [%d] = %v, want %v", i, bw[i], orig[i])
		}
	}
	// 指定 s
	back2, err := IRFFT2(r, []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(back2.Shape(), []int{3, 4}) {
		t.Fatalf("s 指定形状 %v", back2.Shape())
	}
	// 奇数长度:rfft2 得 n/2+1,irfft2 默认还原成 2*(n-1)
	odd := mustMat(t, []int{3, 5}, []float64{
		0, 1, 2, 3, 4,
		5, 6, 7, 8, 9,
		10, 11, 12, 13, 14,
	})
	ro, err := RFFT2(odd)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ro.Shape(), []int{3, 3}) {
		t.Fatalf("奇数 rfft2 形状 %v, want [3 3]", ro.Shape())
	}
	bo, err := IRFFT2(ro, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(bo.Shape(), []int{3, 4}) {
		t.Fatalf("奇数默认还原形状 %v, want [3 4](2*(n-1))", bo.Shape())
	}
	bo5, err := IRFFT2(ro, []int{3, 5})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(bo5.Shape(), []int{3, 5}) {
		t.Fatalf("s=(3,5) 形状 %v", bo5.Shape())
	}
	// 错误路径
	if _, err := RFFT2(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("1-D rfft2 应报错")
	}
	if _, err := RFFT2(nil); err == nil {
		t.Fatal("nil 应报错")
	}
	if _, err := IRFFT2(nil, nil); err == nil {
		t.Fatal("nil irfft2 应报错")
	}
	if _, err := IRFFT2(r, []int{3}); err == nil {
		t.Fatal("s 长度不符应报错")
	}
}

// RFFTN / IRFFTN:三维与指定轴。
func TestRFFTN(t *testing.T) {
	b := mustMat(t, []int{2, 3, 4}, []float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	})
	r, err := RFFTN(b, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r.Shape(), []int{2, 3, 3}) {
		t.Fatalf("rfftn 形状 %v, want [2 3 3]", r.Shape())
	}
	back, err := IRFFTN(r, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(back.Shape(), []int{2, 3, 4}) {
		t.Fatalf("irfftn 形状 %v", back.Shape())
	}
	bw := win[float64](engineWindow(back.AsContiguous()))
	orig := win[float64](engineWindow(b.AsContiguous()))
	for i := range orig {
		if math.Abs(bw[i]-orig[i]) > 1e-9 {
			t.Fatalf("rfftn 往返 [%d] = %v, want %v", i, bw[i], orig[i])
		}
	}
	// 指定轴 (0,2)
	ra, err := RFFTN(b, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ra.Shape(), []int{2, 3, 3}) {
		t.Fatalf("指定轴 rfftn 形状 %v", ra.Shape())
	}
	ba, err := IRFFTN(ra, nil, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ba.Shape(), []int{2, 3, 4}) {
		t.Fatalf("指定轴 irfftn 形状 %v", ba.Shape())
	}
	baw := win[float64](engineWindow(ba.AsContiguous()))
	for i := range orig {
		if math.Abs(baw[i]-orig[i]) > 1e-9 {
			t.Fatalf("指定轴往返 [%d] = %v, want %v", i, baw[i], orig[i])
		}
	}
	// 错误路径
	if _, err := RFFTN(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}
	if _, err := IRFFTN(nil, nil, nil); err == nil {
		t.Fatal("nil irfftn 应报错")
	}
	if _, err := IRFFTN(r, []int{1, 2}, []int{0, 1, 2}); err == nil {
		t.Fatal("s 与 axes 长度不符应报错")
	}
}

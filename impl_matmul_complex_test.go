package num

import (
	"math/cmplx"
	"testing"
)

// MatMul 的复数路径:2-D、1-D 提升、batch 广播、实×复提升。
func TestMatMulComplex(t *testing.T) {
	a, _ := FromSlice([]complex128{1 + 2i, 0 - 1i, 3 + 0i, 4 + 1i})
	am, err := a.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := FromSlice([]complex128{1 + 0i, 2 + 1i, 0 + 3i, 1 - 1i})
	bm, err := b.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	got, err := MatMul(am, bm)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dtype() != DComplex128 {
		t.Fatalf("dtype = %s", got.Dtype())
	}
	// 手算 (1+2i)(1+0i) + (-i)(0+3i) = 1+2i + 3 = 4+2i 等
	gw := complexValues(t, got)
	av := complexValues(t, am)
	bv := complexValues(t, bm)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			var want complex128
			for k := 0; k < 2; k++ {
				want += av[i*2+k] * bv[k*2+j]
			}
			if cmplx.Abs(gw[i*2+j]-want) > 1e-12 {
				t.Fatalf("[%d,%d] = %v, want %v", i, j, gw[i*2+j], want)
			}
		}
	}
	// 1-D × 2-D → 1-D
	v, _ := FromSlice([]complex128{1 + 0i, 0 + 1i})
	r, err := MatMul(v, bm)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r.Shape(), []int{2}) {
		t.Fatalf("1-D×2-D 形状 %v", r.Shape())
	}
	// 2-D × 1-D → 1-D
	r2, err := MatMul(am, v)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r2.Shape(), []int{2}) {
		t.Fatalf("2-D×1-D 形状 %v", r2.Shape())
	}
	// batch: (2,2,2) × (2,2)
	batched, _ := FromSlice([]complex128{
		1 + 0i, 2 + 0i, 3 + 0i, 4 + 0i,
		0 + 1i, 1 + 0i, 0 + 0i, 1 + 1i,
	})
	ba, err := batched.Reshape([]int{2, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	rb, err := MatMul(ba, bm)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(rb.Shape(), []int{2, 2, 2}) {
		t.Fatalf("batch 形状 %v", rb.Shape())
	}
	// batch 内的第一块应与单独相乘一致
	first, err := ba.Sub(Slice(0, 1))
	if err != nil {
		t.Fatal(err)
	}
	f1, err := first.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	want1, err := MatMul(f1, bm)
	if err != nil {
		t.Fatal(err)
	}
	got1, err := rb.Sub(Slice(0, 1))
	if err != nil {
		t.Fatal(err)
	}
	g1, err := got1.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	w1 := complexValues(t, g1)
	ww := complexValues(t, want1)
	for i := range ww {
		if cmplx.Abs(w1[i]-ww[i]) > 1e-12 {
			t.Fatalf("batch 块 [%d] = %v, want %v", i, w1[i], ww[i])
		}
	}
	// 实 × 复 → 复
	realM := mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1})
	mix, err := MatMul(realM, bm)
	if err != nil {
		t.Fatal(err)
	}
	if mix.Dtype() != DComplex128 {
		t.Fatalf("实×复 dtype = %s", mix.Dtype())
	}
	// 内维不匹配仍应报错
	if _, err := MatMul(am, mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("内维不匹配应报错")
	}
}

// Dot 也要能吃复数(内部走同一套 matmul 语义)。
func TestDotComplex(t *testing.T) {
	a, _ := FromSlice([]complex128{1 + 1i, 2 + 0i})
	b, _ := FromSlice([]complex128{1 - 1i, 3 + 0i})
	d, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}
	v, err := d.Item()
	if err != nil {
		t.Fatal(err)
	}
	// (1+1i)(1-1i) + 2*3 = 2 + 6 = 8
	if z, ok := v.(complex128); !ok || cmplx.Abs(z-8) > 1e-12 {
		t.Fatalf("dot = %v, want 8", v)
	}
}

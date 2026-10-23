package num

import (
	"math"
	"testing"
)

// FFTN / IFFT2:与 numpy 2.5.4 对齐(正向不缩放,逆向按各轴长度之积缩放)。
func TestFFTN(t *testing.T) {
	a := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	f, err := FFTN(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(f.Shape(), []int{2, 3}) {
		t.Fatalf("形状 %v", f.Shape())
	}
	got := win[complex128](engineWindow(f.AsContiguous()))
	// numpy:[[15+0j, -3+1.7320508j, -3-1.7320508j], [-9+0j, 0, 0]]
	want := []complex128{
		15, complex(-3, 1.7320508075688772), complex(-3, -1.7320508075688772),
		-9, 0, 0,
	}
	for i := range want {
		if math.Hypot(real(got[i]-want[i]), imag(got[i]-want[i])) > 1e-9 {
			t.Fatalf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	// ifftn(fftn(x)) == x
	back, err := IFFTN(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	bw := win[complex128](engineWindow(back.AsContiguous()))
	orig := win[float64](engineWindow(a.AsContiguous()))
	for i := range orig {
		if math.Hypot(real(bw[i]-complex(orig[i], 0)), imag(bw[i]-complex(orig[i], 0))) > 1e-12 {
			t.Fatalf("往返 [%d] = %v, want %v", i, bw[i], orig[i])
		}
	}
	// fft2 == fftn(全部轴),3-D 上两者不同
	f2, err := FFT2(a)
	if err != nil {
		t.Fatal(err)
	}
	if !ArrayEqual(f2, f) {
		t.Fatal("2-D 时 fft2 应等于 fftn")
	}
	b := mustMat(t, []int{2, 3, 4}, []float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	})
	// 只对最后两轴:形状不变,前导维独立
	f3, err := FFT2(b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(f3.Shape(), []int{2, 3, 4}) {
		t.Fatalf("fft2 3-D 形状 %v", f3.Shape())
	}
	// fftn 指定轴 (0,2) 的往返
	fx, err := FFTN(b, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	rt, err := IFFTN(fx, []int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	rw := win[complex128](engineWindow(rt.AsContiguous()))
	ob := win[float64](engineWindow(b.AsContiguous()))
	for i := range ob {
		if math.Hypot(real(rw[i]-complex(ob[i], 0)), imag(rw[i]-complex(ob[i], 0))) > 1e-12 {
			t.Fatalf("轴指定往返 [%d] = %v, want %v", i, rw[i], ob[i])
		}
	}
	// 错误路径
	if _, err := FFTN(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}
	if _, err := FFTN(a, []int{0, 0}); err == nil {
		t.Fatal("重复轴应报错")
	}
	if _, err := FFTN(a, []int{5}); err == nil {
		t.Fatal("轴越界应报错")
	}
	if _, err := FFT2(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("1-D fft2 应报错")
	}
	if _, err := IFFT2(nil); err == nil {
		t.Fatal("nil ifft2 应报错")
	}
}

// NanToNum:默认与自定义替换值。
func TestNanToNum(t *testing.T) {
	a := bArr(t, []float64{math.NaN(), math.Inf(1), math.Inf(-1), 1})
	got, err := NanToNum(a, NanToNumOptions{})
	if err != nil {
		t.Fatal(err)
	}
	w := win[float64](engineWindow(got.AsContiguous()))
	if w[0] != 0 || w[1] != math.MaxFloat64 || w[2] != -math.MaxFloat64 || w[3] != 1 {
		t.Fatalf("默认替换 = %v", w)
	}
	// 输入不被修改
	if src := win[float64](engineWindow(a.AsContiguous())); !math.IsNaN(src[0]) {
		t.Fatal("入参被修改")
	}
	// 自定义
	c, err := NanToNum(a, NanToNumOptions{Nan: 9, PosInf: 8, NegInf: -8, HasNan: true, HasPosInf: true, HasNegInf: true})
	if err != nil {
		t.Fatal(err)
	}
	cw := win[float64](engineWindow(c.AsContiguous()))
	if cw[0] != 9 || cw[1] != 8 || cw[2] != -8 {
		t.Fatalf("自定义替换 = %v", cw)
	}
	// f32 / f16 保持 dtype
	f32a, _ := FromSlice([]float32{float32(math.NaN()), 2})
	f32r, err := NanToNum(f32a, NanToNumOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if f32r.Dtype() != DFloat32 {
		t.Fatalf("f32 dtype %s", f32r.Dtype())
	}
	if v, _ := f32r.Item(0); v.(float32) != 0 {
		t.Fatalf("f32 NaN 未替换: %v", v)
	}
	f16a := f16Arr(t, math.NaN(), 2)
	f16r, err := NanToNum(f16a, NanToNumOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if f16r.Dtype() != DFloat16 {
		t.Fatalf("f16 dtype %s", f16r.Dtype())
	}
	// 整型拒绝
	if _, err := NanToNum(bArr(t, []int64{1}), NanToNumOptions{}); err == nil {
		t.Fatal("整型应报错")
	}
	if _, err := NanToNum(nil, NanToNumOptions{}); err == nil {
		t.Fatal("nil 应报错")
	}
}

// DiagIndices / DiagIndicesFrom。
func TestDiagIndices(t *testing.T) {
	d, err := DiagIndices(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(d) != 2 {
		t.Fatalf("ndim=2 应返回 2 个数组,得 %d", len(d))
	}
	wantI64(t, d[0], 0, 1, 2)
	wantI64(t, d[1], 0, 1, 2)
	d3, err := DiagIndices(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(d3) != 3 {
		t.Fatalf("ndim=3 应返回 3 个数组,得 %d", len(d3))
	}
	for _, a := range d3 {
		wantI64(t, a, 0, 1)
	}
	// 边角
	if _, err := DiagIndices(-1, 2); err == nil {
		t.Fatal("负 n 应报错")
	}
	if _, err := DiagIndices(2, 0); err == nil {
		t.Fatal("ndim=0 应报错")
	}
	// from
	f, err := DiagIndicesFrom(mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 2 {
		t.Fatalf("2-D from 应返回 2 个数组")
	}
	wantI64(t, f[0], 0, 1)
	// 非方阵
	if _, err := DiagIndicesFrom(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
	if _, err := DiagIndicesFrom(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("1-D 应报错")
	}
	if _, err := DiagIndicesFrom(nil); err == nil {
		t.Fatal("nil 应报错")
	}
}

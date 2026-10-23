package num

import (
	"math"
	"testing"
)

// 集合运算:期望值取自 numpy 2.5.4(注意 NaN 不参与集合匹配)。
func TestSetOps(t *testing.T) {
	a := bArr(t, []int32{3, 1, 2, 5})
	b := bArr(t, []int32{2, 3, 7})

	in, err := IsIn(a, b)
	if err != nil {
		t.Fatal(err)
	}
	w := win[uint8](engineWindow(in.AsContiguous()))
	for i, want := range []uint8{1, 0, 1, 0} {
		if w[i] != want {
			t.Fatalf("isin[%d] = %d, want %d", i, w[i], want)
		}
	}
	// 二维输入:形状保持
	m := mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})
	in2, err := IsIn(m, bArr(t, []int32{2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(in2.Shape(), []int{2, 2}) {
		t.Fatalf("isin 形状 %v", in2.Shape())
	}
	w2 := win[uint8](engineWindow(in2.AsContiguous()))
	for i, want := range []uint8{0, 1, 1, 0} {
		if w2[i] != want {
			t.Fatalf("isin 2-D[%d] = %d, want %d", i, w2[i], want)
		}
	}

	inter, err := Intersect1D(a, b)
	if err != nil {
		t.Fatal(err)
	}
	wantI32(t, inter, 2, 3)
	u, err := Union1D(a, b)
	if err != nil {
		t.Fatal(err)
	}
	wantI32(t, u, 1, 2, 3, 5, 7)
	d, err := Setdiff1D(a, b)
	if err != nil {
		t.Fatal(err)
	}
	wantI32(t, d, 1, 5)
	x, err := Setxor1D(a, b)
	if err != nil {
		t.Fatal(err)
	}
	wantI32(t, x, 1, 5, 7)

	// NaN:不参与匹配
	na := bArr(t, []float64{math.NaN()})
	nb := bArr(t, []float64{math.NaN()})
	ni, err := IsIn(na, nb)
	if err != nil {
		t.Fatal(err)
	}
	if v := win[uint8](engineWindow(ni.AsContiguous()))[0]; v != 0 {
		t.Fatalf("isin(NaN, NaN) = %d, want 0(numpy 同)", v)
	}
	nx, err := Intersect1D(bArr(t, []float64{math.NaN(), 1}), bArr(t, []float64{math.NaN(), 2}))
	if err != nil {
		t.Fatal(err)
	}
	if nx.Size() != 0 {
		t.Fatalf("intersect1d 应丢掉 NaN,得 %v", nx)
	}

	// 非数值 dtype → 报错
	if _, err := IsIn(bArr(t, []complex128{1 + 0i}), a); err == nil {
		t.Fatal("复数应报错")
	}
	if _, err := Union1D(nil, a); err == nil {
		t.Fatal("nil 应报错")
	}
}

// Trapz / Polyval。
func TestTrapzPolyval(t *testing.T) {
	y := bArr(t, []float64{1, 2, 3, 4})
	if s, err := Trapz(y, nil, 1); err != nil || s != 7.5 {
		t.Fatalf("trapz = %v err=%v, want 7.5", s, err)
	}
	if s, err := Trapz(y, nil, 2); err != nil || s != 15 {
		t.Fatalf("trapz dx=2 = %v, want 15", s)
	}
	xs := bArr(t, []float64{0, 1, 3, 6})
	if s, err := Trapz(y, xs, 1); err != nil || s != 17 {
		t.Fatalf("trapz x = %v, want 17", s)
	}
	// 别名
	if s, err := Trapezoid(y, nil, 1); err != nil || s != 7.5 {
		t.Fatalf("trapezoid = %v", s)
	}
	// 单点 / 空
	if s, err := Trapz(bArr(t, []float64{5}), nil, 1); err != nil || s != 0 {
		t.Fatalf("单点积分 = %v", s)
	}
	// 长度不符
	if _, err := Trapz(y, bArr(t, []float64{0, 1}), 1); err == nil {
		t.Fatal("x 长度不符应报错")
	}
	if _, err := Trapz(y, nil, 0); err == nil {
		t.Fatal("dx=0 应报错")
	}

	// polyval:2x² - x + 3 在 0,1,2 → 3,4,9
	p := bArr(t, []float64{2, -1, 3})
	x := bArr(t, []float64{0, 1, 2})
	pv, err := Polyval(p, x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, pv, 3, 4, 9)
	// 空系数 → 全 0
	if o, err := Polyval(mustNew(t, []int{0}, DFloat64), x); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 0, 0, 0)
	}
	// 非 1-D 系数
	if _, err := Polyval(mustMat(t, []int{1, 2}, []float64{1, 2}), x); err == nil {
		t.Fatal("二维系数应报错")
	}
}

// 最后补的 4 个:Absolute / Any / Invert / Cross(numpy 2.x 口径)。
func TestFinalFour(t *testing.T) {
	// Absolute = abs
	if o, err := Absolute(bArr(t, []float64{-1, 2})); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 1, 2)
	}
	// Any(与 AnyTrue 同义)
	if o, err := Any(bArr(t, []int32{0, 0, 1}), nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); v != true {
		t.Fatalf("any = %v, want true", v)
	}
	if o, err := Any(bArr(t, []int32{0, 0}), nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); v != false {
		t.Fatalf("any = %v, want false", v)
	}
	// 按轴
	m := mustMat(t, []int{2, 2}, []int32{0, 1, 0, 0})
	if o, err := Any(m, []int{1}, false); err != nil {
		t.Fatal(err)
	} else {
		w := win[uint8](engineWindow(o.AsContiguous()))
		if w[0] != 1 || w[1] != 0 {
			t.Fatalf("any axis=1 = %v", w)
		}
	}
	// Invert:整型按位取反,bool 取反,浮点报错
	if o, err := Invert(bArr(t, []int32{1, -2})); err != nil {
		t.Fatal(err)
	} else {
		wantI32(t, o, -2, 1)
	}
	if o, err := Invert(bArr(t, []bool{true, false})); err != nil {
		t.Fatal(err)
	} else {
		w := win[uint8](engineWindow(o.AsContiguous()))
		if w[0] != 0 || w[1] != 1 {
			t.Fatalf("invert bool = %v, want [0 1]", w)
		}
	}
	if _, err := Invert(bArr(t, []float64{1})); err == nil {
		t.Fatal("浮点 invert 应报错(numpy 同)")
	}
	// Cross:3 维叉积
	cr, err := Cross(bArr(t, []float64{1, 2, 3}), bArr(t, []float64{4, 5, 6}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, cr, -3, 6, -3)
	// 批量 + 广播:[[1,0,0],[0,1,0]] × [0,0,1] → [[0,-1,0],[1,0,0]]
	batch := mustMat(t, []int{2, 3}, []float64{1, 0, 0, 0, 1, 0})
	cb, err := Cross(batch, bArr(t, []float64{0, 0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(cb.Shape(), []int{2, 3}) {
		t.Fatalf("cross 批量形状 %v", cb.Shape())
	}
	bWantF64(t, cb.Flatten(), 0, -1, 0, 1, 0, 0)
	// 长度非 3 → 报错(numpy 2.x 同)
	if _, err := Cross(bArr(t, []float64{1, 2}), bArr(t, []float64{3, 4})); err == nil {
		t.Fatal("2 维向量应报错")
	}
	// 前导维不可广播 → 报错
	if _, err := Cross(mustMat(t, []int{2, 3}, []float64{1, 0, 0, 0, 1, 0}),
		mustMat(t, []int{3, 3}, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})); err == nil {
		t.Fatal("前导维不可广播应报错")
	}
}

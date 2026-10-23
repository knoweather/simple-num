package num

import (
	"errors"
	"math"
	"testing"
)

// Deg2Rad/Rad2Deg:numpy 同式常量口径(与 reference 逐位对齐由 xcheck 终审,这里做数值自检)。
func TestDegRadRoundTrip(t *testing.T) {
	x := bArr(t, []float64{0, 90, 180, -45})
	d, err := Deg2Rad(x)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, math.Pi / 2, math.Pi, -math.Pi / 4}
	for i, w := range want {
		g := getI(t, d, i).(float64)
		if math.Abs(g-w) > 1e-15 {
			t.Fatalf("[%d] deg2rad = %v, want ~%v", i, g, w)
		}
	}
	back, err := Rad2Deg(d)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		g := getI(t, back, i).(float64)
		if math.Abs(g-float64([]float64{0, 90, 180, -45}[i])) > 1e-13 {
			t.Fatalf("[%d] rad2deg 往返 = %v", i, g)
		}
	}
}

// 整型按 numpy 解析提升:i8→f16、i16→f32、i32→f64。
func TestDeg2RadIntPromote(t *testing.T) {
	for _, c := range []struct {
		a    *Array
		want Dtype
	}{
		{bArr(t, []int8{90}), DFloat16},
		{bArr(t, []int16{90}), DFloat32},
		{bArr(t, []int32{90}), DFloat64},
	} {
		out, err := Deg2Rad(c.a)
		if err != nil {
			t.Fatal(err)
		}
		if out.Dtype() != c.want {
			t.Fatalf("%s 输入 → %s, want %s", c.a.Dtype(), out.Dtype(), c.want)
		}
	}
}

// Fabs:浮点取绝对值;整型提升为浮点(numpy 同)。
func TestFabs(t *testing.T) {
	x := bArr(t, []float64{-3, 3, math.Copysign(0, -1), math.NaN()})
	out, err := Fabs(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 3, 3, 0, math.NaN())
	if v, _ := out.Item(2); math.Signbit(v.(float64)) {
		t.Fatal("-0.0 的 fabs 应为 +0")
	}
	i8 := bArr(t, []int8{-5})
	outI, err := Fabs(i8)
	if err != nil {
		t.Fatal(err)
	}
	if outI.Dtype() != DFloat16 { // np.fabs(int8) → float16
		t.Fatalf("fabs(int8) dtype %s, want float16", outI.Dtype())
	}
}

// Sign:浮点/整型/复数三套口径。
func TestSign(t *testing.T) {
	x := bArr(t, []float64{-2, 0, 3, math.NaN()})
	out, err := Sign(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, -1, 0, 1, math.NaN())
	// numpy 的 sign 用 (x>0)-(x<0),-0.0 得 +0(不是 -0)
	neg := bArr(t, []float64{math.Copysign(0, -1)})
	if o, err := Sign(neg); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); math.Signbit(v.(float64)) {
		t.Fatal("sign(-0.0) 应为 +0")
	}

	i := bArr(t, []int32{-5, 0, 7})
	oi, err := Sign(i)
	if err != nil {
		t.Fatal(err)
	}
	if oi.Dtype() != DInt32 {
		t.Fatalf("sign(int32) dtype %s", oi.Dtype())
	}
	wantI32(t, oi, -1, 0, 1)

	c := bArr(t, []complex128{3 + 4i, 0})
	oc, err := Sign(c)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := oc.Item(0); v != complex128(0.6+0.8i) {
		t.Fatalf("sign(3+4i) = %v, want 0.6+0.8i", v)
	}
	if v, _ := oc.Item(1); v != complex128(0) {
		t.Fatalf("sign(0) = %v, want 0", v)
	}

	// bool 无定义(numpy 同样报错)
	if _, err := Sign(bArr(t, []bool{true})); !errors.Is(err, ErrDtype) {
		t.Fatalf("sign(bool) 应 ErrDtype,得 %v", err)
	}
}

// Arctan2:四象限 + 整型提升 + 广播。
func TestArctan2(t *testing.T) {
	y := bArr(t, []float64{1, -1, 1, 0})
	x := bArr(t, []float64{1, -1, -1, 0})
	out, err := Arctan2(y, x)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.Pi / 4, -3 * math.Pi / 4, 3 * math.Pi / 4, 0}
	for i, w := range want {
		if g := getI(t, out, i).(float64); math.Abs(g-w) > 1e-15 {
			t.Fatalf("[%d] atan2 = %v, want %v", i, g, w)
		}
	}
	// int8 输入 → float16(numpy 解析)
	outI, err := Arctan2(bArr(t, []int8{1}), bArr(t, []int8{1}))
	if err != nil {
		t.Fatal(err)
	}
	if outI.Dtype() != DFloat16 {
		t.Fatalf("arctan2(int8,int8) dtype %s, want float16", outI.Dtype())
	}
	// 广播:y (2,1) × x (1,3)
	yy := mustSlice(t, []float64{1, -1})
	yy, _ = yy.Reshape([]int{2, 1})
	xx := mustSlice(t, []float64{1, 0, -1})
	xx, _ = xx.Reshape([]int{1, 3})
	if out, err := Arctan2(yy, xx); err != nil {
		t.Fatal(err)
	} else if out.Shape()[0] != 2 || out.Shape()[1] != 3 {
		t.Fatalf("广播形状 %v, want [2 3]", out.Shape())
	}
}

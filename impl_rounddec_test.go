package num

// Round(a, decimals) 单测(B)。期望值 = numpy 2.5.4 实测探针字面量,并与
// "乘 10**decimals(元素同精度标量)→ RoundToEven → 除同一标量"手算逐位一致。
// ±0 符号另显式断言;门槛沿用 rTol(f64 逐位)/1e-6(f32)/1e-3(f16)。

import (
	"math"
	"testing"
)

// decimals = 0 即 Rint:银行家舍入 + -0.5 → -0.0。
func TestRoundDecF64Bankers(t *testing.T) {
	x := bArr(t, []float64{0.5, 1.5, 2.5, 3.5, 4.5, -0.5})
	out, err := Round(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, 0, 2, 2, 4, 4, math.Copysign(0, -1))
	if g := getI(t, out, 5).(float64); !math.Signbit(g) {
		t.Fatalf("Round(-0.5, 0) = %v, want -0.0", g)
	}
}

// decimals = 2(f64 标量 100)手算:2.675×100 积回绕到精确 267.5 → 268(偶)→ 2.68;
// 1.005×100 = 100.49999999999999 → 100 → 1.00;3.14159×100 = 314.159 → 314 → 3.14;
// 1.015×100 = 101.499999999999985789(低于 tie)→ 101 → 1.01。
func TestRoundDecF64Decimals2(t *testing.T) {
	x := bArr(t, []float64{2.675, 1.005, 3.14159, 1.015})
	out, err := Round(x, 2)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, 2.68, 1.00, 3.14, 1.01)
}

// decimals = -2(f64 标量 0.01,非精确二进制)手算:1250 → 12.5 → 12 →
// 1199.9999999999998;−1250 → −12.5 → −12 → −1199.9999999999998;
// 1499 → 14.99 → 15 → 1499.9999999999998(皆在 rTol 内即 1200/−1200/1500)。
func TestRoundDecF64NegDecimals(t *testing.T) {
	x := bArr(t, []float64{1250, -1250, 1499})
	out, err := Round(x, -2)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, 1200, -1200, 1500)
}

// f32 同精度标量:f32(2.675)×100 回绕到 267.5(tie,267 奇)→ 268 → 2.68;
// f32(1.005)×100 回绕到 100.5(tie,100 偶)→ 100 → 1.00;
// f32(1.015)×100 回绕到 101.5(tie,101 奇)→ 102 → 1.02,与 f64 的 1.01 分歧。
func TestRoundDecF32(t *testing.T) {
	x := bArr(t, []float32{0.5, 1.5, 2.5, 3.5, 4.5, -0.5})
	out, err := Round(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat32, 1e-6, 0, 2, 2, 4, 4, float32(math.Copysign(0, -1)))
	if g := itemAt(t, out, 5).(float32); !math.Signbit(float64(g)) {
		t.Fatalf("Round(-0.5f, 0) = %v, want -0.0", g)
	}

	y := bArr(t, []float32{2.675, 1.005, 1.015})
	out, err = Round(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat32, 1e-6, 2.68, 1.00, 1.02)
}

// f16(HALF 策略,保 dtype):decimals=0 银行家舍入;decimals=1 手算
// 1.25×10 = 12.5 → 12 → 1.2(f16 回绕 1.2001953125)。
func TestRoundDecF16(t *testing.T) {
	x, err := Cast(bArr(t, []float64{2.5, 3.5, -2.5, 4.5, -0.5}), DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	out, err := Round(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 1e-3, 2, 4, -2, 4, float32(math.Copysign(0, -1)))
	if g := itemAt(t, out, 4).(float32); !math.Signbit(float64(g)) {
		t.Fatalf("Round(-0.5h, 0) = %v, want -0.0", g)
	}

	y, err := Cast(bArr(t, []float64{1.25, -1.25}), DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Round(y, 1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 1e-3, 1.2, -1.2)
}

// 整型/bool 恒等(保 dtype,numpy 对整型输入原样返回)。
func TestRoundDecIntIdentity(t *testing.T) {
	x := bArr(t, []int32{-3, 0, 5})
	out, err := Round(x, 2)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt32 {
		t.Fatalf("dtype = %s, want int32", out.Dtype())
	}
	for i, w := range []int32{-3, 0, 5} {
		if g := getI(t, out, i).(int32); g != w {
			t.Fatalf("[%d] = %d, want %d", i, g, w)
		}
	}

	b := bArr(t, []bool{true, false})
	out, err = Round(b, -1)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, false)
}

// 复数按分量取整(c128 逐位;decimals=2 手算 1.005×100 = 100.49999999999999 → 100、
// 2.675×100 = 267.5 → 268 → 2.68)。
func TestRoundDecComplex(t *testing.T) {
	x := cArr128(t, 1.5+2.5i, 0.5+0.5i, -0.5-1.5i)
	out, err := Round(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantC128(t, out, 2+2i, 0+0i, complex(math.Copysign(0, -1), -2))

	y := bArr(t, []complex128{1.005 + 2.675i})
	out, err = Round(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	bWantC128(t, out, 1.00+2.68i)

	z := bArr(t, []complex64{1.5 + 2.5i, -0.5 - 1.5i})
	out, err = Round(z, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DComplex64 {
		t.Fatalf("dtype = %s, want complex64", out.Dtype())
	}
	if g := itemAt(t, out, 0).(complex64); real(g) != 2 || imag(g) != 2 {
		t.Fatalf("[0] = %v, want 2+2i", g)
	}
	g := itemAt(t, out, 1).(complex64)
	if !math.Signbit(float64(real(g))) || imag(g) != -2 {
		t.Fatalf("[1] = %v, want -0-2i", g)
	}
}

// 边角:NaN/±Inf/±0 保值(含 -0 符号)。
func TestRoundDecEdges(t *testing.T) {
	x := bArr(t, []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1), 2.5})
	out, err := Round(x, 2)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1), 2.5)
	if g := getI(t, out, 3).(float64); !math.Signbit(g) {
		t.Fatalf("Round(-0.0, 2) = %v, want -0.0", g)
	}
}

// 形状:0-d、2-d、转置视图(非连续)、size-0。
func TestRoundDecShapes(t *testing.T) {
	z, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := z.SetItem([]int{}, 2.5); err != nil {
		t.Fatal(err)
	}
	out, err := Round(z, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.NDim() != 0 || out.Size() != 1 {
		t.Fatalf("0-d: ndim=%d size=%d", out.NDim(), out.Size())
	}
	if v, err := out.Item(); err != nil || v.(float64) != 2 {
		t.Fatalf("0-d Item = %v, want 2", v)
	}

	m, err := FromSlice([]float64{0.5, 1.5, 3.5, 4.5})
	if err != nil {
		t.Fatal(err)
	}
	m, err = m.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	out, err = Round(m, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("2-d shape = %v", out.Shape())
	}
	for i, w := range [][]float64{{0, 2}, {4, 4}} {
		for j, v := range w {
			if g := getI(t, out, i, j).(float64); g != v {
				t.Fatalf("[%d,%d] = %v, want %v", i, j, g, v)
			}
		}
	}

	tt, err := m.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	out, err = Round(tt, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("transpose shape = %v", out.Shape())
	}
	for i, w := range [][]float64{{0, 4}, {2, 4}} {
		for j, v := range w {
			if g := getI(t, out, i, j).(float64); g != v {
				t.Fatalf("[%d,%d] = %v, want %v", i, j, g, v)
			}
		}
	}

	e, err := New([]int{0}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Round(e, 3)
	if err != nil {
		t.Fatal(err)
	}
	if out.Size() != 0 || out.Dtype() != DFloat64 {
		t.Fatalf("size-0: size=%d dtype=%s", out.Size(), out.Dtype())
	}
}

// 不支持 dtype → ErrDtype。
func TestRoundDecErrDtype(t *testing.T) {
	x, err := New([]int{2}, DTime("us"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = Round(x, 0)
	wantRoot(t, err, ErrDtype, "round datetime64")

	s, err := New([]int{2}, DBytes(3))
	if err != nil {
		t.Fatal(err)
	}
	_, err = Round(s, 2)
	wantRoot(t, err, ErrDtype, "round S3")
}

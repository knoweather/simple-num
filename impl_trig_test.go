package num

// 单测(B)。期望值 = numpy 2.5.4 实测探针字面量;rel 门槛
// f8 4e-16 / f4 1e-6 / f16 1e-3( dtype 自适应)。NaN 断言双方皆 NaN,
// ±Inf 精确相等,其余按 rel。

import (
	"math"
	"testing"
)

// itemAt 取第 i 个元素(0-d 数组用无参 Item)。
func itemAt(t *testing.T, a *Array, i int) any {
	t.Helper()
	if a.NDim() == 0 {
		v, err := a.Item()
		if err != nil {
			t.Fatalf("Item(): %v", err)
		}
		return v
	}
	return getI(t, a, i)
}

func bWantF64Rel(t *testing.T, a *Array, tol float64, want ...float64) {
	t.Helper()
	if a.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", a.Dtype())
	}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		g := itemAt(t, a, i).(float64)
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("[%d] = %v, want NaN", i, g)
			}
			continue
		}
		if math.IsInf(w, 0) {
			if g != w {
				t.Fatalf("[%d] = %v, want %v", i, g, w)
			}
			continue
		}
		if d := math.Abs(g - w); d > tol*math.Max(1, math.Abs(w)) {
			t.Fatalf("[%d] = %v, want %v(rel %.2e > %.2e)", i, g, w, d/math.Max(1, math.Abs(w)), tol)
		}
	}
}

// bWantF32Rel f32/f16 输出(Item 均返回 float32)。
func bWantF32Rel(t *testing.T, a *Array, wantDT Dtype, tol float64, want ...float32) {
	t.Helper()
	if a.Dtype() != wantDT {
		t.Fatalf("dtype = %s, want %s", a.Dtype(), wantDT)
	}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		g := itemAt(t, a, i).(float32)
		if math.IsNaN(float64(w)) {
			if !math.IsNaN(float64(g)) {
				t.Fatalf("[%d] = %v, want NaN", i, g)
			}
			continue
		}
		if d := math.Abs(float64(g) - float64(w)); d > tol*math.Max(1, math.Abs(float64(w))) {
			t.Fatalf("[%d] = %v, want %v(rel %.2e > %.2e)", i, g, w, d/math.Max(1, math.Abs(float64(w))), tol)
		}
	}
}

// f8 超越 reference(probe 字面量)。
func TestTrigF64(t *testing.T) {
	x := bArr(t, []float64{1, 10, 710, -745})
	out, err := Exp(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, 4e-16, 2.718281828459045, 22026.465794806718, math.Inf(1), 5e-324)

	x = bArr(t, []float64{math.E, 0, -1})
	out, _ = Log(x)
	bWantF64Rel(t, out, 4e-16, 1, math.Inf(-1), math.NaN())

	x = bArr(t, []float64{100, 0, -1})
	out, _ = Log10(x)
	bWantF64Rel(t, out, 4e-16, 2, math.Inf(-1), math.NaN())

	x = bArr(t, []float64{2, -1, math.Copysign(0, -1)})
	out, _ = Sqrt(x)
	bWantF64Rel(t, out, 4e-16, 1.4142135623730951, math.NaN(), math.Copysign(0, -1))
	if g := getI(t, out, 2).(float64); !math.Signbit(g) {
		t.Fatalf("sqrt(-0.0) = %v, want -0.0", g)
	}

	x = bArr(t, []float64{1, math.Inf(1)})
	out, _ = Sin(x)
	bWantF64Rel(t, out, 4e-16, 0.8414709848078965, math.NaN())
	out, _ = Cos(x)
	bWantF64Rel(t, out, 4e-16, 0.5403023058681398, math.NaN())
	out, _ = Tan(x)
	bWantF64Rel(t, out, 4e-16, 1.5574077246549023, math.NaN())

	x = bArr(t, []float64{0.5, 2})
	out, _ = Asin(x)
	bWantF64Rel(t, out, 4e-16, 0.5235987755982989, math.NaN())
	out, _ = Acos(x)
	bWantF64Rel(t, out, 4e-16, 1.0471975511965979, math.NaN())

	x = bArr(t, []float64{1, math.Inf(1), math.Inf(-1)})
	out, _ = Atan(x)
	bWantF64Rel(t, out, 4e-16, 0.7853981633974483, 1.5707963267948966, -1.5707963267948966)
}

// f4 超越(probe:f32 值 = float32(double 计算))。
func TestTrigF32(t *testing.T) {
	x := bArr(t, []float32{1})
	out, err := Exp(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat32, 1e-6, 2.7182819843292236)

	x = bArr(t, []float32{2})
	out, _ = Sqrt(x)
	bWantF32Rel(t, out, DFloat32, 1e-6, 1.4142135381698608)

	x = bArr(t, []float32{1})
	out, _ = Sin(x)
	bWantF32Rel(t, out, DFloat32, 1e-6, 0.8414710164070129)
	out, _ = Tan(x)
	bWantF32Rel(t, out, DFloat32, 1e-6, 1.5574077367782593)
	out, _ = Atan(x)
	bWantF32Rel(t, out, DFloat32, 1e-6, 0.7853981852531433)
}

// 整型/bool 输入的类型解析:b/i8/u8→f16,i16/u16→f32,其余→f64。
func TestTrigIntResolve(t *testing.T) {
	// i8 → f16:exp(1) 的 f16 值 = 2.71875
	out, err := Exp(bArr(t, []int8{1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 1e-3, 2.71875)

	// bool → f16
	out, err = Exp(bArr(t, []bool{true, false}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 1e-3, 2.71875, 1)

	// i16 → f32
	out, err = Exp(bArr(t, []int16{1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat32, 1e-6, 2.7182819843292236)

	// i32 / u32 → f64
	out, err = Exp(bArr(t, []int32{1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, 4e-16, 2.718281828459045)

	out, err = Exp(bArr(t, []uint32{1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, 4e-16, 2.718281828459045)

	// i8 sin → f16(probe:0.84130859375)
	out, err = Sin(bArr(t, []int8{1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 1e-3, 0.84130859375)
}

// abs 保 dtype(bool/整型/浮点)。
func TestAbs(t *testing.T) {
	// i8:MinInt 回绕
	out, err := Abs(bArr(t, []int8{-128, -1, 0, 5}))
	if err != nil {
		t.Fatal(err)
	}
	bWantI8(t, out, -128, 1, 0, 5)

	// f8:-0.0 → +0.0;NaN → NaN
	out, err = Abs(bArr(t, []float64{math.Copysign(0, -1), math.NaN(), -1.5, 2}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, 4e-16, 0, math.NaN(), 1.5, 2)
	if g := getI(t, out, 0).(float64); math.Signbit(g) {
		t.Fatalf("abs(-0.0) = %v, want +0.0", g)
	}

	// NaN payload 保留(C fabs 语义,probe:0xFFF8…0123 → 0x7FF8…0123)
	pn := bArr(t, []float64{math.Float64frombits(0xfff8000000000123)})
	out, err = Abs(pn)
	if err != nil {
		t.Fatal(err)
	}
	if g := getI(t, out, 0).(float64); math.Float64bits(g) != 0x7ff8000000000123 {
		t.Fatalf("abs(payload NaN) bits = %#x, want 0x7ff8000000000123", math.Float64bits(g))
	}

	// bool 恒等
	out, err = Abs(bArr(t, []bool{true, false}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, false)

	// u4 保 dtype
	out, err = Abs(bArr(t, []uint32{7, 0}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DUint32 {
		t.Fatalf("dtype = %s, want uint32", out.Dtype())
	}
	if g := getI(t, out, 0).(uint32); g != 7 {
		t.Fatalf("[0] = %d, want 7", g)
	}

	// f16 保 dtype(HALF 策略)
	out, err = Abs(bArr(t, []float32{0})) // 先造 f16 需要 Cast
	if err != nil {
		t.Fatal(err)
	}
	_ = out
	f16arr, err := Cast(bArr(t, []float64{-2.5, 1.5}), DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Abs(f16arr)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 1e-3, 2.5, 1.5)
}

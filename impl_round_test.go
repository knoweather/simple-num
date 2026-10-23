package num

// 单测(B)。期望值 = numpy 2.5.4 实测探针字面量。
// 取整结果都是精确整值,rel 门槛用 4e-16(等价逐位);±0 符号另显式断言。

import (
	"math"
	"testing"
)

const rTol = 4e-16

// f8 取整 4 的银行家舍入向量(probe 原样)。
func TestRoundF64Bankers(t *testing.T) {
	vals := []float64{0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5}
	x := bArr(t, vals)

	out, err := Floor(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, 0, 1, 2, 3, -1, -2, -3, 4)

	out, _ = Ceil(x)
	bWantF64Rel(t, out, rTol, 1, 2, 3, 4, math.Copysign(0, -1), -1, -2, 5)

	out, _ = Trunc(x)
	bWantF64Rel(t, out, rTol, 0, 1, 2, 3, math.Copysign(0, -1), -1, -2, 4)

	out, _ = Rint(x)
	bWantF64Rel(t, out, rTol, 0, 2, 2, 4, math.Copysign(0, -1), -2, -2, 4)
}

// 边角:±Inf/NaN/±0 保值(含 -0 符号)。
func TestRoundF64Edges(t *testing.T) {
	x := bArr(t, []float64{math.Inf(1), math.Inf(-1), math.NaN(), math.Copysign(0, -1), 0, -1.7, 1.7})

	out, err := Floor(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, math.Inf(1), math.Inf(-1), math.NaN(), math.Copysign(0, -1), 0, -2, 1)
	if g := getI(t, out, 3).(float64); !math.Signbit(g) {
		t.Fatalf("floor(-0.0) = %v, want -0.0", g)
	}

	out, _ = Ceil(x)
	bWantF64Rel(t, out, rTol, math.Inf(1), math.Inf(-1), math.NaN(), math.Copysign(0, -1), 0, -1, 2)
	if g := getI(t, out, 3).(float64); !math.Signbit(g) {
		t.Fatalf("ceil(-0.0) = %v, want -0.0", g)
	}

	out, _ = Trunc(x)
	bWantF64Rel(t, out, rTol, math.Inf(1), math.Inf(-1), math.NaN(), math.Copysign(0, -1), 0, -1, 1)
	if g := getI(t, out, 3).(float64); !math.Signbit(g) {
		t.Fatalf("trunc(-0.0) = %v, want -0.0", g)
	}

	out, _ = Rint(x)
	bWantF64Rel(t, out, rTol, math.Inf(1), math.Inf(-1), math.NaN(), math.Copysign(0, -1), 0, -2, 2)
	if g := getI(t, out, 3).(float64); !math.Signbit(g) {
		t.Fatalf("rint(-0.0) = %v, want -0.0", g)
	}
}

// f4 取整(probe:f32 值)。
func TestRoundF32(t *testing.T) {
	x := bArr(t, []float32{2.5, 3.5, -2.5, 1.1, -1.1})

	out, err := Floor(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat32, 0, 2, 3, -3, 1, -2)
	out, _ = Ceil(x)
	bWantF32Rel(t, out, DFloat32, 0, 3, 4, -2, 2, -1)
	out, _ = Trunc(x)
	bWantF32Rel(t, out, DFloat32, 0, 2, 3, -2, 1, -1)
	out, _ = Rint(x)
	bWantF32Rel(t, out, DFloat32, 0, 2, 4, -2, 1, -1)
}

// f16 取整(HALF 策略,保 dtype)。
func TestRoundF16(t *testing.T) {
	x, err := Cast(bArr(t, []float64{2.5, 3.5, -2.5}), DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	out, err := Floor(x)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 0, 2, 3, -3)
	out, _ = Ceil(x)
	bWantF32Rel(t, out, DFloat16, 0, 3, 4, -2)
	out, _ = Trunc(x)
	bWantF32Rel(t, out, DFloat16, 0, 2, 3, -2)
	out, _ = Rint(x)
	bWantF32Rel(t, out, DFloat16, 0, 2, 4, -2)
}

// floor/ceil/trunc 对整型/bool 恒等(保 dtype,numpy 整型循环)。
func TestRoundIntIdentity(t *testing.T) {
	x := bArr(t, []int8{-3, 0, 5})
	out, err := Floor(x)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt8 {
		t.Fatalf("dtype = %s, want int8", out.Dtype())
	}
	bWantI8(t, out, -3, 0, 5)

	b := bArr(t, []bool{true, false})
	out, err = Ceil(b)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, false)
}

// rint 整型/bool 输入的类型解析( 同表)。
func TestRintIntResolve(t *testing.T) {
	out, err := Rint(bArr(t, []int8{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 0, 1, 2)

	out, err = Rint(bArr(t, []uint32{3}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, 3)

	out, err = Rint(bArr(t, []bool{true, false}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 0, 1, 0)

	// 大值:2^52 + 0.5 → 2^52(偶数侧);1e16+1 → 1e16
	big := bArr(t, []float64{math.Pow(2, 52) + 0.5, 1e16 + 1})
	out, err = Rint(big)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, out, rTol, math.Pow(2, 52), 1e16)
}

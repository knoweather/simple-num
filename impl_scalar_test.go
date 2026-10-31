package num

import (
	"math"
	"math/cmplx"
	"testing"
)

// FromSlice 的泛型推断:Go 类型 → dtype 严格对应,不发生"统一升 float64"。
func TestGenericConstructDtypes(t *testing.T) {
	cases := []struct {
		name string
		got  Dtype
		want Dtype
	}{
		{"int", mustDtype(t, func() (*Array, error) { return FromSlice([]int{1}) }), DInt64},
		{"int8", mustDtype(t, func() (*Array, error) { return FromSlice([]int8{1}) }), DInt8},
		{"int16", mustDtype(t, func() (*Array, error) { return FromSlice([]int16{1}) }), DInt16},
		{"int32", mustDtype(t, func() (*Array, error) { return FromSlice([]int32{1}) }), DInt32},
		{"int64", mustDtype(t, func() (*Array, error) { return FromSlice([]int64{1}) }), DInt64},
		{"uint8", mustDtype(t, func() (*Array, error) { return FromSlice([]uint8{1}) }), DUint8},
		{"uint16", mustDtype(t, func() (*Array, error) { return FromSlice([]uint16{1}) }), DUint16},
		{"uint32", mustDtype(t, func() (*Array, error) { return FromSlice([]uint32{1}) }), DUint32},
		{"uint64", mustDtype(t, func() (*Array, error) { return FromSlice([]uint64{1}) }), DUint64},
		{"float32", mustDtype(t, func() (*Array, error) { return FromSlice([]float32{1}) }), DFloat32},
		{"float64", mustDtype(t, func() (*Array, error) { return FromSlice([]float64{1}) }), DFloat64},
		{"complex64", mustDtype(t, func() (*Array, error) { return FromSlice([]complex64{1i}) }), DComplex64},
		{"complex128", mustDtype(t, func() (*Array, error) { return FromSlice([]complex128{1i}) }), DComplex128},
		{"bool", mustDtype(t, func() (*Array, error) { return FromSlice([]bool{true}) }), DBool},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Fatalf("%s: dtype = %s, want %s", c.name, c.got, c.want)
		}
	}
	// 字符串切片
	if a, err := FromSlice([]string{"ab", "cd"}); err != nil {
		t.Fatalf("字符串: %v", err)
	} else if a.Dtype().Kind != Bytes && a.Dtype().Kind != Unicode {
		t.Fatalf("字符串 dtype = %s", a.Dtype())
	}
}

func mustDtype(t *testing.T, fn func() (*Array, error)) Dtype {
	t.Helper()
	a, err := fn()
	if err != nil {
		t.Fatal(err)
	}
	return a.Dtype()
}

// 标量运算:dtype 由 Go 类型决定(与 numpy 的弱标量不同,已在文档说明)。
func TestScalarOps(t *testing.T) {
	// int8 数组 + int8 标量 → int8,且按 int8 回绕(127+1 = -128)
	i8 := bArr(t, []int8{127})
	r, err := AddScalar(i8, int8(1))
	if err != nil {
		t.Fatal(err)
	}
	if r.Dtype() != DInt8 {
		t.Fatalf("int8+int8 dtype = %s", r.Dtype())
	}
	if v := win[int8](engineWindow(r.AsContiguous()))[0]; v != -128 {
		t.Fatalf("int8 127+1 = %d, want -128(回绕)", v)
	}
	// int8 数组 + Go int → int64(提升)
	r2, err := AddScalar(i8, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r2.Dtype() != DInt64 {
		t.Fatalf("int8+int dtype = %s, want int64", r2.Dtype())
	}
	// float32 保持 float32
	f32 := bArr(t, []float32{0.1})
	r3, err := MulScalar(f32, float32(3))
	if err != nil {
		t.Fatal(err)
	}
	if r3.Dtype() != DFloat32 {
		t.Fatalf("f32*f32 dtype = %s", r3.Dtype())
	}
	if v := win[float32](engineWindow(r3.AsContiguous()))[0]; math.Abs(float64(v)-0.3) > 1e-6 {
		t.Fatalf("0.1*3 = %v", v)
	}
	// 复数标量
	c := bArr(t, []complex128{1 + 1i})
	r4, err := MulScalar(c, 2+0i)
	if err != nil {
		t.Fatal(err)
	}
	if v := win[complex128](engineWindow(r4.AsContiguous()))[0]; cmplx.Abs(v-(2+2i)) > 1e-12 {
		t.Fatalf("复数字标量乘 = %v", v)
	}
	// 方法形式 + 链式
	chained := bArr(t, []float64{1, 2, 3}).
		MustAddScalar(10.0).
		MustMulScalar(2.0).
		MustSubScalar(1.0).
		MustDivScalar(2.0).
		MustSumAll().
		MustFloat()
	// ((1+10)*2-1)/2 = 10.5; (2+10)*2-1)/2 = 11.5; (3+10)*2-1)/2 = 12.5 → 34.5
	if math.Abs(chained-34.5) > 1e-12 {
		t.Fatalf("标量链式结果 = %v, want 34.5", chained)
	}
	// 幂
	p, err := PowScalar(bArr(t, []float64{2, 3}), 2.0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, p, 4, 9)
	// 错误路径:不支持的 Go 类型
	if _, err := bArr(t, []float64{1}).AddScalar("x"); err == nil {
		t.Fatal("字符串标量应报错")
	}
	if _, err := AddScalar(nil, 1.0); err == nil {
		t.Fatal("nil 数组应报错")
	}
}

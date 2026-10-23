package num

// T7/T8 单测(B)。期望值 = numpy 2.5.4 实测探针字面量(与 xcheck/reference fixtures
// 同源, §5)。f64/f32 逐位比对(bits),NaN 断言双方皆 NaN(payload 由
// xcheck npy gate 逐字节兜底)。getI/rootOf 借用测试辅助(同包)。

import (
	"errors"
	"math"
	"testing"
)

func bArr[T any](t *testing.T, xs []T) *Array {
	t.Helper()
	a, err := FromSlice(xs)
	if err != nil {
		t.Fatalf("FromSlice[%T]: %v", xs, err)
	}
	return a
}

// bWantF64 逐位比对;NaN 只要求双方皆 NaN。
func bWantF64(t *testing.T, a *Array, want ...float64) {
	t.Helper()
	if a.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", a.Dtype())
	}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		g := getI(t, a, i).(float64)
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("[%d] = %v (%#x), want NaN", i, g, math.Float64bits(g))
			}
			continue
		}
		if math.Float64bits(g) != math.Float64bits(w) {
			t.Fatalf("[%d] = %v (%#x), want %v (%#x)", i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}

func bWantF32(t *testing.T, a *Array, want ...float32) {
	t.Helper()
	if a.Dtype() != DFloat32 {
		t.Fatalf("dtype = %s, want float32", a.Dtype())
	}
	for i, w := range want {
		g := getI(t, a, i).(float32)
		if math.IsNaN(float64(w)) {
			if !math.IsNaN(float64(g)) {
				t.Fatalf("[%d] = %v, want NaN", i, g)
			}
			continue
		}
		if math.Float32bits(g) != math.Float32bits(w) {
			t.Fatalf("[%d] = %v (%#x), want %v (%#x)", i, g, math.Float32bits(g), w, math.Float32bits(w))
		}
	}
}

func bWantI64(t *testing.T, a *Array, want ...int64) {
	t.Helper()
	if a.Dtype() != DInt64 {
		t.Fatalf("dtype = %s, want int64", a.Dtype())
	}
	for i, w := range want {
		if g := getI(t, a, i).(int64); g != w {
			t.Fatalf("[%d] = %d, want %d", i, g, w)
		}
	}
}

func bWantI8(t *testing.T, a *Array, want ...int8) {
	t.Helper()
	if a.Dtype() != DInt8 {
		t.Fatalf("dtype = %s, want int8", a.Dtype())
	}
	for i, w := range want {
		if g := getI(t, a, i).(int8); g != w {
			t.Fatalf("[%d] = %d, want %d", i, g, w)
		}
	}
}

func bWantU64(t *testing.T, a *Array, want ...uint64) {
	t.Helper()
	if a.Dtype() != DUint64 {
		t.Fatalf("dtype = %s, want uint64", a.Dtype())
	}
	for i, w := range want {
		if g := getI(t, a, i).(uint64); g != w {
			t.Fatalf("[%d] = %d, want %d", i, g, w)
		}
	}
}

func bWantBool(t *testing.T, a *Array, want ...bool) {
	t.Helper()
	if a.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", a.Dtype())
	}
	for i, w := range want {
		if g := getI(t, a, i).(bool); g != w {
			t.Fatalf("[%d] = %v, want %v", i, g, w)
		}
	}
}

// wantRoot 错误根哨兵断言。
func wantRoot(t *testing.T, err error, sentinel error, what string) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: 期望报错", what)
	}
	if r := rootOf(err); !errors.Is(r, sentinel) {
		t.Fatalf("%s: 错误根 = %v, want %v (err=%v)", what, r, sentinel, err)
	}
}

// —— add/subtract/multiply ——

func TestAddF64SameShape(t *testing.T) {
	x := bArr(t, []float64{1.5, -2.25, 3.125, 0})
	y := bArr(t, []float64{0.5, 0.25, -6, 0})
	out, err := Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 2, -2, -2.875, 0)
}

// 广播:(3,1)+(1,4) → (3,4),numpy 文档同款算例。
func TestAddI64Broadcast(t *testing.T) {
	col := bArr(t, []int64{1, 2, 3})
	x, _ := col.Reshape([]int{3, 1})
	row := bArr(t, []int64{10, 20, 30, 40})
	y, _ := row.Reshape([]int{1, 4})
	out, err := Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{3, 4}) || out.Dtype() != DInt64 {
		t.Fatalf("out %v %s", out.Shape(), out.Dtype())
	}
	if g := getI(t, out, 2, 3); g != int64(43) {
		t.Fatalf("[2,3] = %v want 43", g)
	}
	if g := getI(t, out, 0, 1); g != int64(21) {
		t.Fatalf("[0,1] = %v want 21", g)
	}
}

// 晋级:i8 + i64 → i64(numpy promote 同款)。
func TestAddPromoteI8I64(t *testing.T) {
	x := bArr(t, []int8{-1, 2})
	y := bArr(t, []int64{10, 20})
	out, err := Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt64 {
		t.Fatalf("dtype = %s, want int64", out.Dtype())
	}
	bWantI64(t, out, 9, 22)
}

// bool×bool → bool(OR;)。
func TestAddBoolBool(t *testing.T) {
	x := bArr(t, []bool{true, true, false})
	y := bArr(t, []bool{true, false, false})
	out, err := Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, true, false)
}

// bool×bool multiply → bool(AND;)。
func TestMultiplyBoolBool(t *testing.T) {
	x := bArr(t, []bool{true, false, true})
	y := bArr(t, []bool{true, true, false})
	out, err := Multiply(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, false, false)
}

// bool×bool subtract → ErrDtype(numpy boolean subtract 同拒)。
func TestSubtractBoolBoolErr(t *testing.T) {
	x := bArr(t, []bool{true})
	y := bArr(t, []bool{true})
	_, err := Subtract(x, y)
	wantRoot(t, err, ErrDtype, "Subtract(bool,bool)")
}

// 整型回绕:i8 100*3 = 44(probe:numpy 同值)。
func TestMultiplyI8Wrap(t *testing.T) {
	x := bArr(t, []int8{100, -100})
	y := bArr(t, []int8{3, 3})
	out, err := Multiply(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantI8(t, out, 44, -44)
}

// 整型回绕:i8 -128 - 1 = 127。
func TestSubtractI8Wrap(t *testing.T) {
	x := bArr(t, []int8{-128})
	y := bArr(t, []int8{1})
	out, err := Subtract(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantI8(t, out, 127)
}

// 整型回绕:u64 2^64-1 + 1 = 0。
func TestAddU64Wrap(t *testing.T) {
	x := bArr(t, []uint64{math.MaxUint64, 3})
	y := bArr(t, []uint64{1, 1})
	out, err := Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantU64(t, out, 0, 4)
}

// 0-size 维:add 保持形状,不 panic。
// engineWindow 对 size-0 返回空窗。
func TestAddZeroSize(t *testing.T) {
	x := bArr(t, []float64{})
	out, err := Add(x, x)
	if err != nil {
		t.Fatal(err)
	}
	if out.Size() != 0 || !shapeEqual(out.Shape(), []int{0}) {
		t.Fatalf("out %v size %d", out.Shape(), out.Size())
	}
}

// —— divide ——

// divide i64/i64 → f64(numpy 类型解析,);含零与 MinInt64 边角。
func TestDivI64(t *testing.T) {
	x := bArr(t, []int64{math.MinInt64, -7, -1, 0, 7})
	y := bArr(t, []int64{-1, 0, 2, 3, -2})
	out, err := Div(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", out.Dtype())
	}
	// numpy: [9.223372036854776e+18, -inf, -0.5, 0.0, -3.5]
	bWantF64(t, out, 9.223372036854776e+18, math.Inf(-1), -0.5, 0.0, -3.5)
}

// divide bool/bool → f64;[T,F,F]/[F,F,F] = [inf,nan,nan]。
func TestDivBool(t *testing.T) {
	x := bArr(t, []bool{true, false, false})
	y := bArr(t, []bool{false, false, false})
	out, err := Div(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", out.Dtype())
	}
	bWantF64(t, out, math.Inf(1), math.NaN(), math.NaN())
}

// divide f4/f4 → f4(IEEE)。
func TestDivF32(t *testing.T) {
	x := bArr(t, []float32{7.5, -7.5, 1})
	y := bArr(t, []float32{2.5, 2.5, 0})
	out, err := Div(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat32 {
		t.Fatalf("dtype = %s, want float32", out.Dtype())
	}
	bWantF32(t, out, 3, -3, float32(math.Inf(1)))
}

// 晋级:divide(f4, i8) → f4(promote(f4,i8)=f4,非整型不预 Cast)。
func TestDivF32I8Promote(t *testing.T) {
	x := bArr(t, []float32{7.5})
	y := bArr(t, []int8{2})
	out, err := Div(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat32 {
		t.Fatalf("dtype = %s, want float32", out.Dtype())
	}
	bWantF32(t, out, 3.75)
}

// 晋级:divide(i4, f8) → f8(promote(i4,f8)=f8)。
func TestDivI32F64Promote(t *testing.T) {
	x := bArr(t, []int32{7})
	y := bArr(t, []float64{2})
	out, err := Div(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", out.Dtype())
	}
	bWantF64(t, out, 3.5)
}

// —— floor_divide / remainder(整型) ——

// i64 边角向量(probe:MinInt64/-1 回绕、零除 → 0、负号 floor)。
func TestFloorDivI64Edges(t *testing.T) {
	x := bArr(t, []int64{math.MinInt64, -7, -1, 0, 7})
	y := bArr(t, []int64{-1, 0, 2, 3, -2})
	out, err := FloorDivide(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, math.MinInt64, 0, -1, 0, -4)
}

func TestRemainderI64Edges(t *testing.T) {
	x := bArr(t, []int64{math.MinInt64, -7, -1, 0, 7})
	y := bArr(t, []int64{-1, 0, 2, 3, -2})
	out, err := Remainder(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 0, 0, 1, 0, -1)
}

// u64 零除 → 0(,numpy 2.5.4 实测)。
func TestFloorDivU64Zero(t *testing.T) {
	x := bArr(t, []uint64{math.MaxUint64, 7})
	y := bArr(t, []uint64{0, 3})
	out, err := FloorDivide(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantU64(t, out, 0, 2)
}

// —— floor_divide / remainder(浮点:npy_divmod 复刻) ——

// f64 边角向量(probe 原样:±inf、±0、NaN)。
func TestFloorDivF64Edges(t *testing.T) {
	x := bArr(t, []float64{7.5, -7.5, math.Inf(1), math.Inf(-1), math.NaN(), 5, -5, 0})
	y := bArr(t, []float64{math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1), 2, math.Inf(1), math.Inf(-1), 0})
	out, err := FloorDivide(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 0, 0, math.Inf(1), math.Inf(1), math.NaN(), 0, 0, math.NaN())
}

func TestRemainderF64Edges(t *testing.T) {
	x := bArr(t, []float64{7.5, -7.5, math.Inf(1), math.Inf(-1), math.NaN(), 5, -5, 0})
	y := bArr(t, []float64{math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1), 2, math.Inf(1), math.Inf(-1), 0})
	out, err := Remainder(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 7.5, -7.5, math.NaN(), math.NaN(), math.NaN(), 5, -5, math.NaN())
}

// snap 特例:numpy floor_divide(-5,+inf) = -1.0(非 -0.0);
// floor_divide(-7.5,-inf) = +0.0(div==0 分支 copysign(0, a/b))。
func TestFloorDivInfSnap(t *testing.T) {
	x := bArr(t, []float64{-5, 5, -7.5, 7.5})
	y := bArr(t, []float64{math.Inf(1), math.Inf(-1), math.Inf(-1), math.Inf(1)})
	out, err := FloorDivide(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, -1, -1, 0, 0)
	if g := getI(t, out, 3).(float64); math.Signbit(g) {
		t.Fatalf("[3] = %v, want +0.0", g)
	}
	// b==0 分支走 IEEE:7.5/-0.0 = -Inf
	x2 := bArr(t, []float64{7.5})
	y2 := bArr(t, []float64{math.Copysign(0, -1)})
	out2, err := FloorDivide(x2, y2)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out2, math.Inf(-1))
}

// remainder 符号修正:rem(-5,+inf) = +inf;rem(5,-inf) = -inf。
func TestRemainderInfSign(t *testing.T) {
	x := bArr(t, []float64{-5, 5})
	y := bArr(t, []float64{math.Inf(1), math.Inf(-1)})
	out, err := Remainder(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, math.Inf(1), math.Inf(-1))
}

// f32 逐位:7.5//2 = 3、-7.5//2 = -4、snap(-5f,+inf) = -1。
func TestFloorDivF32(t *testing.T) {
	x := bArr(t, []float32{7.5, -7.5, -5})
	y := bArr(t, []float32{2, 2, float32(math.Inf(1))})
	out, err := FloorDivide(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat32 {
		t.Fatalf("dtype = %s, want float32", out.Dtype())
	}
	bWantF32(t, out, 3, -4, -1)
}

// —— power ——

// i64:含回绕(2^63 → MinInt64;3^40 回绕)与 0^0=1。
func TestPowerI64(t *testing.T) {
	x := bArr(t, []int64{2, 3, -2, 0})
	y := bArr(t, []int64{63, 40, 10, 0})
	out, err := Power(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, math.MinInt64, -6289078614652622815, 1024, 1)
}

// 整型负指数 → fail-loud(:包 ErrDtype; 加 ErrValue 并列)。
func TestPowerNegExpErr(t *testing.T) {
	x := bArr(t, []int64{2})
	y := bArr(t, []int64{-1})
	_, err := Power(x, y)
	wantRoot(t, err, ErrDtype, "Power(2,-1)")
	if !errors.Is(err, ErrValue) { // 第五哨兵:值域错误可分类
		t.Errorf("Power(2,-1) 应 errors.Is(ErrValue),得:%v", err)
	}
	if !errors.Is(err, errPowNegExp) { // 内部哨兵保持可判
		t.Errorf("Power(2,-1) 应 errors.Is(errPowNegExp),得:%v", err)
	}
}

// f64:2^-1=0.5;(-8)^(1/3)=NaN;0^-1=+Inf;1^inf=1。
func TestPowerF64(t *testing.T) {
	x := bArr(t, []float64{2, -8, 0, 1})
	y := bArr(t, []float64{-1, 1.0 / 3.0, -1, math.Inf(1)})
	out, err := Power(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 0.5, math.NaN(), math.Inf(1), 1)
}

// f32:2^0.5 逐位 = float32(math.Pow(2,0.5))(probe:1.4142135381698608)。
func TestPowerF32(t *testing.T) {
	x := bArr(t, []float32{2, 9})
	y := bArr(t, []float32{0.5, 0.5})
	out, err := Power(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat32 {
		t.Fatalf("dtype = %s, want float32", out.Dtype())
	}
	bWantF32(t, out, float32(math.Pow(2, 0.5)), 3)
}

// bool×bool power → int8(numpy:[1,0,1])。
func TestPowerBoolBool(t *testing.T) {
	x := bArr(t, []bool{true, false, true})
	y := bArr(t, []bool{true, true, false})
	out, err := Power(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt8 {
		t.Fatalf("dtype = %s, want int8", out.Dtype())
	}
	bWantI8(t, out, 1, 0, 1)
}

// —— NaN 位型(npFmod 复刻 UCRT;)——

func bWantBitsF64(t *testing.T, a *Array, want ...uint64) {
	t.Helper()
	for i, w := range want {
		g := getI(t, a, i).(float64)
		if math.Float64bits(g) != w {
			t.Fatalf("[%d] bits = %#x, want %#x", i, math.Float64bits(g), w)
		}
	}
}

func bWantBitsF32(t *testing.T, a *Array, want ...uint32) {
	t.Helper()
	for i, w := range want {
		g := getI(t, a, i).(float32)
		if math.Float32bits(g) != w {
			t.Fatalf("[%d] bits = %#x, want %#x", i, math.Float32bits(g), w)
		}
	}
}

func bF64(v float64) *Array { a, _ := FromSlice([]float64{v}); return a }

// f64:±Inf 被除数 → 静态 NaN(随符号);±0 除数 → 正 qNaN;payload 传播。
func TestDivmodNaNPayloadsF64(t *testing.T) {
	inf, ninf := math.Inf(1), math.Inf(-1)
	two := 2.0
	out, err := FloorDivide(bF64(inf), bF64(two))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0x7ff8000000000000)
	out, err = FloorDivide(bF64(ninf), bF64(two))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0xfff8000000000000)

	out, err = Remainder(bF64(inf), bF64(0))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0x7ff8000000000000)
	out, err = Remainder(bF64(ninf), bF64(math.Copysign(0, -1)))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0xfff8000000000000)
	out, err = Remainder(bF64(0), bF64(0))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0x7ff8000000000000)

	// 自定义 NaN payload 原样传播(C fmod 语义;numpy 实测)。
	payload := math.Float64frombits(0x7ff8000000000123)
	out, err = FloorDivide(bF64(payload), bF64(two))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0x7ff8000000000123)
	out, err = Remainder(bF64(payload), bF64(two))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0x7ff8000000000123)
	// b==0 的 floor_divide 走 a/b:NaN payload 经 IEEE 除法传播。
	out, err = FloorDivide(bF64(payload), bF64(0))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF64(t, out, 0x7ff8000000000123)
}

// f32:inf/±0 → 0xFFC00000;标准 nan 传播 0x7FC00000。
func TestDivmodNaNPayloadsF32(t *testing.T) {
	inf, ninf := float32(math.Inf(1)), float32(math.Inf(-1))
	two, zero := float32(2), float32(0)
	out, err := FloorDivide(bArr(t, []float32{inf}), bArr(t, []float32{two}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF32(t, out, 0xffc00000)
	out, err = FloorDivide(bArr(t, []float32{ninf}), bArr(t, []float32{two}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF32(t, out, 0xffc00000)
	out, err = Remainder(bArr(t, []float32{inf}), bArr(t, []float32{zero}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF32(t, out, 0xffc00000)
	out, err = Remainder(bArr(t, []float32{zero}), bArr(t, []float32{zero}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF32(t, out, 0xffc00000)
	out, err = FloorDivide(bArr(t, []float32{float32(math.NaN())}), bArr(t, []float32{two}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF32(t, out, 0x7fc00000)
	// 自定义 f32 NaN payload 传播。
	p := math.Float32frombits(0x7fc00001)
	out, err = Remainder(bArr(t, []float32{p}), bArr(t, []float32{two}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBitsF32(t, out, 0x7fc00001)
}

// —— 引擎错误路径(经真实算子复核)——

func TestArithShapeErr(t *testing.T) {
	x := bArr(t, []int64{1, 2, 3, 4})
	y := bArr(t, []int64{1, 2, 3})
	if _, err := Add(x, y); err == nil {
		t.Error("形状不相容应报错")
	} else if rootOf(err) != ErrShape {
		t.Errorf("错误根应为 ErrShape:%v", err)
	}
}

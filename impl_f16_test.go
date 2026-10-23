package num

// 单测(B):f16 算子面(HALF 策略)。
// 期望值 = numpy 2.5.4 实测探针字面量(/§3);f16 输出按 float32 视图
// 逐位断言(Item 对 f16 返回 float32)。

import (
	"math"
	"testing"
)

func f16Arr(t *testing.T, vals ...float64) *Array {
	t.Helper()
	a, err := FromSlice(vals)
	if err != nil {
		t.Fatal(err)
	}
	h, err := Cast(a, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// f16 四则(probe:f16 位型)。
func TestF16Arith(t *testing.T) {
	a := f16Arr(t, 1.5, 0.1, -2.5)
	b := f16Arr(t, 2.25, 0.2, 0.0)

	out, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, out, DFloat16, 0, 3.75, 0.2998046875, -2.5)

	out, _ = Subtract(a, b)
	bWantF32Rel(t, out, DFloat16, 0, -0.75, -0.0999755859375, -2.5)

	out, _ = Multiply(a, b)
	bWantF32Rel(t, out, DFloat16, 0, 3.375, 0.019989013671875, float32(math.Copysign(0, -1)))

	out, _ = Div(a, b)
	bWantF32Rel(t, out, DFloat16, 0, 0.66650390625, 0.5, float32(math.Inf(-1)))

	out, _ = FloorDivide(a, b)
	bWantF32Rel(t, out, DFloat16, 0, 0, 0, float32(math.Inf(-1)))

	out, _ = Remainder(a, b)
	bWantF32Rel(t, out, DFloat16, 0, 1.5, 0.0999755859375, float32(math.NaN()))

	out, _ = Power(a, b)
	bWantF32Rel(t, out, DFloat16, 1e-3, 2.490234375, 0.630859375, 1)
}

// f16 比较。
func TestF16Cmp(t *testing.T) {
	a := f16Arr(t, 1.5, -0.5, math.NaN())
	b := f16Arr(t, 2.5, 0.5, 1)
	out, err := Less(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, true, false)

	out, _ = NotEqual(a, b)
	bWantBool(t, out, true, true, true)
}

// f16 混合提升(promote 表)。
func TestF16Promote(t *testing.T) {
	h := f16Arr(t, 1)
	out, err := Add(h, bArr(t, []int8{1}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat16 {
		t.Fatalf("f16+i8 dtype = %s, want float16", out.Dtype())
	}
	out, err = Add(h, bArr(t, []float32{1}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat32 {
		t.Fatalf("f16+f4 dtype = %s, want float32", out.Dtype())
	}
	out, err = Add(h, bArr(t, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("f16+f8 dtype = %s, want float64", out.Dtype())
	}
}

// f16 求和(f32 pairwise 累加 → f16)。
func TestF16Sum(t *testing.T) {
	a := f16Arr(t, 1.5, 0.1, -2.5)
	out, err := Sum(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat16 {
		t.Fatalf("dtype = %s, want float16", out.Dtype())
	}
	bWantF32Rel(t, out, DFloat16, 0, -0.89990234375)
}

// f16 聚合 mean/var/std(HALF 策略:输出 dtype = f16,内部按 f32 计算后 cast 回 f16;
// numpy 实测 mean/var/std(f16) → f16)。期望值 = numpy 2.5.4 探针字面量。
func TestF16MeanVarStd(t *testing.T) {
	a := f16Arr(t, 1, 2, 3, 4, 5)

	m, err := Mean(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if m.Dtype() != DFloat16 {
		t.Fatalf("mean dtype = %s, want float16", m.Dtype())
	}
	bWantF32Rel(t, m, DFloat16, 0, 3.0)

	v, err := Var(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v.Dtype() != DFloat16 {
		t.Fatalf("var dtype = %s, want float16", v.Dtype())
	}
	bWantF32Rel(t, v, DFloat16, 0, 2.0)

	s, err := Std(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if s.Dtype() != DFloat16 {
		t.Fatalf("std dtype = %s, want float16", s.Dtype())
	}
	bWantF32Rel(t, s, DFloat16, 1e-2, 1.4140625) // f16(sqrt(2)) = 1.4140625

	// axis=0 二维:[[1,2,3],[4,5,6]]
	a2 := f16Arr(t, 1, 2, 3, 4, 5, 6)
	x, err := a2.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	m2, err := Mean(x, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, m2, DFloat16, 0, 2.5, 3.5, 4.5)
	v2, err := Var(x, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, v2, DFloat16, 0, 2.25, 2.25, 2.25)
	s2, err := Std(x, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, s2, DFloat16, 1e-2, 1.5, 1.5, 1.5)

	// 0-size 全归约 → NaN(f16)
	empty, err := New([]int{0}, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	me, err := Mean(empty, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if me.NDim() != 0 || !math.IsNaN(float64(itemAt(t, me, 0).(float32))) {
		t.Fatalf("mean(空 f16) = %v, want NaN", itemAt(t, me, 0))
	}
	ve, err := Var(empty, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(float64(itemAt(t, ve, 0).(float32))) {
		t.Fatalf("var(空 f16) = %v, want NaN", itemAt(t, ve, 0))
	}
}

// f16 聚合 min/max/argmin/argmax/any/all(f32 内核;f16 值在 f32 精确表示,
// 转回无损失)。期望值 = numpy 2.5.4 探针字面量。
func TestF16MinMaxArgAnyAll(t *testing.T) {
	a := f16Arr(t, 3, 1, 5, 2, 4)

	mn, err := Min(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if mn.Dtype() != DFloat16 {
		t.Fatalf("min dtype = %s, want float16", mn.Dtype())
	}
	bWantF32Rel(t, mn, DFloat16, 0, 1.0)
	mx, err := Max(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if mx.Dtype() != DFloat16 {
		t.Fatalf("max dtype = %s, want float16", mx.Dtype())
	}
	bWantF32Rel(t, mx, DFloat16, 0, 5.0)

	amin, err := ArgMin(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if amin.Dtype() != DInt64 {
		t.Fatalf("argmin dtype = %s, want int64", amin.Dtype())
	}
	if v := itemAt(t, amin, 0).(int64); v != 1 {
		t.Fatalf("argmin = %v, want 1", v)
	}
	amax, err := ArgMax(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, amax, 0).(int64); v != 2 {
		t.Fatalf("argmax = %v, want 2", v)
	}

	any0, err := AnyTrue(f16Arr(t, 0, 0, 0), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, any0, 0).(bool); v {
		t.Fatalf("any([0,0,0]) = %v, want false", v)
	}
	any1, err := AnyTrue(f16Arr(t, 0, 1, 0), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, any1, 0).(bool); !v {
		t.Fatalf("any([0,1,0]) = %v, want true", v)
	}
	all1, err := AllTrue(f16Arr(t, 1, 1, 1), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, all1, 0).(bool); !v {
		t.Fatalf("all([1,1,1]) = %v, want true", v)
	}
	all0, err := AllTrue(f16Arr(t, 1, 0, 1), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, all0, 0).(bool); v {
		t.Fatalf("all([1,0,1]) = %v, want false", v)
	}

	// NaN:min/max 传播 NaN,argmin 首个 NaN 位置胜出
	an := f16Arr(t, 3, math.NaN(), 1)
	mnN, err := Min(an, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(float64(itemAt(t, mnN, 0).(float32))) {
		t.Fatalf("min(NaN) = %v, want NaN", itemAt(t, mnN, 0))
	}
	amN, err := ArgMin(an, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, amN, 0).(int64); v != 1 {
		t.Fatalf("argmin(NaN) = %v, want 1", v)
	}

	// axis=0 二维
	a2 := f16Arr(t, 1, 5, 3, 4, 2, 6)
	x, err := a2.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	mn2, err := Min(x, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF32Rel(t, mn2, DFloat16, 0, 1.0, 2.0, 3.0)

	// 0-size:min/max/argmin → ErrShape(numpy raise);any/all → false/true
	empty, err := New([]int{0}, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Min(empty, nil, false); rootOf(err) != ErrShape {
		t.Fatalf("min(空 f16) err = %v, want ErrShape", err)
	}
	if _, err := ArgMax(empty, nil, false); rootOf(err) != ErrShape {
		t.Fatalf("argmax(空 f16) err = %v, want ErrShape", err)
	}
	anyE, err := AnyTrue(empty, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, anyE, 0).(bool); v {
		t.Fatalf("any(空) = %v, want false", v)
	}
	allE, err := AllTrue(empty, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := itemAt(t, allE, 0).(bool); !v {
		t.Fatalf("all(空) = %v, want true", v)
	}
}

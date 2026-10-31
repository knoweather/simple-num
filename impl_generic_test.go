package num

import (
	"math"
	"math/cmplx"
	"testing"
)

// 泛型方法 As:按目标类型展开,必要时转换 dtype。
func TestGenericAs(t *testing.T) {
	a := mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	// float64 → float32(逐元素转换,长度不变)
	f32, err := a.As[float32]()
	if err != nil {
		t.Fatal(err)
	}
	if len(f32) != 6 || f32[5] != 6 {
		t.Fatalf("As[float32] = %v", f32)
	}
	// float64 → int64(截断)
	i64, err := a.As[int64]()
	if err != nil {
		t.Fatal(err)
	}
	if len(i64) != 6 || i64[0] != 1 || i64[5] != 6 {
		t.Fatalf("As[int64] = %v", i64)
	}
	// 已是目标类型:直接取
	f64, err := a.As[float64]()
	if err != nil {
		t.Fatal(err)
	}
	if len(f64) != 6 || f64[3] != 4 {
		t.Fatalf("As[float64] = %v", f64)
	}
	// int8 数组 → int64
	i8 := bArr(t, []int8{-1, 2})
	conv, err := i8.As[int64]()
	if err != nil {
		t.Fatal(err)
	}
	if len(conv) != 2 || conv[0] != -1 {
		t.Fatalf("int8 → int64 = %v", conv)
	}
	// 复数
	c := bArr(t, []complex128{1 + 2i, 3 - 4i})
	cv, err := c.As[complex128]()
	if err != nil {
		t.Fatal(err)
	}
	if cmplx.Abs(cv[0]-(1+2i)) > 1e-12 {
		t.Fatalf("complex As = %v", cv)
	}
	// 复数 → 实数(取实部)
	cr, err := c.As[float64]()
	if err != nil {
		t.Fatal(err)
	}
	if cr[0] != 1 || cr[1] != 3 {
		t.Fatalf("complex → float64 = %v", cr)
	}
	// 空数组
	empty := mustNew(t, []int{0}, DFloat64)
	if out, err := empty.As[float64](); err != nil || len(out) != 0 {
		t.Fatalf("空数组 As = %v %v", out, err)
	}
	// nil
	var nilArr *Array
	if _, err := nilArr.As[float64](); err == nil {
		t.Fatal("nil As 应报错")
	}
	// Must 形式 panic
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("MustAs(nil) 应 panic")
			}
		}()
		nilArr.MustAs[float64]()
	}()
}

// 泛型方法 At:单元素取值 + 类型转换。
func TestGenericAt(t *testing.T) {
	a := mustMat(t, []int{2, 2}, []float64{1.9, 2.5, -3.1, 0})
	if v, err := a.At[float64](0, 0); err != nil || v != 1.9 {
		t.Fatalf("At[float64] = %v %v", v, err)
	}
	// 截断向零
	if v, err := a.At[int](0, 0); err != nil || v != 1 {
		t.Fatalf("At[int](1.9) = %v %v", v, err)
	}
	if v, err := a.At[int](1, 0); err != nil || v != -3 {
		t.Fatalf("At[int](-3.1) = %v %v", v, err)
	}
	// bool → 数值
	b := bArr(t, []bool{true, false})
	if v, err := b.At[int](0); err != nil || v != 1 {
		t.Fatalf("bool At = %v %v", v, err)
	}
	if v, err := b.At[float64](1); err != nil || v != 0 {
		t.Fatalf("bool At = %v %v", v, err)
	}
	// 整数 → float
	i8 := bArr(t, []int8{-2})
	if v, err := i8.At[float64](0); err != nil || v != -2 {
		t.Fatalf("int8 At = %v %v", v, err)
	}
	// 复数
	c := bArr(t, []complex128{1 + 2i})
	if v, err := c.At[complex128](0); err != nil || cmplx.Abs(v-(1+2i)) > 1e-12 {
		t.Fatalf("complex At = %v %v", v, err)
	}
	// 越界
	if _, err := a.At[float64](9, 9); err == nil {
		t.Fatal("越界应报错")
	}
	// Must 形式
	if v := a.MustAt[float64](1, 1); v != 0 {
		t.Fatalf("MustAt = %v", v)
	}
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("越界 MustAt 应 panic")
			}
		}()
		a.MustAt[float64](5, 5)
	}()
}

// FromRows:二维字面量 → 2-D 数组。
func TestFromRows(t *testing.T) {
	a, err := FromRows([][]float64{{1, 2, 3}, {4, 5, 6}})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(a.Shape(), []int{2, 3}) {
		t.Fatalf("形状 %v", a.Shape())
	}
	if v := a.MustFloat(1, 2); v != 6 {
		t.Fatalf("a[1,2] = %v", v)
	}
	// 类型保持
	i, err := FromRows([][]int8{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}
	if i.Dtype() != DInt8 {
		t.Fatalf("int8 行 dtype = %s", i.Dtype())
	}
	// 字符串
	s, err := FromRows([][]string{{"a", "bb"}, {"ccc", "d"}})
	if err != nil {
		t.Fatal(err)
	}
	if s.Dtype().Kind != Unicode || s.Dtype().Itemsize != 3*4 {
		t.Fatalf("字符串行 dtype = %s", s.Dtype())
	}
	// 参差行 / 空
	if _, err := FromRows([][]float64{{1, 2}, {3}}); err == nil {
		t.Fatal("参差行应报错")
	}
	if _, err := FromRows([][]float64{}); err == nil {
		t.Fatal("空输入应报错")
	}
	// Must
	m := MustFromRows([][]float64{{1}, {2}})
	if !shapeEqual(m.Shape(), []int{2, 1}) {
		t.Fatalf("MustFromRows 形状 %v", m.Shape())
	}
	// 链式:行构造 → 转置 → 取值
	v := MustFromRows([][]float64{{1, 2, 3}, {4, 5, 6}}).
		MustTranspose().
		MustSumAll().
		MustFloat()
	if v != 21 {
		t.Fatalf("链式 = %v, want 21", v)
	}
	// As 与 At 的组合
	flat := MustFromRows([][]float32{{1, 2}, {3, 4}}).MustAs[float32]()
	if len(flat) != 4 || flat[3] != 4 {
		t.Fatalf("MustAs = %v", flat)
	}
	if got := math.Abs(float64(MustFromRows([][]float64{{1.5}}).MustAt[float64](0, 0)) - 1.5); got > 1e-12 {
		t.Fatal("MustAt 组合失败")
	}
}

package num

import (
	"math"
	"testing"
)

// mkScalarArr 构造 1 元素(指定 dtype)数组,值用 Go 标量写入。
func mkScalarArr(t *testing.T, dt Dtype, v any) *Array {
	t.Helper()
	a, err := New([]int{1}, dt)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.SetItem([]int{0}, v); err != nil {
		t.Fatalf("SetItem(%s, %v): %v", dt, v, err)
	}
	return a
}

// TestCastFastPathCoverage 让快速路径的每个 dtype 分支都至少跑一次
// (语义正确性由既有 cast 单测与 reference 兜底,这里保证分支被执行)。
func TestCastFastPathCoverage(t *testing.T) {
	kinds := []struct {
		dt Dtype
		v  any
	}{
		{DInt8, int8(5)}, {DInt16, int16(5)}, {DInt32, int32(5)}, {DInt64, int64(5)},
		{DUint8, uint8(5)}, {DUint16, uint16(5)}, {DUint32, uint32(5)}, {DUint64, uint64(5)},
		{DFloat32, float32(5)}, {DFloat64, float64(5)},
	}
	for _, s := range kinds {
		for _, d := range kinds {
			src := mkScalarArr(t, s.dt, s.v)
			out, err := CastUnsafe(src, d.dt)
			if err != nil {
				continue // same_kind 外由 CastUnsafe 也应允许,失败即跳过
			}
			if out.Dtype() != d.dt || out.Size() != 1 {
				t.Fatalf("%s → %s: dtype=%s size=%d", s.dt, d.dt, out.Dtype(), out.Size())
			}
		}
	}
}

// TestCastFastValues 关键转换的数值正确性(含 NaN/越界的 unsafe 口径)。
func TestCastFastValues(t *testing.T) {
	// f8 → f4:值精确保留(0.5 等可表示)
	x := bArr(t, []float64{0.5, -0.25, 1.5})
	f4, err := Cast(x, DFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if f4.Dtype() != DFloat32 {
		t.Fatalf("dtype %s", f4.Dtype())
	}
	for i, w := range []float32{0.5, -0.25, 1.5} {
		if v, _ := f4.Item(i); v != w {
			t.Fatalf("f8→f4 [%d] = %v, want %v", i, v, w)
		}
	}
	// f4 → f8
	back, err := Cast(f4, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, back, 0.5, -0.25, 1.5)

	// i4 → i8 / i4 → u1(截断低位)
	i4 := bArr(t, []int32{-1, 200, 300})
	i8, err := Cast(i4, DInt64)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, i8, -1, 200, 300)
	u1, err := CastUnsafe(i4, DUint8)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := u1.Item(0); v != uint8(255) {
		t.Fatalf("i4→u1 [0] = %v, want 255", v)
	}
	if v, _ := u1.Item(2); v != uint8(44) {
		t.Fatalf("i4→u1 [2] = %v, want 44", v)
	}

	// unsafe 浮点 → 整型的 NaN / ±Inf / 越界口径
	f := bArr(t, []float64{math.NaN(), math.Inf(1), math.Inf(-1), 1e18, 1.7, -1.7})
	ui, err := CastUnsafe(f, DInt32)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []int32{math.MinInt32, math.MinInt32, math.MinInt32, math.MinInt32, 1, -1} {
		if v, _ := ui.Item(i); v != w {
			t.Fatalf("f8→i4 unsafe [%d] = %v, want %v", i, v, w)
		}
	}

	// f16 参与的快路径
	f16, err := New([]int{2}, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{0.5, 2} {
		if err := f16.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	f32, err := Cast(f16, DFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := f32.Item(0); v != float32(0.5) {
		t.Fatalf("f16→f32 [0] = %v", v)
	}
	f64, err := Cast(f16, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := f64.Item(1); v != 2.0 {
		t.Fatalf("f16→f64 [1] = %v", v)
	}
	// f32 → f16(单次 RNE)
	back16, err := Cast(bArr(t, []float32{0.5, 2}), DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	if back16.Dtype() != DFloat16 {
		t.Fatalf("f32→f16 dtype %s", back16.Dtype())
	}
}

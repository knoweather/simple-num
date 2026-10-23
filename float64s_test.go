package num

import "testing"

// TestFromFloat64sSharesMemory 验证零拷贝语义:构造后双向写入可见,且视图与
// 原切片同基址。
func TestFromFloat64sSharesMemory(t *testing.T) {
	x := []float64{1, 2, 3}
	a, err := FromFloat64s(x)
	if err != nil {
		t.Fatalf("FromFloat64s: %v", err)
	}
	if got := a.Shape(); len(got) != 1 || got[0] != 3 {
		t.Fatalf("Shape = %v, want [3]", got)
	}
	if a.Dtype() != DFloat64 {
		t.Fatalf("Dtype = %v, want f64", a.Dtype())
	}

	if err := a.SetItem([]int{1}, 9.0); err != nil {
		t.Fatalf("SetItem: %v", err)
	}
	if x[1] != 9 {
		t.Fatalf("数组写入未反映到原切片: x[1] = %v", x[1])
	}

	x[2] = 7
	v, ok := a.Float64s()
	if !ok {
		t.Fatal("Float64s 应成功")
	}
	if len(v) != 3 || v[2] != 7 {
		t.Fatalf("Float64s = %v, want [1 9 7]", v)
	}
	if &v[0] != &x[0] {
		t.Fatal("Float64s 不是同基址视图")
	}
}

// TestFloat64sRejects 验证非连续/非 f64 时 ok=false,调用方可退回通用路径。
func TestFloat64sRejects(t *testing.T) {
	a, err := New([]int{2, 3}, DFloat64)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	tr, err := a.Transpose(1, 0)
	if err != nil {
		t.Fatalf("Transpose: %v", err)
	}
	if _, ok := tr.Float64s(); ok {
		t.Fatal("非连续视图不应返回 Float64s")
	}

	b, err := FromSlice([]int{1, 2, 3})
	if err != nil {
		t.Fatalf("FromSlice: %v", err)
	}
	if _, ok := b.Float64s(); ok {
		t.Fatal("非 f64 数组不应返回 Float64s")
	}
}

// TestFloat64sEmpty 验证空数组返回空视图而不是 panic。
func TestFloat64sEmpty(t *testing.T) {
	a, err := FromFloat64s(nil)
	if err != nil {
		t.Fatalf("FromFloat64s(nil): %v", err)
	}
	v, ok := a.Float64s()
	if !ok || len(v) != 0 {
		t.Fatalf("空数组 Float64s = %v,%v; want [],true", v, ok)
	}
}

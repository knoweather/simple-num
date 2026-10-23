package num

import "testing"

// maskBits 取掩码的 uint8 视图(便于逐位断言)。
func maskBits(m *MaskedArray) []uint8 {
	return win[uint8](engineWindow(m.MaskArray().AsContiguous()))
}

// 8 个谓词构造器:掩码与 numpy 快照一致,且数据原样保留。
func TestMaskedPredicates(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	cases := []struct {
		name string
		fn   func() (*MaskedArray, error)
		want []uint8
	}{
		{"equal", func() (*MaskedArray, error) { return MaskedEqual(a, 2.0) }, []uint8{0, 1, 0, 0}},
		{"not_equal", func() (*MaskedArray, error) { return MaskedNotEqual(a, 2.0) }, []uint8{1, 0, 1, 1}},
		{"less", func() (*MaskedArray, error) { return MaskedLess(a, 3.0) }, []uint8{1, 1, 0, 0}},
		{"greater", func() (*MaskedArray, error) { return MaskedGreater(a, 3.0) }, []uint8{0, 0, 0, 1}},
		{"less_equal", func() (*MaskedArray, error) { return MaskedLessEqual(a, 3.0) }, []uint8{1, 1, 1, 0}},
		{"greater_equal", func() (*MaskedArray, error) { return MaskedGreaterEqual(a, 3.0) }, []uint8{0, 0, 1, 1}},
		{"inside", func() (*MaskedArray, error) { return MaskedInside(a, 2.0, 3.0) }, []uint8{0, 1, 1, 0}},
		{"outside", func() (*MaskedArray, error) { return MaskedOutside(a, 2.0, 3.0) }, []uint8{1, 0, 0, 1}},
	}
	for _, c := range cases {
		got, err := c.fn()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		bits := maskBits(got)
		if len(bits) != len(c.want) {
			t.Fatalf("%s: 掩码长度 %d", c.name, len(bits))
		}
		for i := range c.want {
			if bits[i] != c.want[i] {
				t.Fatalf("%s: mask[%d] = %d, want %d", c.name, i, bits[i], c.want[i])
			}
		}
		// 数据保留
		for i, w := range []float64{1, 2, 3, 4} {
			v, _ := got.Data().Flatten().Item(i)
			if v != w {
				t.Fatalf("%s: data[%d] = %v, want %v", c.name, i, v, w)
			}
		}
	}

	// 整型 + 二维
	m := mustMat(t, []int{2, 2}, []int32{1, 5, 3, 9})
	mi, err := MaskedGreater(m, int32(4))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(mi.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v", mi.Shape())
	}
	if mi.Count() != 2 { // 5 与 9
		t.Fatalf("count = %d, want 2", mi.Count())
	}
	// 标量类型不匹配 → 报错
	if _, err := MaskedEqual(m, "x"); err == nil {
		t.Fatal("非法标量应报错")
	}
}

// 谓词的错误分支:非法标量应 fail-loud(每个构造器都走同一条标量构造路径)。
func TestMaskedPredicateErrors(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3})
	type bad struct {
		name string
		fn   func() (*MaskedArray, error)
	}
	for _, c := range []bad{
		{"equal", func() (*MaskedArray, error) { return MaskedEqual(a, "x") }},
		{"not_equal", func() (*MaskedArray, error) { return MaskedNotEqual(a, "x") }},
		{"less", func() (*MaskedArray, error) { return MaskedLess(a, "x") }},
		{"less_equal", func() (*MaskedArray, error) { return MaskedLessEqual(a, "x") }},
		{"greater", func() (*MaskedArray, error) { return MaskedGreater(a, "x") }},
		{"greater_equal", func() (*MaskedArray, error) { return MaskedGreaterEqual(a, "x") }},
		{"inside", func() (*MaskedArray, error) { return MaskedInside(a, "x", 2.0) }},
		{"outside", func() (*MaskedArray, error) { return MaskedOutside(a, 1.0, "x") }},
	} {
		if _, err := c.fn(); err == nil {
			t.Fatalf("%s: 非法标量应报错", c.name)
		}
	}
	// 掩码谓词对 0-d 输入也应工作
	sc, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := sc.SetItem([]int{}, 5.0); err != nil {
		t.Fatal(err)
	}
	// 5 > 4 → 该元素被掩码:未掩码计数 0、IsMasked 为真
	if m, err := MaskedGreater(sc, 4.0); err != nil {
		t.Fatal(err)
	} else {
		if m.Count() != 0 {
			t.Fatalf("被掩码时 count = %d, want 0", m.Count())
		}
		if !m.IsMasked() {
			t.Fatal("应标记为被掩码")
		}
	}
	// 5 不大于 10 → 不掩码
	if m, err := MaskedGreater(sc, 10.0); err != nil {
		t.Fatal(err)
	} else {
		if m.Count() != 1 {
			t.Fatalf("未掩码时 count = %d, want 1", m.Count())
		}
		if m.IsMasked() {
			t.Fatal("不应被标记")
		}
	}
}

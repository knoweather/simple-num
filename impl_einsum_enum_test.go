package num

import "testing"

// 枚举路径(含重复标签)的各 dtype 分支:einsumAccum 的类型化累加。
// 快路径接管了无重复标签的多操作数场景,这里专门走 "ii->i" 这类取对角。
func TestEinsumEnumDtypeSweep(t *testing.T) {
	type tc struct {
		name string
		arr  *Array
	}
	cases := []tc{
		{"i8", mustMat(t, []int{2, 2}, []int8{1, 2, 3, 4})},
		{"i16", mustMat(t, []int{2, 2}, []int16{1, 2, 3, 4})},
		{"i32", mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})},
		{"i64", mustMat(t, []int{2, 2}, []int64{1, 2, 3, 4})},
		{"u8", mustMat(t, []int{2, 2}, []uint8{1, 2, 3, 4})},
		{"u16", mustMat(t, []int{2, 2}, []uint16{1, 2, 3, 4})},
		{"u32", mustMat(t, []int{2, 2}, []uint32{1, 2, 3, 4})},
		{"u64", mustMat(t, []int{2, 2}, []uint64{1, 2, 3, 4})},
		{"f32", mustMat(t, []int{2, 2}, []float32{1, 2, 3, 4})},
		{"f64", mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})},
	}
	for _, c := range cases {
		out, err := Einsum("ii->i", c.arr)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !shapeEqual(out.Shape(), []int{2}) {
			t.Fatalf("%s 形状 %v", c.name, out.Shape())
		}
		if out.Dtype() != c.arr.Dtype() {
			t.Fatalf("%s dtype %s, want %s", c.name, out.Dtype(), c.arr.Dtype())
		}
		v0, err := numToFloat(out, 0)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		v1, err := numToFloat(out, 1)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if v0 != 1 || v1 != 4 {
			t.Fatalf("%s 对角 = %v,%v, want 1,4", c.name, v0, v1)
		}
	}
}

// 掩码数组四则与访问器(MaskedSubtract/Multiply/Divide、NDim/Dtype)。
func TestMaskedArithAccessors(t *testing.T) {
	a, err := NewMasked(bArr(t, []float64{1, 2, 3}), bArr(t, []bool{false, false, true}))
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewMasked(bArr(t, []float64{10, 20, 30}), bArr(t, []bool{false, true, false}))
	if err != nil {
		t.Fatal(err)
	}
	if a.NDim() != 1 {
		t.Fatalf("NDim = %d, want 1", a.NDim())
	}
	if a.Dtype() != DFloat64 {
		t.Fatalf("Dtype = %s", a.Dtype())
	}
	if a.Size() != 3 {
		t.Fatalf("Size = %d", a.Size())
	}
	// 减法:数据 1-10, 2-20, 3-30;掩码并集
	sub, err := MaskedSubtract(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, sub.Data().Flatten(), -9, -18, -27)
	if sub.Count() != 1 { // 只有第 0 个未掩码
		t.Fatalf("减法未掩码计数 = %d, want 1", sub.Count())
	}
	// 乘法
	mul, err := MaskedMultiply(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mul.Data().Flatten(), 10, 40, 90)
	// 除法
	div, err := MaskedDivide(b, a)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, div.Data().Flatten(), 10, 10, 10)
	// 掩码形状不匹配 → 报错
	bad, err := NewMasked(bArr(t, []float64{1, 1}), bArr(t, []bool{false, false}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := MaskedSubtract(a, bad); err == nil {
		t.Fatal("形状不匹配应报错")
	}
}

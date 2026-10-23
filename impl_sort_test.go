package num

import (
	"math"
	"testing"
)

// Sort:升序、NaN 置尾、按轴。
func TestSort(t *testing.T) {
	a := bArr(t, []float64{3, 1, 2})
	out, err := Sort(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 1, 2, 3)

	nan := bArr(t, []float64{2, math.NaN(), 1})
	o2, err := Sort(nan, nil)
	if err != nil {
		t.Fatal(err)
	}
	if v0, _ := o2.Item(0); v0 != 1.0 {
		t.Fatalf("[0] = %v, want 1", v0)
	}
	if v2, _ := o2.Item(2); !math.IsNaN(v2.(float64)) {
		t.Fatalf("[2] 应为 NaN(numpy 把 NaN 排最后),得 %v", v2)
	}

	m := mustSlice(t, []float64{3, 1, 2, 6, 5, 4})
	m, _ = m.Reshape([]int{2, 3})
	sa, err := Sort(m, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, sa.Flatten(), 1, 2, 3, 4, 5, 6)
	s0, err := Sort(m, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, s0.Flatten(), 3, 1, 2, 6, 5, 4)

	i := bArr(t, []int32{5, -2, 7})
	o3, err := Sort(i, nil)
	if err != nil {
		t.Fatal(err)
	}
	if o3.Dtype() != DInt32 {
		t.Fatalf("dtype %s", o3.Dtype())
	}
	wantI32(t, o3, -2, 5, 7)
}

// Argsort:返回下标。
func TestArgsort(t *testing.T) {
	a := bArr(t, []float64{30, 10, 20})
	idx, err := Argsort(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Dtype() != DInt64 {
		t.Fatalf("dtype %s, want int64", idx.Dtype())
	}
	bWantI64(t, idx, 1, 2, 0)

	m := mustSlice(t, []float64{3, 1, 2, 6, 5, 4})
	m, _ = m.Reshape([]int{2, 3})
	ia, err := Argsort(m, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ia.Shape(), []int{2, 3}) {
		t.Fatalf("形状 %v", ia.Shape())
	}
	bWantI64(t, ia.Flatten(), 1, 2, 0, 2, 1, 0)
}

// Median:奇偶两种、按轴。
func TestMedian(t *testing.T) {
	odd := bArr(t, []float64{3, 1, 2})
	if v, err := Median(odd, nil, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, v.Flatten(), 2)
	}
	even := bArr(t, []float64{4, 1, 3, 2})
	if v, err := Median(even, nil, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, v.Flatten(), 2.5)
	}
	m := mustSlice(t, []float64{1, 3, 2, 4})
	m, _ = m.Reshape([]int{2, 2})
	if v, err := Median(m, []int{1}, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, v.Flatten(), 2, 3)
	}
}

// Percentile:线性插值口径。
func TestPercentile(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	for _, c := range []struct {
		q    float64
		want float64
	}{{0, 1}, {25, 1.75}, {50, 2.5}, {75, 3.25}, {100, 4}} {
		v, err := Percentile(a, c.q, nil, false)
		if err != nil {
			t.Fatal(err)
		}
		if g := getI(t, v.Flatten(), 0).(float64); math.Abs(g-c.want) > 1e-12 {
			t.Fatalf("q=%v → %v, want %v", c.q, g, c.want)
		}
	}
	if _, err := Percentile(a, 101, nil, false); err == nil {
		t.Fatal("q 越界应报错")
	}
}

// Diff:一阶/二阶、按轴、长度不足。
func TestDiff(t *testing.T) {
	a := bArr(t, []float64{1, 2, 4, 7})
	d1, err := Diff(a, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, d1, 1, 2, 3)
	d2, err := Diff(a, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, d2, 1, 1)

	m := mustSlice(t, []float64{1, 2, 3, 10, 20, 30})
	m, _ = m.Reshape([]int{2, 3})
	d0, err := Diff(m, 1, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, d0.Flatten(), 9, 18, 27)

	short := bArr(t, []float64{5})
	se, err := Diff(short, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if se.Size() != 0 {
		t.Fatalf("单元素 diff 应为空,得 %v", se.Shape())
	}
}

// Unique:排序去重。
func TestUnique(t *testing.T) {
	a := bArr(t, []float64{3, 1, 3, 2, 1})
	u, err := Unique(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, u, 1, 2, 3)

	m := mustSlice(t, []int32{5, 5, 2, 9, 2})
	um, err := Unique(m)
	if err != nil {
		t.Fatal(err)
	}
	if um.Dtype() != DInt32 {
		t.Fatalf("dtype %s", um.Dtype())
	}
	wantI32(t, um, 2, 5, 9)
}

// TestSortAllKinds 覆盖 sort/argsort 的全部 dtype 分支(含 f16/bool/无符号/窄整)。
func TestSortAllKinds(t *testing.T) {
	// 有序整型:逐个 kind 走一遍
	i8 := bArr(t, []int8{5, -3, 0})
	if o, err := Sort(i8, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != int8(-3) {
		t.Fatalf("i8 sort[0] = %v", v)
	}
	i16 := bArr(t, []int16{300, -1})
	if o, err := Sort(i16, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != int16(-1) {
		t.Fatalf("i16 sort[0] = %v", v)
	}
	i64 := bArr(t, []int64{9, -9})
	if o, err := Sort(i64, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != int64(-9) {
		t.Fatalf("i64 sort[0] = %v", v)
	}
	u8 := bArr(t, []uint8{200, 3})
	if o, err := Sort(u8, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != uint8(3) {
		t.Fatalf("u8 sort[0] = %v", v)
	}
	u16 := bArr(t, []uint16{60000, 1})
	if o, err := Sort(u16, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != uint16(1) {
		t.Fatalf("u16 sort[0] = %v", v)
	}
	u32 := bArr(t, []uint32{4000000000, 7})
	if o, err := Sort(u32, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != uint32(7) {
		t.Fatalf("u32 sort[0] = %v", v)
	}
	u64 := bArr(t, []uint64{1 << 40, 5})
	if o, err := Sort(u64, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != uint64(5) {
		t.Fatalf("u64 sort[0] = %v", v)
	}
	f32 := bArr(t, []float32{2.5, -1.5})
	if o, err := Sort(f32, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != float32(-1.5) {
		t.Fatalf("f32 sort[0] = %v", v)
	}
	// f16:经 f32 中转排序
	f16, err := New([]int{3}, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{2, 0.5, -1} {
		if err := f16.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	if o, err := Sort(f16, nil); err != nil {
		t.Fatal(err)
	} else if o.Dtype() != DFloat16 {
		t.Fatalf("f16 sort dtype %s", o.Dtype())
	}
	// bool:false 在前
	bo := bArr(t, []bool{true, false, true})
	if o, err := Sort(bo, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != false {
		t.Fatalf("bool sort[0] = %v", v)
	}
	// argsort 各分支
	for _, a := range []*Array{i8, i16, i64, u8, u16, u32, u64, f32, bo, f16} {
		if _, err := Argsort(a, nil); err != nil {
			t.Fatalf("argsort %s: %v", a.Dtype(), err)
		}
	}
	// 不支持类型 → ErrDtype
	if _, err := Sort(bArr(t, []complex128{1 + 2i, 3 + 4i}), nil); err == nil {
		t.Fatal("复数 sort 应报错")
	}
	if _, err := Argsort(bArr(t, []complex128{1 + 2i, 3 + 4i}), nil); err == nil {
		t.Fatal("复数 argsort 应报错")
	}
}

// TestStatEdges 覆盖 keepdims、DifnegErr、Unique 0-d 等边角。
func TestStatEdges(t *testing.T) {
	m := mustSlice(t, []float64{1, 3, 2, 4})
	m, _ = m.Reshape([]int{2, 2})
	if o, err := Median(m, []int{1}, true); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(o.Shape(), []int{2, 1}) {
		t.Fatalf("median keepdims 形状 %v, want [2 1]", o.Shape())
	}
	if o, err := Percentile(m, 50, []int{0}, false); err != nil {
		t.Fatal(err)
	} else if o.Size() != 2 {
		t.Fatalf("percentile axis=0 形状 %v", o.Shape())
	}
	if _, err := Diff(m, -1, nil); err == nil {
		t.Fatal("diff n<0 应报错")
	}
	if o, err := Diff(m, 0, nil); err != nil {
		t.Fatal(err)
	} else if o.Size() != m.Size() {
		t.Fatalf("diff n=0 应原样返回")
	}
	z := mustNew(t, []int{}, DFloat64)
	if err := z.SetItem([]int{}, 7.0); err != nil {
		t.Fatal(err)
	}
	if u, err := Unique(z); err != nil {
		t.Fatal(err)
	} else if u.Size() != 1 {
		t.Fatalf("0-d unique 应得 1 元素,得 %d", u.Size())
	}
	if v, err := Median(z, nil, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, v.Flatten(), 7)
	}
}

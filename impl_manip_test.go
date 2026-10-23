package num

import "testing"

// 切分族:array_split(不等分)/ split 索引版 / h·v·dsplit。
func TestSplitFamily(t *testing.T) {
	a := bArr(t, []float64{0, 1, 2, 3, 4, 5})
	parts, err := ArraySplit(a, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 4 {
		t.Fatalf("段数 %d, want 4", len(parts))
	}
	want := [][]float64{{0, 1}, {2, 3}, {4}, {5}}
	for i, w := range want {
		if !shapeEqual(parts[i].Shape(), []int{len(w)}) {
			t.Fatalf("第 %d 段形状 %v", i, parts[i].Shape())
		}
		bWantF64(t, parts[i], w...)
	}
	// split 按索引
	sp, err := SplitAt(a, []int{2, 5}, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, sp[0], 0, 1)
	bWantF64(t, sp[1], 2, 3, 4)
	bWantF64(t, sp[2], 5)

	// hsplit:1-D 沿轴 0
	h, err := HSplit(a, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(h) != 3 || h[2].Size() != 2 {
		t.Fatalf("hsplit 1-D 结果异常")
	}
	bWantF64(t, h[0], 0, 1)

	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	// hsplit 2-D 沿轴 1 → 3 个 (2,1)
	h2, err := HSplit(m, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(h2[0].Shape(), []int{2, 1}) {
		t.Fatalf("hsplit 2-D 形状 %v", h2[0].Shape())
	}
	bWantF64(t, h2[0].Flatten(), 0, 3)
	// vsplit 沿轴 0 → 2 个 (1,3)
	v, err := VSplit(m, 2)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, v[1].Flatten(), 3, 4, 5)
	// dsplit 需要 3 维
	c := mustMat(t, []int{2, 2, 2}, []float64{0, 1, 2, 3, 4, 5, 6, 7})
	dsp, err := DSplit(c, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(dsp[0].Shape(), []int{2, 2, 1}) {
		t.Fatalf("dsplit 形状 %v", dsp[0].Shape())
	}
	bWantF64(t, dsp[0].Flatten(), 0, 2, 4, 6)
	if _, err := DSplit(m, 2); err == nil {
		t.Fatal("2 维 dsplit 应报错")
	}
	if _, err := VSplit(bArr(t, []float64{1}), 1); err == nil {
		t.Fatal("1 维 vsplit 应报错")
	}
	// 段数为 0
	if _, err := ArraySplit(a, 0, 0); err == nil {
		t.Fatal("sections=0 应报错")
	}
}

// Append / Insert / Delete。
func TestAppendInsertDelete(t *testing.T) {
	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	// append axis=0
	ap, err := Append(m, mustMat(t, []int{1, 3}, []float64{9, 9, 9}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ap.Shape(), []int{3, 3}) {
		t.Fatalf("append 形状 %v", ap.Shape())
	}
	bWantF64(t, ap.Flatten(), 0, 1, 2, 3, 4, 5, 9, 9, 9)
	// append axis=-1 → 展平
	af, err := Append(m, bArr(t, []float64{9, 9}), -1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(af.Shape(), []int{8}) {
		t.Fatalf("展平 append 形状 %v", af.Shape())
	}

	a := bArr(t, []float64{0, 1, 2, 3, 4, 5})
	// insert 展平
	ins, err := Insert(a, 2, 99.0, -1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, ins, 0, 1, 99, 2, 3, 4, 5)
	// insert axis=0 标量 → 整行
	ins2, err := Insert(m, 1, 99.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ins2.Shape(), []int{3, 3}) {
		t.Fatalf("insert 形状 %v", ins2.Shape())
	}
	bWantF64(t, ins2.Flatten(), 0, 1, 2, 99, 99, 99, 3, 4, 5)
	// insert axis=0 多值行
	ins3, err := Insert(m, 1, bArr(t, []float64{7, 8, 9}), 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, ins3.Flatten(), 0, 1, 2, 7, 8, 9, 3, 4, 5)

	// delete
	dl, err := Delete(m, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(dl.Shape(), []int{1, 3}) {
		t.Fatalf("delete 形状 %v", dl.Shape())
	}
	bWantF64(t, dl.Flatten(), 3, 4, 5)
	// delete 展平 + 负索引
	df, err := Delete(a, -1, -1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, df, 0, 1, 2, 3, 4)
	// 越界
	if _, err := Delete(a, 99, -1); err == nil {
		t.Fatal("越界删除应报错")
	}
}

// ColumnStack / DStack。
func TestColumnDStack(t *testing.T) {
	c1 := bArr(t, []float64{1, 2})
	c2 := bArr(t, []float64{3, 4})
	cs, err := ColumnStack(c1, c2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(cs.Shape(), []int{2, 2}) {
		t.Fatalf("column_stack 形状 %v", cs.Shape())
	}
	bWantF64(t, cs.Flatten(), 1, 3, 2, 4)

	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	cs2, err := ColumnStack(m, m)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(cs2.Shape(), []int{2, 6}) {
		t.Fatalf("column_stack 2-D 形状 %v", cs2.Shape())
	}
	bWantF64(t, cs2.Flatten(), 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5)

	ds, err := DStack(c1, c2)
	if err != nil {
		t.Fatal(err)
	}
	// numpy:shape (1,2,2),值 [[[1,3],[2,4]]]
	if !shapeEqual(ds.Shape(), []int{1, 2, 2}) {
		t.Fatalf("dstack 形状 %v", ds.Shape())
	}
	bWantF64(t, ds.Flatten(), 1, 3, 2, 4)
	// 2-D → (m,n,1)
	ds2, err := DStack(m, m)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ds2.Shape(), []int{2, 3, 2}) {
		t.Fatalf("dstack 2-D 形状 %v", ds2.Shape())
	}
}

// PutAlongAxis / Copy / CopyTo。
func TestPutAlongAxisAndCopy(t *testing.T) {
	d, err := New([]int{2, 3}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	idx := mustMat(t, []int{2, 1}, []int64{0, 2})
	vals := mustMat(t, []int{2, 1}, []float64{5, 6})
	if err := PutAlongAxis(d, idx, vals, 1); err != nil {
		t.Fatal(err)
	}
	// numpy:[[5,0,0],[0,0,6]]
	bWantF64(t, d.Flatten(), 5, 0, 0, 0, 0, 6)

	// 单值 values 复用
	d2, err := New([]int{2, 2}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	single, err := New([]int{1}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := single.SetItem([]int{0}, 7.0); err != nil {
		t.Fatal(err)
	}
	if err := PutAlongAxis(d2, mustMat(t, []int{2, 2}, []int64{0, 1, 1, 0}), single, 1); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, d2.Flatten(), 7, 7, 7, 7)

	// 越界 / nil
	if err := PutAlongAxis(d2, mustMat(t, []int{2, 2}, []int64{0, 9, 1, 0}), single, 1); err == nil {
		t.Fatal("越界下标应报错")
	}
	if err := PutAlongAxis(nil, idx, vals, 1); err == nil {
		t.Fatal("nil 应报错")
	}

	// Copy:独立副本
	src := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	cp := Copy(src)
	if cp == nil {
		t.Fatal("Copy 返回 nil")
	}
	if err := cp.SetItem([]int{0, 0}, 99.0); err != nil {
		t.Fatal(err)
	}
	if v, _ := src.Item(0, 0); v != 1.0 {
		t.Fatalf("Copy 应独立,原数组被改: %v", v)
	}
	// CopyTo
	dst, err := New([]int{2, 2}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := CopyTo(dst, src); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dst.Flatten(), 1, 2, 3, 4)
	// 形状不符
	if err := CopyTo(dst, bArr(t, []float64{1})); err == nil {
		t.Fatal("形状不符应报错")
	}
}

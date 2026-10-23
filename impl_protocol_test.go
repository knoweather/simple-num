package num

import (
	"testing"
)

// BinaryOut / UnaryOut:写回已有 dst。
func TestOutProtocol(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3})
	b := bArr(t, []float64{10, 20, 30})
	dst, err := New([]int{3}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := BinaryOut(Add, dst, a, b); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dst, 11, 22, 33)

	// 一元 out=
	if err := UnaryOut(Abs, dst, bArr(t, []float64{-1, -2, -3})); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dst, 1, 2, 3)

	// 形状不符 → ErrShape
	bad, err := New([]int{2}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := BinaryOut(Add, bad, a, b); err == nil {
		t.Fatal("形状不符应报错")
	}

	// dtype 不同 → 按目标 dtype 落值
	di, err := New([]int{3}, DInt32)
	if err != nil {
		t.Fatal(err)
	}
	if err := BinaryOut(Add, di, bArr(t, []float64{1.9, 2.9, 3.9}),
		bArr(t, []float64{0, 0, 0})); err != nil {
		t.Fatal(err)
	}
	wantI32(t, di, 1, 2, 3)
}

// BinaryWhere:只写条件为真的位置,其余保持 dst 原值。
func TestWhereProtocol(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	b := bArr(t, []float64{10, 20, 30, 40})
	cond := bArr(t, []bool{true, false, true, false})
	dst := bArr(t, []float64{-1, -1, -1, -1})
	if err := BinaryWhere(Add, dst, a, b, cond); err != nil {
		t.Fatal(err)
	}
	// 位置 0、2 被写;1、3 保持 -1
	bWantF64(t, dst, 11, -1, 33, -1)

	// 非 bool 条件按 !=0 取真
	dst2 := bArr(t, []float64{0, 0, 0, 0})
	if err := BinaryWhere(Add, dst2, a, b, bArr(t, []int32{1, 0, 2, 0})); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dst2, 11, 0, 33, 0)

	// 一元 where=
	dst3 := bArr(t, []float64{0, 0, 0, 0})
	if err := UnaryWhere(Abs, dst3, bArr(t, []float64{-1, -2, -3, -4}), cond); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dst3, 1, 0, 3, 0)
}

// Reduce:全归约、按轴、keepdims。
func TestReduce(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	s, err := Reduce(Add, a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if s.NDim() != 0 {
		t.Fatalf("全归约应为 0-d,得 %v", s.Shape())
	}
	if v, _ := s.Item(); v != 10.0 {
		t.Fatalf("sum = %v, want 10", v)
	}

	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	r0, err := Reduce(Add, m, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r0.Flatten(), 4, 6)

	// keepdims
	kd, err := Reduce(Add, m, []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(kd.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims 形状 %v, want [2 1]", kd.Shape())
	}
	bWantF64(t, kd.Flatten(), 3, 7)

	// multiply.reduce
	p, err := Reduce(Multiply, a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := p.Item(); v != 24.0 {
		t.Fatalf("prod = %v, want 24", v)
	}

	// 空轴 → ErrShape
	empty := mustNew(t, []int{0}, DFloat64)
	if _, err := Reduce(Add, empty, nil, false); err == nil {
		t.Fatal("空轴应报错")
	}
}

// Accumulate:前缀和 / 前缀积。
func TestAccumulate(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	s, err := Accumulate(Add, a, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, s, 1, 3, 6, 10)

	p, err := Accumulate(Multiply, a, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, p, 1, 2, 6, 24)

	// 按列累积
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	c, err := Accumulate(Add, m, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, c.Flatten(), 1, 2, 4, 6)

	// 单元素 / 0-d
	if o, err := Accumulate(Add, bArr(t, []float64{7}), nil); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 7)
	}
}

// UfuncOuter:逐对外积。
func TestUfuncOuter(t *testing.T) {
	a := bArr(t, []float64{1, 2})
	b := bArr(t, []float64{3, 4})
	out, err := UfuncOuter(Add, a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", out.Shape())
	}
	bWantF64(t, out.Flatten(), 4, 5, 5, 6)

	// multiply.outer 等价于 np.outer
	mo, err := UfuncOuter(Multiply, a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mo.Flatten(), 3, 4, 6, 8)
	// 与既有 Outer 一致
	ref, err := Outer(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, ref.Flatten(), 3, 4, 6, 8)
}

// 协议层边界:重复轴、0-d、空数组、nil 入参、dtype 不一致。
func TestProtocolEdges(t *testing.T) {
	// Reduce:重复轴去重
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	if o, err := Reduce(Add, m, []int{0, 0}, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o.Flatten(), 4, 6)
	}
	// Reduce 0-d:原样返回
	z, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := z.SetItem([]int{}, 5.0); err != nil {
		t.Fatal(err)
	}
	if o, err := Reduce(Add, z, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); v != 5.0 {
		t.Fatalf("0-d reduce = %v", v)
	}
	// Reduce nil
	if _, err := Reduce(Add, nil, nil, false); err == nil {
		t.Fatal("nil 数组应报错")
	}
	// Accumulate 0-d / 空
	if o, err := Accumulate(Add, z, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); v != 5.0 {
		t.Fatalf("0-d accumulate = %v", v)
	}
	if _, err := Accumulate(Add, nil, nil); err == nil {
		t.Fatal("nil 数组应报错")
	}
	empty := mustNew(t, []int{0}, DFloat64)
	if o, err := Accumulate(Add, empty, nil); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatal("空数组 accumulate 应为空")
	}
	// UfuncOuter:空输入与 nil
	if o, err := UfuncOuter(Add, empty, bArr(t, []float64{1})); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(o.Shape(), []int{0, 1}) {
		t.Fatalf("空 outer 形状 %v", o.Shape())
	}
	if _, err := UfuncOuter(Add, nil, empty); err == nil {
		t.Fatal("nil 应报错")
	}
	// UnaryOut dtype 不一致 / UnaryWhere 形状不符 / nil
	d32, err := New([]int{3}, DFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if err := UnaryOut(Abs, d32, bArr(t, []float64{-1.5, -2.5, -3.5})); err != nil {
		t.Fatal(err)
	}
	if v, _ := d32.Item(0); v != float32(1.5) {
		t.Fatalf("f32 落值 = %v", v)
	}
	if err := UnaryWhere(Abs, d32, bArr(t, []float64{1, 2}), bArr(t, []bool{true, false})); err == nil {
		t.Fatal("形状不符应报错")
	}
	if err := UnaryOut(Abs, nil, d32); err == nil {
		t.Fatal("nil dst 应报错")
	}
	if err := BinaryWhere(Add, nil, d32, d32, d32); err == nil {
		t.Fatal("nil 应报错")
	}
}

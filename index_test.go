package num

import (
	"errors"
	"testing"
)

// 基本索引单测。形状期望全部来自 numpy 2.5.4 实测(gen_index.py,)。

func shape3(t *testing.T) *Array {
	t.Helper()
	a, err := Arange(0, 24, 1, DInt64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	r, err := a.Reshape([]int{2, 3, 4})
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	return r
}

func wantShape(t *testing.T, a *Array, want ...int) {
	t.Helper()
	if !shapeEqual(a.Shape(), want) {
		t.Fatalf("shape = %v, want %v", a.Shape(), want)
	}
}

func int64At(t *testing.T, a *Array, i ...int) int64 {
	t.Helper()
	v, err := a.Item(i...)
	if err != nil {
		t.Fatalf("Item%v: %v", i, err)
	}
	return v.(int64)
}

func TestSubShapes(t *testing.T) {
	a := shape3(t)
	cases := []struct {
		name string
		idx  []Indexer
		want []int
	}{
		{"a[1:2]", []Indexer{Slice(1, 2)}, []int{1, 3, 4}},
		{"a[1:2,...,None]", []Indexer{Slice(1, 2), Ellipsis(), NewAxis()}, []int{1, 3, 4, 1}},
		{"a[...,0]", []Indexer{Ellipsis(), Idx(0)}, []int{2, 3}},
		{"a[0,...,1]", []Indexer{Idx(0), Ellipsis(), Idx(1)}, []int{3}},
		{"a[None]", []Indexer{NewAxis()}, []int{1, 2, 3, 4}},
		{"a[-1]", []Indexer{Idx(-1)}, []int{3, 4}},
		{"a[1:5]", []Indexer{Slice(1, 5)}, []int{1, 3, 4}},
		{"a[::2]", []Indexer{SliceStep(SliceNone, SliceNone, 2)}, []int{1, 3, 4}},
		{"a[1]", []Indexer{Idx(1)}, []int{3, 4}},
		{"a[...]", []Indexer{Ellipsis()}, []int{2, 3, 4}},
		{"a[:]", nil, []int{2, 3, 4}},
		{"a[1,2]", []Indexer{Idx(1), Idx(2)}, []int{4}},
		{"a[1:1]", []Indexer{Slice(1, 1)}, []int{0, 3, 4}},
		{"a[-1,...,-1]", []Indexer{Idx(-1), Ellipsis(), Idx(-1)}, []int{3}},
	}
	for _, c := range cases {
		v, err := a.Sub(c.idx...)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		wantShape(t, v, c.want...)
	}
}

func TestSubOneDimBounds(t *testing.T) {
	x, err := Arange(0, 5, 1, DInt64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	cases := []struct {
		idx  []Indexer
		want []int
	}{
		{[]Indexer{Slice(-10, 3)}, []int{3}},
		{[]Indexer{Slice(3, 1)}, []int{0}},
		{[]Indexer{Slice(1, 10)}, []int{4}},
		{[]Indexer{Slice(-2, SliceNone)}, []int{2}},
		{[]Indexer{SliceStep(SliceNone, SliceNone, 2)}, []int{3}},
	}
	for i, c := range cases {
		v, err := x.Sub(c.idx...)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		wantShape(t, v, c.want...)
	}
	// a[::2] → [0 2 4](实测)
	v, err := x.Sub(SliceStep(SliceNone, SliceNone, 2))
	if err != nil {
		t.Fatalf("step slice: %v", err)
	}
	for i, w := range []int64{0, 2, 4} {
		if got := int64At(t, v, i); got != w {
			t.Fatalf("a[::2][%d] = %d, want %d", i, got, w)
		}
	}
}

// 视图零拷贝:改动视图即改动原数组。
func TestSubIsView(t *testing.T) {
	a := shape3(t)
	v, err := a.Sub(Idx(0), Slice(1, 3))
	if err != nil {
		t.Fatalf("Sub: %v", err)
	}
	if v.buf == nil || a.buf == nil {
		t.Fatalf("buf nil")
	}
	if err := v.SetSubScalar(int64(99)); err != nil {
		t.Fatalf("SetSubScalar: %v", err)
	}
	// a[0,1:3,:] 全变 99;a[0,0,0] 不受影响
	if got := int64At(t, a, 0, 1, 0); got != 99 {
		t.Fatalf("a[0,1,0] = %d, want 99", got)
	}
	if got := int64At(t, a, 0, 2, 3); got != 99 {
		t.Fatalf("a[0,2,3] = %d, want 99", got)
	}
	if got := int64At(t, a, 0, 0, 0); got != 0 {
		t.Fatalf("a[0,0,0] = %d, want 0(未写)", got)
	}
}

func TestSubErrors(t *testing.T) {
	a := shape3(t)
	bad := []struct {
		name string
		idx  []Indexer
	}{
		{"整点越界", []Indexer{Idx(2)}},
		{"整点负越界", []Indexer{Idx(-3)}},
		{"索引项过多", []Indexer{Idx(0), Idx(0), Idx(0), Idx(0)}},
		{"两个省略号", []Indexer{Ellipsis(), Ellipsis()}},
		{"负步长", []Indexer{SliceStep(0, 3, -1)}},
		{"步长 0", []Indexer{SliceStep(0, 3, 0)}},
		{"零值索引项", []Indexer{{}}},
		{"省略号外索引过多", []Indexer{Idx(0), Idx(0), Idx(0), Ellipsis(), Idx(0)}},
	}
	for _, c := range bad {
		if _, err := a.Sub(c.idx...); !errors.Is(err, ErrIndex) {
			t.Fatalf("%s: err = %v, want ErrIndex", c.name, err)
		}
	}
}

// 赋值 unsafe cast 口径(实测 numpy:a[:2] = f64 数组 → 截断)。
func TestSetSubAssignCast(t *testing.T) {
	a, err := Zeros([]int{2, 3}, DInt64)
	if err != nil {
		t.Fatalf("Zeros: %v", err)
	}
	src, err := FromSlice([]float64{1.5, 2.5, 3.5})
	if err != nil {
		t.Fatalf("FromSlice: %v", err)
	}
	if err := a.SetSub(src, Idx(0)); err != nil {
		t.Fatalf("SetSub: %v", err)
	}
	for j, w := range []int64{1, 2, 3} {
		if got := int64At(t, a, 0, j); got != w {
			t.Fatalf("a[0,%d] = %d, want %d(截断)", j, got, w)
		}
	}

	// 广播 (1,3) → 行
	b, _ := Zeros([]int{2, 3}, DInt64)
	row, err := Full([]int{1, 3}, int64(7), DInt64)
	if err != nil {
		t.Fatalf("Full: %v", err)
	}
	if err := b.SetSub(row, Idx(1)); err != nil {
		t.Fatalf("SetSub bcast: %v", err)
	}
	if got := int64At(t, b, 1, 2); got != 7 {
		t.Fatalf("b[1,2] = %d, want 7", got)
	}

	// 标量(Python float)写入整型 → 截断
	c, _ := Zeros([]int{3}, DInt64)
	if err := c.SetSubScalar(1.5, Slice(SliceNone, SliceNone)); err != nil {
		t.Fatalf("SetSubScalar: %v", err)
	}
	if got := int64At(t, c, 0); got != 1 {
		t.Fatalf("c[0] = %d, want 1", got)
	}
}

// 重叠赋值:a[1:] = a[:-1] → numpy 实测 [0 0 1 2 3 4]。
func TestSetSubOverlap(t *testing.T) {
	g, err := Arange(0, 6, 1, DInt64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	src, err := g.Sub(Slice(0, 5))
	if err != nil {
		t.Fatalf("Sub: %v", err)
	}
	if err := g.SetSub(src, Slice(1, 6)); err != nil {
		t.Fatalf("SetSub: %v", err)
	}
	for i, w := range []int64{0, 0, 1, 2, 3, 4} {
		if got := int64At(t, g, i); got != w {
			t.Fatalf("g[%d] = %d, want %d", i, got, w)
		}
	}
}

func TestSetSubErrors(t *testing.T) {
	a := shape3(t)
	src, _ := Zeros([]int{5}, DInt64)
	if err := a.SetSub(src, Idx(0)); !errors.Is(err, ErrShape) {
		t.Fatalf("形状不可广播 err = %v, want ErrShape", err)
	}
	if err := a.SetSub(nil, Idx(0)); !errors.Is(err, ErrDtype) {
		t.Fatalf("nil 源 err = %v, want ErrDtype", err)
	}
	if err := a.SetSubScalar(int64(1), Idx(9)); !errors.Is(err, ErrIndex) {
		t.Fatalf("越界 err = %v, want ErrIndex", err)
	}
}

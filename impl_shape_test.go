package num

import "testing"

// *_like:同形状同 dtype。
func TestLikeFamily(t *testing.T) {
	a := mustSlice(t, []float64{1, 2, 3, 4})
	a, _ = a.Reshape([]int{2, 2})
	for name, f := range map[string]func(*Array) (*Array, error){
		"zeros": ZerosLike, "ones": OnesLike, "empty": EmptyLike,
	} {
		o, err := f(a)
		if err != nil {
			t.Fatal(err)
		}
		if o.Dtype() != a.Dtype() || len(o.Shape()) != 2 || o.Shape()[0] != 2 {
			t.Fatalf("%s_like 形状/dtype 不符: %v %s", name, o.Shape(), o.Dtype())
		}
	}
	o, err := FullLike(a, 7.0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o.Flatten(), 7, 7, 7, 7)
}

// atleast_*:注意 1-D 的 atleast_3d 是 (1,n,1)。
func TestAtleast(t *testing.T) {
	// 0-d
	z := mustNew(t, []int{}, DFloat64)
	if err := z.SetItem([]int{}, 5.0); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		f    func(*Array) (*Array, error)
		want []int
	}{
		{Atleast1D, []int{1}},
		{Atleast2D, []int{1, 1}},
		{Atleast3D, []int{1, 1, 1}},
	} {
		o, err := c.f(z)
		if err != nil {
			t.Fatal(err)
		}
		if !shapeEqual(o.Shape(), c.want) {
			t.Fatalf("0-d → %v, want %v", o.Shape(), c.want)
		}
	}
	// 1-D
	a := bArr(t, []float64{1, 2, 3})
	if o, _ := Atleast1D(a); !shapeEqual(o.Shape(), []int{3}) {
		t.Fatalf("1-D atleast_1d → %v", o.Shape())
	}
	if o, _ := Atleast2D(a); !shapeEqual(o.Shape(), []int{1, 3}) {
		t.Fatalf("1-D atleast_2d → %v", o.Shape())
	}
	if o, _ := Atleast3D(a); !shapeEqual(o.Shape(), []int{1, 3, 1}) {
		t.Fatalf("1-D atleast_3d → %v, want [1 3 1]", o.Shape())
	}
	// 2-D → 3-D
	m := mustSlice(t, []float64{1, 2, 3, 4})
	m, _ = m.Reshape([]int{2, 2})
	if o, _ := Atleast3D(m); !shapeEqual(o.Shape(), []int{2, 2, 1}) {
		t.Fatalf("2-D atleast_3d → %v, want [2 2 1]", o.Shape())
	}
}

// Flip/FlipUD/FlipLR。
func TestFlip(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	o, err := Flip(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o, 4, 3, 2, 1)

	m := mustSlice(t, []float64{1, 2, 3, 4, 5, 6})
	m, _ = m.Reshape([]int{2, 3})
	ud, err := FlipUD(m)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, ud.Flatten(), 4, 5, 6, 1, 2, 3)
	lr, err := FlipLR(m)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, lr.Flatten(), 3, 2, 1, 6, 5, 4)
	both, err := Flip(m, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, both.Flatten(), 6, 5, 4, 3, 2, 1)

	if _, err := FlipLR(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("fliplr 1-D 应报错")
	}
}

// Roll:负移位与 axis=None(先拉平)。
func TestRoll(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	o, err := Roll(a, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o, 4, 1, 2, 3)
	o2, err := Roll(a, -1, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o2, 2, 3, 4, 1)

	m := mustSlice(t, []float64{1, 2, 3, 4, 5, 6})
	m, _ = m.Reshape([]int{2, 3})
	r0, err := Roll(m, 1, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r0.Flatten(), 4, 5, 6, 1, 2, 3)
	r1, err := Roll(m, 1, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r1.Flatten(), 3, 1, 2, 6, 4, 5)
}

// Repeat / Tile。
func TestRepeatTile(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3})
	r, err := Repeat(a, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r, 1, 1, 2, 2, 3, 3)

	m := mustSlice(t, []float64{1, 2, 3, 4})
	m, _ = m.Reshape([]int{2, 2})
	r0, err := Repeat(m, 2, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r0.Shape(), []int{4, 2}) {
		t.Fatalf("repeat axis=0 形状 %v, want [4 2]", r0.Shape())
	}
	bWantF64(t, r0.Flatten(), 1, 2, 1, 2, 3, 4, 3, 4)

	t1, err := Tile(a, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, t1, 1, 2, 3, 1, 2, 3)

	t2, err := Tile(m, []int{2, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(t2.Shape(), []int{4, 2}) {
		t.Fatalf("tile 形状 %v, want [4 2]", t2.Shape())
	}
	bWantF64(t, t2.Flatten(), 1, 2, 3, 4, 1, 2, 3, 4)

	// reps 短于 ndim:前置补 1(tile(a,1) 对 2-D 等价 tile(a,(1,1)))
	t3, err := Tile(m, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(t3.Shape(), []int{2, 2}) {
		t.Fatalf("tile reps 过短: %v", t3.Shape())
	}
}

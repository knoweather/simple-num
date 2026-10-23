package num

import "testing"

// broadcastShapes:numpy 右对齐;broadcast_test 中期望值取自 numpy 文档口径
// (np.broadcast_shapes 文档示例,内容为无争议定案)。
func TestBroadcastShapes(t *testing.T) {
	cases := []struct {
		x, y []int
		want []int
	}{
		{[]int{2, 3}, []int{3}, []int{2, 3}},
		{[]int{3}, []int{2, 3}, []int{2, 3}},
		{[]int{3, 1}, []int{1, 4}, []int{3, 4}},
		{[]int{}, []int{5}, []int{5}},
		{[]int{5}, []int{}, []int{5}},
		{[]int{}, []int{}, []int{}},
		{[]int{1}, []int{3, 4, 5}, []int{3, 4, 5}},
		{[]int{}, []int{}, []int{}},
		{[]int{2, 2, 1}, []int{1, 3}, []int{2, 2, 3}},
	}
	for _, c := range cases {
		got, err := broadcastShapes(c.x, c.y)
		if err != nil {
			t.Fatalf("broadcast %v × %v: %v", c.x, c.y, err)
		}
		if !shapeEqual(got, c.want) {
			t.Errorf("broadcast %v × %v = %v, want %v", c.x, c.y, got, c.want)
		}
	}
	bad := [][2][]int{
		{{2, 2}, {2, 3}},
		{{4}, {3}},
		{{1, 2, 3}, {2, 3, 1}},
	}
	for _, c := range bad {
		if _, err := broadcastShapes(c[0], c[1]); !isErrShape(err) {
			t.Errorf("broadcast %v × %v 应 ErrShape", c[0], c[1])
		}
	}
}

// BroadcastTo:零拷贝视图、广播维 stride 0、读⨉ 均一致。
func TestBroadcastTo(t *testing.T) {
	a := mustSlice(t, []float64{10, 20, 30})
	v, err := a.BroadcastTo([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(v.Shape(), []int{2, 3}) {
		t.Fatalf("形状 %v", v.Shape())
	}
	if v.Strides()[0] != 0 {
		t.Fatalf("广播维步长应 0,得 %v", v.Strides())
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if x, err := v.Item(i, j); err != nil || x != float64(10*(j+1)) {
				t.Fatalf("[%d,%d]=%v err=%v", i, j, x, err)
			}
		}
	}
	// 不兼容 → ErrShape
	if _, err := a.BroadcastTo([]int{2, 2}); !isErrShape(err) {
		t.Errorf("应 ErrShape:%v", err)
	}
	// 0-d 广播到任意形
	s := mustNew(t, []int{}, DFloat64)
	if err := s.SetItem(nil, 3.5); err != nil {
		t.Fatal(err)
	}
	w, err := s.BroadcastTo([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	if x, _ := w.Item(1, 0); x != 3.5 {
		t.Fatalf("0-d 广播读 %v", x)
	}
}

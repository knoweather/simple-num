package num

import (
	"math"
	"testing"
)

// Pad:constant 与四种映射模式。
func TestPad(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3})
	// constant:两侧各补 1,值 9
	c, err := Pad(a, [][2]int{{1, 1}}, "constant", 9)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(c.Shape(), []int{5}) {
		t.Fatalf("形状 %v, want [5]", c.Shape())
	}
	bWantF64(t, c, 9, 1, 2, 3, 9)

	// edge:两端复制边缘值
	e, err := Pad(a, [][2]int{{1, 2}}, "edge", 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, e, 1, 1, 2, 3, 3, 3)

	// reflect:不重复边缘 … 2 1 | 1 2 3 | 2 1 …
	r, err := Pad(a, [][2]int{{2, 2}}, "reflect", 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r, 3, 2, 1, 2, 3, 2, 1)

	// symmetric:重复边缘 … 3 2 1 | 1 2 3 | 3 2 …
	s, err := Pad(a, [][2]int{{2, 2}}, "symmetric", 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, s, 2, 1, 1, 2, 3, 3, 2)

	// wrap:循环
	w, err := Pad(a, [][2]int{{2, 2}}, "wrap", 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, w, 2, 3, 1, 2, 3, 1, 2)

	// 2-D 每轴不同宽度
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	mc, err := Pad(m, [][2]int{{1, 0}, {0, 1}}, "constant", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(mc.Shape(), []int{3, 3}) {
		t.Fatalf("2-D 形状 %v, want [3 3]", mc.Shape())
	}
	bWantF64(t, mc.Flatten(), 0, 0, 0, 1, 2, 0, 3, 4, 0)

	// 非法 mode
	if _, err := Pad(a, [][2]int{{1, 1}}, "bogus", 0); err == nil {
		t.Fatal("未知 mode 应报错")
	}
}

// Histogram:等分箱、边界归属、退化区间。
func TestHistogram(t *testing.T) {
	a := bArr(t, []float64{0, 0.1, 0.2, 0.9, 1.0})
	out, err := Histogram(a, 2, math.NaN(), math.NaN()) // 自动 range [0,1]
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("应返回 (hist, edges),得 %d", len(out))
	}
	if out[0].Dtype() != DInt64 {
		t.Fatalf("hist dtype %s, want int64", out[0].Dtype())
	}
	// [0,0.5) → 3 个(0,0.1,0.2);[0.5,1] → 2 个(0.9,1.0)
	bWantI64(t, out[0], 3, 2)
	// edges = [0, 0.5, 1]
	if e, _ := out[1].Item(0); e != 0.0 {
		t.Fatalf("edges[0] = %v", e)
	}
	if e, _ := out[1].Item(2); e != 1.0 {
		t.Fatalf("edges[2] = %v, want 1", e)
	}

	// 显式 range
	o2, err := Histogram(bArr(t, []float64{0, 5, 10}), 2, 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, o2[0], 1, 2) // [0,5) → {0};[5,10] → {5,10}

	// 常数数组:全部落第 0 箱
	o3, err := Histogram(bArr(t, []float64{3, 3, 3}), 4, math.NaN(), math.NaN())
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := o3[0].Item(0); v != int64(3) {
		t.Fatalf("常数数组应全落首箱,得 %v", v)
	}
}

// Select:首个为真胜出;default 参与 dtype 提升。
func TestSelect(t *testing.T) {
	c1 := bArr(t, []bool{true, false, false})
	c2 := bArr(t, []bool{false, true, false})
	ch1 := bArr(t, []float64{1, 1, 1})
	ch2 := bArr(t, []float64{2, 2, 2})
	def, err := New([]int{}, DInt32)
	if err != nil {
		t.Fatal(err)
	}
	if err := def.SetItem([]int{}, int32(-1)); err != nil {
		t.Fatal(err)
	}
	out, err := Select([]*Array{c1, c2}, []*Array{ch1, ch2}, def)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype %s, want float64(default i4 与 f8 提升)", out.Dtype())
	}
	bWantF64(t, out.Flatten(), 1, 2, -1)

	// 广播:cond 标量 true
	cTrue, err := New([]int{}, DBool)
	if err != nil {
		t.Fatal(err)
	}
	if err := cTrue.SetItem([]int{}, true); err != nil {
		t.Fatal(err)
	}
	b2, err := Select([]*Array{cTrue}, []*Array{bArr(t, []float64{7, 8})}, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, b2.Flatten(), 7, 8)

	if _, err := Select(nil, nil, nil); err == nil {
		t.Fatal("空 conds 应报错")
	}
}

// PutMask:原地覆盖 + values 循环取用 + 非连续视图回退。
func TestPutMask(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	m := bArr(t, []bool{true, false, true, false})
	v := bArr(t, []float64{9, 8})
	if err := PutMask(a, m, v); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, a, 9, 2, 9, 4) // numpy 口径:按**下标**取用 values[i%%len] → 位置 0→9、位置 2→9

	// 二维 + 同形 mask(numpy 要求 mask 与 a 同形)
	m2 := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	mask2 := mustMat(t, []int{2, 2}, []bool{false, true, false, true})
	if err := PutMask(m2, mask2, bArr(t, []float64{0})); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, m2.Flatten(), 1, 0, 3, 0)

	// 非连续视图(转置)也要写回真实内存
	base := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	tr, err := base.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	// tr = [[1,3],[2,4]];mask 同形全真 → 全写成 5
	all := mustMat(t, []int{2, 2}, []bool{true, true, true, true})
	if err := PutMask(tr, all, bArr(t, []float64{5})); err != nil {
		t.Fatal(err)
	}
	// base 应全为 5(经视图写回)
	for i, w := range []float64{5, 5, 5, 5} {
		if g, _ := base.Item(i/2, i%2); g != w {
			t.Fatalf("非连续视图写回失败:base[%d]=%v", i, g)
		}
	}
}

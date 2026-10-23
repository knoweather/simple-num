package num

import (
	"math"
	"testing"
)

// pad 统计/线性过渡模式:与 numpy 快照逐位对齐。
func TestPadStatModes(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3})

	// linear_ramp
	r1, err := PadWith(a, [][2]int{{2, 0}}, PadOpts{Mode: "linear_ramp"})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r1, 0, 0.5, 1, 2, 3)

	r2, err := PadWith(a, [][2]int{{2, 2}}, PadOpts{Mode: "linear_ramp"})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r2, 0, 0.5, 1, 2, 3, 1.5, 0)

	r3, err := PadWith(a, [][2]int{{0, 3}}, PadOpts{Mode: "linear_ramp", EndAfter: 9})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r3, 1, 2, 3, 5, 7, 9)

	// 统计模式
	for _, c := range []struct {
		mode string
		want []float64
	}{
		{"maximum", []float64{3, 3, 1, 2, 3, 3}},
		{"minimum", []float64{1, 1, 1, 2, 3, 1}},
		{"mean", []float64{2, 2, 1, 2, 3, 2}},
		{"median", []float64{2, 2, 1, 2, 3, 2}},
	} {
		out, err := PadWith(a, [][2]int{{2, 1}}, PadOpts{Mode: c.mode})
		if err != nil {
			t.Fatalf("%s: %v", c.mode, err)
		}
		bWantF64(t, out, c.want...)
	}

	// 整型:输出保持整型,linear_ramp 按截断
	i := bArr(t, []int32{1, 2, 3})
	ir, err := PadWith(i, [][2]int{{2, 0}}, PadOpts{Mode: "linear_ramp"})
	if err != nil {
		t.Fatal(err)
	}
	if ir.Dtype() != DInt32 {
		t.Fatalf("linear_ramp 整型 dtype %s, want int32", ir.Dtype())
	}
	wantI32(t, ir, 0, 0, 1, 2, 3)

	im, err := PadWith(i, [][2]int{{2, 1}}, PadOpts{Mode: "maximum"})
	if err != nil {
		t.Fatal(err)
	}
	if im.Dtype() != DInt32 {
		t.Fatalf("maximum 整型 dtype %s", im.Dtype())
	}
	wantI32(t, im, 3, 3, 1, 2, 3, 3)
}

// pad 二维:按"线"统计。
func TestPadStat2D(t *testing.T) {
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})

	// 轴 0 maximum(每列取最大)
	o1, err := PadWith(m, [][2]int{{1, 1}, {0, 0}}, PadOpts{Mode: "maximum"})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o1.Shape(), []int{4, 2}) {
		t.Fatalf("形状 %v, want [4 2]", o1.Shape())
	}
	bWantF64(t, o1.Flatten(), 3, 4, 1, 2, 3, 4, 3, 4)

	// 轴 1 mean(每行取均值)
	o2, err := PadWith(m, [][2]int{{0, 0}, {1, 1}}, PadOpts{Mode: "mean"})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o2.Shape(), []int{2, 4}) {
		t.Fatalf("形状 %v, want [2 4]", o2.Shape())
	}
	bWantF64(t, o2.Flatten(), 1.5, 1, 2, 1.5, 3.5, 3, 4, 3.5)

	// 轴 0 linear_ramp(每列独立斜坡)
	o3, err := PadWith(m, [][2]int{{1, 1}, {0, 0}}, PadOpts{Mode: "linear_ramp"})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o3.Flatten(), 0, 0, 1, 2, 3, 4, 0, 0)

	// 不支持的 mode
	if _, err := PadWith(m, nil, PadOpts{Mode: "bogus"}); err == nil {
		t.Fatal("未知 mode 应报错")
	}
	// 0-d
	if _, err := PadWith(mustNew(t, []int{}, DFloat64), nil, PadOpts{Mode: "mean"}); err == nil {
		t.Fatal("0-d 应报错")
	}
}

// UniqueAxis:按切片去重(字典序排序)。
func TestUniqueAxis(t *testing.T) {
	u := mustMat(t, []int{4, 2}, []int32{1, 2, 1, 2, 3, 4, 1, 2})
	o, err := UniqueAxis(u, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", o.Shape())
	}
	wantI32(t, o.Flatten(), 1, 2, 3, 4)

	// 乱序输入也按字典序排序
	u2 := mustMat(t, []int{3, 2}, []int32{3, 4, 1, 2, 3, 4})
	o2, err := UniqueAxis(u2, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantI32(t, o2.Flatten(), 1, 2, 3, 4)

	// 沿轴 1
	u3 := mustMat(t, []int{2, 2}, []float64{1, 2, 1, 3})
	o3, err := UniqueAxis(u3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o3.Shape(), []int{2, 2}) {
		t.Fatalf("axis=1 形状 %v, want [2 2]", o3.Shape())
	}
	bWantF64(t, o3.Flatten(), 1, 2, 1, 3)

	// 1-D:等价普通 unique
	v := bArr(t, []int32{3, 1, 3})
	ov, err := UniqueAxis(v, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantI32(t, ov, 1, 3)

	// 0-d 报错
	if _, err := UniqueAxis(mustNew(t, []int{}, DFloat64), 0); err == nil {
		t.Fatal("0-d 应报错")
	}
	// 轴越界
	if _, err := UniqueAxis(u, 5); err == nil {
		t.Fatal("轴越界应报错")
	}
	// NaN 参与比较(排序不崩,且 NaN 行排最后)
	n := mustMat(t, []int{2, 1}, []float64{math.NaN(), 1})
	if _, err := UniqueAxis(n, 0); err != nil {
		t.Fatalf("NaN 场景不应报错: %v", err)
	}
}

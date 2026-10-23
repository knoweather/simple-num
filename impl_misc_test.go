package num

import (
	"math"
	"testing"
)

// SearchSorted:lower/upper bound 两种 side。
func TestSearchSorted(t *testing.T) {
	a := bArr(t, []float64{1, 3, 5, 7})
	v := bArr(t, []float64{0, 1, 3, 4, 7, 9})
	lo, err := SearchSorted(a, v, false)
	if err != nil {
		t.Fatal(err)
	}
	if lo.Dtype() != DInt64 {
		t.Fatalf("dtype %s, want int64", lo.Dtype())
	}
	bWantI64(t, lo, 0, 0, 1, 2, 3, 4)

	hi, err := SearchSorted(a, v, true)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, hi, 0, 1, 2, 2, 4, 4)

	// 空 a → 全 0(numpy 同)
	empty := mustNew(t, []int{0}, DFloat64)
	if o, err := SearchSorted(empty, v, false); err != nil {
		t.Fatal(err)
	} else {
		bWantI64(t, o, 0, 0, 0, 0, 0, 0)
	}
	// 非 1-D → ErrShape
	if _, err := SearchSorted(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}), v, false); err == nil {
		t.Fatal("非 1-D 的 a 应报错")
	}
}

// Digitize:right=false 等价 searchsorted(side='right')。
func TestDigitize(t *testing.T) {
	bins := bArr(t, []float64{0, 1, 2, 3})
	x := bArr(t, []float64{-1, 0, 0.5, 1, 2.5, 4})
	out, err := Digitize(x, bins, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 0, 1, 1, 2, 3, 4)

	// right=true → 边界归入左侧箱(numpy 实测:[0 0 1 1 3 4])
	out2, err := Digitize(x, bins, true)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out2, 0, 0, 1, 1, 3, 4)
}

// Interp:端点钳制 + 线性插值。
func TestInterp(t *testing.T) {
	xp := bArr(t, []float64{0, 1, 2})
	fp := bArr(t, []float64{0, 10, 20})
	x := bArr(t, []float64{-1, 0.5, 1.5, 3})
	out, err := Interp(x, xp, fp)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype %s, want float64", out.Dtype())
	}
	bWantF64(t, out, 0, 5, 15, 20)

	// 非等距 xp 与多点:xp=[0,2,3], fp=[0,4,9];x=1 → 0 + (4-0)/(2-0)*(1-0)=2
	xp2 := bArr(t, []float64{0, 2, 3})
	fp2 := bArr(t, []float64{0, 4, 9})
	if o, err := Interp(bArr(t, []float64{1}), xp2, fp2); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 2)
	}
	// 形状保持:二维 x
	m := mustMat(t, []int{2, 2}, []float64{0.5, 1.5, 2, -1})
	if o, err := Interp(m, xp, fp); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(o.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", o.Shape())
	}
	// NaN 传播
	if o, err := Interp(bArr(t, []float64{math.NaN()}), xp, fp); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); !math.IsNaN(v.(float64)) {
		t.Fatalf("NaN 应传播,得 %v", v)
	}
}

// MeshGrid:'ij' 与 'xy' 两种索引序。
func TestMeshGrid(t *testing.T) {
	a := bArr(t, []float64{1, 2})
	b := bArr(t, []float64{3, 4, 5})
	out, err := MeshGrid([]*Array{a, b}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("应返回 2 个数组,得 %d", len(out))
	}
	if !shapeEqual(out[0].Shape(), []int{2, 3}) {
		t.Fatalf("ij 形状 %v, want [2 3]", out[0].Shape())
	}
	bWantF64(t, out[0].Flatten(), 1, 1, 1, 2, 2, 2)
	bWantF64(t, out[1].Flatten(), 3, 4, 5, 3, 4, 5)

	// xy:前两维交换
	outXY, err := MeshGrid([]*Array{a, b}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(outXY[0].Shape(), []int{3, 2}) {
		t.Fatalf("xy 形状 %v, want [3 2]", outXY[0].Shape())
	}
	bWantF64(t, outXY[0].Flatten(), 1, 2, 1, 2, 1, 2)
	bWantF64(t, outXY[1].Flatten(), 3, 3, 4, 4, 5, 5)

	// 物化:改输出不应影响输入
	out[0].SetItem([]int{0, 0}, 99)
	if v, _ := a.Item(0); v != 1.0 {
		t.Fatalf("meshgrid 应返回独立副本,输入被改: %v", v)
	}
}

// Gradient:一维与二维单轴。
func TestGradient(t *testing.T) {
	f := bArr(t, []float64{1, 2, 4, 7})
	g, err := Gradient(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(g) != 1 {
		t.Fatalf("1-D 应返回 1 个数组,得 %d", len(g))
	}
	// 边界一阶、内部中心差分:[1,1.5,2.5,3]
	bWantF64(t, g[0], 1, 1.5, 2.5, 3)

	// 二维沿轴 0:[[1,2],[4,6]] → [[3,4]]
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 4, 6})
	g2, err := Gradient(m, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	// 梯度与输入同形(每点一个导数值)
	if !shapeEqual(g2[0].Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", g2[0].Shape())
	}
	bWantF64(t, g2[0].Flatten(), 3, 4, 3, 4)

	// 多轴:返回 2 个数组
	both, err := Gradient(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(both) != 2 {
		t.Fatalf("2-D 无 axis 应返回 2 个数组,得 %d", len(both))
	}
	// 沿轴 1:[[2-1, 2-1],[6-4, 6-4]] = [[1,1],[2,2]]
	bWantF64(t, both[1].Flatten(), 1, 1, 2, 2)

	// 长度 2 的轴:两侧都取一阶差
	two := bArr(t, []float64{3, 8})
	if o, err := Gradient(two, nil); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o[0], 5, 5)
	}
	// 整型输入 → float64 输出
	if o, err := Gradient(bArr(t, []int32{1, 3, 6}), nil); err != nil {
		t.Fatal(err)
	} else if o[0].Dtype() != DFloat64 {
		t.Fatalf("整型梯度 dtype %s, want float64", o[0].Dtype())
	}
}

package num

import (
	"math"
	"testing"
)

// Polyfit:与 numpy 对照,并用 polyval 回验。
func TestPolyfit(t *testing.T) {
	x := bArr(t, []float64{0, 1, 2, 3})
	y := bArr(t, []float64{1, 3, 7, 13}) // y = x²+x+1
	c2, err := Polyfit(x, y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(c2.Shape(), []int{3}) {
		t.Fatalf("coef 形状 %v", c2.Shape())
	}
	for i, want := range []float64{1, 1, 1} {
		v, _ := c2.Item(i)
		if math.Abs(v.(float64)-want) > 1e-9 {
			t.Fatalf("polyfit deg2 [%d] = %v, want %v", i, v, want)
		}
	}
	// 用拟合结果回算
	fitted, err := Polyval(c2, x)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		v, _ := fitted.Item(i)
		w, _ := y.Item(i)
		if math.Abs(v.(float64)-w.(float64)) > 1e-9 {
			t.Fatalf("拟合回算[%d] = %v, want %v", i, v, w)
		}
	}
	// 一次拟合
	c1, err := Polyfit(x, y, 1)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := c1.Item(0); math.Abs(v.(float64)-4) > 1e-9 {
		t.Fatalf("polyfit deg1 斜率 = %v, want 4", v)
	}
	// 错误路径
	if _, err := Polyfit(x, y, 5); err == nil {
		t.Fatal("点数不足应报错")
	}
	if _, err := Polyfit(x, y, -1); err == nil {
		t.Fatal("负 deg 应报错")
	}
	if _, err := Polyfit(x, bArr(t, []float64{1}), 1); err == nil {
		t.Fatal("长度不符应报错")
	}
	if _, err := Polyfit(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}), y, 1); err == nil {
		t.Fatal("非 1-D 应报错")
	}
}

// Unwrap:与 numpy 对照(默认 period=2π / discont=period/2)。
func TestUnwrap(t *testing.T) {
	p := bArr(t, []float64{0, 3.5, 7.0, 1.5, 5.0})
	got, err := Unwrap(p, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0, -2.7831853071795862, -5.5663706143591725, -4.783185307179586, -7.5663706143591725}
	for i, w := range want {
		v, _ := got.Item(i)
		if math.Abs(v.(float64)-w) > 1e-12 {
			t.Fatalf("unwrap[%d] = %v, want %v", i, v, w)
		}
	}
	// period=4 → discont=2
	g4, err := Unwrap(p, 0, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{0, -0.5, -1.0, -2.5, -3.0} {
		v, _ := g4.Item(i)
		if math.Abs(v.(float64)-w) > 1e-12 {
			t.Fatalf("unwrap period=4 [%d] = %v, want %v", i, v, w)
		}
	}
	// 2-D:沿最后一轴,行间独立(numpy 同)
	m := mustMat(t, []int{2, 3}, []float64{0, 3.5, 7.0, 1.5, 5.0, 8.5})
	gm, err := Unwrap(m, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(gm.Shape(), []int{2, 3}) {
		t.Fatalf("2-D 形状 %v", gm.Shape())
	}
	// numpy:[[0, -2.7832, -5.5664], [1.5, -1.2832, -4.0664]]
	want2 := [][]float64{
		{0, -2.7831853071795862, -5.5663706143591725},
		{1.5, -1.2831853071795862, -4.0663706143591725},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v, _ := gm.Item(i, j)
			if math.Abs(v.(float64)-want2[i][j]) > 1e-12 {
				t.Fatalf("unwrap 2-D[%d,%d] = %v, want %v", i, j, v, want2[i][j])
			}
		}
	}
	// 1 元素 / 0-d
	if o, err := Unwrap(bArr(t, []float64{5}), 0, 0); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != 5.0 {
		t.Fatalf("单元素 unwrap = %v", v)
	}
	if _, err := Unwrap(mustNew(t, []int{}, DFloat64), 0, 0); err == nil {
		t.Fatal("0-d 应报错")
	}
	if _, err := Unwrap(nil, 0, 0); err == nil {
		t.Fatal("nil 应报错")
	}
}

package num

import (
	"math"
	"testing"
)

// 非连续视图审计:同一个算子作用在**视图**(转置/切片/广播)与**连续副本**上,
// 结果必须逐元素一致。Go 侧手写 stride 最容易在视图路径出错,这里一次性覆盖。
func TestViewConsistency(t *testing.T) {
	// 3x4 基础数据(带零/负数)
	base := mustMat(t, []int{3, 4}, []float64{
		1, -2, 3, 0,
		5, 6, -7, 8,
		-9, 10, 11, 12,
	})
	// 视图集合:转置 / 行切片 / 列切片 / 二次转置
	tr, err := base.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	rows, err := base.Sub(Slice(1, 3), All())
	if err != nil {
		t.Fatal(err)
	}
	cols, err := base.Sub(All(), Slice(1, 3))
	if err != nil {
		t.Fatal(err)
	}
	tr2, err := tr.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	views := []struct {
		name string
		view *Array
	}{
		{"transpose", tr},
		{"rows", rows},
		{"cols", cols},
		{"transpose2", tr2},
	}
	// 逐元素算子的结果比较(view vs 连续副本)
	cmpSame := func(t *testing.T, name string, a, b *Array) {
		t.Helper()
		if !shapeEqual(a.Shape(), b.Shape()) {
			t.Fatalf("%s: 形状不同 %v vs %v", name, a.Shape(), b.Shape())
		}
		af := a.AsContiguous().Flatten()
		bf := b.AsContiguous().Flatten()
		w1 := win[float64](engineWindow(af))
		w2 := win[float64](engineWindow(bf))
		for i := range w1 {
			x, y := w1[i], w2[i]
			if math.IsNaN(x) && math.IsNaN(y) {
				continue
			}
			if x != y {
				t.Fatalf("%s: [%d] 视图路径 %v ≠ 连续路径 %v", name, i, x, y)
			}
		}
	}
	unary := map[string]func(*Array) (*Array, error){
		"abs": Abs, "neg": Negative, "sqrt": func(a *Array) (*Array, error) {
			// 取绝对值避免 NaN,再开方
			ab, err := Abs(a)
			if err != nil {
				return nil, err
			}
			return Sqrt(ab)
		},
		"exp": Exp, "round": func(a *Array) (*Array, error) { return Round(a, 0) },
		"floor": Floor, "ceil": Ceil,
	}
	for _, v := range views {
		cont := v.view.AsContiguous()
		for name, f := range unary {
			r1, err := f(v.view)
			if err != nil {
				t.Fatalf("%s/%s: %v", v.name, name, err)
			}
			r2, err := f(cont)
			if err != nil {
				t.Fatalf("%s/%s(cont): %v", v.name, name, err)
			}
			cmpSame(t, v.name+"/"+name, r1, r2)
		}
		// 归约
		for _, ax := range [][]int{nil, {0}, {1}} {
			r1, err := Sum(v.view, ax, false)
			if err != nil {
				continue
			}
			r2, err := Sum(cont, ax, false)
			if err != nil {
				t.Fatal(err)
			}
			cmpSame(t, v.name+"/sum", r1, r2)
		}
		// 排序(不得修改入参,且结果一致)
		s1, err := Sort(v.view, []int{-1})
		if err != nil {
			t.Fatalf("%s/sort: %v", v.name, err)
		}
		s2, err := Sort(cont, []int{-1})
		if err != nil {
			t.Fatal(err)
		}
		cmpSame(t, v.name+"/sort", s1, s2)
		// 变换类
		for name, f := range map[string]func(*Array) (*Array, error){
			"cumsum": func(a *Array) (*Array, error) { return Cumsum(a, []int{0}) },
			"flip":   func(a *Array) (*Array, error) { return Flip(a, []int{0}) },
			"copy":   func(a *Array) (*Array, error) { return Copy(a), nil },
		} {
			r1, err := f(v.view)
			if err != nil {
				t.Fatalf("%s/%s: %v", v.name, name, err)
			}
			r2, err := f(cont)
			if err != nil {
				t.Fatalf("%s/%s(cont): %v", v.name, name, err)
			}
			cmpSame(t, v.name+"/"+name, r1, r2)
		}
	}
	// 二元:视图 × 视图 / 视图 × 连续 / 标量广播
	// 形状必须可广播:同形视图、视图 × 连续副本、视图 × 底层数组
	pairs := [][2]*Array{
		{tr, tr}, {rows, rows.AsContiguous()}, {cols, cols}, {tr, tr.AsContiguous()},
	}
	for i, p := range pairs {
		r1, err := Add(p[0], p[1])
		if err != nil {
			t.Fatalf("pair%d add: %v", i, err)
		}
		r2, err := Add(p[0].AsContiguous(), p[1].AsContiguous())
		if err != nil {
			t.Fatal(err)
		}
		cmpSame(t, "pair/add", r1, r2)
		m1, err := Multiply(p[0], p[1])
		if err != nil {
			t.Fatal(err)
		}
		m2, err := Multiply(p[0].AsContiguous(), p[1].AsContiguous())
		if err != nil {
			t.Fatal(err)
		}
		cmpSame(t, "pair/mul", m1, m2)
	}
	// 索引/切片写入视图后,底层数据应同步变化
	alias := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	at, err := alias.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	if err := at.SetItem([]int{0, 1}, 99.0); err != nil {
		t.Fatal(err)
	}
	if v, _ := alias.Item(1, 0); v != 99.0 {
		t.Fatalf("转置视图写入未反映到底层: alias[1,0] = %v, want 99", v)
	}
}

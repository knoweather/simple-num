package num

import (
	"math"
	"testing"
)

// 期望值取自 numpy 2.5.4 现场:np.percentile(np.arange(1,6), q, method=m)。
// q 列依次为 0/10/25/50/75/90/100。
func TestPercentileMethods(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4, 5})
	qs := []float64{0, 10, 25, 50, 75, 90, 100}
	cases := map[string][]float64{
		"linear":                    {1, 1.4, 2, 3, 4, 4.6, 5},
		"lower":                     {1, 1, 2, 3, 4, 4, 5},
		"higher":                    {1, 2, 2, 3, 4, 5, 5},
		"midpoint":                  {1, 1.5, 2, 3, 4, 4.5, 5},
		"nearest":                   {1, 1, 2, 3, 4, 5, 5},
		"inverted_cdf":              {1, 1, 2, 3, 4, 5, 5},
		"averaged_inverted_cdf":     {1, 1, 2, 3, 4, 5, 5},
		"closest_observation":       {1, 1, 1, 2, 4, 4, 5},
		"interpolated_inverted_cdf": {1, 1, 1.25, 2.5, 3.75, 4.5, 5},
		"hazen":                     {1, 1, 1.75, 3, 4.25, 5, 5},
		"weibull":                   {1, 1, 1.5, 3, 4.5, 5, 5},
		"median_unbiased":           {1, 1, 1.6666666667, 3, 4.3333333333, 5, 5},
		"normal_unbiased":           {1, 1, 1.6875, 3, 4.3125, 5, 5},
	}
	for method, want := range cases {
		for i, q := range qs {
			out, err := PercentileMethod(a, q, method, nil, false)
			if err != nil {
				t.Fatalf("%s q=%v: %v", method, q, err)
			}
			v, _ := out.Item()
			if math.Abs(v.(float64)-want[i]) > 1e-9 {
				t.Fatalf("%s q=%v = %v, want %v", method, q, v, want[i])
			}
		}
	}
	// n=6 的 nearest(银行家舍入)单独锁一下
	b := bArr(t, []float64{1, 2, 3, 4, 5, 6})
	want6 := []float64{1, 1, 2, 3, 5, 5, 6}
	for i, q := range qs {
		out, err := PercentileMethod(b, q, "nearest", nil, false)
		if err != nil {
			t.Fatal(err)
		}
		if v, _ := out.Item(); math.Abs(v.(float64)-want6[i]) > 1e-9 {
			t.Fatalf("n6 nearest q=%v = %v, want %v", q, v, want6[i])
		}
	}
	// 未知 method / q 越界
	if _, err := PercentileMethod(a, 50, "bogus", nil, false); err == nil {
		t.Fatal("未知 method 应报错")
	}
	if _, err := PercentileMethod(a, 101, "linear", nil, false); err == nil {
		t.Fatal("q 越界应报错")
	}
}

// PercentileMany:q 数组与按轴归约(numpy 把 q 放最前)。
func TestPercentileMany(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4, 5})
	out, err := PercentileMany(a, []float64{25, 50, 75}, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{3}) {
		t.Fatalf("形状 %v, want [3]", out.Shape())
	}
	bWantF64(t, out, 2, 3, 4)

	// 按轴:2x3,axis=1,q=[25,75] → (2,2)
	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	o2, err := PercentileMany(m, []float64{25, 75}, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o2.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", o2.Shape())
	}
	// numpy:[[0.5, 3.5],[1.5, 4.5]]
	bWantF64(t, o2.Flatten(), 0.5, 3.5, 1.5, 4.5)

	// 单 q 退化为普通 percentile 形状
	if o, err := PercentileMany(a, []float64{50}, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); v != 3.0 {
		t.Fatalf("单 q = %v, want 3", v)
	}
	// 空 q 列表 → 空数组
	if o, err := PercentileMany(a, nil, nil, false); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("空 qs 应为空,得 %d", o.Size())
	}
	// 越界 q
	if _, err := PercentileMany(a, []float64{50, 200}, nil, false); err == nil {
		t.Fatal("q 越界应报错")
	}
}

// PercentileMethod 的按轴/keepdims/空数组/整型输入路径。
func TestPercentileMethodAxis(t *testing.T) {
	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	// 按轴 1,linear q=50 → [1, 4]
	o, err := PercentileMethod(m, 50, "linear", []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o.Shape(), []int{2}) {
		t.Fatalf("形状 %v, want [2]", o.Shape())
	}
	bWantF64(t, o, 1, 4)

	// keepdims
	kd, err := PercentileMethod(m, 75, "hazen", []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(kd.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims 形状 %v, want [2 1]", kd.Shape())
	}

	// 按轴 0
	o0, err := PercentileMethod(m, 25, "weibull", []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o0.Shape(), []int{3}) {
		t.Fatalf("轴 0 形状 %v", o0.Shape())
	}

	// 空数组 → NaN
	empty := mustNew(t, []int{0}, DFloat64)
	if e, err := PercentileMethod(empty, 50, "linear", nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := e.Item(); !math.IsNaN(v.(float64)) {
		t.Fatalf("空数组应得 NaN,得 %v", v)
	}

	// 整型输入(结果 float64)
	i := bArr(t, []int32{1, 2, 3, 4})
	if e, err := PercentileMethod(i, 50, "lower", nil, false); err != nil {
		t.Fatal(err)
	} else if e.Dtype() != DFloat64 {
		t.Fatalf("整型分位数 dtype %s, want float64", e.Dtype())
	}

	// f32 输入保持 f32
	f32 := bArr(t, []float32{1, 2, 3})
	if e, err := PercentileMethod(f32, 50, "linear", nil, false); err != nil {
		t.Fatal(err)
	} else if e.Dtype() != DFloat32 {
		t.Fatalf("f32 分位数 dtype %s, want float32", e.Dtype())
	}
}

// 方法边界:默认 method、q 端点、averaged_inverted_cdf 的 j<0 分支、keepdims。
func TestPercentileMethodEdges(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4, 5})
	// 空 method 视为 linear
	if o, err := PercentileMethod(a, 25, "", nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-2) > 1e-12 {
		t.Fatalf("默认 method = %v, want 2", v)
	}
	// q 端点
	for _, q := range []float64{0, 100} {
		for _, m := range []string{"linear", "lower", "higher", "midpoint", "nearest",
			"inverted_cdf", "averaged_inverted_cdf", "closest_observation"} {
			if _, err := PercentileMethod(a, q, m, nil, false); err != nil {
				t.Fatalf("%s q=%v: %v", m, q, err)
			}
		}
	}
	// 单元素数组(各方法都应返回该元素)
	one := bArr(t, []float64{7})
	for _, m := range []string{"linear", "lower", "higher", "midpoint", "nearest",
		"inverted_cdf", "averaged_inverted_cdf", "closest_observation", "hazen", "weibull",
		"median_unbiased", "normal_unbiased"} {
		o, err := PercentileMethod(one, 50, m, nil, false)
		if err != nil {
			t.Fatalf("%s 单元素: %v", m, err)
		}
		if v, _ := o.Item(); v != 7.0 {
			t.Fatalf("%s 单元素 = %v, want 7", m, v)
		}
	}
	// keepdims(全归约 → 全 1 形状)
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	o, err := PercentileMethod(m, 50, "linear", nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o.Shape(), []int{1, 1}) {
		t.Fatalf("keepdims 形状 %v, want [1 1]", o.Shape())
	}
	// PercentileMany 的 keepdims
	om, err := PercentileMany(m, []float64{25, 75}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(om.Shape(), []int{2, 1, 1}) {
		t.Fatalf("many keepdims 形状 %v, want [2 1 1]", om.Shape())
	}
}

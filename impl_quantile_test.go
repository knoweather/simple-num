package num

import (
	"math"
	"testing"
)

// Quantile 族:q∈[0,1] 与 percentile 同源。
func TestQuantile(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4, 5})
	// numpy:np.quantile(a, 0.25) = 2.0
	if o, err := Quantile(a, 0.25, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-2) > 1e-12 {
		t.Fatalf("quantile(0.25) = %v, want 2", v)
	}
	// method 透传:weibull 0.25 → 1.5
	if o, err := QuantileMethod(a, 0.25, "weibull", nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-1.5) > 1e-12 {
		t.Fatalf("quantile weibull = %v, want 1.5", v)
	}
	// q 数组
	om, err := QuantileMany(a, []float64{0.25, 0.5}, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(om.Shape(), []int{2}) {
		t.Fatalf("形状 %v, want [2]", om.Shape())
	}
	bWantF64(t, om, 2, 3)
	// 越界
	if _, err := Quantile(a, 1.5, nil, false); err == nil {
		t.Fatal("q>1 应报错")
	}
	if _, err := QuantileMany(a, []float64{0.5, 2}, nil, false); err == nil {
		t.Fatal("q 越界应报错")
	}
}

// NanPercentile / NanQuantile:剔除 NaN 后计算。
func TestNanPercentile(t *testing.T) {
	a := bArr(t, []float64{1, 2, math.NaN(), 4, 5})
	// 非 NaN 为 [1,2,4,5];numpy:50 → 3,25 → 1.75
	if o, err := NanPercentile(a, 50, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-3) > 1e-12 {
		t.Fatalf("nanpercentile 50 = %v, want 3", v)
	}
	if o, err := NanPercentile(a, 25, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-1.75) > 1e-12 {
		t.Fatalf("nanpercentile 25 = %v, want 1.75", v)
	}
	// nanquantile 0.75 → 4.25
	if o, err := NanQuantile(a, 0.75, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-4.25) > 1e-12 {
		t.Fatalf("nanquantile 0.75 = %v, want 4.25", v)
	}

	// 按轴:numpy [[1,nan,3],[4,5,6]] axis=1 → [2, 5]
	m := mustMat(t, []int{2, 3}, []float64{1, math.NaN(), 3, 4, 5, 6})
	om, err := NanPercentile(m, 50, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, om, 2, 5)

	// 全 NaN → NaN
	all := bArr(t, []float64{math.NaN(), math.NaN()})
	if o, err := NanPercentile(all, 50, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); !math.IsNaN(v.(float64)) {
		t.Fatalf("全 NaN 应得 NaN,得 %v", v)
	}

	// 无 NaN 时与普通 percentile 一致
	clean := bArr(t, []float64{1, 2, 3, 4})
	p1, err := NanPercentile(clean, 50, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := Percentile(clean, 50, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	v1, _ := p1.Item()
	v2, _ := p2.Item()
	if v1 != v2 {
		t.Fatalf("无 NaN 时两者应一致: %v vs %v", v1, v2)
	}

	// keepdims + 整型输入(f64 输出)
	i := bArr(t, []int32{1, 2, 3})
	if o, err := NanPercentile(i, 50, nil, true); err != nil {
		t.Fatal(err)
	} else if o.Dtype() != DFloat64 {
		t.Fatalf("整型 dtype %s, want float64", o.Dtype())
	}
	// q 越界
	if _, err := NanPercentile(a, 101, nil, false); err == nil {
		t.Fatal("q 越界应报错")
	}
	if _, err := NanQuantile(a, -0.1, nil, false); err == nil {
		t.Fatal("q 越界应报错")
	}
	// 空数组 → NaN
	empty := mustNew(t, []int{0}, DFloat64)
	if o, err := NanPercentile(empty, 50, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); !math.IsNaN(v.(float64)) {
		t.Fatalf("空数组应得 NaN,得 %v", v)
	}
}

// quantile/nan* 的按轴、keepdims、f32 与空形状路径。
func TestQuantileAxisPaths(t *testing.T) {
	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	// quantile 按轴
	o, err := Quantile(m, 0.5, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o, 1, 4)
	// quantile keepdims
	kd, err := Quantile(m, 0.25, []int{0}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(kd.Shape(), []int{1, 3}) {
		t.Fatalf("keepdims 形状 %v, want [1 3]", kd.Shape())
	}
	// nanpercentile keepdims(f32 输入 → f32 输出)
	f32 := mustMat(t, []int{2, 2}, []float32{1, 2, 3, 4})
	nk, err := NanPercentile(f32, 50, []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(nk.Shape(), []int{2, 1}) {
		t.Fatalf("nan keepdims 形状 %v, want [2 1]", nk.Shape())
	}
	if nk.Dtype() != DFloat32 {
		t.Fatalf("f32 输入 dtype %s, want float32", nk.Dtype())
	}
	// nanquantile 按轴
	withNaN := mustMat(t, []int{2, 2}, []float64{1, math.NaN(), 3, 4})
	oq, err := NanQuantile(withNaN, 0.5, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	// 第 0 行只剩 [1] → 1;第 1 行 [3,4] → 3.5
	bWantF64(t, oq, 1, 3.5)

	// 空轴的 nan 变体:0 行 → 空输出
	zeroRows := mustNew(t, []int{0, 2}, DFloat64)
	if o, err := NanPercentile(zeroRows, 50, []int{1}, false); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("0 行应得空输出,得 %d", o.Size())
	}
	// 归约轴长度为 0:分位数无定义 → ErrShape(numpy 在这里同样报错)。
	// 顺带记录:此前 axisRows 会因 cont.Size/n 除零而 panic,已修。
	zeroCols := mustNew(t, []int{2, 0}, DFloat64)
	if _, err := Quantile(zeroCols, 0.5, []int{1}, false); err == nil {
		t.Fatal("0 长度归约轴应报错")
	}
}

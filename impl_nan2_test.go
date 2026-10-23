package num

import (
	"math"
	"testing"
)

// NanMedian:跳过 NaN 的中位数(全 NaN → NaN)。
func TestNanMedian(t *testing.T) {
	a := bArr(t, []float64{3, math.NaN(), 1, 2, math.NaN()})
	o, err := NanMedian(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := o.Item(); v != 2.0 {
		t.Fatalf("nanmedian = %v, want 2", v)
	}
	// 空轴的 NaN 已被剔除
	m := mustMat(t, []int{2, 3}, []float64{3, math.NaN(), 1, 5, 2, 4})
	om, err := NanMedian(m, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, om, 2, 4)
	// 全 NaN → NaN
	all := bArr(t, []float64{math.NaN(), math.NaN()})
	if v, err := NanMedian(all, nil, false); err != nil {
		t.Fatal(err)
	} else if x, _ := v.Item(); !math.IsNaN(x.(float64)) {
		t.Fatalf("全 NaN 应得 NaN,得 %v", x)
	}
	// keepdims
	kd, err := NanMedian(m, []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(kd.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims 形状 %v", kd.Shape())
	}
}

// NanArgMin / NanArgMax:下标 int64,跳过 NaN,全 NaN 报错。
func TestNanArg(t *testing.T) {
	a := bArr(t, []float64{3, math.NaN(), 1, 2, math.NaN()})
	mn, err := NanArgMin(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if mn.Dtype() != DInt64 {
		t.Fatalf("dtype %s, want int64", mn.Dtype())
	}
	if v, _ := mn.Item(); v != int64(2) {
		t.Fatalf("nanargmin = %v, want 2", v)
	}
	mx, err := NanArgMax(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := mx.Item(); v != int64(0) {
		t.Fatalf("nanargmax = %v, want 0", v)
	}

	m := mustMat(t, []int{2, 3}, []float64{3, math.NaN(), 1, 5, 2, 4})
	// axis=1 → [2, 1]
	am, err := NanArgMin(m, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, am, 2, 1)
	// axis=0 → [1, 1, 1]
	ax, err := NanArgMax(m, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, ax, 1, 1, 1)

	// keepdims
	kd, err := NanArgMin(m, []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(kd.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims 形状 %v", kd.Shape())
	}

	// 全 NaN → 报错(numpy 抛 ValueError)
	all := bArr(t, []float64{math.NaN(), math.NaN()})
	if _, err := NanArgMin(all, nil, false); err == nil {
		t.Fatal("全 NaN 求下标应报错")
	}
	// 0-d / 空数组 / 空轴
	if _, err := NanArgMin(mustNew(t, []int{}, DFloat64), nil, false); err != nil {
		// 0-d 全归约会走 nanArgOf,size=1 → 允许(非 NaN 时返回 0)
		t.Fatalf("0-d 不应报错: %v", err)
	}
	if _, err := NanArgMin(mustNew(t, []int{0}, DFloat64), nil, false); err == nil {
		t.Fatal("空数组应报错")
	}
	if _, err := NanArgMin(mustNew(t, []int{2, 0}, DFloat64), []int{1}, false); err == nil {
		t.Fatal("空轴应报错")
	}
	// 0 行的轴:空输出
	if o, err := NanArgMin(mustNew(t, []int{0, 2}, DFloat64), []int{1}, false); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("0 行应得空输出,得 %d", o.Size())
	}
	// nil 数组
	if _, err := NanArgMax(nil, nil, false); err == nil {
		t.Fatal("nil 应报错")
	}
}

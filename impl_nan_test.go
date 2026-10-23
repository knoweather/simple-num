package num

import (
	"math"
	"testing"
)

// NanSum:NaN 视作 0;整型委托 Sum。
func TestNanSum(t *testing.T) {
	a := bArr(t, []float64{1, math.NaN(), 3})
	out, err := NanSum(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out.Flatten(), 4)

	all := bArr(t, []float64{math.NaN(), math.NaN()})
	if out, err := NanSum(all, nil, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, out.Flatten(), 0)
	}

	// 轴归约:[[1,nan],[3,4]] axis=0 → [4,4]
	m := mustSlice(t, []float64{1, math.NaN(), 3, 4})
	m, err = m.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	o2, err := NanSum(m, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o2.Flatten(), 4, 4)

	// 整型(无 NaN)与 Sum 等价
	i := bArr(t, []int32{1, 2, 3})
	oi, err := NanSum(i, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, oi.Flatten(), 6)
}

// NanMax/NanMin:跳过 NaN;全 NaN 返回 NaN。
func TestNanMaxMin(t *testing.T) {
	a := bArr(t, []float64{1, math.NaN(), 3, 2})
	out, err := NanMax(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out.Flatten(), 3)
	if out, err := NanMin(a, nil, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, out.Flatten(), 1)
	}

	allNaN := bArr(t, []float64{math.NaN(), math.Float64frombits(0x7ff8000000000003)})
	if out, err := NanMax(allNaN, nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := out.Flatten().Item(0); !math.IsNaN(v.(float64)) {
		t.Fatalf("全 NaN 的 nanmax 应 NaN,得 %v", v)
	}

	// 轴归约:[[1,nan],[5,2]] axis=1 → [1,5]
	m := mustSlice(t, []float64{1, math.NaN(), 5, 2})
	m, err = m.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	o2, err := NanMax(m, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o2.Flatten(), 1, 5)

	// 整型委托(min/max 保输入 dtype)
	i := bArr(t, []int32{-3, 7})
	if o, err := NanMax(i, nil, false); err != nil {
		t.Fatal(err)
	} else {
		wantI32(t, o.Flatten(), 7)
	}
}

// NanMean/NanVar/NanStd:剔除 NaN 后再算(等价 numpy 的 where=~isnan)。
func TestNanMeanVarStd(t *testing.T) {
	a := bArr(t, []float64{1, math.NaN(), 3, 5}) // 非 NaN:[1,3,5]
	mean, err := NanMean(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mean.Flatten(), 3)
	v, err := NanVar(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if g := getI(t, v.Flatten(), 0).(float64); math.Abs(g-8.0/3.0) > 1e-12 {
		t.Fatalf("nanvar = %v, want %v", g, 8.0/3.0)
	}
	s, err := NanStd(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if g := getI(t, s.Flatten(), 0).(float64); math.Abs(g-math.Sqrt(8.0/3.0)) > 1e-12 {
		t.Fatalf("nanstd = %v, want %v", g, math.Sqrt(8.0/3.0))
	}

	// 轴归约:[[1,nan],[3,5]] axis=1 → mean [1,4] / var [0,1] / std [0,1]
	m := mustSlice(t, []float64{1, math.NaN(), 3, 5})
	m, err = m.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	mo, err := NanMean(m, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mo.Flatten(), 1, 4)
	vo, err := NanVar(m, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, vo.Flatten(), 0, 1)

	// 全 NaN → NaN
	all := bArr(t, []float64{math.NaN(), math.NaN()})
	if o, err := NanMean(all, nil, false); err != nil {
		t.Fatal(err)
	} else if g := getI(t, o.Flatten(), 0).(float64); !math.IsNaN(g) {
		t.Fatalf("全 NaN 的 nanmean 应 NaN,得 %v", g)
	}
}

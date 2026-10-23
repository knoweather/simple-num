package num

import (
	"math"
	"testing"
)

// IsClose:numpy 判定式逐字对齐。
func TestIsCloseBasic(t *testing.T) {
	a := bArr(t, []float64{1.0, 1.0, math.NaN(), math.Inf(1), math.Inf(1), 0})
	b := bArr(t, []float64{1.0, 1.000009, math.NaN(), math.Inf(1), 1.0, 1e-9})
	out, err := IsClose(a, b, 1e-5, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype %s, want bool", out.Dtype())
	}
	// np.isclose(0, 1e-9, 1e-5, 1e-8) == True(atol 覆盖),已用 numpy 实测确认
	bWantBool(t, out.Flatten(), true, true, false, true, false, true)
}

// 边界:atol/rtol 各自起作用;NaN 恒 False;±Inf 只有同值才 True。
func TestIsCloseEdges(t *testing.T) {
	a := bArr(t, []float64{0, 0, math.Inf(1), math.Inf(-1), math.NaN()})
	b := bArr(t, []float64{1e-9, 1e-6, math.Inf(1), math.Inf(1), 1})
	out, err := IsClose(a, b, 0, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out.Flatten(), true, false, true, false, false)

	// 纯 rtol:1.0 vs 1.0000001(rtol=1e-5, atol=0)→ True
	if o, err := IsClose(bArr(t, []float64{1.0}), bArr(t, []float64{1.0000001}), 1e-5, 0); err != nil {
		t.Fatal(err)
	} else {
		bWantBool(t, o.Flatten(), true)
	}
}

// f32 保持 f32 精度;整型输入按 numpy 解析提升。
func TestIsCloseDtype(t *testing.T) {
	a := bArr(t, []float32{1, 2})
	b := bArr(t, []float32{1, 2.000001})
	if out, err := IsClose(a, b, 1e-5, 1e-8); err != nil {
		t.Fatal(err)
	} else {
		bWantBool(t, out.Flatten(), true, true)
	}
	if out, err := IsClose(bArr(t, []int32{1, 2}), bArr(t, []int32{1, 3}), 1e-5, 1e-8); err != nil {
		t.Fatal(err)
	} else {
		bWantBool(t, out.Flatten(), true, false)
	}
}

// AllClose:全真才真;空数组视为真(numpy 同)。
func TestAllClose(t *testing.T) {
	ok, err := AllClose(bArr(t, []float64{1, 2, 3}), bArr(t, []float64{1, 2.0000001, 3}), 1e-5, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("应判全近")
	}
	if ok, err := AllClose(bArr(t, []float64{1}), bArr(t, []float64{2}), 1e-5, 1e-8); err != nil {
		t.Fatal(err)
	} else if ok {
		t.Fatal("1 vs 2 不应判近")
	}
	empty := mustNew(t, []int{0}, DFloat64)
	if ok, err := AllClose(empty, empty, 1e-5, 1e-8); err != nil {
		t.Fatal(err)
	} else if !ok {
		t.Fatal("空数组 allclose 应为真")
	}
}

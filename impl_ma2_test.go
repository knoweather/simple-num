package num

import (
	"math"
	"testing"
)

// MaskedInvalid / MaskedWhere。
func TestMaskedInvalidWhere(t *testing.T) {
	a := bArr(t, []float64{1, 2, math.NaN(), math.Inf(1), 5})
	mi, err := MaskedInvalid(a)
	if err != nil {
		t.Fatal(err)
	}
	// numpy:[False, False, True, True, False]
	for i, w := range []bool{false, false, true, true, false} {
		has := false
		if mw := win[uint8](engineWindow(mi.MaskArray().AsContiguous())); mw[i] != 0 {
			has = true
		}
		if has != w {
			t.Fatalf("masked_invalid[%d] = %v, want %v", i, has, w)
		}
	}
	// 非浮点:掩码全 false
	if o, err := MaskedInvalid(bArr(t, []int32{1, 2})); err != nil {
		t.Fatal(err)
	} else if o.IsMasked() {
		t.Fatal("整型不应被标记")
	}

	// masked_where
	b := bArr(t, []float64{1, 2, 3, 4})
	mw, err := MaskedWhere(bArr(t, []bool{false, true, false, true}), b)
	if err != nil {
		t.Fatal(err)
	}
	got := win[uint8](engineWindow(mw.MaskArray().AsContiguous()))
	for i, w := range []uint8{0, 1, 0, 1} {
		if got[i] != w {
			t.Fatalf("masked_where[%d] = %d, want %d", i, got[i], w)
		}
	}
	if mw.Count() != 2 {
		t.Fatalf("count = %d, want 2", mw.Count())
	}
	// 形状不符
	// 真正不可广播的形状(3 与 2 无法右对齐)
	if _, err := MaskedWhere(bArr(t, []bool{true, false, true}), mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})); err == nil {
		t.Fatal("形状不可广播应报错")
	}
}

// 掩码 Median / Var / Std:跳过被掩码元素(与 numpy 快照对齐)。
func TestMaskedMedianVarStd(t *testing.T) {
	m, err := NewMasked(bArr(t, []float64{5, 1, 9, 3}), bArr(t, []bool{false, false, true, false}))
	if err != nil {
		t.Fatal(err)
	}
	// numpy:median 3、var 8/3、std sqrt(8/3)
	med, err := m.Median(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := med.Data().Flatten().Item(0); v != 3.0 {
		t.Fatalf("ma.median = %v, want 3", v)
	}
	vr, err := m.Var(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := vr.Data().Flatten().Item(0); math.Abs(v.(float64)-8.0/3.0) > 1e-12 {
		t.Fatalf("ma.var = %v, want 8/3", v)
	}
	sd, err := m.Std(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := sd.Data().Flatten().Item(0); math.Abs(v.(float64)-math.Sqrt(8.0/3.0)) > 1e-12 {
		t.Fatalf("ma.std = %v, want sqrt(8/3)", v)
	}

	// 按轴
	m2, err := NewMasked(
		mustMat(t, []int{2, 2}, []float64{1, 5, 3, 9}),
		mustMat(t, []int{2, 2}, []bool{false, true, false, false}))
	if err != nil {
		t.Fatal(err)
	}
	amm, err := m2.Median([]int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, amm.Data().Flatten(), 1, 6)
	// 第 0 行只有 1 个有效值 → 未被掩码
	if amm.IsMasked() {
		t.Fatal("单值线不应被掩码")
	}
	asd, err := m2.Std([]int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, asd.Data().Flatten(), 1, 0)

	// 全掩码 → 结果被掩码
	allM, err := NewMasked(bArr(t, []float64{1, 2}), bArr(t, []bool{true, true}))
	if err != nil {
		t.Fatal(err)
	}
	am, err := allM.Median(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !am.IsMasked() {
		t.Fatal("全掩码线应得掩码结果")
	}
	// keepdims
	m3, err := NewMasked(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}),
		mustMat(t, []int{2, 2}, []bool{false, false, true, false}))
	if err != nil {
		t.Fatal(err)
	}
	kd, err := m3.Median([]int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(kd.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims 形状 %v", kd.Shape())
	}
	// 空轴
	if _, err := m3.Median([]int{1}, false); err != nil {
		t.Fatal(err)
	}
	if _, err := (&MaskedArray{data: mustNew(t, []int{2, 0}, DFloat64),
		mask: mustNew(t, []int{2, 0}, DBool)}).Median([]int{1}, false); err == nil {
		t.Fatal("空轴应报错")
	}
}

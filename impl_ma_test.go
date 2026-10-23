package num

import (
	"math"
	"testing"
)

// mkMasked 构造数据+掩码的 MaskedArray。
func mkMasked(t *testing.T, data, mask []float64) *MaskedArray {
	t.Helper()
	d := bArr(t, data)
	mb := make([]bool, len(mask))
	for i, v := range mask {
		mb[i] = v != 0
	}
	mm := bArr(t, mb)
	ma, err := NewMasked(d, mm)
	if err != nil {
		t.Fatal(err)
	}
	return ma
}

// Filled / Compressed / Count / IsMasked。
func TestMaskedBasics(t *testing.T) {
	m := mkMasked(t, []float64{1, 2, 3, 4}, []float64{0, 1, 0, 1})
	f, err := m.Filled(0.0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, f, 1, 0, 3, 0)

	c, err := m.Compressed()
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, c, 1, 3)

	if m.Count() != 2 {
		t.Fatalf("count = %d, want 2", m.Count())
	}
	if !m.IsMasked() {
		t.Fatal("应报告 IsMasked")
	}
	none := AsMasked(bArr(t, []float64{1, 2}))
	if none.IsMasked() {
		t.Fatal("全 False 掩码不应 IsMasked")
	}
	if none.Count() != 2 {
		t.Fatalf("count = %d", none.Count())
	}
}

// 掩码归约:跳过无效值。
func TestMaskedReductions(t *testing.T) {
	m := mkMasked(t, []float64{1, 2, 3, 4}, []float64{0, 1, 0, 1})
	s, err := m.Sum(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, s.Data().Flatten(), 4)
	if s.IsMasked() {
		t.Fatal("非全掩码的 sum 不应被掩码")
	}
	mean, err := m.Mean(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mean.Data().Flatten(), 2)
	mn, err := m.Min(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mn.Data().Flatten(), 1)
	mx, err := m.Max(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mx.Data().Flatten(), 3)
}

// 全掩码 → 结果被掩码(numpy 的 masked constant)。
func TestMaskedAllMasked(t *testing.T) {
	m := mkMasked(t, []float64{1, 2}, []float64{1, 1})
	s, err := m.Sum(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !s.IsMasked() {
		t.Fatal("全掩码切片的结果应被掩码")
	}
	if s.Count() != 0 {
		t.Fatalf("应无有效元素,得 %d", s.Count())
	}
	if m.Count() != 0 {
		t.Fatalf("源数组应无有效元素")
	}
}

// 2-D 轴归约。
func TestMaskedAxis(t *testing.T) {
	d := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	mb := mustMat(t, []int{2, 2}, []bool{false, true, false, false})
	m, err := NewMasked(d, mb)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.Sum([]int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, s.Data().Flatten(), 4, 4) // 第 1 列只有 4 有效

	mean, err := m.Mean([]int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, mean.Data().Flatten(), 1, 3.5)
}

// 二元运算:掩码取并集。
func TestMaskedBinary(t *testing.T) {
	a := mkMasked(t, []float64{1, 2, 3, 4}, []float64{0, 1, 0, 0})
	b := mkMasked(t, []float64{1, 1, 1, 1}, []float64{1, 0, 0, 0})
	sum, err := MaskedAdd(a, b)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sum.Filled(0.0)
	if err != nil {
		t.Fatal(err)
	}
	// 掩码并集:[T,T,F,F] → 有效位 2,3 → 值 3+1=4、4+1=5
	bWantF64(t, got, 0, 0, 4, 5)
	if sum.Count() != 2 {
		t.Fatalf("并集掩码应有 2 个有效位,得 %d", sum.Count())
	}

	// 广播 + 掩码广播
	col := mkMasked(t, []float64{10, 20}, []float64{0, 1})
	m2 := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	mask2 := mustMat(t, []int{2, 2}, []bool{false, false, false, false})
	mm, err := NewMasked(m2, mask2)
	if err != nil {
		t.Fatal(err)
	}
	out, err := MaskedAdd(mm, col)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("广播形状 %v", out.Shape())
	}
}

// 极值填充口径:整型不应被填充值污染。
func TestMaskedIntMinMax(t *testing.T) {
	d := bArr(t, []int32{5, 7, 9})
	mb := bArr(t, []bool{false, true, false})
	m, err := NewMasked(d, mb)
	if err != nil {
		t.Fatal(err)
	}
	mn, err := m.Min(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := mn.Data().Flatten().Item(0); v != int32(5) {
		t.Fatalf("masked min = %v, want 5", v)
	}
	mx, err := m.Max(nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := mx.Data().Flatten().Item(0); v != int32(9) {
		t.Fatalf("masked max = %v, want 9", v)
	}
	// 掩码 dtype 非 bool 时自动转换
	mi, err := NewMasked(bArr(t, []int32{1, 2}), bArr(t, []int32{0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if mi.Count() != 1 {
		t.Fatalf("非 bool 掩码应自动转 bool,得 count=%d", mi.Count())
	}
	_ = math.Inf
}

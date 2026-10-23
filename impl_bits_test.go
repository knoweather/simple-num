package num

import (
	"math"
	"testing"
)

// 位运算:与 numpy 逐值一致(非法移位量得 0、有符号算术右移)。
func TestBitwise(t *testing.T) {
	a := bArr(t, []int32{6, 3, -4})
	b := bArr(t, []int32{3, 5, 1})
	cases := []struct {
		name string
		fn   func(x, y *Array) (*Array, error)
		want []int32
	}{
		{"and", BitwiseAnd, []int32{2, 1, 0}},
		{"or", BitwiseOr, []int32{7, 7, -3}},
		{"xor", BitwiseXor, []int32{5, 6, -3}},
		{"lshift", LeftShift, []int32{48, 96, -8}},
		{"rshift", RightShift, []int32{0, 0, -2}},
	}
	for _, c := range cases {
		got, err := c.fn(a, b)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		wantI32(t, got, c.want...)
	}
	// bool:按位运算等价逻辑运算
	ba := bArr(t, []bool{true, false})
	bb := bArr(t, []bool{true, true})
	if o, err := BitwiseAnd(ba, bb); err != nil {
		t.Fatal(err)
	} else {
		w := win[uint8](engineWindow(o.AsContiguous()))
		if w[0] != 1 || w[1] != 0 {
			t.Fatalf("bool and = %v, want [1 0]", w)
		}
	}
	if o, err := BitwiseXor(ba, bb); err != nil {
		t.Fatal(err)
	} else {
		w := win[uint8](engineWindow(o.AsContiguous()))
		if w[0] != 0 || w[1] != 1 {
			t.Fatalf("bool xor = %v, want [0 1]", w)
		}
	}
	// 浮点/复数拒绝(numpy 抛 TypeError)
	fa := bArr(t, []float64{1})
	if _, err := BitwiseAnd(fa, fa); err == nil {
		t.Fatal("浮点位运算应报错")
	}
	// 非法移位:>= 位宽 或 < 0 → 0
	s8 := bArr(t, []int8{1})
	if o, err := LeftShift(s8, bArr(t, []int8{8})); err != nil {
		t.Fatal(err)
	} else {
		w := win[int8](engineWindow(o.AsContiguous()))
		if w[0] != 0 {
			t.Fatalf("i8<<8 = %d, want 0", w[0])
		}
	}
	if o, err := LeftShift(s8, bArr(t, []int8{-1})); err != nil {
		t.Fatal(err)
	} else {
		w := win[int8](engineWindow(o.AsContiguous()))
		if w[0] != 0 {
			t.Fatalf("i8<<-1 = %d, want 0", w[0])
		}
	}
	// 算术右移
	if o, err := RightShift(bArr(t, []int8{-8}), bArr(t, []int8{1})); err != nil {
		t.Fatal(err)
	} else {
		w := win[int8](engineWindow(o.AsContiguous()))
		if w[0] != -4 {
			t.Fatalf("-8>>1 = %d, want -4", w[0])
		}
	}
}

// Rot90 / Average / PtP:与 numpy 快照对齐。
func TestRot90AveragePtP(t *testing.T) {
	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	r1, err := Rot90(m, 1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r1.Shape(), []int{3, 2}) {
		t.Fatalf("rot90 形状 %v, want [3 2]", r1.Shape())
	}
	bWantF64(t, r1.Flatten(), 2, 5, 1, 4, 0, 3)

	r2, _ := Rot90(m, 2, 0, 1)
	bWantF64(t, r2.Flatten(), 5, 4, 3, 2, 1, 0)
	r3, _ := Rot90(m, 3, 0, 1)
	if !shapeEqual(r3.Shape(), []int{3, 2}) {
		t.Fatalf("k=3 形状 %v", r3.Shape())
	}
	// 负数 k 等价 mod 4
	rNeg, _ := Rot90(m, -3, 0, 1)
	if !ArrayEqual(rNeg, r1) {
		t.Fatal("k=-3 应等价 k=1")
	}
	// 交换轴顺序
	rAx, _ := Rot90(m, 1, 1, 0)
	bWantF64(t, rAx.Flatten(), 3, 0, 4, 1, 5, 2)

	// average:1-D 权重按轴加权
	av, err := Average(bArr(t, []float64{1, 2, 3}), nil, bArr(t, []float64{1, 2, 2}))
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := av.Item(); math.Abs(v.(float64)-2.2) > 1e-12 {
		t.Fatalf("average = %v, want 2.2", v)
	}
	// 按轴:[[1,2],[3,4]] axis=1 权重 [1,3] → [1.75, 3.75]
	m2 := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	av2, err := Average(m2, []int{1}, bArr(t, []float64{1, 3}))
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := av2.Item(0); math.Abs(v.(float64)-1.75) > 1e-12 {
		t.Fatalf("average axis=1 [0] = %v, want 1.75", v)
	}
	if v, _ := av2.Item(1); math.Abs(v.(float64)-3.75) > 1e-12 {
		t.Fatalf("average axis=1 [1] = %v, want 3.75", v)
	}
	// 权重形状不符
	if _, err := Average(m2, []int{1}, bArr(t, []float64{1, 2, 3})); err == nil {
		t.Fatal("权重长度不符应报错")
	}

	// ptp
	if p, err := PtP(bArr(t, []float64{3, 1, 4, 1, 5}), nil, false); err != nil {
		t.Fatal(err)
	} else if v, _ := p.Item(); v != 4.0 {
		t.Fatalf("ptp = %v, want 4", v)
	}
	if p, err := PtP(mustMat(t, []int{2, 2}, []float64{3, 1, 4, 5}), []int{0}, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, p, 1, 4)
	}
}

// LogSpace / GeomSpace / NanProd / NanCumsum / Kron / Slogdet。
func TestSpacesAndMore(t *testing.T) {
	ls, err := LogSpace(0, 2, 3, 10)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{1, 10, 100} {
		v, _ := ls.Item(i)
		if math.Abs(v.(float64)-w) > 1e-12 {
			t.Fatalf("logspace[%d] = %v, want %v", i, v, w)
		}
	}
	ls2, _ := LogSpace(0, 3, 4, 2)
	for i, w := range []float64{1, 2, 4, 8} {
		v, _ := ls2.Item(i)
		if math.Abs(v.(float64)-w) > 1e-12 {
			t.Fatalf("logspace base2[%d] = %v, want %v", i, v, w)
		}
	}
	gs, err := GeomSpace(1, 1000, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{1, 10, 100, 1000} {
		v, _ := gs.Item(i)
		if math.Abs(v.(float64)-w) > 1e-9 {
			t.Fatalf("geomspace[%d] = %v, want %v", i, v, w)
		}
	}
	// 异号/含零 → 报错
	if _, err := GeomSpace(-1, 10, 3); err == nil {
		t.Fatal("异号应报错")
	}
	if _, err := GeomSpace(0, 10, 3); err == nil {
		t.Fatal("含零应报错")
	}

	// nanprod:忽略 NaN;全 NaN → 1
	np, err := NanProd(bArr(t, []float64{1, math.NaN(), 3}), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := np.Item(); v != 3.0 {
		t.Fatalf("nanprod = %v, want 3", v)
	}
	np2, err := NanProd(bArr(t, []float64{math.NaN(), math.NaN()}), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := np2.Item(); v != 1.0 {
		t.Fatalf("全 NaN nanprod = %v, want 1", v)
	}
	// nancumsum:NaN 当 0
	ncs, err := NanCumsum(bArr(t, []float64{1, math.NaN(), 3}), nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, ncs, 1, 1, 4)

	// kron:2-D
	a := mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})
	b := mustMat(t, []int{2, 2}, []int32{0, 5, 6, 7})
	k, err := Kron(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(k.Shape(), []int{4, 4}) {
		t.Fatalf("kron 形状 %v", k.Shape())
	}
	wantI32(t, k.Flatten(), 0, 5, 0, 10, 6, 7, 12, 14, 0, 15, 0, 20, 18, 21, 24, 28)
	// kron:1-D × 1-D → 1-D
	k1, err := Kron(bArr(t, []float64{1, 2}), bArr(t, []float64{3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(k1.Shape(), []int{4}) {
		t.Fatalf("1-D kron 形状 %v", k1.Shape())
	}
	bWantF64(t, k1, 3, 4, 6, 8)

	// slogdet:det=-2 → (-1, ln2)
	s, ld, err := Slogdet(mustMat(t, []int{2, 2}, []float64{4, 3, 2, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if s != -1 {
		t.Fatalf("slogdet sign = %v, want -1", s)
	}
	if math.Abs(ld-math.Log(2)) > 1e-12 {
		t.Fatalf("slogdet logdet = %v, want ln2", ld)
	}
	// 奇异 → (0, -Inf)
	s2, ld2, err := Slogdet(mustMat(t, []int{2, 2}, []float64{1, 2, 2, 4}))
	if err != nil {
		t.Fatal(err)
	}
	if s2 != 0 || !math.IsInf(ld2, -1) {
		t.Fatalf("奇异 slogdet = (%v,%v), want (0,-Inf)", s2, ld2)
	}
}

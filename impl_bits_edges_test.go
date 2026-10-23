package num

import (
	"math"
	"testing"
)

// 覆盖新算子在各 dtype 与边角上的分支(位运算全 dtype、各 error 路径等)。
func TestBitwiseAllKinds(t *testing.T) {
	// 每种整型都要走自己的内核
	type pair struct {
		name string
		a, b *Array
	}
	cases := []pair{
		{"i8", mustMat(t, []int{3}, []int8{6, 3, -4}), mustMat(t, []int{3}, []int8{3, 5, 1})},
		{"i16", mustMat(t, []int{3}, []int16{6, 3, -4}), mustMat(t, []int{3}, []int16{3, 5, 1})},
		{"i32", mustMat(t, []int{3}, []int32{6, 3, -4}), mustMat(t, []int{3}, []int32{3, 5, 1})},
		{"i64", mustMat(t, []int{3}, []int64{6, 3, -4}), mustMat(t, []int{3}, []int64{3, 5, 1})},
		{"u8", mustMat(t, []int{3}, []uint8{6, 3, 4}), mustMat(t, []int{3}, []uint8{3, 5, 1})},
		{"u16", mustMat(t, []int{3}, []uint16{6, 3, 4}), mustMat(t, []int{3}, []uint16{3, 5, 1})},
		{"u32", mustMat(t, []int{3}, []uint32{6, 3, 4}), mustMat(t, []int{3}, []uint32{3, 5, 1})},
		{"u64", mustMat(t, []int{3}, []uint64{6, 3, 4}), mustMat(t, []int{3}, []uint64{3, 5, 1})},
	}
	for _, c := range cases {
		for _, f := range []struct {
			name string
			fn   func(x, y *Array) (*Array, error)
		}{
			{"and", BitwiseAnd}, {"or", BitwiseOr}, {"xor", BitwiseXor},
			{"lsh", LeftShift}, {"rsh", RightShift},
		} {
			if _, err := f.fn(c.a, c.b); err != nil {
				t.Fatalf("%s/%s: %v", c.name, f.name, err)
			}
		}
	}
	// bool 的 or / xor
	ba := bArr(t, []bool{true, false})
	bb := bArr(t, []bool{false, false})
	if o, err := BitwiseOr(ba, bb); err != nil {
		t.Fatal(err)
	} else {
		w := win[uint8](engineWindow(o.AsContiguous()))
		if w[0] != 1 || w[1] != 0 {
			t.Fatalf("bool or = %v", w)
		}
	}
	if _, err := BitwiseXor(ba, bb); err != nil {
		t.Fatal(err)
	}
	// 复数拒绝
	ca := bArr(t, []complex128{1 + 1i})
	if _, err := BitwiseOr(ca, ca); err == nil {
		t.Fatal("复数为运算应报错")
	}
	// 广播:标量 × 数组
	if o, err := BitwiseAnd(bArr(t, []int32{6}), mustMat(t, []int{2}, []int32{3, 5})); err != nil {
		t.Fatal(err)
	} else {
		wantI32(t, o, 2, 4)
	}
}

// 新工具算子的边角与错误路径。
func TestMisc3Edges(t *testing.T) {
	// Rot90:k=0 返回副本;k=4 等价 0;非法维数/同轴报错
	m := mustMat(t, []int{2, 3}, []float64{0, 1, 2, 3, 4, 5})
	r0, err := Rot90(m, 0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !ArrayEqual(r0, m) {
		t.Fatal("k=0 应与原数组相同")
	}
	r4, err := Rot90(m, 4, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !ArrayEqual(r4, m) {
		t.Fatal("k=4 应与原数组相同")
	}
	if _, err := Rot90(bArr(t, []float64{1, 2}), 1, 0, 1); err == nil {
		t.Fatal("1-D rot90 应报错")
	}
	if _, err := Rot90(m, 1, 0, 0); err == nil {
		t.Fatal("同轴应报错")
	}
	if _, err := Rot90(m, 1, 0, 5); err == nil {
		t.Fatal("越界轴应报错")
	}
	if _, err := Rot90(nil, 1, 0, 1); err == nil {
		t.Fatal("nil 应报错")
	}

	// Average:同形权重(逐元素加权)与 nil 权重
	m2 := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	same, err := Average(m2, nil, mustMat(t, []int{2, 2}, []float64{1, 1, 1, 3}))
	if err != nil {
		t.Fatal(err)
	}
	// sum(a*w)/sum(w) = (1+2+3+12)/6 = 3
	if v, _ := same.Item(); math.Abs(v.(float64)-3.0) > 1e-12 {
		t.Fatalf("average 同形权重 = %v, want 3", v)
	}
	if o, err := Average(m2, nil, nil); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(); math.Abs(v.(float64)-2.5) > 1e-12 {
		t.Fatalf("average 无权重 = %v, want 2.5", v)
	}
	if _, err := Average(nil, nil, nil); err == nil {
		t.Fatal("nil 数组应报错")
	}

	// PtP 报错路径(轴越界)
	if _, err := PtP(m2, []int{5}, false); err == nil {
		t.Fatal("PtP 轴越界应报错")
	}

	// LogSpace / GeomSpace 边角
	if _, err := LogSpace(0, 1, -1, 10); err == nil {
		t.Fatal("num<0 应报错")
	}
	if o, err := LogSpace(0, 1, 2, 0); err != nil { // base<=0 → 视为 10
		t.Fatal(err)
	} else if v, _ := o.Item(1); math.Abs(v.(float64)-10) > 1e-12 {
		t.Fatalf("base=0 应回落 10,得 %v", v)
	}
	if _, err := GeomSpace(1, -100, 3); err == nil {
		t.Fatal("异号应报错")
	}
	if _, err := GeomSpace(1, 100, -1); err == nil {
		t.Fatal("num<0 应报错")
	}
	// 同为负:走 sign 分支
	if o, err := GeomSpace(-1, -1000, 4); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); math.Abs(v.(float64)+1) > 1e-9 {
		t.Fatalf("geomspace(-1,-1000)[0] = %v, want -1", v)
	}

	// NanProd:按轴 + keepdims + 空轴报错
	ax := mustMat(t, []int{2, 2}, []float64{1, math.NaN(), 3, 4})
	if o, err := NanProd(ax, []int{1}, false); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 1, 12)
	}
	if o, err := NanProd(ax, []int{1}, true); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(o.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims 形状 %v", o.Shape())
	}
	if _, err := NanProd(mustNew(t, []int{2, 0}, DFloat64), []int{1}, false); err == nil {
		t.Fatal("空轴应报错")
	}
	if _, err := NanProd(nil, nil, false); err == nil {
		t.Fatal("nil 应报错")
	}
	// 0 行 → 空输出
	if o, err := NanProd(mustNew(t, []int{0, 2}, DFloat64), []int{1}, false); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("0 行应得空输出,得 %d", o.Size())
	}

	// NanCumsum:按轴 + nil
	if o, err := NanCumsum(ax, []int{0}); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(o.Shape(), []int{2, 2}) {
		t.Fatalf("nancumsum 形状 %v", o.Shape())
	}
	if _, err := NanCumsum(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}

	// Kron:1-D × 2-D(提升)与错误路径
	k1, err := Kron(bArr(t, []float64{1, 2}), mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(k1.Shape(), []int{2, 4}) {
		t.Fatalf("1-D×2-D kron 形状 %v, want [2 4]", k1.Shape())
	}
	if _, err := Kron(mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
		bArr(t, []float64{1})); err == nil {
		t.Fatal("3-D kron 应报错")
	}
	if _, err := Kron(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}
	// 复数 → 明确报错
	if _, err := Kron(bArr(t, []complex128{1 + 1i}), bArr(t, []complex128{1 + 1i})); err == nil {
		t.Fatal("复数 kron 应报错")
	}

	// Slogdet:非方阵报错;正行列式
	if _, _, err := Slogdet(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
	s, ld, err := Slogdet(mustMat(t, []int{2, 2}, []float64{2, 0, 0, 3}))
	if err != nil {
		t.Fatal(err)
	}
	if s != 1 || math.Abs(ld-math.Log(6)) > 1e-12 {
		t.Fatalf("slogdet 对角 = (%v,%v), want (1,ln6)", s, ld)
	}
}

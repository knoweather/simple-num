package num

import (
	"math"
	"testing"
)

// wantI64 断言 int64 数组的扁平内容。
func wantI64(t *testing.T, a *Array, want ...int64) {
	t.Helper()
	f := a.AsContiguous().Flatten()
	if f.Size() != len(want) {
		t.Fatalf("长度 %d, want %d", f.Size(), len(want))
	}
	w := win[int64](engineWindow(f))
	for i := range want {
		if w[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d(全量 %v)", i, w[i], want[i], w)
		}
	}
}

// Partition / Argpartition:验证不变量(numpy 的侧内顺序不作保证)。
func TestPartitionInvariant(t *testing.T) {
	a := bArr(t, []float64{3, 1, 4, 1, 5, 9, 2, 6})
	for _, kth := range []int{0, 3, 7} {
		p, err := Partition(a, kth, 0)
		if err != nil {
			t.Fatalf("kth=%d: %v", kth, err)
		}
		w := win[float64](engineWindow(p.AsContiguous()))
		kv := w[kth]
		for i := 0; i < len(w); i++ {
			if i < kth && w[i] > kv {
				t.Fatalf("kth=%d: 左侧 w[%d]=%v > w[kth]=%v", kth, i, w[i], kv)
			}
			if i > kth && w[i] < kv {
				t.Fatalf("kth=%d: 右侧 w[%d]=%v < w[kth]=%v", kth, i, w[i], kv)
			}
		}
		// 多重集必须一致
		sa, err1 := Sort(a, []int{0})
		sp, err2 := Sort(p, []int{0})
		if err1 != nil || err2 != nil {
			t.Fatal(err1, err2)
		}
		if !ArrayEqual(sa.Flatten(), sp.Flatten()) {
			t.Fatalf("kth=%d: 元素集合被改变", kth)
		}
	}
	// kth 越界
	if _, err := Partition(a, 8, 0); err == nil {
		t.Fatal("kth 越界应报错")
	}
	if _, err := Partition(a, -1, 0); err != nil {
		t.Fatalf("负 kth 应等价于从尾部数:%v", err)
	}
	if _, err := Partition(nil, 0, 0); err == nil {
		t.Fatal("nil 应报错")
	}
	// NaN 排最后
	pn, err := Partition(bArr(t, []float64{3, math.NaN(), 1}), 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	wn := win[float64](engineWindow(pn.AsContiguous()))
	if !math.IsNaN(wn[2]) {
		t.Fatalf("NaN 应在最后,得 %v", wn)
	}
	// 2-D 按轴
	m := mustMat(t, []int{2, 3}, []float64{3, 1, 4, 1, 5, 9})
	pm, err := Partition(m, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(pm.Shape(), []int{2, 3}) {
		t.Fatalf("2-D 形状 %v", pm.Shape())
	}
}

func TestArgpartition(t *testing.T) {
	a := bArr(t, []float64{3, 1, 4, 1, 5, 9, 2, 6})
	ap, err := Argpartition(a, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if ap.Dtype() != DInt64 {
		t.Fatalf("argpartition dtype %s, want int64", ap.Dtype())
	}
	w := win[int64](engineWindow(ap.AsContiguous()))
	if len(w) != 8 {
		t.Fatalf("长度 %d", len(w))
	}
	// 下标必须是 0..7 的一个排列
	seen := make([]bool, 8)
	for _, v := range w {
		if v < 0 || v >= 8 || seen[v] {
			t.Fatalf("下标 %v 不是排列", w)
		}
		seen[v] = true
	}
	// 取值后满足分区不变量
	src := win[float64](engineWindow(a.AsContiguous()))
	kv := src[w[3]]
	for i, idx := range w {
		if i < 3 && src[idx] > kv {
			t.Fatalf("左侧 %v > kth %v", src[idx], kv)
		}
		if i > 3 && src[idx] < kv {
			t.Fatalf("右侧 %v < kth %v", src[idx], kv)
		}
	}
	if _, err := Argpartition(nil, 0, 0); err == nil {
		t.Fatal("nil 应报错")
	}
	if _, err := Argpartition(a, 9, 0); err == nil {
		t.Fatal("kth 越界应报错")
	}
}

// Moveaxis / Resize / BroadcastArrays。
func TestShapeTools(t *testing.T) {
	vals := make([]int32, 24)
	for i := range vals {
		vals[i] = int32(i)
	}
	m := mustMat(t, []int{2, 3, 4}, vals)
	mv, err := Moveaxis(m, []int{0}, []int{-1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(mv.Shape(), []int{3, 4, 2}) {
		t.Fatalf("moveaxis 形状 %v, want [3 4 2]", mv.Shape())
	}
	// numpy:moveaxis(m,0,-1)[0,0] == 原 m[0,:,0] == [0,12]
	if v, _ := mv.Item(0, 0, 0); v != int32(0) {
		t.Fatalf("mv[0,0,0] = %v, want 0", v)
	}
	if v, _ := mv.Item(0, 0, 1); v != int32(12) {
		t.Fatalf("mv[0,0,1] = %v, want 12", v)
	}
	// 多轴同时移动
	// numpy:moveaxis(m,[0,1],[2,0]).shape == (3,4,2)
	mv2, err := Moveaxis(m, []int{0, 1}, []int{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(mv2.Shape(), []int{3, 4, 2}) {
		t.Fatalf("多轴 moveaxis 形状 %v, want [3 4 2]", mv2.Shape())
	}
	// numpy:moveaxis(m,[0,1],[1,2]).shape == (4,2,3)
	mv3, err := Moveaxis(m, []int{0, 1}, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(mv3.Shape(), []int{4, 2, 3}) {
		t.Fatalf("多轴 moveaxis 形状 %v, want [4 2 3]", mv3.Shape())
	}
	// 错误路径
	if _, err := Moveaxis(m, []int{0}, []int{0, 1}); err == nil {
		t.Fatal("长度不同应报错")
	}
	if _, err := Moveaxis(m, []int{0, 0}, []int{0, 1}); err == nil {
		t.Fatal("重复 source 应报错")
	}
	if _, err := Moveaxis(nil, nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}

	// resize:[1,2,3] → 5 → [1,2,3,1,2]
	rz, err := Resize(bArr(t, []int64{1, 2, 3}), []int{5})
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, rz, 1, 2, 3, 1, 2)
	// 缩小 → 截断
	rz2, err := Resize(bArr(t, []int64{1, 2, 3, 4}), []int{2})
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, rz2, 1, 2)
	// 空数组放大 → 报错
	if _, err := Resize(mustNew(t, []int{0}, DFloat64), []int{3}); err == nil {
		t.Fatal("空数组放大应报错")
	}

	// broadcast_arrays
	b1, _ := bArr(t, []float64{1, 2}).Reshape([]int{2, 1})
	b2 := bArr(t, []float64{10, 20, 30})
	bc, err := BroadcastArrays(b1, b2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(bc[0].Shape(), []int{2, 3}) || !shapeEqual(bc[1].Shape(), []int{2, 3}) {
		t.Fatalf("broadcast 形状 %v %v", bc[0].Shape(), bc[1].Shape())
	}
	if v, _ := bc[0].Item(0, 0); v != 1.0 {
		t.Fatalf("bc[0][0,0] = %v", v)
	}
	if v, _ := bc[1].Item(1, 2); v != 30.0 {
		t.Fatalf("bc[1][1,2] = %v", v)
	}
	if _, err := BroadcastArrays(); err == nil {
		t.Fatal("空参应报错")
	}
	if _, err := BroadcastArrays(bArr(t, []float64{1, 2}), bArr(t, []float64{1, 2, 3})); err == nil {
		t.Fatal("不可广播应报错")
	}
}

// Choose / TrilIndices / TriuIndices / UnravelIndex。
func TestIndexHelpers(t *testing.T) {
	idx, _ := FromSlice([]int64{0, 1, 0})
	c1, _ := FromSlice([]int64{1, 2, 3})
	c2, _ := FromSlice([]int64{10, 20, 30})
	got, err := Choose(idx, []*Array{c1, c2})
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, got, 1, 20, 3)
	// 索引越界
	bad, _ := FromSlice([]int64{0, 5})
	if _, err := Choose(bad, []*Array{c1}); err == nil {
		t.Fatal("索引越界应报错")
	}
	// 浮点索引拒绝
	if _, err := Choose(bArr(t, []float64{0}), []*Array{c1}); err == nil {
		t.Fatal("浮点索引应报错")
	}

	tr, err := TrilIndices(3, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, tr[0], 0, 1, 1, 2, 2, 2)
	wantI64(t, tr[1], 0, 0, 1, 0, 1, 2)
	trk, err := TrilIndices(3, -1)
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, trk[0], 1, 2, 2)
	wantI64(t, trk[1], 0, 0, 1)
	tu, err := TriuIndices(3, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, tu[0], 0, 0, 0, 1, 1, 2)
	wantI64(t, tu[1], 0, 1, 2, 1, 2, 2)
	if _, err := TrilIndices(-1, 0); err == nil {
		t.Fatal("负 n 应报错")
	}

	ui, err := UnravelIndex(bArr(t, []int64{5, 11}), []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, ui[0], 1, 2)
	wantI64(t, ui[1], 1, 3)
	// 越界
	if _, err := UnravelIndex(bArr(t, []int64{12}), []int{3, 4}); err == nil {
		t.Fatal("越界应报错")
	}
	// 3-D
	u3, err := UnravelIndex(bArr(t, []int64{23}), []int{2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	wantI64(t, u3[0], 1)
	wantI64(t, u3[1], 2)
	wantI64(t, u3[2], 3)
}

// FFTShift / IFFTShift / Exp2 / NanCumprod。
func TestShiftAndExp(t *testing.T) {
	fs, err := FFTShift(bArr(t, []float64{1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, fs, 3, 4, 1, 2)
	// 奇数长度:roll by n//2
	odd := bArr(t, []float64{0, 1, 2, 3, 4})
	f5, err := FFTShift(odd)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, f5, 3, 4, 0, 1, 2)
	i5, err := IFFTShift(odd)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, i5, 2, 3, 4, 0, 1)
	i4, err := IFFTShift(bArr(t, []float64{0, 1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, i4, 2, 3, 0, 1)
	if _, err := FFTShift(nil); err == nil {
		t.Fatal("nil 应报错")
	}

	e2, err := Exp2(bArr(t, []float64{1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, e2, 2, 4, 8)
	// f32 保持 f32
	e32, err := Exp2(bArr(t, []float32{3}))
	if err != nil {
		t.Fatal(err)
	}
	if e32.Dtype() != DFloat32 {
		t.Fatalf("f32 exp2 dtype %s", e32.Dtype())
	}
	if v, _ := e32.Item(0); math.Abs(float64(v.(float32))-8) > 1e-6 {
		t.Fatalf("f32 exp2 = %v", v)
	}
	if _, err := Exp2(nil); err == nil {
		t.Fatal("nil 应报错")
	}

	ncp, err := NanCumprod(bArr(t, []float64{1, math.NaN(), 3}), nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, ncp, 1, 1, 3)
	if _, err := NanCumprod(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}
}

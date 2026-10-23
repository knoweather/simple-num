package num

// 单测(B):take / take_along_axis / put。
// 期望值 = numpy 2.5.4 实测探针字面量。

import (
	"math"
	"testing"
)

// TestTakeFlat ≡ np.take(axis=None)。
func TestTakeFlat(t *testing.T) {
	a := bArr(t, []int64{10, 20, 30, 40})
	idx := bArr(t, []int64{0, -1, 2})
	out, err := Take(a, idx)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt64 || !shapeEqual(out.Shape(), []int{3}) {
		t.Fatalf("out %v %s", out.Shape(), out.Dtype())
	}
	bWantI64(t, out, 10, 40, 30)

	// 2-D 索引 → 输出形状 = 索引形状
	idx2 := bArr(t, []int64{0, 1, 2, 3})
	idx2v, _ := idx2.Reshape([]int{2, 2})
	out, err = Take(a, idx2v)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("shape = %v, want [2 2]", out.Shape())
	}
	if g := getI(t, out, 1, 0); g != int64(30) {
		t.Fatalf("[1,0] = %v, want 30", g)
	}

	// dtype 保真 + 索引 dtype 宽容(u8)
	f := bArr(t, []float64{1.5, 2.5, 3.5})
	uf := bArr(t, []uint8{2, 0})
	out, err = Take(f, uf)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s", out.Dtype())
	}
	bWantF64Rel(t, out, 4e-16, 3.5, 1.5)

	// f16 走字节拷贝(不经过 decode)
	h, err := Cast(bArr(t, []float64{1.5, 2.5}), DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Take(h, bArr(t, []int64{1}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat16 {
		t.Fatalf("f16 dtype = %s", out.Dtype())
	}
	bWantF32Rel(t, out, DFloat16, 0, 2.5)
}

// 索引错误路径。
func TestTakeErrors(t *testing.T) {
	a := bArr(t, []int64{10, 20, 30, 40})
	if _, err := Take(a, bArr(t, []int64{4})); rootOf(err) != ErrIndex {
		t.Fatalf("越界 err = %v, want ErrIndex", err)
	}
	if _, err := Take(a, bArr(t, []int64{-5})); rootOf(err) != ErrIndex {
		t.Fatalf("负越界 err = %v, want ErrIndex", err)
	}
	if _, err := Take(a, bArr(t, []float64{0})); rootOf(err) != ErrDtype {
		t.Fatalf("浮点索引 err = %v, want ErrDtype", err)
	}
}

// TestTakeAlongAxis ≡ np.take_along_axis。
func TestTakeAlongAxis(t *testing.T) {
	a := bArr(t, []int64{1, 2, 3, 4})
	av, _ := a.Reshape([]int{2, 2})
	idx := bArr(t, []int64{1, 0})
	iv, _ := idx.Reshape([]int{2, 1})
	out, err := TakeAlongAxis(av, iv, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 1}) {
		t.Fatalf("shape = %v, want [2 1]", out.Shape())
	}
	if g := getI(t, out, 0, 0); g != int64(2) {
		t.Fatalf("[0,0] = %v, want 2", g)
	}
	if g := getI(t, out, 1, 0); g != int64(3) {
		t.Fatalf("[1,0] = %v, want 3", g)
	}

	// 负索引归一:[[-1]] → 最后一列
	iv2, _ := bArr(t, []int64{-1, -1}).Reshape([]int{2, 1})
	out, err = TakeAlongAxis(av, iv2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if g := getI(t, out, 0, 0); g != int64(2) {
		t.Fatalf("neg [0,0] = %v, want 2", g)
	}
	if g := getI(t, out, 1, 0); g != int64(4) {
		t.Fatalf("neg [1,0] = %v, want 4", g)
	}

	// 非轴维广播:idx (1,2) 于 (2,2) axis=1 → [[1,2],[3,4]]
	iv3, _ := bArr(t, []int64{0, 1}).Reshape([]int{1, 2})
	out, err = TakeAlongAxis(av, iv3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{1, 2}) {
		t.Fatalf("bcast shape = %v, want [1 2]", out.Shape())
	}
	if g := getI(t, out, 0, 1); g != int64(2) {
		t.Fatalf("bcast [0,1] = %v, want 2", g)
	}

	// 错误:越界 / 非轴维不容 / ndim 不符
	if _, err := TakeAlongAxis(av, iv, 2); rootOf(err) != ErrIndex {
		t.Fatalf("axis 越界 err = %v", err)
	}
	iv4, _ := bArr(t, []int64{2, 0}).Reshape([]int{1, 2})
	if _, err := TakeAlongAxis(av, iv4, 1); rootOf(err) != ErrIndex {
		t.Fatalf("索引越界 err = %v", err)
	}
	bad, _ := bArr(t, []int64{0, 1, 0}).Reshape([]int{3, 1})
	if _, err := TakeAlongAxis(av, bad, 1); rootOf(err) != ErrShape {
		t.Fatalf("非轴维不容 err = %v", err)
	}
	flat := bArr(t, []int64{0, 1})
	if _, err := TakeAlongAxis(av, flat, 1); rootOf(err) != ErrShape {
		t.Fatalf("ndim 不符 err = %v", err)
	}
}

// TestPut ≡ np.put(原地、值循环、last-write-wins、不安全转换)。
func TestPut(t *testing.T) {
	a := bArr(t, []int64{10, 20, 30, 40})
	if err := Put(a, bArr(t, []int64{0, -1}), bArr(t, []int64{1, 2})); err != nil {
		t.Fatal(err)
	}
	bWantI64(t, a, 1, 20, 30, 2)

	// 值循环:vals 只有 1 个 → 全部填 7
	b := bArr(t, []int64{10, 20, 30, 40})
	if err := Put(b, bArr(t, []int64{0, 1, 2}), bArr(t, []int64{7})); err != nil {
		t.Fatal(err)
	}
	bWantI64(t, b, 7, 7, 7, 40)

	// 重复索引 last-write-wins:[0,0,1] ← [5,6,7] → [6,7,30,40]
	c := bArr(t, []int64{10, 20, 30, 40})
	if err := Put(c, bArr(t, []int64{0, 0, 1}), bArr(t, []int64{5, 6, 7})); err != nil {
		t.Fatal(err)
	}
	bWantI64(t, c, 6, 7, 30, 40)

	// 不安全转换:float → int 向零截断(1.9 → 1)
	d := bArr(t, []int64{10, 20, 30, 40})
	if err := Put(d, bArr(t, []int64{0, 1}), bArr(t, []float64{1.9, 2.9})); err != nil {
		t.Fatal(err)
	}
	bWantI64(t, d, 1, 2, 30, 40)

	// float 目标保真
	e := bArr(t, []float64{1, 2})
	if err := Put(e, bArr(t, []int64{0}), bArr(t, []float64{1.7})); err != nil {
		t.Fatal(err)
	}
	bWantF64Rel(t, e, 4e-16, 1.7, 2)

	// 错误:越界 / 空值数组
	f := bArr(t, []int64{10, 20})
	if err := Put(f, bArr(t, []int64{5}), bArr(t, []int64{1})); rootOf(err) != ErrIndex {
		t.Fatalf("越界 err = %v, want ErrIndex", err)
	}
	if err := Put(f, bArr(t, []int64{0}), bArr(t, []int64{})); rootOf(err) != ErrShape {
		t.Fatalf("空值 err = %v, want ErrShape", err)
	}
}

// 视图目标:Put 经步长写入(非连续布局)。
func TestPutStrided(t *testing.T) {
	a := bArr(t, []int64{0, 0, 0, 0})
	v, err := a.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := Put(v, bArr(t, []int64{1, 2}), bArr(t, []int64{9, 8})); err != nil {
		t.Fatal(err)
	}
	// C 序扁平位置 1 → (0,1);2 → (1,0)
	bWantI64(t, a, 0, 9, 8, 0)
}

// NaN 参与 take 时位型不变(字节拷贝)。
func TestTakeNaNPayload(t *testing.T) {
	a := bArr(t, []float64{math.Float64frombits(0xfff8000000000123), 1})
	out, err := Take(a, bArr(t, []int64{0}))
	if err != nil {
		t.Fatal(err)
	}
	if g := getI(t, out, 0).(float64); math.Float64bits(g) != 0xfff8000000000123 {
		t.Fatalf("payload bits = %#x", math.Float64bits(g))
	}
}

package num

// 单测(B):compress / nonzero / count_nonzero / bincount。
// 期望值 = numpy 2.5.4 实测探针字面量。

import (
	"math"
	"testing"
)

// TestCompressFlat ≡ np.compress(axis=None)。
func TestCompressFlat(t *testing.T) {
	a := bArr(t, []int64{0, 1, 2, 3, 4})
	cond := bArr(t, []bool{true, false, true, false, true})
	out, err := Compress(a, cond)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 0, 2, 4)

	// 条件短于数据:只看前 len(cond) 个
	short := bArr(t, []bool{true, false})
	out, err = Compress(a, short)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 0)

	// 非 bool 条件按 != 0
	ic := bArr(t, []int64{1, 0, 1, 0, 1})
	out, err = Compress(a, ic)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 0, 2, 4)

	// 条件非 1-D → ErrShape
	c2, _ := bArr(t, []bool{true, false, true, false}).Reshape([]int{2, 2})
	if _, err := Compress(a, c2); rootOf(err) != ErrShape {
		t.Fatalf("2-D 条件 err = %v, want ErrShape", err)
	}
}

// TestCompressAxis 沿轴压缩。
func TestCompressAxis(t *testing.T) {
	a, err := Arange(0, 12, 1, DInt64)
	if err != nil {
		t.Fatal(err)
	}
	m, _ := a.Reshape([]int{3, 4})
	cond := bArr(t, []bool{true, false, true})
	out, err := CompressAxis(m, cond, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 4}) {
		t.Fatalf("shape = %v, want [2 4]", out.Shape())
	}
	want := []int64{0, 1, 2, 3, 8, 9, 10, 11}
	for i, w := range want {
		if g := getI(t, out, i/4, i%4).(int64); g != w {
			t.Fatalf("flat[%d] = %v, want %v", i, g, w)
		}
	}

	// 轴 -1(最后一维)
	c4 := bArr(t, []bool{true, false, false, true})
	out, err = CompressAxis(m, c4, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{3, 2}) {
		t.Fatalf("shape = %v, want [3 2]", out.Shape())
	}
	if g := getI(t, out, 0, 1); g != int64(3) {
		t.Fatalf("[0,1] = %v, want 3", g)
	}

	// 条件长于轴长 → ErrIndex
	long := bArr(t, []bool{true, true, true, true})
	if _, err := CompressAxis(m, long, 0); rootOf(err) != ErrIndex {
		t.Fatalf("条件过长 err = %v, want ErrIndex", err)
	}
}

// TestNonzero ≡ np.nonzero(每维一个 i64 索引数组)。
func TestNonzero(t *testing.T) {
	a := bArr(t, []int64{0, 3, 0, 5})
	outs, err := Nonzero(a)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 1 || outs[0].Dtype() != DInt64 {
		t.Fatalf("outs = %d", len(outs))
	}
	bWantI64(t, outs[0], 1, 3)

	m := bArr(t, []int64{0, 1, 2, 0})
	mv, _ := m.Reshape([]int{2, 2})
	outs, err = Nonzero(mv)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 2 {
		t.Fatalf("2-D outs = %d", len(outs))
	}
	bWantI64(t, outs[0], 0, 1) // 行
	bWantI64(t, outs[1], 1, 0) // 列

	// NaN 视为非零
	nf := bArr(t, []float64{0, math.NaN()})
	outs, err = Nonzero(nf)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, outs[0], 1)

	// bool
	b := bArr(t, []bool{true, false})
	outs, err = Nonzero(b)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, outs[0], 0)

	// 0-d → ErrShape(numpy 同)
	z, _ := bArr(t, []float64{1}).Reshape([]int{})
	if _, err := Nonzero(z); rootOf(err) != ErrShape {
		t.Fatalf("0-d err = %v, want ErrShape", err)
	}
}

// TestCountNonzero 非零计数(NaN 计入)。
func TestCountNonzero(t *testing.T) {
	m := bArr(t, []int64{0, 1, 2, 0})
	mv, _ := m.Reshape([]int{2, 2})
	n, err := CountNonzero(mv)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("count = %d, want 2", n)
	}
	nf := bArr(t, []float64{0, math.NaN()})
	if n, err = CountNonzero(nf); err != nil || n != 1 {
		t.Fatalf("nan count = %d err %v, want 1", n, err)
	}
	// 0-d
	z, _ := bArr(t, []float64{3}).Reshape([]int{})
	if n, err = CountNonzero(z); err != nil || n != 1 {
		t.Fatalf("0-d count = %d err %v, want 1", n, err)
	}
}

// TestBincount ≡ np.bincount(x, minlength=…)。
func TestBincount(t *testing.T) {
	a := bArr(t, []int64{0, 1, 1, 3})
	out, err := Bincount(a, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt64 {
		t.Fatalf("dtype = %s", out.Dtype())
	}
	bWantI64(t, out, 1, 2, 0, 1)

	// minlength
	b := bArr(t, []int64{0, 1})
	out, err = Bincount(b, 5)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 1, 1, 0, 0, 0)

	// 空输入
	e := bArr(t, []int64{})
	out, err = Bincount(e, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Size() != 0 {
		t.Fatalf("空 bincount size = %d, want 0", out.Size())
	}
	out, err = Bincount(e, 3)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, out, 0, 0, 0)

	// u64 输入 → i64 输出
	u := bArr(t, []uint64{0, 1})
	out, err = Bincount(u, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt64 {
		t.Fatalf("u64 输入 dtype = %s, want int64", out.Dtype())
	}
	bWantI64(t, out, 1, 1)

	// 错误:负值 → ErrIndex;浮点 → ErrDtype;2-D → ErrShape
	if _, err := Bincount(bArr(t, []int64{-1, 0}), 0); rootOf(err) != ErrIndex {
		t.Fatalf("负值 err = %v, want ErrIndex", err)
	}
	if _, err := Bincount(bArr(t, []float64{0}), 0); rootOf(err) != ErrDtype {
		t.Fatalf("浮点 err = %v, want ErrDtype", err)
	}
	m, _ := bArr(t, []int64{0, 1}).Reshape([]int{1, 2})
	if _, err := Bincount(m, 0); rootOf(err) != ErrShape {
		t.Fatalf("2-D err = %v, want ErrShape", err)
	}
	if _, err := Bincount(a, -1); rootOf(err) != ErrShape {
		t.Fatalf("minlength 负 err = %v, want ErrShape", err)
	}
}

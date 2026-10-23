package num

import (
	"errors"
	"math"
	"testing"
)

// cond 广播 (2,3)、x 广播 (3,)、y 为 0-d —— numpy 右对齐广播的典型形态。
func TestWhereBasic(t *testing.T) {
	cond := mustSlice(t, []bool{true, false, true, false, true, false})
	cond, err := cond.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	x := bArr(t, []float64{1, 2, 3}) // 广播到列
	y := mustNew(t, []int{}, DFloat64)
	if err := y.SetItem([]int{}, -1.0); err != nil {
		t.Fatal(err)
	}
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.Shape(), []int{2, 3}; !shapeEqual(got, want) {
		t.Fatalf("shape %v, want %v", got, want)
	}
	bWantF64(t, out.Flatten(), 1, -1, 3, -1, 2, -1)
}

// 结果 dtype = promote(x, y):i32 与 f8 → f8。
func TestWherePromote(t *testing.T) {
	cond := mustSlice(t, []bool{true, false, true, false})
	x := bArr(t, []int32{7, 7, 7, 7})
	y := bArr(t, []float64{0.5, 0.5, 0.5, 0.5})
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype %s, want float64", out.Dtype())
	}
	bWantF64(t, out, 7, 0.5, 7, 0.5)
}

// cond 非 bool:按 != 0 取真。
func TestWhereTruthyCond(t *testing.T) {
	cond := bArr(t, []int32{0, 2, 0, -3})
	x := bArr(t, []float64{1, 1, 1, 1})
	y := bArr(t, []float64{9, 9, 9, 9})
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 9, 1, 9, 1)
}

// NaN 计为真(-0.0 为假),与 numpy truthiness 一致。
func TestWhereNaNTruthy(t *testing.T) {
	cond := bArr(t, []float64{math.NaN(), math.Copysign(0, -1), 0, 1})
	x := bArr(t, []float64{1, 2, 3, 4})
	y := bArr(t, []float64{-1, -2, -3, -4})
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 1, -2, -3, 4)
}

// 全 0-d → 0-d 结果。
func TestWhereZeroDim(t *testing.T) {
	cond := mustNew(t, []int{}, DBool)
	if err := cond.SetItem([]int{}, true); err != nil {
		t.Fatal(err)
	}
	x := mustNew(t, []int{}, DFloat64)
	if err := x.SetItem([]int{}, 2.5); err != nil {
		t.Fatal(err)
	}
	y := mustNew(t, []int{}, DFloat64)
	if err := y.SetItem([]int{}, -2.5); err != nil {
		t.Fatal(err)
	}
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.NDim() != 0 || out.Size() != 1 {
		t.Fatalf("0-d 期望,得到 shape %v", out.Shape())
	}
	v, err := out.Item()
	if err != nil {
		t.Fatal(err)
	}
	if v != 2.5 {
		t.Fatalf("0-d 值 = %v, want 2.5", v)
	}
}

// 元素搬运与 dtype 无关:c128 也走同一路径。
func TestWhereComplex(t *testing.T) {
	cond := mustSlice(t, []bool{true, false})
	x := bArr(t, []complex128{1 + 2i, 3 + 4i})
	y := bArr(t, []complex128{9 + 9i, 8 + 8i})
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	got0, err := out.Item(0)
	if err != nil {
		t.Fatal(err)
	}
	if got0 != complex128(1+2i) {
		t.Fatalf("[0] = %v, want 1+2i", got0)
	}
	got1, err := out.Item(1)
	if err != nil {
		t.Fatal(err)
	}
	if got1 != complex128(8+8i) {
		t.Fatalf("[1] = %v, want 8+8i", got1)
	}
}

// 不可广播 → ErrShape;S/U 无真值语义 → ErrDtype。
func TestWhereErrors(t *testing.T) {
	cond := mustSlice(t, []bool{true, false, true, false})
	cond, err := cond.Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	x := bArr(t, []float64{1, 2, 3})
	_, err = Where(cond, x, x)
	if !errors.Is(err, ErrShape) {
		t.Fatalf("不可广播应 ErrShape,得 %v", err)
	}
	scond, err := New([]int{2}, DBytes(3))
	if err != nil {
		t.Fatal(err)
	}
	if err := scond.SetItem([]int{0}, "a"); err != nil {
		t.Fatal(err)
	}
	y := bArr(t, []float64{1, 2})
	if _, err := Where(scond, y, y); !errors.Is(err, ErrDtype) {
		t.Fatalf("字符串 cond 应 ErrDtype,得 %v", err)
	}
}

// 只在真侧的位型被取出:±0 与 NaN payload 原样传递。
func TestWhereBitPreserving(t *testing.T) {
	cond := mustSlice(t, []bool{true, true})
	x := bArr(t, []float64{math.Copysign(0, -1), math.Float64frombits(0x7ff8000000000001)})
	y := bArr(t, []float64{9, 9})
	out, err := Where(cond, x, y)
	if err != nil {
		t.Fatal(err)
	}
	v0, _ := out.Item(0)
	if !math.Signbit(v0.(float64)) { // -0.0 位型保留
		t.Fatalf("[0] 应为 -0.0,得 %v", v0)
	}
	v1, _ := out.Item(1)
	if math.Float64bits(v1.(float64)) != 0x7ff8000000000001 { // NaN payload 按位传递
		t.Fatalf("[1] payload 位型变了: %x", math.Float64bits(v1.(float64)))
	}
}

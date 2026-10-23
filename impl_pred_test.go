package num

// 谓词/逻辑系单测(B)。期望值 = numpy 2.5.4 口径(写法与 impl_where_test 同,
// bArr / bWantBool / getI / wantRoot 复用同包测试辅助)。

import (
	"math"
	"testing"
)

// isnan:f8 五元素典型面(NaN / +Inf / -Inf / 0 / 1)。
func TestPredIsNanF64(t *testing.T) {
	a := bArr(t, []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0, 1})
	out, err := IsNan(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false, false, false, false)
}

// isnan:f4 面(NaN 判定在 f32 位型上)。
func TestPredIsNanF32(t *testing.T) {
	a := bArr(t, []float32{float32(math.NaN()), float32(math.Inf(1)), 0, -1})
	out, err := IsNan(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false, false, false)
}

// isnan/isinf/isfinite:f16 面(经 f16ToF32 升格判定,位型完整)。
func TestPredIsNanF16(t *testing.T) {
	a := mustNew(t, []int{3}, DFloat16)
	for i, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), 1.5} {
		if err := a.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	n, err := IsNan(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, n, true, false, false)
	i, err := IsInf(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, i, false, true, false)
	f, err := IsFinite(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, f, false, false, true)
}

// isnan:复数按分量(任一分量 NaN 即真);c64 同口径。
func TestPredIsNanComplex(t *testing.T) {
	a := bArr(t, []complex128{1 + 2i, complex(math.NaN(), 0), complex(0, math.NaN())})
	out, err := IsNan(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true, true)

	b := bArr(t, []complex64{1 + 2i, complex(float32(math.NaN()), 0)})
	out, err = IsNan(b)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true)
}

// isinf:f8 与复数任一分量 Inf。
func TestPredIsInf(t *testing.T) {
	a := bArr(t, []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0})
	out, err := IsInf(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true, true, false)

	c := bArr(t, []complex128{complex(math.Inf(1), 0), complex(0, math.Inf(-1)), 1 + 2i})
	out, err = IsInf(c)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, true, false)
}

// isfinite:NaN/Inf 皆假;复数须两分量皆有限。
func TestPredIsFinite(t *testing.T) {
	a := bArr(t, []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0})
	out, err := IsFinite(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, false, false, true)

	c := bArr(t, []complex128{complex(math.Inf(1), 0), complex(math.NaN(), 0), 1 + 2i})
	out, err = IsFinite(c)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, false, true)
}

// 整型/bool/S/M8 无 ufunc 循环 → ErrDtype(numpy TypeError 同口径)。
func TestPredErrDtype(t *testing.T) {
	for _, tc := range []struct {
		name string
		fn   func(*Array) (*Array, error)
	}{
		{"isnan", IsNan},
		{"isinf", IsInf},
		{"isfinite", IsFinite},
	} {
		for _, a := range []*Array{
			bArr(t, []int32{1, 2}),
			bArr(t, []bool{true, false}),
			mustNew(t, []int{2}, DBytes(3)),
			mustNew(t, []int{2}, DTime("us")),
		} {
			_, err := tc.fn(a)
			wantRoot(t, err, ErrDtype, tc.name+" "+a.Dtype().String())
		}
	}
}

// logical_and:整型非零即真、bool×bool、(2,1)×(1,3) 右对齐广播。
func TestLogicalAnd(t *testing.T) {
	x := bArr(t, []int32{0, 1, 2})
	y := bArr(t, []int32{1, 1, 0})
	out, err := LogicalAnd(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true, false)

	bx := bArr(t, []bool{false, true, false})
	by := bArr(t, []bool{true, true, false})
	out, err = LogicalAnd(bx, by)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true, false)

	bx, err = mustSlice(t, []bool{false, true}).Reshape([]int{2, 1})
	if err != nil {
		t.Fatal(err)
	}
	by, err = mustSlice(t, []bool{false, true, false}).Reshape([]int{1, 3})
	if err != nil {
		t.Fatal(err)
	}
	out, err = LogicalAnd(bx, by)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.Shape(), []int{2, 3}; !shapeEqual(got, want) {
		t.Fatalf("shape %v, want %v", got, want)
	}
	bWantBool(t, out.Flatten(), false, false, false, false, true, false)
}

// logical_or / logical_xor:同广播口径;NaN 为真。
func TestLogicalOrXor(t *testing.T) {
	x := bArr(t, []int32{0, 0, 2})
	y := bArr(t, []int32{1, 0, 0})
	out, err := LogicalOr(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false, true)

	out, err = LogicalXor(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false, true)

	// NaN 为真:NaN&0 → False,NaN&1 → True。
	n := bArr(t, []float64{math.NaN(), math.NaN()})
	m := bArr(t, []float64{0, 1})
	out, err = LogicalAnd(n, m)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true)
}

// logical_not:一元同形状 bool;NaN 取真后取反为假,-0.0 为假。
func TestLogicalNot(t *testing.T) {
	a := bArr(t, []float64{math.NaN(), 0, math.Copysign(0, -1), 1})
	out, err := LogicalNot(a)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.Shape(), []int{4}; !shapeEqual(got, want) {
		t.Fatalf("shape %v, want %v", got, want)
	}
	bWantBool(t, out, false, true, true, false)

	b := bArr(t, []bool{true, false})
	out, err = LogicalNot(b)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true)
}

// 二元先 promote 再取真值:i4×f8 → f8 上判非零;复数任一分量非零为真。
func TestLogicalPromoteAndComplex(t *testing.T) {
	x := bArr(t, []int32{0, 1})
	y := bArr(t, []float64{0.0, 2.0})
	out, err := LogicalAnd(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true)

	c := bArr(t, []complex128{0 + 1i, 0 + 0i})
	out, err = LogicalOr(c, bArr(t, []int32{0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, true)
}

// 错误:S 无真值语义 → ErrDtype;不可广播 → ErrShape。
func TestLogicalErrors(t *testing.T) {
	s := mustNew(t, []int{2}, DBytes(3))
	if err := s.SetItem([]int{0}, "a"); err != nil {
		t.Fatal(err)
	}
	_, err := LogicalAnd(s, bArr(t, []bool{true, false}))
	wantRoot(t, err, ErrDtype, "logical_and S")
	_, err = LogicalNot(s)
	wantRoot(t, err, ErrDtype, "logical_not S")
	x, err := mustSlice(t, []bool{true, false, true, false}).Reshape([]int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	_, err = LogicalAnd(x, bArr(t, []bool{true, false, true}))
	wantRoot(t, err, ErrShape, "logical_and 广播")
}

// 非连续视图走归一路径(Transpose 视图上的判定与拷贝结果一致)。
func TestPredOnStridedView(t *testing.T) {
	a := mustNew(t, []int{2, 3}, DFloat64)
	for i := 0; i < 6; i++ {
		if err := a.SetItem([]int{i / 3, i % 3}, float64(i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.SetItem([]int{0, 1}, math.NaN()); err != nil {
		t.Fatal(err)
	}
	v, err := a.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	out, err := IsNan(v)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.Shape(), []int{3, 2}; !shapeEqual(got, want) {
		t.Fatalf("shape %v, want %v", got, want)
	}
	bWantBool(t, out.Flatten(), false, false, true, false, false, false)

	n, err := LogicalNot(v)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, n.Flatten(), true, false, false, false, false, false)
}

// size-0 输入:输出空 bool 数组(空窗,非 nil)。
func TestPredEmpty(t *testing.T) {
	e := bArr(t, []float64{})
	out, err := IsFinite(e)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool || out.Size() != 0 {
		t.Fatalf("空输入应得 0 元素 bool,得 %s/%d", out.Dtype(), out.Size())
	}
	out, err = LogicalAnd(e, e)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool || out.Size() != 0 {
		t.Fatalf("空输入应得 0 元素 bool,得 %s/%d", out.Dtype(), out.Size())
	}
}

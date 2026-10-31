package num

import (
	"math"
	"testing"
)

// 把 Must API 全部调用一遍(既是覆盖率网,也是"链式饮食"回归)。
func TestMustCoverageSweep(t *testing.T) {
	base := func() *Array { return MustFromSlice([]float64{1, 2, 3, 4, 5, 6}) }
	// 构造
	MustZeros([]int{2, 2}, DFloat64)
	MustOnes([]int{2, 2}, DFloat64)
	MustEmpty([]int{2, 2}, DFloat64)
	MustFull([]int{2}, 7.0, DFloat64)
	MustArange(0, 3, 1, DFloat64)
	MustLinSpace(0, 1, 3, true)
	MustEye(2, DFloat64)
	MustIdentity(2)

	// 形状与视图
	m := base().MustReshape(2, 3)
	m.MustTranspose()
	m.MustSqueeze()
	base().MustExpandDims(0)
	m.MustSwapAxes(0, 1)
	m.MustSub(Slice(0, 1))
	m.MustBroadcastTo(4, 2, 3)
	m.MustCopy()
	m.MustCast(DFloat32) // same_kind:float→float;float→int 需 unsafe 版本
	base().MustCastUnsafe(DInt64)

	// 算术与一元
	a := base()
	a.MustAdd(a)
	a.MustSubtract(a)
	a.MustMultiply(a)
	a.MustDivide(a)
	a.MustAbs()
	a.MustNegative()
	a.MustExp()
	a.MustLog()
	a.MustSqrt()
	a.MustSin()
	a.MustCos()
	a.MustFloor()
	a.MustCeil()
	a.MustRound()
	a.MustRound(2)
	a.MustClip(0.0, 4.0)
	a.MustSort()
	a.MustDiag(0)

	// 线代
	mm := MustFromSlice([]float64{1, 2, 3, 4}).MustReshape(2, 2)
	mm.MustMatMul(mm)
	base().MustDot(base())
	mm.MustDet()
	mm.MustInv()
	mm.MustSolve(MustFromSlice([]float64{1, 0}))

	// 归约
	a.MustSum([]int{0}, false)
	a.MustMean([]int{0}, true)
	a.MustProd(nil, false)
	a.MustMax(nil, false)
	a.MustMin(nil, false)
	a.MustStd(nil, false)
	a.MustVar(nil, false)
	a.MustArgMax(nil, false)
	a.MustArgMin(nil, false)
	a.MustCumsum([]int{0})
	a.MustCumprod([]int{0})
	a.MustMeanAll()
	a.MustMaxAll()
	a.MustMinAll()

	// 取值
	a.MustItem(0)
	a.MustFloat(0)
	MustFromSlice([]int64{7}).MustInt(0)
	MustFromSlice([]int32{3}).MustInt(0)
	MustFromSlice([]int16{3}).MustInt(0)
	MustFromSlice([]int8{3}).MustInt(0)
	MustFromSlice([]uint64{3}).MustInt(0)
	MustFromSlice([]uint8{3}).MustInt(0)
	MustFromSlice([]float32{3}).MustInt(0)
	MustFromSlice([]bool{true}).MustInt(0)
	MustFromSlice([]float32{3}).MustFloat(0)
	MustFromSlice([]int8{3}).MustFloat(0)
	MustFromSlice([]uint8{3}).MustFloat(0)
	MustFromSlice([]bool{true}).MustFloat(0)

	// 包级泛型标量运算
	ix := MustFromSlice([]int8{1})
	if v, err := AddScalar(ix, int8(2)); err != nil || v.MustInt(0) != 3 {
		t.Fatalf("AddScalar: %v %v", v, err)
	}
	if _, err := SubScalar(ix, int8(1)); err != nil {
		t.Fatal(err)
	}
	if _, err := MulScalar(ix, int8(2)); err != nil {
		t.Fatal(err)
	}
	if _, err := DivScalar(bArr(t, []float64{4}), 2.0); err != nil {
		t.Fatal(err)
	}
	if _, err := PowScalar(bArr(t, []float64{2}), 3.0); err != nil {
		t.Fatal(err)
	}
	// 方法形式(非 Must)
	if _, err := bArr(t, []float64{1}).AddScalar(1.0); err != nil {
		t.Fatal(err)
	}
	if _, err := bArr(t, []float64{1}).SubScalar(1.0); err != nil {
		t.Fatal(err)
	}
	if _, err := bArr(t, []float64{1}).MulScalar(2.0); err != nil {
		t.Fatal(err)
	}
	if _, err := bArr(t, []float64{1}).DivScalar(2.0); err != nil {
		t.Fatal(err)
	}
	if _, err := bArr(t, []float64{1}).PowScalar(2.0); err != nil {
		t.Fatal(err)
	}
	// 标量支持的全部 Go 类型
	for _, v := range []any{
		int(1), int8(1), int16(1), int32(1), int64(1),
		uint(1), uint8(1), uint16(1), uint32(1), uint64(1),
		float32(1), float64(1), complex64(1), complex128(1), true,
	} {
		if _, err := scalarArr(v); err != nil {
			t.Fatalf("scalarArr(%T): %v", v, err)
		}
	}
	// 不支持的标量类型
	if _, err := scalarArr("x"); err == nil {
		t.Fatal("字符串标量应报错")
	}
	// nil 保护
	if _, err := Add(nil, bArr(t, []float64{1})); err == nil {
		t.Fatal("Add(nil,·) 应报错而不是 panic")
	}
	if _, err := Subtract(bArr(t, []float64{1}), nil); err == nil {
		t.Fatal("Subtract(·,nil) 应报错")
	}
	if _, err := Exp(nil); err == nil {
		t.Fatal("Exp(nil) 应报错")
	}
	// 链式结果正确性抽查
	got := MustArange(1, 5, 1, DFloat64).MustMulScalar(10.0).MustSumAll().MustFloat()
	if math.Abs(got-100) > 1e-12 { // (1+2+3+4)*10
		t.Fatalf("链式结果 = %v, want 100", got)
	}
}

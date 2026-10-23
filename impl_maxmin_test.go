package num

import (
	"errors"
	"math"
	"testing"
)

// 常规取大/取小。
func TestMaxMinBasic(t *testing.T) {
	x := bArr(t, []float64{1, 5, -3, 7})
	y := bArr(t, []float64{2, 0.5, -9, 7})
	out, err := Maximum(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, 2, 5, -3, 7)
	if out, err := Minimum(x, y); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, out, 1, 0.5, -9, 7)
	}
}

// NaN 按位传播:返回原操作数(保 payload)。
func TestMaxMinNaN(t *testing.T) {
	nanA := math.Float64frombits(0x7ff8000000000001)
	nanB := math.Float64frombits(0x7ff8000000000002)
	x := bArr(t, []float64{nanA, 1, nanB, math.Inf(1)})
	y := bArr(t, []float64{5, nanB, 3, 1})
	for _, f := range []func(*Array, *Array) (*Array, error){Maximum, Minimum} {
		out, err := f(x, y)
		if err != nil {
			t.Fatal(err)
		}
		// [0] 左 NaN → 左侧 payload
		if v, _ := out.Item(0); math.Float64bits(v.(float64)) != 0x7ff8000000000001 {
			t.Fatalf("[0] 应传播左侧 NaN payload,得 %x", math.Float64bits(v.(float64)))
		}
		// [1] 右 NaN → 右侧 payload
		if v, _ := out.Item(1); math.Float64bits(v.(float64)) != 0x7ff8000000000002 {
			t.Fatalf("[1] 应传播右侧 NaN payload,得 %x", math.Float64bits(v.(float64)))
		}
	}
}

// 相等取右侧(±0 的符号随右侧),与 numpy 的 a>b/a<b 口径一致。
func TestMaxMinZeroSign(t *testing.T) {
	neg := math.Copysign(0, -1)
	x := bArr(t, []float64{neg, 0})
	y := bArr(t, []float64{0, neg})
	out, err := Maximum(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := out.Item(0); math.Signbit(v.(float64)) { // -0.0 > 0.0 为假 → 取右侧 +0.0
		t.Fatal("[0] maximum(-0,+0) 应得 +0")
	}
	if v, _ := out.Item(1); !math.Signbit(v.(float64)) { // 0.0 > -0.0 为假 → 取右侧 -0.0
		t.Fatal("[1] maximum(+0,-0) 应得 -0")
	}
}

// 整型与混合 promote。
func TestMaxMinIntAndPromote(t *testing.T) {
	a := bArr(t, []int32{1, 5, -3})
	b := bArr(t, []int32{2, 0, -9})
	out, err := Maximum(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", out.Dtype())
	}
	wantI32(t, out, 2, 5, -3)

	mix, err := Maximum(a, bArr(t, []float64{2, 0.5, -9}))
	if err != nil {
		t.Fatal(err)
	}
	if mix.Dtype() != DFloat64 {
		t.Fatalf("混合 dtype %s, want float64", mix.Dtype())
	}
	bWantF64(t, mix, 2, 5, -3)
}

// 广播与 0-d。
func TestMaxMinBroadcast(t *testing.T) {
	x := bArr(t, []float64{1, 2, 3, 4, 5, 6})
	x, err := x.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	y := bArr(t, []float64{0, 4, 2}) // 广播到列
	out, err := Maximum(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out.Flatten(), 1, 4, 3, 4, 5, 6)

	z := mustNew(t, []int{}, DFloat64)
	if err := z.SetItem([]int{}, 2.5); err != nil {
		t.Fatal(err)
	}
	if out, err := Minimum(x, z); err != nil {
		t.Fatal(err)
	} else if out.Shape()[0] != 2 || out.Shape()[1] != 3 {
		t.Fatalf("0-d 广播形状错误: %v", out.Shape())
	}
}

// 复数无大小语义(numpy 侧也非稳定口径)→ ErrDtype。
func TestMaxMinComplexRejected(t *testing.T) {
	c := bArr(t, []complex128{1 + 2i})
	if _, err := Maximum(c, c); !errors.Is(err, ErrDtype) {
		t.Fatalf("复数 maximum 应 ErrDtype,得 %v", err)
	}
}

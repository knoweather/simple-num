package num

import (
	"errors"
	"math"
	"testing"
)

// wantI32 断言 int32 数组逐值相等(夹取结果保持 i4 dtype)。
func wantI32(t *testing.T, a *Array, want ...int32) {
	t.Helper()
	if a.Dtype() != DInt32 {
		t.Fatalf("dtype = %s, want int32", a.Dtype())
	}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	flat := a.Flatten()
	for i, w := range want {
		g := getI(t, flat, i).(int32)
		if g != w {
			t.Fatalf("[%d] = %v, want %v", i, g, w)
		}
	}
}

// 常规夹取:f8 数组 + f8 边界。
func TestClipBasic(t *testing.T) {
	a := bArr(t, []float64{-2, -0.5, 0, 1, 3})
	out, err := Clip(a, -0.5, 1.5)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out, -0.5, -0.5, 0, 1, 1.5)
}

// NaN 传播:NaN 不参与比较,原样保留。
func TestClipNaN(t *testing.T) {
	a := bArr(t, []float64{math.NaN(), 5, -5})
	out, err := Clip(a, 0.0, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	v0, _ := out.Item(0)
	if !math.IsNaN(v0.(float64)) {
		t.Fatalf("[0] 应为 NaN,得 %v", v0)
	}
	bWantF64(t, out, math.NaN(), 1, 0)
}

// lo > hi → 恒为 hi(numpy 实测行为)。
func TestClipLoGtHi(t *testing.T) {
	a := bArr(t, []int32{1, 2, 3})
	out, err := Clip(a, int32(3), int32(1))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", out.Dtype())
	}
	wantI32(t, out, 1, 1, 1)
}

// 单侧不限:nil 表示 numpy 的 None。
func TestClipHalfOpen(t *testing.T) {
	a := bArr(t, []float64{-5, 0, 5})
	if out, err := Clip(a, nil, 1.0); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, out, -5, 0, 1)
	}
	if out, err := Clip(a, -1.0, nil); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, out, -1, 0, 5)
	}
	// 两侧都 nil → 拷贝(值相同、底层缓冲不共享)
	cp, err := Clip(a, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, cp, -5, 0, 5)
	if len(a.buf) > 0 && len(cp.buf) > 0 && &a.buf[0] == &cp.buf[0] {
		t.Fatal("双侧 nil 应返回拷贝,不能共享缓冲")
	}
}

// dtype promote:f8 数组 + int32 边界 → f8;i4 数组 + i4 边界 → i4。
func TestClipPromote(t *testing.T) {
	a := bArr(t, []float64{0.5, 1.5, 2.5})
	out, err := Clip(a, int32(1), int32(2))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype %s, want float64", out.Dtype())
	}
	bWantF64(t, out, 1, 1.5, 2)

	b := bArr(t, []int32{-1, 0, 7})
	out2, err := Clip(b, int32(0), int32(5))
	if err != nil {
		t.Fatal(err)
	}
	if out2.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", out2.Dtype())
	}
	wantI32(t, out2, 0, 0, 5)
}

// 0-d、size-0、±0 位型。
func TestClipEdges(t *testing.T) {
	z := mustNew(t, []int{}, DFloat64)
	if err := z.SetItem([]int{}, 5.0); err != nil {
		t.Fatal(err)
	}
	if out, err := Clip(z, 0.0, 1.0); err != nil {
		t.Fatal(err)
	} else {
		v, _ := out.Item()
		if v != 1.0 {
			t.Fatalf("0-d clip = %v, want 1", v)
		}
	}
	empty := mustNew(t, []int{0}, DFloat64)
	if out, err := Clip(empty, 0.0, 1.0); err != nil {
		t.Fatal(err)
	} else if out.Size() != 0 {
		t.Fatalf("size-0 结果 size = %d", out.Size())
	}
	neg := bArr(t, []float64{math.Copysign(0, -1)})
	if out, err := Clip(neg, -1.0, 1.0); err != nil {
		t.Fatal(err)
	} else {
		v, _ := out.Item(0)
		if !math.Signbit(v.(float64)) {
			t.Fatalf("-0.0 位型应保留,得 %v", v)
		}
	}
}

// 复数无大小语义 → ErrDtype。
func TestClipComplexRejected(t *testing.T) {
	a := bArr(t, []complex128{1 + 2i})
	if _, err := Clip(a, 0.0, 1.0); !errors.Is(err, ErrDtype) {
		t.Fatalf("复数应 ErrDtype,得 %v", err)
	}
}

// TestClipAllKinds 覆盖 clipInto 的全部 dtype 分支(含无符号/窄整/f16/bool)。
func TestClipAllKinds(t *testing.T) {
	i8 := bArr(t, []int8{-128, 0, 127})
	if out, err := Clip(i8, int8(-5), int8(5)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(int8) != -5 || getI(t, got, 2).(int8) != 5 {
		t.Fatalf("i8 夹取错误: %v", got)
	}

	i16 := bArr(t, []int16{-1000, 0, 1000})
	if out, err := Clip(i16, int16(-50), int16(50)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(int16) != -50 || getI(t, got, 2).(int16) != 50 {
		t.Fatalf("i16 夹取错误: %v", got)
	}

	i64 := bArr(t, []int64{math.MinInt64, 0, math.MaxInt64})
	if out, err := Clip(i64, int64(-1), int64(1)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(int64) != -1 || getI(t, got, 2).(int64) != 1 {
		t.Fatalf("i64 夹取错误")
	}

	u8 := bArr(t, []uint8{0, 128, 255})
	if out, err := Clip(u8, uint8(10), uint8(200)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(uint8) != 10 || getI(t, got, 2).(uint8) != 200 {
		t.Fatalf("u8 夹取错误")
	}

	u16 := bArr(t, []uint16{0, 30000, 65535})
	if out, err := Clip(u16, uint16(100), uint16(60000)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(uint16) != 100 || getI(t, got, 2).(uint16) != 60000 {
		t.Fatalf("u16 夹取错误")
	}

	u32 := bArr(t, []uint32{0, 4000000000})
	if out, err := Clip(u32, uint32(1), uint32(3999999999)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(uint32) != 1 || getI(t, got, 1).(uint32) != 3999999999 {
		t.Fatalf("u32 夹取错误")
	}

	u64 := bArr(t, []uint64{0, 1 << 63})
	if out, err := Clip(u64, uint64(1), uint64(1<<63-1)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(uint64) != 1 || getI(t, got, 1).(uint64) != 1<<63-1 {
		t.Fatalf("u64 夹取错误")
	}

	f32 := bArr(t, []float32{-2, 0, 2})
	if out, err := Clip(f32, float32(-1), float32(1)); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(float32) != -1 || getI(t, got, 2).(float32) != 1 {
		t.Fatalf("f32 夹取错误")
	}

	// f16:用 New + SetItem 构造(存储为 2 字节半精度)。
	f16, err := New([]int{3}, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{-2, 0, 2} {
		if err := f16.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	if out, err := Clip(f16, float32(-1), float32(1)); err != nil {
		t.Fatal(err)
	} else if out.Dtype() != DFloat32 { // f16 与 f32 边界按 numpy 提升为 f32
		t.Fatalf("f16 结果 dtype %s, want float32", out.Dtype())
	} else if got := out.Flatten(); getI(t, got, 0).(float32) != -1 || getI(t, got, 2).(float32) != 1 {
		t.Fatalf("f16 夹取值错误")
	}

	// bool:夹到 false/true 两个极值。
	b := bArr(t, []bool{false, true})
	if out, err := Clip(b, false, true); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); getI(t, got, 0).(bool) || !getI(t, got, 1).(bool) {
		t.Fatalf("bool 夹取错误")
	}
	if out, err := Clip(b, true, true); err != nil {
		t.Fatal(err)
	} else if got := out.Flatten(); !getI(t, got, 0).(bool) || !getI(t, got, 1).(bool) {
		t.Fatalf("bool 下界 true 应全真")
	}
}

// TestClipNonContiguous 非连续输入走拷贝归一路径。
func TestClipNonContiguous(t *testing.T) {
	a := mustSlice(t, []float64{1, 2, 3, 4, 5, 6})
	m, err := a.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	tr, err := m.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	out, err := Clip(tr, 2.0, 5.0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out.Flatten(), 2, 4, 2, 5, 3, 5)
}

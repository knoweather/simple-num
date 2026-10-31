package num

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// Must 风格:链式调用与 panic 语义。
func TestMustChaining(t *testing.T) {
	// 构造 → 变形 → 转置 → 全归约 → 取标量
	s := MustFromSlice([]float64{1, 2, 3, 4, 5, 6}).
		MustReshape(2, 3).
		MustTranspose().
		MustSumAll()
	if got := s.MustFloat(); got != 21 {
		t.Fatalf("链式求和 = %v, want 21", got)
	}
	// 形状与逐元素
	m := MustZeros([]int{2, 3}, DFloat64).MustAdd(MustOnes([]int{2, 3}, DFloat64))
	if !shapeEqual(m.Shape(), []int{2, 3}) {
		t.Fatalf("形状 %v", m.Shape())
	}
	if v := m.MustSumAll().MustFloat(); v != 6 {
		t.Fatalf("0+1 全 1 矩阵之和 = %v, want 6", v)
	}
	// 转置两次回到原样
	back := m.MustTranspose().MustTranspose()
	if !ArrayEqual(back, m) {
		t.Fatal("两次转置应回到原数组")
	}
	// 切片 + 归约
	sl := MustArange(0, 10, 1, DFloat64).MustSub(Slice(2, 5))
	if got := sl.MustSumAll().MustFloat(); got != 9 { // 2+3+4
		t.Fatalf("切片求和 = %v, want 9", got)
	}
	// 一元算子链
	e := MustLinSpace(0, 1, 5, true).
		MustMultiply(MustFull([]int{5}, 2.0, DFloat64)).
		MustAdd(MustFull([]int{5}, 1.0, DFloat64)).
		MustSqrt()
	if v := e.MustFloat(0); math.Abs(v-1) > 1e-12 {
		t.Fatalf("一元链 [0] = %v, want 1", v)
	}
	// 类型转换 + 取整
	// float→int 属于跨 kind,Cast 会按 same_kind 拒绝(numpy 同),这里用 unsafe 版本
	i := MustFromSlice([]float64{1.9, 2.1}).MustRound().MustCastUnsafe(DInt64)
	if v := i.MustInt(0); v != 2 {
		t.Fatalf("取整 [0] = %v, want 2", v)
	}
	// 线代链
	d := MustEye(3, DFloat64).MustMultiply(MustFull([]int{3, 3}, 2.0, DFloat64)).MustDet()
	if math.Abs(d-8) > 1e-12 {
		t.Fatalf("det(2I) = %v, want 8", d)
	}
	inv := MustEye(2, DFloat64).MustInv()
	if !ArrayEqual(inv, MustEye(2, DFloat64)) {
		t.Fatal("I 的逆应为 I")
	}
	x := MustFromSlice([]float64{2, 0, 0, 4}).MustReshape(2, 2).MustSolve(MustFromSlice([]float64{2, 8}))
	bWantF64(t, x, 1, 2)
	// 排序 / 裁剪 / 最值
	srt := MustFromSlice([]float64{3, 1, 2}).MustSort()
	bWantF64(t, srt, 1, 2, 3)
	cl := MustFromSlice([]float64{-5, 0, 5}).MustClip(-1.0, 1.0)
	bWantF64(t, cl, -1, 0, 1)
	if v := MustFromSlice([]float64{3, 1, 2}).MustMaxAll().MustFloat(); v != 3 {
		t.Fatalf("max = %v", v)
	}
	if v := MustFromSlice([]float64{3, 1, 2}).MustMinAll().MustFloat(); v != 1 {
		t.Fatalf("min = %v", v)
	}
}

// Must 的 panic 语义:带操作名,且能 errors.Is 命中原始哨兵错误。
func TestMustPanics(t *testing.T) {
	catch := func(name string, fn func()) (msg string, err error) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: 未 panic", name)
			}
			if e, ok := r.(error); ok {
				err = e
				msg = e.Error()
				return
			}
			msg = "非 error 类型"
		}()
		fn()
		return "", nil
	}
	// 形状不符
	msg, err := catch("MustReshape", func() {
		MustFromSlice([]float64{1, 2, 3}).MustReshape(2, 2)
	})
	if !errors.Is(err, ErrShape) {
		t.Fatalf("MustReshape 错误未包 ErrShape: %v", err)
	}
	if !strings.Contains(msg, "Reshape") {
		t.Fatalf("panic 信息应含操作名: %q", msg)
	}
	// 轴越界
	_, err = catch("MustTranspose", func() {
		MustFromSlice([]float64{1, 2, 3}).MustReshape(1, 3).MustTranspose(0, 5)
	})
	if !errors.Is(err, ErrIndex) && !errors.Is(err, ErrShape) {
		t.Fatalf("MustTranspose 错误类型: %v", err)
	}
	// 下标越界
	_, err = catch("MustItem", func() {
		MustFromSlice([]float64{1, 2, 3}).MustItem(5)
	})
	if err == nil {
		t.Fatal("MustItem 越界应 panic")
	}
	// 非数值取 float
	_, err = catch("MustFloat", func() {
		MustFromSlice([]string{"x"}).MustFloat(0)
	})
	if !errors.Is(err, ErrDtype) {
		t.Fatalf("MustFloat 字符串应包 ErrDtype: %v", err)
	}
	// 奇异矩阵求逆
	_, err = catch("MustInv", func() {
		MustFromSlice([]float64{1, 2, 2, 4}).MustReshape(2, 2).MustInv()
	})
	if err == nil {
		t.Fatal("奇异矩阵求逆应 panic")
	}
	// 构造失败
	_, err = catch("MustZeros", func() {
		MustZeros([]int{-1, 2}, DFloat64)
	})
	if !errors.Is(err, ErrShape) && !errors.Is(err, ErrValue) {
		t.Fatalf("负维度应报形状/取值错误: %v", err)
	}
	// 非方阵求逆
	_, err = catch("MustInv2", func() {
		MustZeros([]int{2, 3}, DFloat64).MustInv()
	})
	if !errors.Is(err, ErrShape) {
		t.Fatalf("非方阵应报 ErrShape: %v", err)
	}
}

package num

import (
	"math"
	"testing"
)

// same_kind 权限矩阵 spot(逐格取自 numpy 2.5.4 实测,PRINT 见)。
func TestCastMatrixSpots(t *testing.T) {
	cases := []struct {
		from, to Kind
		want     bool
		note     string // numpy 观察注记
	}{
		{Int64, Float32, true, "i8→f4 same_kind 允许(允许失真)"},
		{Float64, Float32, true, "f8→f4 同 kind 收窄:允(经典口径)"},
		{Float32, Float16, true, "f4→f2 收窄:允"},
		{Float16, Float64, true, "f2→f8 加宽:允"},
		{Bool, Uint8, false, "b→u1:same_kind 拒"},
		{Uint8, Int64, true, "u1→i8:允"},
		{Int8, Uint32, false, "i1→u4:拒"},
		{Float64, Int64, false, "f8→i8:拒"},
		{Complex128, Complex64, true, "c16→c8 同 kind 收窄:允"},
		{Complex64, Complex128, true, "c8→c16:允"},
		{Int16, Complex64, true, "i2→c8:允"},
		{Int32, Complex64, true, "i4→c8:允(same_kind 宽松,精度由 promote 面保证)"},
		{Int64, Bool, true, "i8→b1:same_kind 允(非零→True)"},
		{Bool, Int8, true, "b→i1:允"},
		{Int64, Bytes, true, "i8→S:数值 → 文本 same_kind 允(numpy 实测)"},
		{Float32, Unicode, true, "f4→U:允"},
		{Complex128, Bytes, true, "c16→S:允"},
		{Bytes, Unicode, true, "S→U:允(读出按 UTF-8 解码)"},
		{Unicode, Bytes, false, "U→S:拒(unicode 降 ASCII 可能失真)"},
		{Datetime64, Bytes, false, "M8→S:same_kind 拒(datetime 转文本仅 unsafe)"},
		{Bytes, Float64, false, "S→f8:same_kind 拒(仅 unsafe)"},
	}
	for _, c := range cases {
		if got := castAllowed(c.from, c.to); got != c.want {
			t.Errorf("castAllowed(%s→%s) = %v want %v(%s)", c.from, c.to, got, c.want, c.note)
		}
	}
}

func TestCastValues(t *testing.T) {
	// i64 → f64
	x, _ := FromSlice([]int64{1, 2, 3})
	f, err := Cast(x, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{1, 2, 3} {
		if got := getI(t, f, i); got != w {
			t.Fatalf("f[%d]=%v want %v", i, got, w)
		}
	}
	// i8 → i32(同家族收窄允许)
	y, _ := FromSlice([]int8{-5, 3})
	w, err := Cast(y, DInt32)
	if err != nil {
		t.Fatal(err)
	}
	if got := getI(t, w, 0); got != int32(-5) {
		t.Fatalf("w[0]=%v", got)
	}
	// u8 → i64
	z, err := Cast(mustNew(t, []int{1}, DUint8), DInt64)
	if err != nil {
		t.Fatal(err)
	}
	if got := getI(t, z, 0); got != int64(0) {
		t.Fatalf("u→i 值:%v", got)
	}
	// f32 → f64
	v32, _ := FromSlice([]float32{1.5})
	v64, err := Cast(v32, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if got := getI(t, v64, 0); got != float64(float32(1.5)) {
		t.Fatalf("v64[0]=%v", got)
	}
	// b → i8:1/0
	b, _ := FromSlice([]bool{true, false})
	bi, err := Cast(b, DInt8)
	if err != nil {
		t.Fatal(err)
	}
	if got := getI(t, bi, 0); got != int8(1) || getI(t, bi, 1) != int8(0) {
		t.Fatalf("b→i1 = %v / %v", getI(t, bi, 0), getI(t, bi, 1))
	}
	// i → b:非零 → true
	i8, _ := FromSlice([]int8{7, 0})
	ib, err := Cast(i8, DBool)
	if err != nil {
		t.Fatal(err)
	}
	if got := getI(t, ib, 0); got != true || getI(t, ib, 1) != false {
		t.Fatalf("i→b = %v / %v", getI(t, ib, 0), getI(t, ib, 1))
	}
}

func TestCastErrors(t *testing.T) {
	f64a, _ := FromSlice([]float64{1.5})
	if _, err := Cast(f64a, DInt64); err == nil {
		t.Error("f8→i8 应 ErrDtype(same_kind 拒)")
	} else if rootOf(err) != ErrDtype {
		t.Errorf("错误包:%v", err)
	}
	ia, _ := FromSlice([]int64{1})
	if _, err := Cast(ia, DUint8); err == nil {
		t.Error("i→u 应 ErrDtype")
	}
	// 同 dtype 直通返回原对象
	same, err := Cast(ia, DInt64)
	if err != nil || same != ia {
		t.Errorf("同 dtype Cast 应返回原对象(same=%v err=%v)", same, err)
	}
	// 非法 dtype 参数
	if _, err := Cast(ia, Dtype{Kind: Kind(42)}); err == nil {
		t.Error("非法 dtype 应报错")
	}
}

// —— CastUnsafe(unsafe 口径)逐值锁死 ——
// 期望值全部按本机 numpy 2.2.1 实测转录;Cast 侧既有口径另组复核(不许变)。

func unsafeCastTo(t *testing.T, a *Array, dt Dtype) *Array {
	t.Helper()
	got, err := CastUnsafe(a, dt)
	if err != nil {
		t.Fatalf("CastUnsafe %s → %s: %v", a.Dtype(), dt, err)
	}
	if got.Dtype() != dt {
		t.Fatalf("dtype = %s want %s", got.Dtype(), dt)
	}
	return got
}

func checkUnsafe[T any](t *testing.T, a *Array, dt Dtype, want []T) {
	t.Helper()
	got := unsafeCastTo(t, a, dt)
	for i, w := range want {
		if v := getI(t, got, i); v != any(w) {
			t.Errorf("[%d] = %v want %v", i, v, w)
		}
	}
}

func TestCastUnsafeValues(t *testing.T) {
	t.Run("Unsafe_f8_to_i4", func(t *testing.T) {
		// 向零截断;NaN/±Inf/超界 → 目标位宽硬件不定值(amd64 = 0x80000000)
		a := mustSlice(t, []float64{1.7, -1.7, 2.5, math.NaN(), math.Inf(1), math.Inf(-1), 1e18})
		checkUnsafe(t, a, DInt32, []int32{1, -1, 2, -2147483648, -2147483648, -2147483648, -2147483648})
	})
	t.Run("Unsafe_f8_to_i1", func(t *testing.T) {
		a := mustSlice(t, []float64{127.5, 128.0, -1.0, 0.5})
		checkUnsafe(t, a, DInt8, []int8{127, -128, -1, 0})
	})
	t.Run("Unsafe_f8_to_b", func(t *testing.T) {
		a := mustSlice(t, []float64{0.0, 1.0, -2.0, math.NaN()})
		checkUnsafe(t, a, DBool, []bool{false, true, true, true}) // 非零即真,NaN 为真
	})
	t.Run("Unsafe_i1_to_b", func(t *testing.T) {
		a := mustSlice(t, []int8{0, 1, -1})
		checkUnsafe(t, a, DBool, []bool{false, true, true})
	})
	t.Run("Unsafe_i8_to_u1", func(t *testing.T) {
		a := mustSlice(t, []int64{-1, 255, 256, 300})
		checkUnsafe(t, a, DUint8, []uint8{255, 255, 0, 44}) // 补码截低位
	})
	t.Run("Unsafe_i8_to_i1", func(t *testing.T) {
		a := mustSlice(t, []int64{-1, 127, 128})
		checkUnsafe(t, a, DInt8, []int8{-1, 127, -128})
	})
	t.Run("Unsafe_i4_to_u8", func(t *testing.T) {
		a := mustSlice(t, []int32{-1, 0, 7})
		checkUnsafe(t, a, DUint64, []uint64{18446744073709551615, 0, 7})
	})
	t.Run("Unsafe_f8_to_u4", func(t *testing.T) {
		a := mustSlice(t, []float64{-0.5, 2.9})
		checkUnsafe(t, a, DUint32, []uint32{0, 2}) // 向零截断(先有符号转换再按宽度落位)
	})
	t.Run("Unsafe_c16_to_f8", func(t *testing.T) {
		a := mustSlice(t, []complex128{1 + 2i, 0.5 + 3.5i})
		checkUnsafe(t, a, DFloat64, []float64{1.0, 0.5}) // 取实部
	})
	t.Run("Unsafe_c16_to_b", func(t *testing.T) {
		a := mustSlice(t, []complex128{0 + 0i, 1 + 2i})
		checkUnsafe(t, a, DBool, []bool{false, true}) // 任务口径:只看实部
	})
	t.Run("Unsafe_f8_to_c8", func(t *testing.T) {
		a := mustSlice(t, []float64{1.5, -0.5})
		checkUnsafe(t, a, DComplex64, []complex64{1.5 + 0i, -0.5 + 0i})
	})
	t.Run("Unsafe_S3_to_U3", func(t *testing.T) {
		s := mustNew(t, []int{2}, DBytes(3))
		for i, v := range []string{"abc", "de"} {
			if err := s.SetItem([]int{i}, v); err != nil {
				t.Fatal(err)
			}
		}
		got := unsafeCastTo(t, s, DUnicode(3))
		for i, w := range []string{"abc", "de"} {
			if v := getI(t, got, i); v != w {
				t.Errorf("[%d] = %v want %v", i, v, w)
			}
		}
	})
	t.Run("Unsafe_U5_to_S5_nonASCII", func(t *testing.T) {
		// numpy 抛 UnicodeEncodeError;本库不抛,锁死口径 = 非 ASCII 码元 → '?'
		u := mustNew(t, []int{2}, DUnicode(5))
		for i, v := range []string{"aébc", "ok"} {
			if err := u.SetItem([]int{i}, v); err != nil {
				t.Fatal(err)
			}
		}
		got := unsafeCastTo(t, u, DBytes(5))
		for i, w := range []string{"a?bc", "ok"} {
			if v := getI(t, got, i); v != w {
				t.Errorf("[%d] = %v want %v", i, v, w)
			}
		}
	})
	t.Run("Unsafe_M8_to_S_rejected", func(t *testing.T) {
		m := mustNew(t, []int{1}, DTime("s"))
		if err := m.SetItem([]int{0}, int64(1)); err != nil {
			t.Fatal(err)
		}
		_, err := CastUnsafe(m, DBytes(24))
		if err == nil || rootOf(err) != ErrDtype {
			t.Errorf("M8→S 应 ErrDtype(文本渲染未实现,宁拒不冒充): %v", err)
		}
	})
	t.Run("Unsafe_M8_unit_rejected", func(t *testing.T) {
		m := mustNew(t, []int{1}, DTime("s"))
		n := mustNew(t, []int{1}, DTime("ms"))
		e1, e2 := "", ""
		if _, err := Cast(m, n.Dtype()); err == nil || rootOf(err) != ErrDtype {
			t.Errorf("Cast M8[s]→M8[ms] 应 ErrDtype: %v", err)
		} else {
			e1 = err.Error()
		}
		if _, err := CastUnsafe(m, n.Dtype()); err == nil || rootOf(err) != ErrDtype {
			t.Errorf("CastUnsafe M8[s]→M8[ms] 应 ErrDtype: %v", err)
		} else {
			e2 = err.Error()
		}
		if e1 != e2 {
			t.Errorf("文案须与 Cast 同款:%q vs %q", e1, e2)
		}
	})
	t.Run("Unsafe_same_dtype_same_pointer", func(t *testing.T) {
		a := mustSlice(t, []int64{1})
		got, err := CastUnsafe(a, DInt64)
		if err != nil {
			t.Fatal(err)
		}
		if got != a {
			t.Errorf("同 dtype 应返回原对象(got=%v)", got)
		}
	})
	t.Run("Unsafe_Cast_still_rejected", func(t *testing.T) {
		// unsafe 只解锁 CastUnsafe;Cast 侧既有拒绝一字不变
		f := mustSlice(t, []float64{1.5})
		if _, err := Cast(f, DInt32); err == nil || rootOf(err) != ErrDtype {
			t.Errorf("Cast f8→i4 仍应 ErrDtype: %v", err)
		}
		i4 := mustSlice(t, []int32{-1})
		if _, err := Cast(i4, DUint64); err == nil || rootOf(err) != ErrDtype {
			t.Errorf("Cast i4→u8 仍应 ErrDtype: %v", err)
		}
		c16 := mustSlice(t, []complex128{1 + 2i})
		if _, err := Cast(c16, DFloat64); err == nil || rootOf(err) != ErrDtype {
			t.Errorf("Cast c16→f8 仍应 ErrDtype: %v", err)
		}
		u5 := mustNew(t, []int{1}, DUnicode(5))
		if err := u5.SetItem([]int{0}, "abc"); err != nil {
			t.Fatal(err)
		}
		if _, err := Cast(u5, DBytes(5)); err == nil || rootOf(err) != ErrDtype {
			t.Errorf("Cast U5→S5 仍应 ErrDtype: %v", err)
		}
		// 同组合在 CastUnsafe 放行且值正确
		if got, err := CastUnsafe(f, DInt32); err != nil || getI(t, got, 0) != int32(1) {
			t.Errorf("CastUnsafe f8→i4 应放行且 = 1: %v", err)
		}
	})
}

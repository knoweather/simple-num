package num

import (
	"errors"
	"math"
	"testing"
)

// 引擎端到端:stub 内核以 _test_ 前缀注册,与 B 的正式算子名(add/lt/…)隔离,
// 防止注册覆盖次序不确定导致测试相互污染。

func getI(t *testing.T, a *Array, idx ...int) any {
	t.Helper()
	v, err := a.Item(idx...)
	if err != nil {
		t.Fatalf("Item %v: %v", idx, err)
	}
	return v
}

func init() {
	registerBinaryTyped[float64]("_test_add", func(dst, a, b []float64) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	})
	registerBinaryTyped[int64]("_test_add", func(dst, a, b []int64) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	})
	registerBinaryTyped[float64]("_test_div", func(dst, a, b []float64) {
		copy(dst, a)
	})
	registerUnaryTyped[float64]("_test_double", func(dst, src []float64) {
		for i := range dst {
			dst[i] = 2 * src[i]
		}
	})
	registerCompareTyped[float64]("_test_lt", func(dst []bool, a, b []float64) {
		for i := range dst {
			dst[i] = a[i] < b[i]
		}
	})
	registerUnarySpec("_test_double", UnarySpec{Fallback: []Kind{Float64}})
	registerBinarySpec("_test_div", BinarySpec{ForceOut: Float64})

	// 标量快速路径内核(bench 靶子 BenchmarkAddF64BroadcastScalar1e6 走 _test_add):
	// 与全量内核同口径,仅 0-stride 逐元素展开。
	registerBinaryScalarTyped[float64]("_test_add",
		func(dst, a []float64, s float64) {
			for i := range dst {
				dst[i] = s + a[i]
			}
		},
		func(dst, a []float64, s float64) {
			for i := range dst {
				dst[i] = a[i] + s
			}
		})
	registerBinaryScalarTyped[int64]("_test_add",
		func(dst, a []int64, s int64) {
			for i := range dst {
				dst[i] = s + a[i]
			}
		},
		func(dst, a []int64, s int64) {
			for i := range dst {
				dst[i] = a[i] + s
			}
		})
	// 带探针的算子:标量内核被命中即计数 → 断言"何时走快速路径、何时回退"。
	registerBinaryTyped[float64]("_test_scalar", func(dst, a, b []float64) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	})
	registerBinaryScalarTyped[float64]("_test_scalar",
		func(dst, a []float64, s float64) {
			scalarFastHits++
			for i := range dst {
				dst[i] = s + a[i]
			}
		},
		func(dst, a []float64, s float64) {
			scalarFastHits++
			for i := range dst {
				dst[i] = a[i] + s
			}
		})
}

// scalarFastHits 探针计数(_test_scalar 的 ScalarL/ScalarR 内核被命中次数)。
var scalarFastHits int

func TestDispatchBinarySameShape(t *testing.T) {
	x, err := New([]int{2, 2}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	vals := [2][2]float64{{1, 2}, {3, 4}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if err := x.SetItem([]int{i, j}, vals[i][j]); err != nil {
				t.Fatal(err)
			}
		}
	}
	y, _ := FromSlice([]float64{10, 20, 30, 40})
	y2, _ := y.Reshape([]int{2, 2})
	out, err := dispatchBinary("_test_add", x, y2)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("out dtype %s", out.Dtype())
	}
	want := [2][2]float64{{11, 22}, {33, 44}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if got := getI(t, out, i, j); got != want[i][j] {
				t.Fatalf("[%d,%d] = %v want %v", i, j, got, want[i][j])
			}
		}
	}
}

// 广播:(3,1) + (1,4) → (3,4);numpy 文档同款算例。
func TestDispatchBinaryBroadcast(t *testing.T) {
	xa, err := FromSlice([]int64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	x, err := xa.Reshape([]int{3, 1})
	if err != nil {
		t.Fatal(err)
	}
	ya, err := FromSlice([]int64{10, 20, 30, 40})
	if err != nil {
		t.Fatal(err)
	}
	y, err := ya.Reshape([]int{1, 4})
	if err != nil {
		t.Fatal(err)
	}
	out, err := dispatchBinary("_test_add", x, y)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{3, 4}) || out.Dtype() != DInt64 {
		t.Fatalf("out %v %s", out.Shape(), out.Dtype())
	}
	if got := getI(t, out, 2, 3); got != int64(43) {
		t.Fatalf("[2,3] = %v want 43", got)
	}
	if got := getI(t, out, 0, 1); got != int64(21) {
		t.Fatalf("[0,1] = %v want 21", got)
	}
}

// 晋级:int8 + int64 → int64(numpy promote 同款);广播维读交叉安然。
func TestDispatchBinaryPromote(t *testing.T) {
	x, _ := FromSlice([]int8{-1, 2})
	y, _ := FromSlice([]int64{10, 20})
	out, err := dispatchBinary("_test_add", x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt64 {
		t.Fatalf("promote 输出应为 int64,得 %s", out.Dtype())
	}
	if got := getI(t, out, 0); got != int64(9) {
		t.Fatalf("[0] = %v want 9", got)
	}
	if got := getI(t, out, 1); got != int64(22) {
		t.Fatalf("[1] = %v want 22", got)
	}
}

// ForceOut:div 对 int64 输入 → Float64(numpy true_divide 同款)。
func TestDispatchBinaryForceOut(t *testing.T) {
	x, _ := FromSlice([]int64{4, 5})
	out, err := dispatchBinary("_test_div", x, x)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("div 输出应为 float64,得 %s", out.Dtype())
	}
}

// 一元 fallback:int64 输入 → Float64 内核(numpy exp(int) → f64 同构)。
func TestDispatchUnaryFallback(t *testing.T) {
	x, _ := FromSlice([]int64{1, 2, 3})
	out, err := dispatchUnary("_test_double", x)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("int 输入 fallback 应得 f64,得 %s", out.Dtype())
	}
	if got := getI(t, out, 1); got != 4.0 {
		t.Fatalf("[1] = %v want 4", got)
	}
}

// 比较:输出 bool;f64 输入。
func TestDispatchCompare(t *testing.T) {
	x, _ := FromSlice([]float64{1, 3})
	y, _ := FromSlice([]float64{2, 2})
	out, err := dispatchCompare("_test_lt", x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("cmp 输出应为 bool,得 %s", out.Dtype())
	}
	if got := getI(t, out, 0); got != true {
		t.Fatalf("[0] = %v want true", got)
	}
	if got := getI(t, out, 1); got != false {
		t.Fatalf("[1] = %v want false", got)
	}
}

// 错误路径:形状不相容 / 未注册 op / 未注册 dtype。
func TestDispatchErrors(t *testing.T) {
	x, _ := FromSlice([]int64{1, 2, 3, 4})
	y, _ := FromSlice([]int64{1, 2, 3})
	if _, err := dispatchBinary("_test_add", x, y); err == nil {
		t.Error("形状不相容应报错")
	} else if rootOf(err) != ErrShape {
		t.Errorf("错误包应为 ErrShape:%v", err)
	}
	if _, err := dispatchBinary("_test_noop", x, x); err == nil {
		t.Error("未注册 op 应报错")
	}
	// int16 stash:无 int16 内核且 fallback 未含可满足目标 → ErrDtype
	z, _ := FromSlice([]int16{1})
	saved := unaryReg["_test_double"].Fallback
	unaryReg["_test_double"].Fallback = nil
	if _, err := dispatchUnary("_test_double", z); err == nil {
		t.Error("无内核 dtype 应报 ErrDtype")
	} else if rootOf(err) != ErrDtype {
		t.Errorf("错误包:%v", err)
	}
	unaryReg["_test_double"].Fallback = saved
}

// kindOfVal ↔ Typed 注册面:多 T 绑定(注册即校验 T→Kind 映射覆盖)。
func TestTypedRegistrationKinds(t *testing.T) {
	registerUnaryTyped[int32]("_test_dbl_i32", func(dst, src []int32) {
		for i := range dst {
			dst[i] = 2 * src[i]
		}
	})
	registerUnaryTyped[uint8]("_test_dbl_u8", func(dst, src []uint8) {
		for i := range dst {
			dst[i] = src[i] + 1
		}
	})
	registerUnaryTyped[bool]("_test_not", func(dst, src []bool) {
		for i := range dst {
			dst[i] = !src[i]
		}
	})
	registerBinaryTyped[float32]("_test_addf32", func(dst, a, b []float32) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	})

	x32, _ := Full([]int{3}, int64(5), DInt32)
	o32, err := dispatchUnary("_test_dbl_i32", x32)
	if err != nil {
		t.Fatalf("i32: %v", err)
	}
	if v, _ := o32.Item(0); v.(int32) != 10 {
		t.Fatalf("i32 结果 = %v", v)
	}

	x8, _ := Full([]int{2}, int64(7), DUint8)
	o8, err := dispatchUnary("_test_dbl_u8", x8)
	if err != nil {
		t.Fatalf("u8: %v", err)
	}
	if v, _ := o8.Item(0); v.(uint8) != 8 {
		t.Fatalf("u8 结果 = %v", v)
	}

	xb, _ := Full([]int{2}, true, DBool)
	ob, err := dispatchUnary("_test_not", xb)
	if err != nil {
		t.Fatalf("bool: %v", err)
	}
	if v, _ := ob.Item(0); v.(bool) != false {
		t.Fatalf("bool 结果 = %v", v)
	}

	xf, _ := Full([]int{2}, int64(2), DFloat32)
	of, err := dispatchBinary("_test_addf32", xf, xf)
	if err != nil {
		t.Fatalf("f32: %v", err)
	}
	if v, _ := of.Item(0); v.(float32) != 4 {
		t.Fatalf("f32 结果 = %v", v)
	}

	// 未注册 T(string)在 typed 注册口应 panic(程序员错误探针)
	func() {
		defer func() {
			if recover() == nil {
				t.Fatalf("string typed 注册应 panic(内部分支)")
			}
		}()
		registerUnaryTyped[string]("_test_bad", func(dst, src []string) {})
	}()
}

// 回归:size-0 操作数窗为空的非 nil 切片,Add/Less 不再 panic(形状保持)。
func TestDispatchZeroSize(t *testing.T) {
	z, _ := New([]int{0}, DFloat64)
	out, err := Add(z, z)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{0}) || out.Size() != 0 {
		t.Fatalf("Add size-0: shape %v size %d", out.Shape(), out.Size())
	}
	out, err = Less(z, z)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{0}) || out.Dtype() != DBool {
		t.Fatalf("Less size-0: shape %v dtype %s", out.Shape(), out.Dtype())
	}
}

// —— 广播标量快速路径(0-stride 逐元素展开)——

// sameVal 逐元素相等判定:NaN 按"同为 NaN"比对(fmod 的 UCRT 静态 NaN 位型与 Go
// math.Mod 的 NaN payload 不同,),其余逐位相等。
func sameVal(got, want any) bool {
	gf, ok1 := got.(float64)
	wf, ok2 := want.(float64)
	if ok1 && ok2 && math.IsNaN(gf) && math.IsNaN(wf) {
		return true
	}
	return got == want
}

// TestScalarFastPathParity 7 算子 × (右标量/左标量) × f8/i64:期望值纯 Go 手算,
// 并与"把标量复制成整数组走全量路径"的结果逐位对照(快速路径不得改变值语义)。
func TestScalarFastPathParity(t *testing.T) {
	fv := []float64{1.5, -2.25, 0, 7, -8.5, 3}
	iv := []int64{7, -8, 0, 5, -3, 12}

	ops := []struct {
		name  string
		entry func(x, y *Array) (*Array, error)
		f8    func(x, y float64) any
		i64   func(x, y int64) (any, error)
	}{
		{"add", Add,
			func(x, y float64) any { return x + y },
			func(x, y int64) (any, error) { return x + y, nil }},
		{"subtract", Subtract,
			func(x, y float64) any { return x - y },
			func(x, y int64) (any, error) { return x - y, nil }},
		{"multiply", Multiply,
			func(x, y float64) any { return x * y },
			func(x, y int64) (any, error) { return x * y, nil }},
		{"divide", Div, // 含整型 → f8,故 i64 参考按 float64 除
			func(x, y float64) any { return x / y },
			func(x, y int64) (any, error) { return float64(x) / float64(y), nil }},
		{"floor_divide", FloorDivide,
			func(x, y float64) any {
				if y == 0 {
					return x / y
				}
				return math.Floor(x / y)
			},
			func(x, y int64) (any, error) {
				if y == 0 {
					return int64(0), nil
				}
				q := x / y
				if r := x % y; r != 0 && (r < 0) != (y < 0) {
					q--
				}
				return q, nil
			}},
		{"remainder", Remainder,
			func(x, y float64) any {
				m := math.Mod(x, y)
				if m != 0 {
					if (y < 0) != (m < 0) {
						m += y
					}
				} else {
					m = math.Copysign(0, y)
				}
				return m
			},
			func(x, y int64) (any, error) {
				if y == 0 {
					return int64(0), nil
				}
				r := x % y
				if r != 0 && (r < 0) != (y < 0) {
					r += y
				}
				return r, nil
			}},
		{"power", Power,
			func(x, y float64) any { return math.Pow(x, y) },
			func(x, y int64) (any, error) {
				if y < 0 {
					return nil, errPowNegExp
				}
				r := int64(1)
				for k := int64(0); k < y; k++ {
					r *= x
				}
				return r, nil
			}},
	}

	for _, op := range ops {
		for _, dir := range []string{"R", "L"} { // R = 右操作数是标量,L = 左操作数是标量
			// —— f8(0-d 标量 = 原 bench 形态)——
			xa, err := FromSlice(fv)
			if err != nil {
				t.Fatal(err)
			}
			sc, err := Full([]int{}, 2.0, DFloat64)
			if err != nil {
				t.Fatal(err)
			}
			x, y := xa, sc
			if dir == "L" {
				x, y = sc, xa
			}
			out, err := op.entry(x, y)
			if err != nil {
				t.Fatalf("%s %s f8: %v", op.name, dir, err)
			}
			bc, err := Full([]int{len(fv)}, 2.0, DFloat64) // 非标量副本 → 必走全量路径
			if err != nil {
				t.Fatal(err)
			}
			bx, by := xa, bc
			if dir == "L" {
				bx, by = bc, xa
			}
			full, err := op.entry(bx, by)
			if err != nil {
				t.Fatalf("%s %s f8 全量对照: %v", op.name, dir, err)
			}
			for i, a := range fv {
				rx, ry := a, 2.0
				if dir == "L" {
					rx, ry = 2.0, a
				}
				want := op.f8(rx, ry)
				got := getI(t, out, i)
				if !sameVal(got, want) {
					t.Fatalf("%s %s f8 [%d] = %v want %v", op.name, dir, i, got, want)
				}
				if g2 := getI(t, full, i); !sameVal(g2, got) {
					t.Fatalf("%s %s f8 [%d] 快速=%v 全量=%v", op.name, dir, i, got, g2)
				}
			}

			// —— i64 ——
			xaI, err := FromSlice(iv)
			if err != nil {
				t.Fatal(err)
			}
			scI, err := Full([]int{}, int64(7), DInt64)
			if err != nil {
				t.Fatal(err)
			}
			xI, yI := xaI, scI
			if dir == "L" {
				xI, yI = scI, xaI
			}
			var rerr error
			want := make([]any, len(iv))
			for i, a := range iv {
				rx, ry := a, int64(7)
				if dir == "L" {
					rx, ry = int64(7), a
				}
				v, e := op.i64(rx, ry)
				if e != nil {
					rerr = e
					break
				}
				want[i] = v
			}
			outI, err := op.entry(xI, yI)
			if rerr != nil { // power 整型负指数:快速路径错误链与全量路径同口径
				if err == nil {
					t.Fatalf("%s %s i64: 期望错误 %v,得 nil", op.name, dir, rerr)
				}
				if !errors.Is(err, rerr) {
					t.Fatalf("%s %s i64: 错误链 %v 不含 %v", op.name, dir, err, rerr)
				}
				if rootOf(err) != ErrDtype {
					t.Fatalf("%s %s i64: 错误包应为 ErrDtype:%v", op.name, dir, err)
				}
				continue
			}
			if err != nil {
				t.Fatalf("%s %s i64: %v", op.name, dir, err)
			}
			for i := range iv {
				if got := getI(t, outI, i); got != want[i] {
					t.Fatalf("%s %s i64 [%d] = %v want %v", op.name, dir, i, got, want[i])
				}
			}
			// 方向敏感直查(左标量 7/x 与 x/7 必须不同):
			if op.name == "subtract" {
				if dir == "R" && getI(t, outI, 1) != int64(-15) { // -8 - 7
					t.Fatalf("subtract R i64[1] = %v want -15", getI(t, outI, 1))
				}
				if dir == "L" && getI(t, outI, 1) != int64(15) { // 7 - (-8)
					t.Fatalf("subtract L i64[1] = %v want 15", getI(t, outI, 1))
				}
			}
		}
	}

	// 跨 dtype:i4 数组 + f8 标量 → promote f8(标量侧仅 1 元素转码)。
	xa, err := FromSlice([]int32{1, -2, 3})
	if err != nil {
		t.Fatal(err)
	}
	sc, err := Full([]int{}, 2.5, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	out, err := Add(xa, sc)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("跨 dtype 输出应为 float64,得 %s", out.Dtype())
	}
	for i, v := range []float64{3.5, 0.5, 5.5} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("跨 dtype [%d] = %v want %v", i, got, v)
		}
	}
	// 反向(左标量):2.5 - i4。
	out, err = Subtract(sc, xa)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float64{1.5, 4.5, -0.5} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("跨 dtype 左标量 [%d] = %v want %v", i, got, v)
		}
	}
	// shape-(1,) 操作数(非 0-d)同样触发快速路径。
	one, err := FromSlice([]float64{2.0})
	if err != nil {
		t.Fatal(err)
	}
	out, err = Subtract(xa, one)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float64{-1, -4, 1} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("shape-(1,) [%d] = %v want %v", i, got, v)
		}
	}
	// f16( 策略 f16→f32→f16):快速路径与全量内核逐位一致(取 f16 可精确表示的值)。
	fa, err := Full([]int{3}, 1.0, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float64{1.0, 2.0, 4.0} {
		if err := fa.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	fs, err := Full([]int{}, 0.5, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Add(fa, fs)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat16 {
		t.Fatalf("f16 输出应为 f16,得 %s", out.Dtype())
	}
	for i, v := range []float32{1.5, 2.5, 4.5} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("f16 右标量 [%d] = %v want %v", i, got, v)
		}
	}
	out, err = Subtract(fs, fa) // 左标量:0.5 − a
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float32{-0.5, -1.5, -3.5} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("f16 左标量 [%d] = %v want %v", i, got, v)
		}
	}
}

// TestScalarFastPathTriggered 探针:0-d 与 (1,) 标量触发快速路径;双标量(outSize≤1)、
// 非标量、size-0 一律回退(计数不增长)。
func TestScalarFastPathTriggered(t *testing.T) {
	x, err := FromSlice([]float64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	sc, err := Full([]int{}, 10.0, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	one, err := FromSlice([]float64{10.0}) // shape (1,)
	if err != nil {
		t.Fatal(err)
	}

	run := func(t *testing.T, x, y *Array, want []float64, hits int) {
		t.Helper()
		scalarFastHits = 0
		out, err := dispatchBinary("_test_scalar", x, y)
		if err != nil {
			t.Fatal(err)
		}
		if scalarFastHits != hits {
			t.Fatalf("快速路径命中 %d 期望 %d", scalarFastHits, hits)
		}
		for i, v := range want {
			if got := getI(t, out, i); got != v {
				t.Fatalf("[%d] = %v want %v", i, got, v)
			}
		}
	}

	run(t, x, sc, []float64{11, 12, 13, 14}, 1)  // 右标量
	run(t, sc, x, []float64{11, 12, 13, 14}, 1)  // 左标量(加法可交换,探针证明方向判定)
	run(t, x, one, []float64{11, 12, 13, 14}, 1) // shape-(1,)
	run(t, sc, one, []float64{20}, 0)            // 双标量 → outSize==1 → 回退
	run(t, x, one, []float64{11, 12, 13, 14}, 1)
	run(t, x, sc, []float64{11, 12, 13, 14}, 1)

	// 非标量两侧 → 回退(既有路径逐字不变)。
	y, err := FromSlice([]float64{5, 6, 7, 8})
	if err != nil {
		t.Fatal(err)
	}
	run(t, x, y, []float64{6, 8, 10, 12}, 0)

	// size-0 → 回退( 口径不变)。
	z, err := New([]int{0}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	run(t, z, sc, nil, 0)
}

// TestScalarFallbackUnchanged cmp 系与未注册 kind 走回退:值与错误口径不变。
func TestScalarFallbackUnchanged(t *testing.T) {
	x, err := FromSlice([]float64{1, 3, 5})
	if err != nil {
		t.Fatal(err)
	}
	sc, err := Full([]int{}, 4.0, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	// cmp 系无 Scalar 内核 → 全量路径。
	scalarFastHits = 0
	out, err := Less(sc, x)
	if err != nil {
		t.Fatal(err)
	}
	if scalarFastHits != 0 {
		t.Fatalf("cmp 不应命中标量内核,得 %d", scalarFastHits)
	}
	if out.Dtype() != DBool {
		t.Fatalf("cmp 输出应为 bool,得 %s", out.Dtype())
	}
	for i, v := range []bool{false, false, true} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("Less(标量,x) [%d] = %v want %v", i, got, v)
		}
	}
	out, err = Less(x, sc)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []bool{true, true, false} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("Less(x,标量) [%d] = %v want %v", i, got, v)
		}
	}

	// bool×bool 特种算子(add_bool)未注册 Scalar → 回退,OR 口径不变。
	ba, err := FromSlice([]bool{true, false, true})
	if err != nil {
		t.Fatal(err)
	}
	bs, err := Full([]int{}, true, DBool)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Add(ba, bs)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("bool×bool 输出应为 bool,得 %s", out.Dtype())
	}
	for i, v := range []bool{true, true, true} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("bool OR [%d] = %v want %v", i, got, v)
		}
	}

	// 未注册 Scalar kind(只有全量内核)→ 回退:值不变。
	registerBinaryTyped[int64]("_test_noscalar", func(dst, a, b []int64) {
		for i := range dst {
			dst[i] = a[i] - b[i]
		}
	})
	ia, err := FromSlice([]int64{10, -3, 0})
	if err != nil {
		t.Fatal(err)
	}
	is, err := Full([]int{}, int64(4), DInt64)
	if err != nil {
		t.Fatal(err)
	}
	scalarFastHits = 0
	out, err = dispatchBinary("_test_noscalar", ia, is)
	if err != nil {
		t.Fatal(err)
	}
	if scalarFastHits != 0 {
		t.Fatalf("未注册 kind 不应命中标量内核,得 %d", scalarFastHits)
	}
	for i, v := range []int64{6, -7, -4} {
		if got := getI(t, out, i); got != v {
			t.Fatalf("回退 [%d] = %v want %v", i, got, v)
		}
	}

	// 错误口径不变:形状不相容仍 ErrShape;负指数错误链快速/全量同构。
	if _, err := dispatchBinary("_test_add", ia, ia); err != nil {
		t.Fatalf("同形状不应报错: %v", err)
	}
	y3, err := FromSlice([]int64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	_, err = dispatchBinary("_test_add", ia, y3)
	if err == nil {
		t.Error("形状不相容应报错")
	} else if rootOf(err) != ErrShape {
		t.Errorf("错误包应为 ErrShape:%v", err)
	}
	neg, err := Full([]int{}, int64(-1), DInt64)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Power(ia, neg) // 快速路径
	if err == nil {
		t.Fatal("整型负指数应报错(快速路径)")
	} else if !errors.Is(err, errPowNegExp) || rootOf(err) != ErrDtype {
		t.Errorf("快速路径错误链:%v", err)
	}
	bc, err := Full([]int{3}, int64(-1), DInt64) // 非标量副本 → 全量路径
	if err != nil {
		t.Fatal(err)
	}
	_, err = Power(ia, bc)
	if err == nil {
		t.Fatal("整型负指数应报错(全量路径)")
	} else if !errors.Is(err, errPowNegExp) || rootOf(err) != ErrDtype {
		t.Errorf("全量路径错误链:%v", err)
	}
}

// TestScalarBroadcastBigArray 广播大数组(两侧都非标量)仍走原路径:结果与既有
// 全量口径一致(xcheck 165 案兜底),且探针证明未走快速路径。
func TestScalarBroadcastBigArray(t *testing.T) {
	x, err := FromSlice([]float64{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	x, err = x.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	y, err := FromSlice([]float64{10, 20, 30})
	if err != nil {
		t.Fatal(err)
	}
	scalarFastHits = 0
	out, err := Add(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if scalarFastHits != 0 {
		t.Fatalf("广播大数组不应命中标量内核,得 %d", scalarFastHits)
	}
	if !shapeEqual(out.Shape(), []int{2, 3}) {
		t.Fatalf("shape %v", out.Shape())
	}
	want := [][]float64{{11, 22, 33}, {14, 25, 36}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if got := getI(t, out, i, j); got != want[i][j] {
				t.Fatalf("[%d,%d] = %v want %v", i, j, got, want[i][j])
			}
		}
	}
}

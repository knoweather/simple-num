package num

// 单测(B):聚合 9 算子与 numpy 逐位/相对对拍。
// 证据来源:xcheck/reference/cases/agg_*.npy(gen_reference.py + numpy 2.5.4)。
// 覆盖:dtype 规则、axis(单/多/负/keepdims)、多轴展平 pairwise、NaN 口径、
// 0-size、错误路径。

import (
	"bytes"
	"math"
	"testing"
)

func aggEntry(op string) func(*Array, []int, bool) (*Array, error) {
	switch op {
	case "sum":
		return Sum
	case "mean":
		return Mean
	case "var":
		return Var
	case "std":
		return Std
	case "min":
		return Min
	case "max":
		return Max
	case "argmin":
		return ArgMin
	case "argmax":
		return ArgMax
	case "any":
		return AnyTrue
	case "all":
		return AllTrue
	}
	return nil
}

// relArr 相对阈值逐元素比较(f32/f64;支持 0-d)。
func relArr(t *testing.T, got, ex *Array, tol float64) error {
	t.Helper()
	if got.Size() != ex.Size() {
		t.Fatalf("size %d vs %d", got.Size(), ex.Size())
	}
	for i := 0; i < got.Size(); i++ {
		var gv, ev any
		var err error
		if got.NDim() == 0 {
			if gv, err = got.Item(); err != nil {
				return err
			}
			if ev, err = ex.Item(); err != nil {
				return err
			}
		} else {
			if gv, err = got.Item(i); err != nil {
				return err
			}
			if ev, err = ex.Item(i); err != nil {
				return err
			}
		}
		var g, e float64
		switch v := gv.(type) {
		case float64:
			g = v
			e, _ = ev.(float64)
		case float32:
			g = float64(v)
			if e32, ok := ev.(float32); ok {
				e = float64(e32)
			}
		default:
			t.Fatalf("rel 区出现非浮点 %T", gv)
		}
		if math.IsNaN(e) {
			if !math.IsNaN(g) {
				t.Fatalf("[%d] = %v, want NaN", i, g)
			}
			continue
		}
		if d := math.Abs(g - e); d > tol*math.Max(1, math.Abs(e)) {
			t.Fatalf("[%d] = %v, want %v(rel %.2e > %.2e)", i, g, e, d/math.Max(1, math.Abs(e)), tol)
		}
	}
	return nil
}

// TestAggRef 逐案对拍。
func TestAggRef(t *testing.T) {
	cases := []struct {
		name string
		op   string
		axis []int
		keep bool
		rel  bool
		tol  float64
	}{
		{"agg_sum_i8_axis0", "sum", []int{0}, false, false, 0},
		{"agg_sum_i8_axis1", "sum", []int{1}, false, false, 0},
		{"agg_sum_bool", "sum", nil, false, false, 0},
		{"agg_sum_u4", "sum", nil, false, false, 0},
		{"agg_sum_f8_axis1", "sum", []int{1}, false, false, 0},
		{"agg_sum_f8_axes01", "sum", []int{0, 1}, false, false, 0},
		{"agg_sum_f8_keepdims", "sum", []int{1}, true, false, 0},
		{"agg_sum_f8_negaxis", "sum", []int{-1}, false, false, 0},
		{"agg_sum_f4_axis0", "sum", []int{0}, false, false, 0},
		{"agg_sum_f8_negzero_axis", "sum", []int{1}, false, false, 0},
		{"agg_mean_f8_axis1", "mean", []int{1}, false, false, 0},
		{"agg_mean_f4_axis0", "mean", []int{0}, false, false, 0},
		{"agg_mean_i8", "mean", nil, false, false, 0},
		{"agg_var_f8_axis1", "var", []int{1}, false, true, 1e-15},
		{"agg_std_f8", "std", nil, false, true, 1e-15},
		{"agg_var_f4", "var", nil, false, true, 1e-6},
		{"agg_mean_f16", "mean", nil, false, false, 0},
		{"agg_var_f16", "var", nil, false, true, 1e-2},
		{"agg_std_f16", "std", nil, false, true, 1e-2},
		{"agg_min_f16", "min", nil, false, false, 0},
		{"agg_max_f16", "max", nil, false, false, 0},
		{"agg_argmin_f16", "argmin", nil, false, false, 0},
		{"agg_argmax_f16", "argmax", nil, false, false, 0},
		{"agg_any_f16", "any", nil, false, false, 0},
		{"agg_all_f16", "all", nil, false, false, 0},
		{"agg_min_f16_nan", "min", nil, false, false, 0},
		{"agg_argmin_f16_nan", "argmin", nil, false, false, 0},
		{"agg_min_f8_nan", "min", nil, false, false, 0},
		{"agg_max_i8_axis1", "max", []int{1}, false, false, 0},
		{"agg_min_bool", "min", nil, false, false, 0},
		{"agg_argmin_f8_nan", "argmin", nil, false, false, 0},
		{"agg_argmax_i8_axis1", "argmax", []int{1}, false, false, 0},
		{"agg_any_f8_nan", "any", nil, false, false, 0},
		{"agg_all_i8", "all", nil, false, false, 0},
		{"agg_all_f8_axis1", "all", []int{1}, false, false, 0},
	}
	for _, c := range cases {
		fn := aggEntry(c.op)
		if fn == nil {
			t.Fatalf("%s: 未知 op %q", c.name, c.op)
		}
		in, err := LoadNPY("xcheck/reference/cases/" + c.name + "_a.npy")
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		ex, err := LoadNPY("xcheck/reference/cases/" + c.name + "_out.npy")
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got, err := fn(in, c.axis, c.keep)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got.Dtype() != ex.Dtype() {
			t.Fatalf("%s: dtype = %s, want %s", c.name, got.Dtype(), ex.Dtype())
		}
		if c.rel {
			if err := relArr(t, got, ex, c.tol); err != nil {
				t.Fatalf("%s: rel 失败: %v", c.name, err)
			}
			continue
		}
		gb, e1 := SaveNPYBytes(got)
		eb, e2 := SaveNPYBytes(ex)
		if e1 != nil || e2 != nil {
			t.Fatalf("%s: save: %v %v", c.name, e1, e2)
		}
		if !bytes.Equal(gb, eb) {
			t.Fatalf("%s: bitwise 失败(形状 %v vs %v)", c.name, got.Shape(), ex.Shape())
		}
	}
}

// dtype 规则。
func TestAggDtypeRules(t *testing.T) {
	i8 := bArr(t, []int8{1, 2, 3})
	checks := []struct {
		got  *Array
		want Dtype
		op   string
	}{
		{mustAgg(t, Sum, i8), DInt64, "sum(i8)"},
		{mustAgg(t, Mean, i8), DFloat64, "mean(i8)"},
		{mustAgg(t, Var, i8), DFloat64, "var(i8)"},
		{mustAgg(t, Min, i8), DInt8, "min(i8)"},
		{mustAgg(t, ArgMax, i8), DInt64, "argmax(i8)"},
		{mustAgg(t, AnyTrue, i8), DBool, "any(i8)"},
	}
	for _, c := range checks {
		if c.got.Dtype() != c.want {
			t.Fatalf("%s dtype = %s, want %s", c.op, c.got.Dtype(), c.want)
		}
	}
	f4 := bArr(t, []float32{1, 2})
	if got := mustAgg(t, Sum, f4); got.Dtype() != DFloat32 {
		t.Fatalf("sum(f4) dtype = %s, want float32", got.Dtype())
	}
	if got := mustAgg(t, Mean, f4); got.Dtype() != DFloat32 {
		t.Fatalf("mean(f4) dtype = %s, want float32", got.Dtype())
	}
}

func mustAgg(t *testing.T, fn func(*Array, []int, bool) (*Array, error), a *Array) *Array {
	t.Helper()
	out, err := fn(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// 0-size 归约维。
func TestAggZeroSize(t *testing.T) {
	empty := bArr(t, []float64{})
	if got := mustAgg(t, Sum, empty); got.NDim() != 0 || getI(t, got) != 0.0 {
		t.Fatalf("sum(空) = %v", got)
	}
	m, err := Mean(empty, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := getI(t, m).(float64); !math.IsNaN(v) {
		t.Fatalf("mean(空) = %v, want NaN", v)
	}
	if _, err := Min(empty, nil, false); rootOf(err) != ErrShape {
		t.Fatalf("min(空) err = %v, want ErrShape", err)
	}
	if _, err := ArgMax(empty, nil, false); rootOf(err) != ErrShape {
		t.Fatalf("argmax(空) err = %v, want ErrShape", err)
	}
	if got := mustAgg(t, AnyTrue, empty); getI(t, got) != false {
		t.Fatalf("any(空) = %v, want false", getI(t, got))
	}
	if got := mustAgg(t, AllTrue, empty); getI(t, got) != true {
		t.Fatalf("all(空) = %v, want true", getI(t, got))
	}

	// (2,0) axis=1:sum → [0,0];min → ErrShape
	z, err := New([]int{2, 0}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Sum(z, []int{1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(s.Shape(), []int{2}) || getI(t, s, 0) != 0.0 || getI(t, s, 1) != 0.0 {
		t.Fatalf("sum((2,0),axis=1) = %v %v", s.Shape(), getI(t, s, 0))
	}
	if _, err := Min(z, []int{1}, false); rootOf(err) != ErrShape {
		t.Fatalf("min((2,0),axis=1) err = %v, want ErrShape", err)
	}
}

// axis 错误路径。
func TestAggAxisErrors(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	x, _ := a.Reshape([]int{2, 2})
	if _, err := Sum(x, []int{2}, false); rootOf(err) != ErrIndex {
		t.Fatalf("axis 越界 err = %v, want ErrIndex", err)
	}
	if _, err := Sum(x, []int{-3}, false); rootOf(err) != ErrIndex {
		t.Fatalf("axis 负越界 err = %v, want ErrIndex", err)
	}
	if _, err := Sum(x, []int{0, 0}, false); rootOf(err) != ErrShape {
		t.Fatalf("axis 重复 err = %v, want ErrShape", err)
	}
}

// keepdims 形状与多轴展平语义。
func TestAggShapes(t *testing.T) {
	a, _ := Arange(0, 24, 1, DFloat64)
	x, _ := a.Reshape([]int{2, 3, 4})
	out, err := Sum(x, []int{1}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 1, 4}) {
		t.Fatalf("keepdims shape = %v, want [2 1 4]", out.Shape())
	}
	// 多轴展平:[[1e16,-1e16],[1,1]] 全归约 = 2.0(展平 pairwise),逐轴会得 1.0
	m := bArr(t, []float64{1e16, -1e16, 1, 1})
	y, _ := m.Reshape([]int{2, 2})
	s, err := Sum(y, []int{0, 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := getI(t, s).(float64); v != 2.0 {
		t.Fatalf("多轴展平 sum = %v, want 2.0(逐轴会得 1.0)", v)
	}
}

// 0-d 输入。
func TestAgg0d(t *testing.T) {
	v := bArr(t, []float64{7.5})
	z, err := v.Reshape([]int{})
	if err != nil {
		t.Fatal(err)
	}
	s, err := Sum(z, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if s.NDim() != 0 || getI(t, s).(float64) != 7.5 {
		t.Fatalf("sum(0-d) = %v", s)
	}
	if _, err := Sum(z, []int{0}, false); rootOf(err) != ErrIndex {
		t.Fatalf("0-d 带 axis err = %v, want ErrIndex", err)
	}
}

// 全 -0 的轴归约:+0.0 恒等元。
func TestAggNegZeroIdentity(t *testing.T) {
	a := bArr(t, []float64{math.Copysign(0, -1), math.Copysign(0, -1)})
	s, err := Sum(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if v := getI(t, s).(float64); math.Signbit(v) {
		t.Fatalf("sum(全 -0) = %v, want +0.0", v)
	}
}

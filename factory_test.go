package num

import (
	"errors"
	"math"
	"testing"
)

// 工厂单测。期望值全部来自 numpy 2.5.4 实测(见;
// 生成脚本 gen_factory.py),逐位比较(f64 用 bits 比较,杜绝容差)。

// f64At 取 1 维 f64 数组元素(测试辅助)。
func f64At(t *testing.T, a *Array, i int) float64 {
	t.Helper()
	v, err := a.Item(i)
	if err != nil {
		t.Fatalf("Item(%d): %v", i, err)
	}
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("Item(%d) = %T(%v), want float64", i, v, v)
	}
	return f
}

func wantF64(t *testing.T, a *Array, want ...float64) {
	t.Helper()
	if a.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", a.Dtype())
	}
	if got := a.Size(); got != len(want) {
		t.Fatalf("size = %d, want %d (shape %v)", got, len(want), a.Shape())
	}
	for i, w := range want {
		g := f64At(t, a, i)
		if math.Float64bits(g) != math.Float64bits(w) {
			t.Fatalf("元素[%d] = %v (%#x), want %v (%#x)", i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}

func TestArangeFloat(t *testing.T) {
	// np.arange(5)
	a, err := Arange(0, 5, 1, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	wantF64(t, a, 0, 1, 2, 3, 4)

	// np.arange(0, 1, 0.1) —— 分离乘加,逐位与 numpy 同(末位 0.30000000000000004 等)
	a, err = Arange(0, 1, 0.1, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	wantF64(t, a,
		0.0, 0.1, 0.2, 0.30000000000000004, 0.4,
		0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9)

	// np.arange(5, 0, -2) → 5,3,1
	a, err = Arange(5, 0, -2, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	wantF64(t, a, 5, 3, 1)

	// np.arange(2.5, 7.5, 1.25) → 2.5,3.75,5,6.25
	a, err = Arange(2.5, 7.5, 1.25, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	wantF64(t, a, 2.5, 3.75, 5, 6.25)

	// np.arange(0, 1, 10) → [0](长度 1)
	a, err = Arange(0, 1, 10, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	wantF64(t, a, 0)
}

func TestArangeEmptyAndErrors(t *testing.T) {
	// np.arange(5, 1) → 空
	a, err := Arange(5, 1, 1, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	if a.Size() != 0 || len(a.Shape()) != 1 || a.Shape()[0] != 0 {
		t.Fatalf("空 arange: size=%d shape=%v", a.Size(), a.Shape())
	}

	// step=0 → ErrZeroDiv(numpy ZeroDivisionError 对应)
	if _, err := Arange(0, 1, 0, DFloat64); !errors.Is(err, ErrZeroDiv) {
		t.Fatalf("step=0 err = %v, want ErrZeroDiv", err)
	}
	// NaN 长度 → ErrShape
	if _, err := Arange(math.NaN(), 1, 1, DFloat64); !errors.Is(err, ErrShape) {
		t.Fatalf("NaN err = %v, want ErrShape", err)
	}
}

func TestArangeIntWrap(t *testing.T) {
	// np.arange(250, 260, 1, uint8) → 250..255,0..3(回绕)
	a, err := Arange(250, 260, 1, DUint8)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	want := []uint8{250, 251, 252, 253, 254, 255, 0, 1, 2, 3}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		v, _ := a.Item(i)
		if v.(uint8) != w {
			t.Fatalf("[%d] = %v, want %v", i, v, w)
		}
	}

	// np.arange(125, 130, 1, int8) → 125,126,127,-128,-127(有符号回绕)
	a, err = Arange(125, 130, 1, DInt8)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	wanti := []int8{125, 126, 127, -128, -127}
	for i, w := range wanti {
		v, _ := a.Item(i)
		if v.(int8) != w {
			t.Fatalf("int8 [%d] = %v, want %v", i, v, w)
		}
	}

	// 大整数精确路径:2^40 起点(超出 f64 逐位可控范围仅靠 int64 直算)
	const base = int64(1) << 40
	a, err = Arange(float64(base), float64(base+3), 1, DInt64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	for i := 0; i < 3; i++ {
		v, _ := a.Item(i)
		if v.(int64) != base+int64(i) {
			t.Fatalf("big int64 [%d] = %v, want %v", i, v, base+int64(i))
		}
	}
}

func TestLinSpaceNumpyExact(t *testing.T) {
	cases := []struct {
		start, stop float64
		n           int
		endpoint    bool
		want        []float64
	}{
		{0, 1, 5, true, []float64{0, 0.25, 0.5, 0.75, 1}},
		{0, 1, 1, true, []float64{0}}, // numpy: n=1 → [start](不置末点)
		{0, 1, 0, true, nil},          // empty
		{0, 1, 5, false, []float64{0, 0.2, 0.4, 0.6000000000000001, 0.8}},
		{0, 1, 4, false, []float64{0, 0.25, 0.5, 0.75}},
		{1, 2, 3, true, []float64{1, 1.5, 2}},
		{-1, 1, 3, true, []float64{-1, 0, 1}},
		{0, 10, 5, true, []float64{0, 2.5, 5, 7.5, 10}},
		// np.linspace(0, 0.3, 4) → i*step+start,step=0.3/3 舍入后逐位如下
		{0, 0.3, 4, true, []float64{0, 0.09999999999999999, 0.19999999999999998, 0.3}},
		{2, 2, 3, true, []float64{2, 2, 2}},
	}
	for _, c := range cases {
		a, err := LinSpace(c.start, c.stop, c.n, c.endpoint)
		if err != nil {
			t.Fatalf("LinSpace(%g,%g,%d,%v): %v", c.start, c.stop, c.n, c.endpoint, err)
		}
		wantF64(t, a, c.want...)
	}

	if _, err := LinSpace(0, 1, -1, true); !errors.Is(err, ErrShape) {
		t.Fatalf("LinSpace(-1) err = %v, want ErrShape", err)
	}
}

func TestEye(t *testing.T) {
	// np.eye(3)
	a, err := Eye(3, DFloat64)
	if err != nil {
		t.Fatalf("Eye: %v", err)
	}
	if !shapeEqual(a.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v", a.Shape())
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			v, _ := a.Item(i, j)
			w := 0.0
			if i == j {
				w = 1
			}
			if v.(float64) != w {
				t.Fatalf("eye[%d,%d] = %v, want %v", i, j, v, w)
			}
		}
	}

	// np.eye(3, dtype=int8)
	ai, err := Eye(3, DInt8)
	if err != nil {
		t.Fatalf("Eye int8: %v", err)
	}
	if v, _ := ai.Item(1, 1); v.(int8) != 1 {
		t.Fatalf("eye int8[1,1] = %v", v)
	}
	if v, _ := ai.Item(1, 2); v.(int8) != 0 {
		t.Fatalf("eye int8[1,2] = %v", v)
	}

	// np.eye(3, dtype=bool)
	ab, err := Eye(3, DBool)
	if err != nil {
		t.Fatalf("Eye bool: %v", err)
	}
	if v, _ := ab.Item(0, 0); v.(bool) != true {
		t.Fatalf("eye bool[0,0] = %v", v)
	}
	if v, _ := ab.Item(0, 1); v.(bool) != false {
		t.Fatalf("eye bool[0,1] = %v", v)
	}

	// eye(0) → 空(numpy: shape (0,0))
	a0, err := Eye(0, DFloat64)
	if err != nil {
		t.Fatalf("Eye(0): %v", err)
	}
	if a0.Size() != 0 || !shapeEqual(a0.Shape(), []int{0, 0}) {
		t.Fatalf("Eye(0): size=%d shape=%v", a0.Size(), a0.Shape())
	}
	if _, err := Eye(-1, DFloat64); !errors.Is(err, ErrShape) {
		t.Fatalf("Eye(-1) err = %v, want ErrShape", err)
	}
}

// TestArangeFloat32 numpy f4 逐位:先 f64 算再收敛到 f32(np.arange 同款双舍入风险已实测对齐)。
func TestArangeFloat32(t *testing.T) {
	a, err := Arange(0, 1, 0.1, DFloat32)
	if err != nil {
		t.Fatalf("Arange f4: %v", err)
	}
	want := []float32{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		v, _ := a.Item(i)
		if math.Float32bits(v.(float32)) != math.Float32bits(w) {
			t.Fatalf("f4[%d] = %v (%#x), want %v (%#x)", i, v, math.Float32bits(v.(float32)), w, math.Float32bits(w))
		}
	}
}

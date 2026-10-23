package num

// 单测(B):复数 c64/c128 四则 / abs / conj / 比较。
// 期望值 = numpy 2.5.4 实测探针字面量。

import (
	"math"
	"testing"
)

func cArr128(t *testing.T, vals ...complex128) *Array {
	t.Helper()
	a, err := FromSlice(vals)
	if err != nil {
		t.Fatal(err)
	}
	if a.Dtype() != DComplex128 {
		t.Fatalf("dtype = %s", a.Dtype())
	}
	return a
}

// bWantC128 逐分量位型比较(NaN 允许)。
func bWantC128(t *testing.T, a *Array, want ...complex128) {
	t.Helper()
	if a.Dtype() != DComplex128 {
		t.Fatalf("dtype = %s, want complex128", a.Dtype())
	}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		g := itemAt(t, a, i).(complex128)
		if !sameF64(real(g), real(w)) || !sameF64(imag(g), imag(w)) {
			t.Fatalf("[%d] = %v (%#x/%#x), want %v (%#x/%#x)", i, g,
				math.Float64bits(real(g)), math.Float64bits(imag(g)), w,
				math.Float64bits(real(w)), math.Float64bits(imag(w)))
		}
	}
}

func sameF64(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// 四则:add/sub/mul 逐位;div 缩放算法(与 numpy 同式)。
func TestComplexArith(t *testing.T) {
	a := cArr128(t, 1+2i, -3.5+0.5i, 0)
	b := cArr128(t, 3-1i, 0.5+2i, 0)

	out, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantC128(t, out, 4+1i, -3+2.5i, 0)

	out, _ = Subtract(a, b)
	bWantC128(t, out, -2+3i, -4-1.5i, 0)

	out, _ = Multiply(a, b)
	bWantC128(t, out, 5+5i, -2.75-6.75i, 0)

	out, _ = Div(a, b)
	// numpy:(0.1+0.7000000000000001j)、(-0.1764705882352941+1.7058823529411764j)
	bWantC128(t, out, complex(0.1, 0.7000000000000001),
		complex(-0.1764705882352941, 1.7058823529411764), complex(math.NaN(), math.NaN()))
}

// 除零:(1+1j)/0 → inf+infj;0/0 → nan+nanj。
func TestComplexDivEdges(t *testing.T) {
	a := cArr128(t, 1+1i, 0)
	z := cArr128(t, 0, 0)
	out, err := Div(a, z)
	if err != nil {
		t.Fatal(err)
	}
	bWantC128(t, out, complex(math.Inf(1), math.Inf(1)), complex(math.NaN(), math.NaN()))
}

// abs:c128 → f64 hypot;c64 → f32。
func TestComplexAbs(t *testing.T) {
	a := cArr128(t, 1+2i, -3.5+0.5i, 0)
	out, err := Abs(a)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s, want float64", out.Dtype())
	}
	bWantF64Rel(t, out, 0, 2.23606797749979, 3.5355339059327378, 0)

	// c64 → f32
	c64, err := Cast(a, DComplex64)
	if err != nil {
		t.Fatal(err)
	}
	out, err = Abs(c64)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DFloat32 {
		t.Fatalf("c64 abs dtype = %s, want float32", out.Dtype())
	}
	bWantF32Rel(t, out, DFloat32, 1e-6, 2.236068, 3.5355339, 0)
}

// conj:虚部取反。
func TestComplexConj(t *testing.T) {
	a := cArr128(t, 1+2i, -3.5+0.5i, 0)
	out, err := Conj(a)
	if err != nil {
		t.Fatal(err)
	}
	bWantC128(t, out, 1-2i, -3.5-0.5i, complex(0, math.Copysign(0, -1)))
}

// 比较:eq/ne 可用;有序比较按 numpy 拒;NaN 参与 eq 恒 False。
func TestComplexCmp(t *testing.T) {
	a := cArr128(t, 1+2i, 0)
	out, err := Equal(a, a)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, true)

	nan := cArr128(t, complex(math.NaN(), 0))
	out, err = Equal(nan, nan)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false)
	out, _ = NotEqual(nan, nan)
	bWantBool(t, out, true)

	if _, err := Less(a, a); rootOf(err) != ErrDtype {
		t.Fatalf("有序比较 err = %v, want ErrDtype", err)
	}
}

// 提升:promote 表(c64+i8 → c64、c64+f8 → c128)。
func TestComplexPromote(t *testing.T) {
	c64, err := Cast(cArr128(t, 1+1i), DComplex64)
	if err != nil {
		t.Fatal(err)
	}
	out, err := Add(c64, bArr(t, []int8{1}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DComplex64 {
		t.Fatalf("c64+i8 dtype = %s, want complex64", out.Dtype())
	}
	out, err = Add(c64, bArr(t, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DComplex128 {
		t.Fatalf("c64+f8 dtype = %s, want complex128", out.Dtype())
	}
	bWantC128(t, out, 2+1i)
}

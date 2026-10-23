package num

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"
)

// detMat 构造与 numpy 参照一致的确定性矩阵:a[i][j] = (i*31+j*17) % 13 - 6。
func detMat(t *testing.T, n int) *Array {
	t.Helper()
	vals := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			vals[i*n+j] = float64((i*31+j*17)%13 - 6)
		}
	}
	a, err := FromSlice(vals)
	if err != nil {
		t.Fatal(err)
	}
	m, err := a.Reshape([]int{n, n})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// complexValues 取数组的扁平复数值。
func complexValues(t *testing.T, a *Array) []complex128 {
	t.Helper()
	f := a.AsContiguous()
	if f.Dtype() != DComplex128 {
		t.Fatalf("期望 complex128,得 %s", f.Dtype())
	}
	out := make([]complex128, f.Size())
	copy(out, win[complex128](engineWindow(f)))
	return out
}

// sortByRealImag 把谱排序(特征值顺序不作保证,比较时必须先排序)。
func sortByRealImag(w []complex128) {
	sort.Slice(w, func(i, j int) bool {
		if real(w[i]) != real(w[j]) {
			return real(w[i]) < real(w[j])
		}
		return imag(w[i]) < imag(w[j])
	})
}

// matchSpectrum 按"最近匹配"比较两组特征值,与顺序无关(谱不保证顺序)。
func matchSpectrum(t *testing.T, got, want []complex128, tol float64) {
	t.Helper()
	used := make([]bool, len(got))
	for _, w := range want {
		found := -1
		for i, g := range got {
			if used[i] {
				continue
			}
			if cmplx.Abs(g-w) <= tol {
				found = i
				break
			}
		}
		if found < 0 {
			t.Fatalf("谱里找不到 %v(容差 %g),实际 %v", w, tol, got)
		}
		used[found] = true
	}
}

// Eigvals:与 numpy 的确定性参照做谱匹配(顺序无关)。
func TestEigvalsAgainstNumpy(t *testing.T) {
	cases := []struct {
		n    int
		want []complex128
	}{
		{2, []complex128{-6.216991, 3.216991}},
		{3, []complex128{-7.105793, -4.392048, 7.497841}},
		{4, []complex128{complex(-7.004064, -4.680931), complex(-7.004064, 4.680931), -2.980814, 7.988941}},
		{5, []complex128{
			complex(-7.079722, -4.701245), complex(-7.079722, 4.701245),
			complex(-1.533786, -1.619639), complex(-1.533786, 1.619639), 12.227017}},
		{6, []complex128{
			complex(-8.156975, -4.700809), complex(-8.156975, 4.700809),
			complex(-2.542111, -2.200572), complex(-2.542111, 2.200572),
			5.002751, 11.39542}},
	}
	for _, c := range cases {
		got, err := Eigvals(detMat(t, c.n))
		if err != nil {
			t.Fatalf("n=%d: %v", c.n, err)
		}
		gw := complexValues(t, got)
		matchSpectrum(t, gw, c.want, 1e-4)
	}
	// 三角、缺陷、伴随矩阵
	extra := []struct {
		name string
		m    *Array
		want []complex128
	}{
		{"tri3", mustMat(t, []int{3, 3}, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6}), []complex128{1, 4, 6}},
		{"defective", mustMat(t, []int{2, 2}, []float64{2, 1, 0, 2}), []complex128{2, 2}},
		{"companion", mustMat(t, []int{3, 3}, []float64{0, 0, 6, 1, 0, -11, 0, 1, 6}),
			[]complex128{1, 2, 3}},
	}
	for _, c := range extra {
		got, err := Eigvals(c.m)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		gw := complexValues(t, got)
		matchSpectrum(t, gw, c.want, 1e-6)
	}
}

// Eig:特征向量满足 A·v = λ·v 且列单位长。
func TestEigResidual(t *testing.T) {
	for _, n := range []int{2, 3, 4, 5, 6} {
		a := detMat(t, n)
		out, err := Eig(a)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		w, v := out[0], out[1]
		if !shapeEqual(w.Shape(), []int{n}) {
			t.Fatalf("n=%d 特征值形状 %v", n, w.Shape())
		}
		if !shapeEqual(v.Shape(), []int{n, n}) {
			t.Fatalf("n=%d 特征向量形状 %v", n, v.Shape())
		}
		lam := complexValues(t, w)
		vec := complexValues(t, v)
		av := win[float64](engineWindow(a.AsContiguous()))
		scale := 0.0
		for _, z := range av {
			scale += math.Abs(z)
		}
		for j := 0; j < n; j++ {
			// 列单位长
			nrm := 0.0
			for i := 0; i < n; i++ {
				nrm += cmplx.Abs(vec[i*n+j]) * cmplx.Abs(vec[i*n+j])
			}
			if math.Abs(math.Sqrt(nrm)-1) > 1e-9 {
				t.Fatalf("n=%d 第 %d 列范数 = %v", n, j, math.Sqrt(nrm))
			}
			// 残差 ||A v - λ v||
			res := 0.0
			for i := 0; i < n; i++ {
				var s complex128
				for k := 0; k < n; k++ {
					s += complex(av[i*n+k], 0) * vec[k*n+j]
				}
				s -= lam[j] * vec[i*n+j]
				res += cmplx.Abs(s) * cmplx.Abs(s)
			}
			if math.Sqrt(res) > 1e-6*math.Max(1, scale) {
				t.Fatalf("n=%d 第 %d 列残差 %.3e(λ=%v)", n, j, math.Sqrt(res), lam[j])
			}
		}
	}
}

// Eig 与 Eigh 在对称矩阵上应一致(交叉验证另一条实现)。
func TestEigMatchesEighOnSymmetric(t *testing.T) {
	sym := mustMat(t, []int{3, 3}, []float64{2, 1, 0, 1, 3, 1, 0, 1, 4})
	w, err := Eigvals(sym)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := Eigvalsh(sym)
	if err != nil {
		t.Fatal(err)
	}
	gw := complexValues(t, w)
	sortByRealImag(gw)
	rw := win[float64](engineWindow(ref.AsContiguous()))
	rwSorted := append([]float64(nil), rw...)
	sort.Float64s(rwSorted)
	for i := range rwSorted {
		if math.Abs(real(gw[i])-rwSorted[i]) > 1e-9 {
			t.Fatalf("[%d] eigvals=%v eigvalsh=%v", i, gw[i], rwSorted[i])
		}
		if math.Abs(imag(gw[i])) > 1e-12 {
			t.Fatalf("[%d] 对称矩阵出现虚部 %v", i, gw[i])
		}
	}
}

// 边角与错误路径。
func TestEigEdges(t *testing.T) {
	// 1×1
	one := mustMat(t, []int{1, 1}, []float64{7})
	w, err := Eigvals(one)
	if err != nil {
		t.Fatal(err)
	}
	if v := complexValues(t, w); len(v) != 1 || v[0] != 7 {
		t.Fatalf("1×1 = %v", v)
	}
	if out, err := Eig(one); err != nil || len(out) != 2 {
		t.Fatalf("1×1 eig: %v %v", out, err)
	}
	// 非方阵 / 1-D / nil
	if _, err := Eigvals(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
	if _, err := Eigvals(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("1-D 应报错")
	}
	if _, err := Eigvals(nil); err == nil {
		t.Fatal("nil 应报错")
	}
	if _, err := Eig(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("eig 非方阵应报错")
	}
	if _, err := Eig(bArr(t, []float64{1})); err == nil {
		t.Fatal("eig 1-D 应报错")
	}
	if _, err := Eig(nil); err == nil {
		t.Fatal("eig nil 应报错")
	}
	// 复数矩阵输入(应支持)
	c, _ := FromSlice([]complex128{1 + 1i, 0, 0, 2 - 1i})
	c2, _ := c.Reshape([]int{2, 2})
	cw, err := Eigvals(c2)
	if err != nil {
		t.Fatal(err)
	}
	got := complexValues(t, cw)
	matchSpectrum(t, got, []complex128{1 + 1i, 2 - 1i}, 1e-9)
}

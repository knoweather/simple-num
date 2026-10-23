package num

import (
	"math"
	"testing"
)

// 线代核心:det / inv / solve / norm(期望值取自 numpy 2.5.4)。
func TestLinalgCore(t *testing.T) {
	a := mustMat(t, []int{3, 3}, []float64{4, 3, 2, 1, 5, 7, 2, 8, 6})
	det, err := Det(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(det-(-84)) > 1e-9 {
		t.Fatalf("det = %v, want -84", det)
	}
	inv, err := Inv(a)
	if err != nil {
		t.Fatal(err)
	}
	wantInv := []float64{
		0.30952380952380953, 0.023809523809523815, -0.13095238095238096,
		-0.09523809523809525, -0.2380952380952381, 0.30952380952380953,
		0.02380952380952381, 0.30952380952380953, -0.20238095238095238,
	}
	iw := inv.AsContiguous().Flatten()
	for i, w := range wantInv {
		v, _ := iw.Item(i)
		if math.Abs(v.(float64)-w) > 1e-12 {
			t.Fatalf("inv[%d] = %v, want %v", i, v, w)
		}
	}
	// A·inv(A) ≈ I
	prod, err := MatMul(a, inv)
	if err != nil {
		t.Fatal(err)
	}
	pw := prod.AsContiguous().Flatten()
	for i := 0; i < 9; i++ {
		v, _ := pw.Item(i)
		expect := 0.0
		if i%4 == 0 {
			expect = 1.0
		}
		if math.Abs(v.(float64)-expect) > 1e-12 {
			t.Fatalf("A·inv(A)[%d] = %v, want %v", i, v, expect)
		}
	}

	// solve 1-D
	b := bArr(t, []float64{1, 2, 3})
	x, err := Solve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	wantX := []float64{-0.03571428571428571, 0.35714285714285715, 0.035714285714285705}
	xw := x.AsContiguous().Flatten()
	for i, w := range wantX {
		v, _ := xw.Item(i)
		if math.Abs(v.(float64)-w) > 1e-12 {
			t.Fatalf("solve[%d] = %v, want %v", i, v, w)
		}
	}
	// solve 2-D 右端
	b2 := mustMat(t, []int{3, 1}, []float64{1, 2, 3})
	x2, err := Solve(a, b2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(x2.Shape(), []int{3, 1}) {
		t.Fatalf("solve 2-D 形状 %v", x2.Shape())
	}

	// 奇异矩阵
	sing := mustMat(t, []int{2, 2}, []float64{1, 2, 2, 4})
	if d, err := Det(sing); err != nil {
		t.Fatal(err)
	} else if math.Abs(d) > 1e-12 {
		t.Fatalf("奇异矩阵 det = %v, want 0", d)
	}
	if _, err := Inv(sing); err == nil {
		t.Fatal("奇异矩阵求逆应报错")
	}
	if _, err := Solve(sing, bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("奇异矩阵求解应报错")
	}
	// 非方阵 / 非 2-D
	if _, err := Det(bArr(t, []float64{1, 2, 3})); err == nil {
		t.Fatal("非 2-D 应报错")
	}
	if _, err := Inv(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
	// 阶数不匹配
	if _, err := Solve(a, bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("b 长度不符应报错")
	}
}

// Norm:向量与矩阵范数。
func TestNorm(t *testing.T) {
	v := bArr(t, []float64{3, -4, 0})
	if n, err := Norm(v, "1"); err != nil || n != 7 {
		t.Fatalf("norm1 = %v err=%v, want 7", n, err)
	}
	if n, err := Norm(v, "2"); err != nil || n != 5 {
		t.Fatalf("norm2 = %v, want 5", n)
	}
	if n, err := Norm(v, "inf"); err != nil || n != 4 {
		t.Fatalf("norminf = %v, want 4", n)
	}
	if n, err := Norm(v, "-inf"); err != nil || n != 0 {
		t.Fatalf("norm -inf = %v, want 0", n)
	}
	if n, err := Norm(v, "0"); err != nil || n != 2 {
		t.Fatalf("norm0 = %v, want 2", n)
	}
	a := mustMat(t, []int{3, 3}, []float64{4, 3, 2, 1, 5, 7, 2, 8, 6})
	if n, err := Norm(a, "fro"); err != nil || math.Abs(n-14.422205101855956) > 1e-12 {
		t.Fatalf("fro = %v", n)
	}
	if n, err := Norm(a, "1"); err != nil || n != 16 {
		t.Fatalf("mat norm1 = %v, want 16", n)
	}
	if n, err := Norm(a, "inf"); err != nil || n != 16 {
		t.Fatalf("mat norminf = %v, want 16", n)
	}
	// 矩阵 2-范数 = 最大奇异值;-2 = 最小奇异值(numpy:13.8968 / 1.76139)
	if n, err := Norm(a, "2"); err != nil {
		t.Fatal(err)
	} else if math.Abs(n-13.896794973767586) > 1e-10 {
		t.Fatalf("矩阵 2-范数 = %v", n)
	}
	if n, err := Norm(a, "-2"); err != nil {
		t.Fatal(err)
	} else if math.Abs(n-1.7613858407187588) > 1e-10 {
		t.Fatalf("矩阵 -2 范数 = %v", n)
	}
	if _, err := Norm(v, "bogus"); err == nil {
		t.Fatal("未知 ord 应报错")
	}
	if _, err := Norm(mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8}), "fro"); err == nil {
		t.Fatal("3-D 应报错")
	}
}

// 别名批:prod / around / array_equal / flatnonzero / argwhere / isreal /
// identity / sinc / ravel / shape / reshape / squeeze / transpose / concat / tanh。
func TestAliasBatch(t *testing.T) {
	// prod(整型提升到 int64)
	i := bArr(t, []int32{1, 2, 3})
	if p, err := Prod(i, nil, false); err != nil {
		t.Fatal(err)
	} else if p.Dtype() != DInt64 {
		t.Fatalf("prod dtype %s, want int64", p.Dtype())
	} else if v, _ := p.Item(); v != int64(6) {
		t.Fatalf("prod = %v, want 6", v)
	}
	// around
	if r, err := Around(bArr(t, []float64{1.234, 2.789}), 1); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, r, 1.2, 2.8)
	}
	// array_equal
	if !ArrayEqual(bArr(t, []float64{1, 2}), bArr(t, []float64{1, 2})) {
		t.Fatal("相同数组应相等")
	}
	if ArrayEqual(bArr(t, []float64{1, 2}), bArr(t, []float64{1, 3})) {
		t.Fatal("不同数组不应相等")
	}
	if ArrayEqual(bArr(t, []float64{1, 2}), bArr(t, []float64{1, 2, 3})) {
		t.Fatal("形状不同不应相等")
	}
	// flatnonzero / argwhere
	m := mustMat(t, []int{2, 2}, []float64{1, 0, 0, 2})
	fn, err := Flatnonzero(m)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, fn, 0, 3)
	aw, err := Argwhere(m)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(aw.Shape(), []int{2, 2}) {
		t.Fatalf("argwhere 形状 %v, want [2 2]", aw.Shape())
	}
	bWantI64(t, aw.Flatten(), 0, 0, 1, 1)
	// isreal
	cz := bArr(t, []complex128{complex(1, 0), complex(1, 1)})
	ir, err := IsReal(cz)
	if err != nil {
		t.Fatal(err)
	}
	w := win[uint8](engineWindow(ir.AsContiguous()))
	if w[0] != 1 || w[1] != 0 {
		t.Fatalf("isreal = %v", w)
	}
	// identity
	id, err := Identity(3)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := id.Item(1, 1); v != 1.0 {
		t.Fatalf("identity 对角 = %v", v)
	}
	// empty
	e, err := Empty([]int{2, 3}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if e.Size() != 6 {
		t.Fatalf("empty size = %d", e.Size())
	}
	// sinc:numpy 值 [1, ~0, 0.6366..., ~0]
	sc, err := Sinc(bArr(t, []float64{0, 1, 0.5, -1}))
	if err != nil {
		t.Fatal(err)
	}
	sw := sc.AsContiguous().Flatten()
	v0, _ := sw.Item(0)
	if v0 != 1.0 {
		t.Fatalf("sinc(0) = %v, want 1", v0)
	}
	v2, _ := sw.Item(2)
	if math.Abs(v2.(float64)-0.6366197723675814) > 1e-12 {
		t.Fatalf("sinc(0.5) = %v", v2)
	}
	v1, _ := sw.Item(1)
	if math.Abs(v1.(float64)) > 1e-15 {
		t.Fatalf("sinc(1) 应约 0,得 %v", v1)
	}
	// tanh
	th, err := Tanh(bArr(t, []float64{0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	tw := th.AsContiguous().Flatten()
	if v, _ := tw.Item(1); math.Abs(v.(float64)-math.Tanh(1)) > 1e-15 {
		t.Fatalf("tanh(1) = %v", v)
	}
	// ravel / shape / reshape / squeeze / transpose / concat
	m2 := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	if r, err := Ravel(m2); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(r.Shape(), []int{4}) {
		t.Fatalf("ravel 形状 %v", r.Shape())
	}
	if sh := Shape(m2); len(sh) != 2 || sh[0] != 2 {
		t.Fatalf("shape = %v", sh)
	}
	if r, err := Reshape(m2, []int{4}); err != nil {
		t.Fatal(err)
	} else if r.Size() != 4 {
		t.Fatalf("reshape size = %d", r.Size())
	}
	sq := mustMat(t, []int{1, 3, 1}, []float64{1, 2, 3})
	if s, err := Squeeze(sq); err != nil {
		t.Fatal(err)
	} else if !shapeEqual(s.Shape(), []int{3}) {
		t.Fatalf("squeeze 形状 %v", s.Shape())
	}
	if tr, err := Transpose(m2); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, tr.Flatten(), 1, 3, 2, 4)
	}
	if cc, err := Concat(bArr(t, []float64{1}), bArr(t, []float64{2})); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, cc, 1, 2)
	}
}

// Convolve / Correlate。
func TestConvolveCorrelate(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	v := bArr(t, []float64{1, 1, 1})
	full, err := Convolve(a, v, "full")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, full, 1, 3, 6, 9, 7, 4)
	same, err := Convolve(a, v, "same")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, same, 3, 6, 9, 7)
	valid, err := Convolve(a, v, "valid")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, valid, 6, 9)

	// 第二个序列更长:numpy valid = [10,13],same = [3,10,13]
	b := bArr(t, []float64{1, 2})
	w := bArr(t, []float64{3, 4, 5})
	fb, err := Convolve(b, w, "full")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, fb, 3, 10, 13, 10)
	sb, err := Convolve(b, w, "same")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, sb, 3, 10, 13)
	vb, err := Convolve(b, w, "valid")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, vb, 10, 13)

	// correlate:v 非对称时与卷积不同
	cr, err := Correlate(bArr(t, []float64{1, 2, 3}), bArr(t, []float64{1, 10}), "full")
	if err != nil {
		t.Fatal(err)
	}
	// numpy:np.correlate([1,2,3],[1,10],"full") = [10, 21, 32, 3]
	bWantF64(t, cr, 10, 21, 32, 3)
	csame, err := Correlate(bArr(t, []float64{1, 2, 3}), bArr(t, []float64{1, 10}), "same")
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, csame, 10, 21, 32)

	// 参数校验
	if _, err := Convolve(a, v, "bogus"); err == nil {
		t.Fatal("未知 mode 应报错")
	}
	if _, err := Convolve(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}), v, "full"); err == nil {
		t.Fatal("2-D 应报错")
	}
	if _, err := Convolve(bArr(t, []float64{}), v, "full"); err == nil {
		t.Fatal("空输入应报错")
	}
}

// ufunc 补充批:与 numpy 逐值对照。
func TestUfuncBatch(t *testing.T) {
	// negative
	n, err := Negative(bArr(t, []float64{1, -2, 0}))
	if err != nil {
		t.Fatal(err)
	}
	// numpy:np.negative([0.]) = -0.0(逐位比对要求保留负零)
	bWantF64(t, n, -1, 2, math.Copysign(0, -1))
	// square 保持 dtype
	sq, err := Square(bArr(t, []int32{2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	if sq.Dtype() != DInt32 {
		t.Fatalf("square dtype %s, want int32", sq.Dtype())
	}
	wantI32(t, sq, 4, 9)
	// reciprocal
	r, err := Reciprocal(bArr(t, []float64{2, 4}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r, 0.5, 0.25)
	// 对数/指数族
	if o, err := Log2(bArr(t, []float64{1, 2, 8})); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 0, 1, 3)
	}
	if o, err := Expm1(bArr(t, []float64{0, 1})); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(1); math.Abs(v.(float64)-1.718281828459045) > 1e-12 {
		t.Fatalf("expm1(1) = %v", v)
	}
	if o, err := Log1p(bArr(t, []float64{0, 1})); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(1); math.Abs(v.(float64)-0.6931471805599453) > 1e-12 {
		t.Fatalf("log1p(1) = %v", v)
	}
	// cbrt 保留负号
	if o, err := Cbrt(bArr(t, []float64{8, -27})); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 2, -3)
	}
	// 反双曲
	if o, err := Arcsinh(bArr(t, []float64{0, 1})); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(1); math.Abs(v.(float64)-0.881373587019543) > 1e-12 {
		t.Fatalf("arcsinh(1) = %v", v)
	}
	// degrees / radians 别名
	if o, err := Degrees(bArr(t, []float64{math.Pi})); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); math.Abs(v.(float64)-180) > 1e-9 {
		t.Fatalf("degrees(pi) = %v", v)
	}
	if o, err := Radians(bArr(t, []float64{180})); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); math.Abs(v.(float64)-math.Pi) > 1e-12 {
		t.Fatalf("radians(180) = %v", v)
	}
	// positive = 副本
	pv, err := Positive(bArr(t, []float64{1, -2}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, pv, 1, -2)
	// real / imag
	cz := bArr(t, []complex128{complex(1, 2)})
	if o, err := Real(cz); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != 1.0 {
		t.Fatalf("real = %v", v)
	}
	if o, err := Imag(cz); err != nil {
		t.Fatal(err)
	} else if v, _ := o.Item(0); v != 2.0 {
		t.Fatalf("imag = %v", v)
	}
	// 实数 imag → 全 0
	if o, err := Imag(bArr(t, []float64{1, 2})); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 0, 0)
	}
}

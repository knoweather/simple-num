package num

import (
	"math"
	"testing"
)

// QR:重构误差、正交性、形状(与 numpy 的 reduced 模式一致)。
func TestQR(t *testing.T) {
	a := mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	out, err := QR(a)
	if err != nil {
		t.Fatal(err)
	}
	Q, R := out[0], out[1]
	if !shapeEqual(Q.Shape(), []int{3, 2}) {
		t.Fatalf("Q 形状 %v, want [3 2]", Q.Shape())
	}
	if !shapeEqual(R.Shape(), []int{2, 2}) {
		t.Fatalf("R 形状 %v, want [2 2]", R.Shape())
	}
	// Q·R == A
	qr, err := MatMul(Q, R)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			g, _ := qr.Item(i, j)
			w, _ := a.Item(i, j)
			if d := math.Abs(g.(float64) - w.(float64)); d > worst {
				worst = d
			}
		}
	}
	if worst > 1e-12 {
		t.Fatalf("Q·R - A 最大误差 %.3e", worst)
	}
	// Q 的列正交归一
	for p := 0; p < 2; p++ {
		for q := 0; q < 2; q++ {
			s := 0.0
			for i := 0; i < 3; i++ {
				x, _ := Q.Item(i, p)
				y, _ := Q.Item(i, q)
				s += x.(float64) * y.(float64)
			}
			want := 0.0
			if p == q {
				want = 1
			}
			if math.Abs(s-want) > 1e-12 {
				t.Fatalf("Q 正交性[%d,%d] = %v, want %v", p, q, s, want)
			}
		}
	}
	// R 是上三角
	for i := 1; i < 2; i++ {
		for j := 0; j < i; j++ {
			if v, _ := R.Item(i, j); v.(float64) != 0 {
				t.Fatalf("R[%d,%d] = %v, want 0(上三角)", i, j, v)
			}
		}
	}
	// 方阵与宽矩阵
	if _, err := QR(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})); err != nil {
		t.Fatal(err)
	}
	wide, err := QR(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(wide[0].Shape(), []int{2, 2}) || !shapeEqual(wide[1].Shape(), []int{2, 3}) {
		t.Fatalf("宽矩阵形状 Q=%v R=%v", wide[0].Shape(), wide[1].Shape())
	}
	// 错误路径
	if _, err := QR(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("1-D 应报错")
	}
	if _, err := QR(nil); err == nil {
		t.Fatal("nil 应报错")
	}
	if _, err := QR(mustNew(t, []int{0, 2}, DFloat64)); err == nil {
		t.Fatal("空矩阵应报错")
	}
}

// Cholesky:L·Lᵀ 重构 与 非正定报错。
func TestCholesky(t *testing.T) {
	s := mustMat(t, []int{2, 2}, []float64{4, 2, 2, 3})
	L, err := Cholesky(s)
	if err != nil {
		t.Fatal(err)
	}
	// numpy:[[2,0],[1,1.414214]]
	if v, _ := L.Item(0, 0); math.Abs(v.(float64)-2) > 1e-12 {
		t.Fatalf("L[0,0] = %v, want 2", v)
	}
	if v, _ := L.Item(1, 0); math.Abs(v.(float64)-1) > 1e-12 {
		t.Fatalf("L[1,0] = %v, want 1", v)
	}
	if v, _ := L.Item(1, 1); math.Abs(v.(float64)-math.Sqrt2) > 1e-12 {
		t.Fatalf("L[1,1] = %v, want sqrt2", v)
	}
	if v, _ := L.Item(0, 1); v.(float64) != 0 {
		t.Fatalf("L[0,1] = %v, want 0", v)
	}
	// L·Lᵀ == S
	Lt, err := L.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	ll, err := MatMul(L, Lt)
	if err != nil {
		t.Fatal(err)
	}
	if !ArrayEqual(ll, s) {
		worst := 0.0
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				x, _ := ll.Item(i, j)
				y, _ := s.Item(i, j)
				if d := math.Abs(x.(float64) - y.(float64)); d > worst {
					worst = d
				}
			}
		}
		if worst > 1e-12 {
			t.Fatalf("L·Lᵀ - S 最大误差 %.3e", worst)
		}
	}
	// 非正定 → 报错
	if _, err := Cholesky(mustMat(t, []int{2, 2}, []float64{1, 2, 2, 1})); err == nil {
		t.Fatal("非正定应报错")
	}
	if _, err := Cholesky(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
	// 3×3 正定
	if _, err := Cholesky(mustMat(t, []int{3, 3}, []float64{4, 2, 0, 2, 5, 1, 0, 1, 3})); err != nil {
		t.Fatal(err)
	}
}

// Pinv:M·pinv(M)·M == M,形状 (n,m)。
func TestPinv(t *testing.T) {
	m := mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	p, err := Pinv(m)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(p.Shape(), []int{2, 3}) {
		t.Fatalf("pinv 形状 %v, want [2 3]", p.Shape())
	}
	// numpy 期望值
	want := [][]float64{
		{-1.333333, -0.333333, 0.666667},
		{1.083333, 0.333333, -0.416667},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v, _ := p.Item(i, j)
			if math.Abs(v.(float64)-want[i][j]) > 1e-6 {
				t.Fatalf("pinv[%d,%d] = %v, want %v", i, j, v, want[i][j])
			}
		}
	}
	// M·P·M == M
	mp, err := MatMul(m, p)
	if err != nil {
		t.Fatal(err)
	}
	mpm, err := MatMul(mp, m)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			x, _ := mpm.Item(i, j)
			y, _ := m.Item(i, j)
			if math.Abs(x.(float64)-y.(float64)) > 1e-9 {
				t.Fatalf("M·pinv(M)·M [%d,%d] = %v, want %v", i, j, x, y)
			}
		}
	}
	// 错误路径
	if _, err := Pinv(bArr(t, []float64{1})); err == nil {
		t.Fatal("1-D 应报错")
	}
	if _, err := Pinv(nil); err == nil {
		t.Fatal("nil 应报错")
	}
}

// Lstsq:超定最小二乘、超定精确解、欠定、秩亏。
func TestLstsq(t *testing.T) {
	a := mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	b := bArr(t, []float64{1, 2, 3})
	out, err := Lstsq(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("lstsq 应返回 4 项,得 %d", len(out))
	}
	x := out[0]
	// numpy:[0, 0.5]
	if v, _ := x.Item(0); math.Abs(v.(float64)) > 1e-9 {
		t.Fatalf("x[0] = %v, want 0", v)
	}
	if v, _ := x.Item(1); math.Abs(v.(float64)-0.5) > 1e-9 {
		t.Fatalf("x[1] = %v, want 0.5", v)
	}
	// rank = 2
	if v, _ := out[2].Item(); v != int64(2) {
		t.Fatalf("rank = %v, want 2", v)
	}
	// residuals:m>n 且满秩 → 残差平方和(此处接近 0)
	if out[1].Size() != 1 {
		t.Fatalf("residuals 长度 %d, want 1", out[1].Size())
	}
	if v, _ := out[1].Item(0); v.(float64) > 1e-9 {
		t.Fatalf("residuals = %v, want ~0", v)
	}
	// 奇异值个数 = min(m,n)
	if out[3].Size() != 2 {
		t.Fatalf("s 长度 %d, want 2", out[3].Size())
	}

	// 超定精确解:m=n → residuals 为空
	o2, err := Lstsq(mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1}), bArr(t, []float64{2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o2[0], 2, 3)
	if o2[1].Size() != 0 {
		t.Fatalf("方阵 residuals 应为空,得 %d", o2[1].Size())
	}
	// 欠定:m<n → 最小范数解,residuals 为空
	o3, err := Lstsq(mustMat(t, []int{2, 3}, []float64{1, 0, 0, 0, 1, 0}), bArr(t, []float64{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o3[0].Shape(), []int{3}) {
		t.Fatalf("欠定 x 形状 %v", o3[0].Shape())
	}
	if o3[1].Size() != 0 {
		t.Fatalf("欠定 residuals 应为空,得 %d", o3[1].Size())
	}
	// 多个右端(2-D b)
	o4, err := Lstsq(mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1}),
		mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o4[0].Shape(), []int{2, 2}) {
		t.Fatalf("多右端 x 形状 %v", o4[0].Shape())
	}
	// 错误路径
	if _, err := Lstsq(bArr(t, []float64{1}), bArr(t, []float64{1})); err == nil {
		t.Fatal("a 非 2-D 应报错")
	}
	if _, err := Lstsq(a, bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("b 行数不符应报错")
	}
	if _, err := Lstsq(nil, b); err == nil {
		t.Fatal("nil 应报错")
	}
}

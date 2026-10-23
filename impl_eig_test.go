package num

import (
	"math"
	"testing"
)

// Eigvalsh / Eigh:与 numpy 特征值一致,且 A·v = λ·v、特征向量正交归一。
func TestEig(t *testing.T) {
	a := mustMat(t, []int{3, 3}, []float64{4, 1, 2, 1, 3, 0, 2, 0, 5})
	want := []float64{1.8548973087995761, 3.4760236029181333, 6.669079088282287}
	w, err := Eigvalsh(a)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(w.Shape(), []int{3}) {
		t.Fatalf("eigvalsh 形状 %v", w.Shape())
	}
	for i, x := range want {
		v, _ := w.Item(i)
		if math.Abs(v.(float64)-x) > 1e-10 {
			t.Fatalf("eigvalsh[%d] = %v, want %v", i, v, x)
		}
	}

	pair, err := Eigh(a)
	if err != nil {
		t.Fatal(err)
	}
	if len(pair) != 2 {
		t.Fatalf("eigh 应返回 (w, v), 得 %d 个", len(pair))
	}
	vals, vecs := pair[0], pair[1]
	if !shapeEqual(vecs.Shape(), []int{3, 3}) {
		t.Fatalf("特征向量形状 %v", vecs.Shape())
	}
	// A·v = λ·v
	for k := 0; k < 3; k++ {
		l, _ := vals.Item(k)
		lam := l.(float64)
		for i := 0; i < 3; i++ {
			s := 0.0
			for j := 0; j < 3; j++ {
				av, _ := a.Item(i, j)
				vv, _ := vecs.Item(j, k)
				s += av.(float64) * vv.(float64)
			}
			vv, _ := vecs.Item(i, k)
			if d := math.Abs(s - lam*vv.(float64)); d > 1e-10 {
				t.Fatalf("A·v - λv 在第 %d 列第 %d 行 = %.3e", k, i, d)
			}
		}
	}
	// 特征向量正交归一
	for p := 0; p < 3; p++ {
		for q := 0; q < 3; q++ {
			s := 0.0
			for i := 0; i < 3; i++ {
				x, _ := vecs.Item(i, p)
				y, _ := vecs.Item(i, q)
				s += x.(float64) * y.(float64)
			}
			expect := 0.0
			if p == q {
				expect = 1
			}
			if d := math.Abs(s - expect); d > 1e-10 {
				t.Fatalf("特征向量内积[%d,%d] 偏差 %.3e", p, q, d)
			}
		}
	}
	// 迹一致
	tr := 0.0
	for i := 0; i < 3; i++ {
		v, _ := a.Item(i, i)
		tr += v.(float64)
	}
	sum := 0.0
	for i := 0; i < 3; i++ {
		v, _ := vals.Item(i)
		sum += v.(float64)
	}
	if math.Abs(tr-sum) > 1e-10 {
		t.Fatalf("特征值之和 %v ≠ 迹 %v", sum, tr)
	}

	// 错误路径
	if _, err := Eigvalsh(mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})); err == nil {
		t.Fatal("非方阵应报错")
	}
	if _, err := Eigvalsh(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("非 2-D 应报错")
	}
}

// Svd:奇异值、重构、正交性,高/宽矩阵都要正确。
func TestSvd(t *testing.T) {
	m := mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	out, err := Svd(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("svd 应返回 (U,S,Vt),得 %d", len(out))
	}
	U, S, Vt := out[0], out[1], out[2]
	if !shapeEqual(U.Shape(), []int{3, 2}) {
		t.Fatalf("U 形状 %v, want [3 2]", U.Shape())
	}
	if !shapeEqual(S.Shape(), []int{2}) {
		t.Fatalf("S 形状 %v", S.Shape())
	}
	if !shapeEqual(Vt.Shape(), []int{2, 2}) {
		t.Fatalf("Vt 形状 %v", Vt.Shape())
	}
	// numpy:9.52551809156511, 0.5143005806586441
	wantS := []float64{9.52551809156511, 0.5143005806586441}
	for i, x := range wantS {
		v, _ := S.Item(i)
		if math.Abs(v.(float64)-x) > 1e-12 {
			t.Fatalf("S[%d] = %v, want %v", i, v, x)
		}
	}
	// 重构
	checkRecon := func(mat, uu, ss, vv *Array, rows, cols, k int) {
		worst := 0.0
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				s := 0.0
				for tt := 0; tt < k; tt++ {
					a, _ := uu.Item(i, tt)
					sv, _ := ss.Item(tt)
					b, _ := vv.Item(tt, j)
					s += a.(float64) * sv.(float64) * b.(float64)
				}
				orig, _ := mat.Item(i, j)
				if d := math.Abs(s - orig.(float64)); d > worst {
					worst = d
				}
			}
		}
		if worst > 1e-10 {
			t.Fatalf("SVD 重构最大误差 %.3e", worst)
		}
	}
	checkRecon(m, U, S, Vt, 3, 2, 2)
	// U 正交
	for p := 0; p < 2; p++ {
		for q := 0; q < 2; q++ {
			s := 0.0
			for i := 0; i < 3; i++ {
				x, _ := U.Item(i, p)
				y, _ := U.Item(i, q)
				s += x.(float64) * y.(float64)
			}
			expect := 0.0
			if p == q {
				expect = 1
			}
			if d := math.Abs(s - expect); d > 1e-10 {
				t.Fatalf("U 正交性[%d,%d] 偏差 %.3e", p, q, d)
			}
		}
	}

	// 宽矩阵(10×20):thin SVD 形状应为 U(10,10) S(10) Vt(10,20)
	vals := make([]float64, 200)
	for i := range vals {
		vals[i] = math.Sin(float64(i) * 1.7)
	}
	x, err := FromSlice(vals)
	if err != nil {
		t.Fatal(err)
	}
	xm, err := x.Reshape([]int{10, 20})
	if err != nil {
		t.Fatal(err)
	}
	o2, err := Svd(xm)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o2[0].Shape(), []int{10, 10}) || !shapeEqual(o2[1].Shape(), []int{10}) ||
		!shapeEqual(o2[2].Shape(), []int{10, 20}) {
		t.Fatalf("宽矩阵 thin SVD 形状: %v %v %v", o2[0].Shape(), o2[1].Shape(), o2[2].Shape())
	}
	checkRecon(xm, o2[0], o2[1], o2[2], 10, 20, 10)

	// Svdvals 等于 S
	sv, err := Svdvals(m)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < sv.Size(); i++ {
		a, _ := sv.Item(i)
		b, _ := S.Item(i)
		if a != b {
			t.Fatalf("svdvals[%d] = %v ≠ S[%d] = %v", i, a, i, b)
		}
	}
	// 错误路径
	if _, err := Svd(bArr(t, []float64{1, 2})); err == nil {
		t.Fatal("1-D 应报错")
	}
	if _, err := Svd(nil); err == nil {
		t.Fatal("nil 应报错")
	}
}

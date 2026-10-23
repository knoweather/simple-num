package num

import (
	"errors"
	"testing"
)

// mustMat 由扁平数据 + 形状构造数组。
func mustMat[T any](t *testing.T, shape []int, xs []T) *Array {
	t.Helper()
	a := bArr(t, xs)
	m, err := a.Reshape(shape)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Dot 的四种形状组合。
func TestDot(t *testing.T) {
	// 1-D · 1-D → 0-d 标量 32
	s, err := Dot(bArr(t, []float64{1, 2, 3}), bArr(t, []float64{4, 5, 6}))
	if err != nil {
		t.Fatal(err)
	}
	if s.NDim() != 0 {
		t.Fatalf("1-D·1-D 应为 0-d,得 %v", s.Shape())
	}
	if v, _ := s.Item(); v != 32.0 {
		t.Fatalf("dot = %v, want 32", v)
	}

	// 2-D · 2-D
	a := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	b := mustMat(t, []int{2, 2}, []float64{5, 6, 7, 8})
	m, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(m.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v", m.Shape())
	}
	bWantF64(t, m.Flatten(), 19, 22, 43, 50)

	// 2-D · 1-D → (2,)
	v, err := Dot(a, bArr(t, []float64{5, 6}))
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, v.Flatten(), 17, 39)

	// 1-D · 2-D → (2,)
	w, err := Dot(bArr(t, []float64{1, 2}), b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, w.Flatten(), 19, 22)

	// 形状不匹配
	if _, err := Dot(a, mustMat(t, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})); !errors.Is(err, ErrShape) {
		t.Fatalf("内维不匹配应 ErrShape,得 %v", err)
	}
}

// MatMul:1-D 提升与降回、batch 广播。
func TestMatMul(t *testing.T) {
	// 1-D · 1-D → 0-d
	s, err := MatMul(bArr(t, []float64{1, 2, 3}), bArr(t, []float64{4, 5, 6}))
	if err != nil {
		t.Fatal(err)
	}
	if s.NDim() != 0 {
		t.Fatalf("1-D·1-D 应 0-d,得 %v", s.Shape())
	}
	if v, _ := s.Item(); v != 32.0 {
		t.Fatalf("matmul = %v, want 32", v)
	}

	a := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	b := mustMat(t, []int{2, 2}, []float64{5, 6, 7, 8})
	m, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, m.Flatten(), 19, 22, 43, 50)

	// 1-D · 2-D → (2,)
	x, err := MatMul(bArr(t, []float64{1, 2}), b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(x.Shape(), []int{2}) {
		t.Fatalf("1-D·2-D 形状 %v, want [2]", x.Shape())
	}
	bWantF64(t, x, 19, 22)

	// 2-D · 1-D → (2,)
	y, err := MatMul(a, bArr(t, []float64{5, 6}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(y.Shape(), []int{2}) {
		t.Fatalf("2-D·1-D 形状 %v, want [2]", y.Shape())
	}
	bWantF64(t, y, 17, 39)

	// batch: (2,2,2) × (2,2) → (2,2,2),b 广播到两个 batch
	bat := mustMat(t, []int{2, 2, 2}, []float64{1, 0, 0, 1, 2, 0, 0, 2}) // 两个单位阵变体
	bm, err := MatMul(bat, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(bm.Shape(), []int{2, 2, 2}) {
		t.Fatalf("batch 形状 %v", bm.Shape())
	}
	// 第一块 = I·b = b;第二块 = diag(2,2)·b = 2b
	bWantF64(t, bm.Flatten(), 5, 6, 7, 8, 10, 12, 14, 16)
}

// Inner:两侧末维内积,结果 a.shape[:-1]+b.shape[:-1]。
func TestInner(t *testing.T) {
	a := mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := mustMat(t, []int{2, 3}, []float64{1, 0, 0, 0, 1, 0})
	out, err := Inner(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", out.Shape())
	}
	// [1,2,3]·[1,0,0]=1;[1,2,3]·[0,1,0]=2;[4,5,6]·[1,0,0]=4;[4,5,6]·[0,1,0]=5
	bWantF64(t, out.Flatten(), 1, 2, 4, 5)

	if _, err := Inner(a, mustMat(t, []int{1, 2}, []float64{1, 2})); !errors.Is(err, ErrShape) {
		t.Fatalf("末维不等应 ErrShape,得 %v", err)
	}
}

// Outer:拉平外积。
func TestOuter(t *testing.T) {
	out, err := Outer(bArr(t, []float64{1, 2}), bArr(t, []float64{3, 4, 5}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 3}) {
		t.Fatalf("形状 %v, want [2 3]", out.Shape())
	}
	bWantF64(t, out.Flatten(), 3, 4, 5, 6, 8, 10)

	// 二维输入也先拉平(numpy 同)
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	o2, err := Outer(m, bArr(t, []float64{1, 10}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o2.Shape(), []int{4, 2}) {
		t.Fatalf("拉平后形状 %v, want [4 2]", o2.Shape())
	}
	bWantF64(t, o2.Flatten(), 1, 10, 2, 20, 3, 30, 4, 40)
}

// 整型:按结果 dtype 累加(与 numpy 的溢出行为一致)。
func TestDotInt(t *testing.T) {
	a := mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})
	b := mustMat(t, []int{2, 2}, []int32{5, 6, 7, 8})
	out, err := Dot(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", out.Dtype())
	}
	wantI32(t, out.Flatten(), 19, 22, 43, 50)

	// 混合 dtype:i4 × f8 → f8
	mixed, err := Dot(a, mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if mixed.Dtype() != DFloat64 {
		t.Fatalf("混合 dtype %s, want float64", mixed.Dtype())
	}
	bWantF64(t, mixed.Flatten(), 1, 2, 3, 4)
}

// TestMatMulParallelPath 覆盖**并行分支**(m·k·n ≥ 2^21):
// 用恒等矩阵与全 1 矩阵做可判定的校验,避免并行优化引入静默错误。
func TestMatMulParallelPath(t *testing.T) {
	const n = 128 // 128^3 = 2^21,正好触发并行阈值
	a, err := New([]int{n, n}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	id, err := New([]int{n, n}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if err := a.SetItem([]int{i, j}, float64(i*n+j)); err != nil {
				t.Fatal(err)
			}
			if i == j {
				if err := id.SetItem([]int{i, j}, 1.0); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	c, err := MatMul(a, id)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g, _ := c.Item(i, j)
			w, _ := a.Item(i, j)
			if g != w {
				t.Fatalf("A·I 在 [%d,%d] 处不符: got %v want %v", i, j, g, w)
			}
		}
	}
	// 全 1 矩阵:每元素应为 n
	ones, err := New([]int{n, n}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if err := ones.SetItem([]int{i, j}, 1.0); err != nil {
				t.Fatal(err)
			}
		}
	}
	p, err := MatMul(ones, ones)
	if err != nil {
		t.Fatal(err)
	}
	for _, ij := range [][2]int{{0, 0}, {n / 2, n / 2}, {n - 1, n - 1}} {
		v, _ := p.Item(ij[0], ij[1])
		if v != float64(n) {
			t.Fatalf("ones×ones[%d,%d] = %v, want %d", ij[0], ij[1], v, n)
		}
	}
}

package num

import (
	"math"
	"testing"
)

// einsum 收缩图:多操作数/多收缩标签的等价性(与手算或 MatMul 对照)。
func TestEinsumFastPath(t *testing.T) {
	// 1) 三操作数链:ij,jk,kl->il 应等于两次 matmul
	c1 := mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	c2 := mustMat(t, []int{3, 2}, []float64{1, 0, 0, 1, 1, 1})
	c3 := mustMat(t, []int{2, 2}, []float64{2, 1, 1, 2})
	got, err := Einsum("ij,jk,kl->il", c1, c2, c3)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := MatMul(c1, c2)
	if err != nil {
		t.Fatal(err)
	}
	want, err := MatMul(m1, c3)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got.Shape(), want.Shape()) {
		t.Fatalf("形状 %v vs %v", got.Shape(), want.Shape())
	}
	for i := 0; i < want.Size(); i++ {
		a, _ := got.AsContiguous().Flatten().Item(i)
		b, _ := want.AsContiguous().Flatten().Item(i)
		if math.Abs(a.(float64)-b.(float64)) > 1e-12 {
			t.Fatalf("链式结果[%d] = %v, want %v", i, a, b)
		}
	}

	// 2) 张量-矩阵:ijk,kl->ijl
	t3 := mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	m2 := mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1})
	o2, err := Einsum("ijk,kl->ijl", t3, m2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o2.Shape(), []int{2, 2, 2}) {
		t.Fatalf("形状 %v, want [2 2 2]", o2.Shape())
	}
	// 与恒等矩阵相乘应保持不变
	for i := 0; i < 8; i++ {
		v, _ := o2.AsContiguous().Flatten().Item(i)
		if v != float64(i+1) {
			t.Fatalf("ijk,kl->ijl 与单位矩阵相乘应不变, [%d] = %v", i, v)
		}
	}

	// 3) 双收缩标签:ijk,ijl->kl
	d1 := mustMat(t, []int{2, 2, 2}, []float64{1, 0, 0, 0, 0, 0, 0, 0})
	d2 := mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	o3, err := Einsum("ijk,ijl->kl", d1, d2)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o3.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", o3.Shape())
	}
	// d1 只有 d1[0,0,0]=1 → result[k,l] = d2[0,0,l](k=0)或 0(k=1)
	for k := 0; k < 2; k++ {
		for l := 0; l < 2; l++ {
			v, _ := o3.Item(k, l)
			var w any = 0.0
			if k == 0 {
				w, _ = d2.Item(0, 0, l)
			}
			if v != w {
				t.Fatalf("ijk,ijl->kl[%d,%d] = %v, want %v", k, l, v, w)
			}
		}
	}

	// 4) 全收缩 ij,jk-> → 0-d
	sq := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	o4, err := Einsum("ij,jk->", sq, sq)
	if err != nil {
		t.Fatal(err)
	}
	if o4.NDim() != 0 {
		t.Fatalf("全收缩应为 0-d,得 %v", o4.Shape())
	}
	// numpy:np.einsum("ij,jk->", A, A) = 54(注意不是 (A@A).sum 的直觉值 60)
	if v, _ := o4.Item(); v != 54.0 {
		t.Fatalf("全收缩 = %v, want 54", v)
	}

	// 5) 单操作数仍走枚举(取对角):ii->i
	dg, err := Einsum("ii->i", sq)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dg, 1, 4)

	// 6) 规模化:与 MatMul 逐元素一致(覆盖快路径的转置/reshape)
	const n = 128
	a, err := New([]int{n, n}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New([]int{n, n}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if err := a.SetItem([]int{i, j}, float64(i+j%7+1)); err != nil {
				t.Fatal(err)
			}
			if err := b.SetItem([]int{i, j}, float64(i%5-j+2)); err != nil {
				t.Fatal(err)
			}
		}
	}
	e, err := Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatal(err)
	}
	mm, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	ew := e.AsContiguous().Flatten()
	mw := mm.AsContiguous().Flatten()
	for i := 0; i < mw.Size(); i++ {
		x, _ := ew.Item(i)
		y, _ := mw.Item(i)
		if math.Abs(x.(float64)-y.(float64)) > 1e-9*math.Max(1, math.Abs(y.(float64))) {
			t.Fatalf("规模化 einsum[%d] = %v, matmul = %v", i, x, y)
		}
	}
}

// 快路径的分支覆盖:外积、多余标签求和、输出置换、标量操作数。
func TestEinsumFastBranches(t *testing.T) {
	// 1) 无共享标签 → 外积 i,j->ij
	a := bArr(t, []float64{1, 2})
	b := bArr(t, []float64{3, 4, 5})
	o, err := Einsum("i,j->ij", a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o.Shape(), []int{2, 3}) {
		t.Fatalf("外积形状 %v, want [2 3]", o.Shape())
	}
	bWantF64(t, o.Flatten(), 3, 4, 5, 6, 8, 10)

	// 2) 多余标签求和:ij,jk->i(对 k 求和)
	sq := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	o2, err := Einsum("ij,jk->i", sq, sq)
	if err != nil {
		t.Fatal(err)
	}
	// numpy:np.einsum("ij,jk->i", A, A) = [17, 37]
	// (我最初按"列和"算成 [16,36] 是错的:应为 sum_j A[i,j]*(行j之和))
	bWantF64(t, o2, 17, 37)

	// 3) 输出置换:ij,jk->ki
	o3, err := Einsum("ij,jk->ki", sq, sq)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o3.Shape(), []int{2, 2}) {
		t.Fatalf("置换形状 %v", o3.Shape())
	}
	// A@A = [[7,10],[15,22]] → 转置 [[7,15],[10,22]]
	bWantF64(t, o3.Flatten(), 7, 15, 10, 22)

	// 4) 标量操作数(0-d 项)→ 逐元素乘
	sc, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := sc.SetItem([]int{}, 3.0); err != nil {
		t.Fatal(err)
	}
	o4, err := Einsum(",ij->ij", sc, sq)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o4.Flatten(), 3, 6, 9, 12)
	// 标量在右侧
	o5, err := Einsum("ij,->ij", sq, sc)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o5.Flatten(), 3, 6, 9, 12)

	// 5) 三个以上操作数的链(覆盖贪心选择循环)
	p := mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1})
	q := mustMat(t, []int{2, 2}, []float64{2, 1, 1, 2})
	r := mustMat(t, []int{2, 2}, []float64{1, 1, 0, 1})
	o6, err := Einsum("ij,jk,kl->il", p, q, r)
	if err != nil {
		t.Fatal(err)
	}
	// p·q·r = q·r = [[2,3],[1,3]]
	bWantF64(t, o6.Flatten(), 2, 3, 1, 3)
}

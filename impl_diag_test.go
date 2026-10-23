package num

import (
	"math"
	"testing"
)

// Diag:1-D 构造 / 2-D 抽取 / k 的正负。
func TestDiag(t *testing.T) {
	v := bArr(t, []float64{1, 2, 3})
	m, err := Diag(v, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(m.Shape(), []int{3, 3}) {
		t.Fatalf("形状 %v, want [3 3]", m.Shape())
	}
	bWantF64(t, m.Flatten(), 1, 0, 0, 0, 2, 0, 0, 0, 3)

	// k=1
	k1, err := Diag(bArr(t, []float64{1, 2}), 1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(k1.Shape(), []int{3, 3}) {
		t.Fatalf("k=1 形状 %v, want [3 3]", k1.Shape())
	}
	bWantF64(t, k1.Flatten(), 0, 1, 0, 0, 0, 2, 0, 0, 0)

	// k=-1
	km1, err := Diag(bArr(t, []float64{1, 2}), -1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, km1.Flatten(), 0, 0, 0, 1, 0, 0, 0, 2, 0)

	// 2-D 抽取
	sq := mustMat(t, []int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	d0, err := Diag(sq, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, d0, 1, 5, 9)
	d1, err := Diag(sq, 1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, d1, 2, 6)
	dm1, err := Diag(sq, -1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dm1, 4, 8)

	// Diagflat:二维输入先拉平
	df, err := Diagflat(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(df.Shape(), []int{4, 4}) {
		t.Fatalf("diagflat 形状 %v, want [4 4]", df.Shape())
	}
	if v, _ := df.Item(3, 3); v != 4.0 {
		t.Fatalf("diagflat[3,3] = %v, want 4", v)
	}
	// 0-d → 报错
	if _, err := Diag(mustNew(t, []int{}, DFloat64), 0); err == nil {
		t.Fatal("0-d 应报错")
	}
}

// Trace:偏移与整型保 dtype。
func TestTrace(t *testing.T) {
	sq := mustMat(t, []int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	tr, err := Trace(sq, 0)
	if err != nil {
		t.Fatal(err)
	}
	if tr.NDim() != 0 {
		t.Fatalf("trace 应为 0-d,得 %v", tr.Shape())
	}
	if v, _ := tr.Item(); v != 15.0 {
		t.Fatalf("trace = %v, want 15", v)
	}
	tr1, err := Trace(sq, 1)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := tr1.Item(); v != 8.0 {
		t.Fatalf("trace offset=1 = %v, want 8", v)
	}
	// 整型保持 dtype
	i := mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})
	ti, err := Trace(i, 0)
	if err != nil {
		t.Fatal(err)
	}
	if ti.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", ti.Dtype())
	}
	if v, _ := ti.Item(); v != int32(5) {
		t.Fatalf("int trace = %v, want 5", v)
	}
	// 1-D → 报错
	if _, err := Trace(bArr(t, []float64{1, 2}), 0); err == nil {
		t.Fatal("1-D trace 应报错")
	}
}

// Triu / Tril。
func TestTriuTril(t *testing.T) {
	sq := mustMat(t, []int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	up, err := Triu(sq, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, up.Flatten(), 1, 2, 3, 0, 5, 6, 0, 0, 9)
	up1, err := Triu(sq, 1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, up1.Flatten(), 0, 2, 3, 0, 0, 6, 0, 0, 0)
	lo, err := Tril(sq, 0)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, lo.Flatten(), 1, 0, 0, 4, 5, 0, 7, 8, 9)
	lo1, err := Tril(sq, -1)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, lo1.Flatten(), 0, 0, 0, 4, 0, 0, 7, 8, 0)

	// 不改入参
	bWantF64(t, sq.Flatten(), 1, 2, 3, 4, 5, 6, 7, 8, 9)

	// N-D:在最后两轴操作
	cube := mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	cu, err := Triu(cube, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(cu.Shape(), []int{2, 2, 2}) {
		t.Fatalf("N-D 形状 %v", cu.Shape())
	}
	// 每个 2x2 块保留上三角:[1,2;0,4] 与 [5,6;0,8]
	bWantF64(t, cu.Flatten(), 1, 2, 0, 4, 5, 6, 0, 8)

	// 1-D → 报错
	if _, err := Triu(bArr(t, []float64{1}), 0); err == nil {
		t.Fatal("1-D 应报错")
	}
}

// FillDiagonal:原地写。
func TestFillDiagonal(t *testing.T) {
	m := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	if err := FillDiagonal(m, 9.0); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, m.Flatten(), 9, 2, 3, 9)
	// 非方阵:只填空到 min(rows,cols)
	r := mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	if err := FillDiagonal(r, 0.0); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r.Flatten(), 0, 2, 3, 4, 0, 6)
	if err := FillDiagonal(bArr(t, []float64{1}), 0.0); err == nil {
		t.Fatal("1-D 应报错")
	}
}

// Cov / CorrCoef:与 numpy 真值对齐。
func TestCovCorrCoef(t *testing.T) {
	// numpy:np.cov([[1,2,3],[4,5,6]]) = [[1,1],[1,1]]
	x := mustMat(t, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	c, err := Cov(x, true, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(c.Shape(), []int{2, 2}) {
		t.Fatalf("cov 形状 %v", c.Shape())
	}
	// numpy 计算:每行方差 = Σ(x-mean)²/(3-1) = (1+0+1)/2 = 1;协方差 = 1
	bWantF64(t, c.Flatten(), 1, 1, 1, 1)

	r, err := CorrCoef(x, true)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, r.Flatten(), 1, 1, 1, 1)

	// 1-D → 0-d 标量(numpy:np.cov([1,2,3]) = 1.0)
	v := bArr(t, []float64{1, 2, 3})
	cv, err := Cov(v, true, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cv.NDim() != 0 {
		t.Fatalf("1-D cov 应为 0-d,得 %v", cv.Shape())
	}
	if g, _ := cv.Item(); g != 1.0 {
		t.Fatalf("1-D cov = %v, want 1", g)
	}
	cr, err := CorrCoef(v, true)
	if err != nil {
		t.Fatal(err)
	}
	if g, _ := cr.Item(); g != 1.0 {
		t.Fatalf("1-D corrcoef = %v, want 1", g)
	}

	// 非完全相关的例子:[[1,2,4],[2,3,7]] → 手算校验
	y := mustMat(t, []int{2, 3}, []float64{1, 2, 4, 2, 3, 7})
	cy, err := Cov(y, true, -1)
	if err != nil {
		t.Fatal(err)
	}
	// row0: mean=7/3, var=( (1-7/3)²+(2-7/3)²+(4-7/3)²)/2 = ((16/9)+(1/9)+(25/9))/2 = (42/9)/2 = 7/3
	// row1: mean=4, var=((2-4)²+(3-4)²+(7-4)²)/2 = (4+1+9)/2 = 7
	// cov = ((1-7/3)(2-4)+(2-7/3)(3-4)+(4-7/3)(7-4))/2 = ((-4/3)(-2)+( -1/3)(-1)+(5/3)(3))/2 = (8/3+1/3+5)/2 = (8/3+1/3+15/3)/2 = (24/3)/2 = 4
	// 浮点末位受累加顺序影响,这里按容差比较(不是逐位)
	for i, w := range []float64{7.0 / 3.0, 4, 4, 7} {
		g, _ := cy.Flatten().Item(i)
		if math.Abs(g.(float64)-w) > 1e-12 {
			t.Fatalf("cov[%d] = %v, want %v", i, g, w)
		}
	}

	// rowvar=false:列是变量
	cf, err := Cov(x, false, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(cf.Shape(), []int{3, 3}) {
		t.Fatalf("rowvar=false 形状 %v, want [3 3]", cf.Shape())
	}
	// 列 (1,4),(2,5),(3,6):每列方差 = ((1-2.5)²+(4-2.5)²)/1 = 4.5;协方差均 4.5
	if v, _ := cf.Item(0, 0); math.Abs(v.(float64)-4.5) > 1e-12 {
		t.Fatalf("rowvar=false cov[0,0] = %v, want 4.5", v)
	}

	// ddof 校验:观测量不足报错
	small := mustMat(t, []int{1, 1}, []float64{5})
	if _, err := Cov(small, true, 1); err == nil {
		t.Fatal("观测量不足应报错")
	}
	// 3-D → 报错
	if _, err := Cov(mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8}), true, -1); err == nil {
		t.Fatal("3-D 应报错")
	}
}

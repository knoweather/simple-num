package num

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// SaveTxt / LoadTxt:往返、分隔符、skiprows、注释、参差行报错。
func TestTxtRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")

	// 2-D 往返(显式格式)
	m := mustMat(t, []int{2, 2}, []float64{1.5, 2.5, 3.5, 4.5})
	if err := SaveTxt(p, m, "%.2f", ","); err != nil {
		t.Fatal(err)
	}
	got, err := LoadTxt(p, DFloat64, ",", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", got.Shape())
	}
	bWantF64(t, got.Flatten(), 1.5, 2.5, 3.5, 4.5)

	// 1-D → 每行一个值
	v := bArr(t, []int32{7, -2, 300})
	p2 := filepath.Join(dir, "v.txt")
	if err := SaveTxt(p2, v, "%d", ""); err != nil {
		t.Fatal(err)
	}
	got2, err := LoadTxt(p2, DInt32, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got2.Shape(), []int{3}) {
		t.Fatalf("单列应为 1-D,得 %v", got2.Shape())
	}
	if got2.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", got2.Dtype())
	}
	wantI32(t, got2, 7, -2, 300)

	// skiprows + 注释 + 空白
	p3 := filepath.Join(dir, "c.txt")
	raw := "# header\n1 2\n\n3 4\n# tail\n"
	if err := os.WriteFile(p3, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	got3, err := LoadTxt(p3, DFloat64, "", 1) // 跳过 "# header" 行
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got3.Shape(), []int{2, 2}) {
		t.Fatalf("形状 %v, want [2 2]", got3.Shape())
	}
	bWantF64(t, got3.Flatten(), 1, 2, 3, 4)

	// 参差行 → ErrShape
	p4 := filepath.Join(dir, "bad.txt")
	if err := os.WriteFile(p4, []byte("1 2\n3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTxt(p4, DFloat64, "", 0); err == nil {
		t.Fatal("列数不一致应报错")
	}
	// 无法解析 → 报错
	p5 := filepath.Join(dir, "bad2.txt")
	if err := os.WriteFile(p5, []byte("1 abc\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTxt(p5, DFloat64, "", 0); err == nil {
		t.Fatal("非法数字应报错")
	}
	// 空文件 → 0 长度 1-D
	p6 := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(p6, []byte("# only comment\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if o, err := LoadTxt(p6, DFloat64, "", 0); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("空文件应得空数组,得 %v", o.Shape())
	}
	// 3-D 保存 → ErrShape
	if err := SaveTxt(filepath.Join(dir, "x.txt"), mustMat(t, []int{2, 2, 2},
		[]float64{1, 2, 3, 4, 5, 6, 7, 8}), "", ""); err == nil {
		t.Fatal("3-D 保存应报错")
	}
}

// Place:按**命中顺序**填充(与 PutMask 的按索引不同)。
func TestPlace(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	m := bArr(t, []bool{true, false, true, false})
	// 对照:PutMask 按索引取 → 位置 0 取 vals[0]=9,位置 2 取 vals[2%2]=9
	pm := bArr(t, []float64{1, 2, 3, 4})
	if err := PutMask(pm, m, bArr(t, []float64{9, 8})); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, pm, 9, 2, 9, 4)

	// Place 按命中顺序取 → 第 1 个命中取 9,第 2 个命中取 8
	if err := Place(a, m, bArr(t, []float64{9, 8})); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, a, 9, 2, 8, 4)

	// vals 不足 → 循环
	b := bArr(t, []float64{1, 1, 1})
	m2 := bArr(t, []bool{true, true, true})
	if err := Place(b, m2, bArr(t, []float64{5})); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, b, 5, 5, 5)

	// 空 vals → 不变
	c := bArr(t, []float64{1, 2})
	if err := Place(c, bArr(t, []bool{true, true}), mustNew(t, []int{0}, DFloat64)); err != nil {
		t.Fatal(err)
	}
	bWantF64(t, c, 1, 2)

	// 非连续视图写回
	base := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	tr, err := base.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	all := mustMat(t, []int{2, 2}, []bool{true, true, true, true})
	if err := Place(tr, all, bArr(t, []float64{7})); err != nil {
		t.Fatal(err)
	}
	for i := range []int{0, 1, 2, 3} {
		if v, _ := base.Item(i/2, i%2); v != 7.0 {
			t.Fatalf("非连续写回失败 base[%d]=%v", i, v)
		}
	}
}

// Piecewise:条件 + 默认分支。
func TestPiecewise(t *testing.T) {
	x := bArr(t, []float64{-2, -1, 0, 1, 2})
	neg := bArr(t, []bool{true, true, false, false, false})
	pos := bArr(t, []bool{false, false, false, true, true})
	// 两个分支 + 默认 0(0 落在默认区)
	out, err := Piecewise(x, []*Array{neg, pos},
		[]UnaryFunc{
			func(a *Array) (*Array, error) { return Multiply(a, bArr(t, []float64{-1})) },
			func(a *Array) (*Array, error) { return Multiply(a, bArr(t, []float64{10})) },
		})
	if err != nil {
		t.Fatal(err)
	}
	// -2,-1 → 取反;1,2 → ×10;0 → 0(未命中)
	bWantF64(t, out, 2, 1, 0, 10, 20)

	// 显式默认分支(最后一个函数)
	out2, err := Piecewise(x, []*Array{neg},
		[]UnaryFunc{
			func(a *Array) (*Array, error) { return Multiply(a, bArr(t, []float64{-1})) },
			func(a *Array) (*Array, error) { return Multiply(a, bArr(t, []float64{100})) },
		})
	if err != nil {
		t.Fatal(err)
	}
	// 负数取反,其余走默认 ×100
	bWantF64(t, out2, 2, 1, 0, 100, 200)

	// 非法:fns 太少 / 太多
	if _, err := Piecewise(x, []*Array{neg, pos}, []UnaryFunc{
		func(a *Array) (*Array, error) { return a, nil },
	}); err == nil {
		t.Fatal("fns 少于 conds 应报错")
	}
	if _, err := Piecewise(x, []*Array{neg}, []UnaryFunc{
		func(a *Array) (*Array, error) { return a, nil },
		func(a *Array) (*Array, error) { return a, nil },
		func(a *Array) (*Array, error) { return a, nil },
	}); err == nil {
		t.Fatal("fns 超过 conds+1 应报错")
	}
	if _, err := Piecewise(x, nil, nil); err == nil {
		t.Fatal("空 conds 应报错")
	}
	_ = strings.TrimSpace
}

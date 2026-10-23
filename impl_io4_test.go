package num

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Genfromtxt:与 numpy 2.5.4 实测对齐(注释跳过、缺省填 NaN、usecols、单列 1-D、参差报错)。
func TestGenfromtxt(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.csv")
	content := "# 注释行\n1.0,2.0,3.0\n,4.0,5.5\n7.5,,9.5\n"
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	// numpy:[[1,2,3],[nan,4,5.5],[7.5,nan,9.5]]
	got, err := Genfromtxt(p, GenfromtxtOptions{Delimiter: ","})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got.Shape(), []int{3, 3}) {
		t.Fatalf("形状 %v, want [3 3]", got.Shape())
	}
	w := win[float64](engineWindow(got.AsContiguous()))
	want := []float64{1, 2, 3, math.NaN(), 4, 5.5, 7.5, math.NaN(), 9.5}
	for i := range want {
		if math.IsNaN(want[i]) {
			if !math.IsNaN(w[i]) {
				t.Fatalf("[%d] = %v, want NaN", i, w[i])
			}
			continue
		}
		if w[i] != want[i] {
			t.Fatalf("[%d] = %v, want %v", i, w[i], want[i])
		}
	}
	// usecols=(0,2) → [[1,3],[nan,5.5],[7.5,9.5]]
	uc, err := Genfromtxt(p, GenfromtxtOptions{Delimiter: ",", Usecols: []int{0, 2}})
	if err != nil {
		t.Fatal(err)
	}
	wuc := win[float64](engineWindow(uc.AsContiguous()))
	if len(wuc) != 6 || wuc[0] != 1 || wuc[1] != 3 || !math.IsNaN(wuc[2]) || wuc[3] != 5.5 {
		t.Fatalf("usecols = %v", wuc)
	}
	// 填充值 0
	fv, err := Genfromtxt(p, GenfromtxtOptions{Delimiter: ",", FillingValue: 0, HasFilling: true})
	if err != nil {
		t.Fatal(err)
	}
	wfv := win[float64](engineWindow(fv.AsContiguous()))
	if wfv[3] != 0 || wfv[7] != 0 {
		t.Fatalf("填充值未生效: %v", wfv)
	}
	// skip_header / skip_footer(物理行,注释行也计入)
	sh, err := Genfromtxt(p, GenfromtxtOptions{Delimiter: ",", SkipHeader: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(sh.Shape(), []int{3, 3}) {
		t.Fatalf("skip_header 形状 %v", sh.Shape())
	}
	sf, err := Genfromtxt(p, GenfromtxtOptions{Delimiter: ",", SkipFooter: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(sf.Shape(), []int{2, 3}) {
		t.Fatalf("skip_footer 形状 %v", sf.Shape())
	}

	// 单列 → 1-D
	p1 := filepath.Join(dir, "one.txt")
	if err := os.WriteFile(p1, []byte("1\n2\n3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	one, err := Genfromtxt(p1, GenfromtxtOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(one.Shape(), []int{3}) {
		t.Fatalf("单列形状 %v, want [3]", one.Shape())
	}
	bWantF64(t, one, 1, 2, 3)

	// 空白分隔(默认)
	pw := filepath.Join(dir, "w.txt")
	if err := os.WriteFile(pw, []byte("1.5  2.5\n3.5 4.5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ww, err := Genfromtxt(pw, GenfromtxtOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(ww.Shape(), []int{2, 2}) {
		t.Fatalf("空白分隔形状 %v", ww.Shape())
	}

	// inf 字面量
	pi := filepath.Join(dir, "inf.txt")
	if err := os.WriteFile(pi, []byte("inf,-inf\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	inf, err := Genfromtxt(pi, GenfromtxtOptions{Delimiter: ","})
	if err != nil {
		t.Fatal(err)
	}
	wi := win[float64](engineWindow(inf.AsContiguous()))
	if !math.IsInf(wi[0], 1) || !math.IsInf(wi[1], -1) {
		t.Fatalf("inf 解析 = %v", wi)
	}

	// 错误路径:参差行 / 非法数值 / 列越界 / 无数据 / 文件不存在
	pr := filepath.Join(dir, "ragged.txt")
	if err := os.WriteFile(pr, []byte("1,2,3\n4,5\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Genfromtxt(pr, GenfromtxtOptions{Delimiter: ","}); err == nil {
		t.Fatal("参差行应报错")
	}
	pb := filepath.Join(dir, "bad.txt")
	if err := os.WriteFile(pb, []byte("1,abc\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Genfromtxt(pb, GenfromtxtOptions{Delimiter: ","}); err == nil {
		t.Fatal("非法数值应报错")
	}
	if _, err := Genfromtxt(p, GenfromtxtOptions{Delimiter: ",", Usecols: []int{9}}); err == nil {
		t.Fatal("列越界应报错")
	}
	pe := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(pe, []byte("# only comments\n\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Genfromtxt(pe, GenfromtxtOptions{}); err == nil {
		t.Fatal("无数据应报错")
	}
	if _, err := Genfromtxt(filepath.Join(dir, "nope.txt"), GenfromtxtOptions{}); err == nil {
		t.Fatal("文件不存在应报错")
	}
}

// GenfromtxtBytes:同一口径,不落盘。
func TestGenfromtxtBytes(t *testing.T) {
	raw := []byte("a,b,c\n1,,3\n")
	// 表头不是数值 → 首行即报错
	if _, err := GenfromtxtBytes(raw, GenfromtxtOptions{Delimiter: ","}); err == nil {
		t.Fatal("非数值表头应报错")
	}
	// 跳过表头后解析
	got2, err := GenfromtxtBytes(raw, GenfromtxtOptions{Delimiter: ",", SkipHeader: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got2.Shape(), []int{1, 3}) {
		t.Fatalf("形状 %v, want [1 3]", got2.Shape())
	}
	w := win[float64](engineWindow(got2.AsContiguous()))
	if w[0] != 1 || !math.IsNaN(w[1]) || w[2] != 3 {
		t.Fatalf("解析结果 %v", w)
	}
	// 空字节 → 报错
	if _, err := GenfromtxtBytes(nil, GenfromtxtOptions{}); err == nil {
		t.Fatal("空数据应报错")
	}
}

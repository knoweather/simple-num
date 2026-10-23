package num

import (
	"os"
	"path/filepath"
	"testing"
)

// npz 往返:多数组、不同 dtype/形状、自动补.npz 后缀。
func TestNpzRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "arc") // 故意不带.npz
	a := bArr(t, []float64{1.5, -2.25, 3})
	b := mustMat(t, []int{2, 2}, []int32{7, 8, 9, 10})
	if err := SaveZ(p, []string{"alpha", "beta"}, []*Array{a, b}); err != nil {
		t.Fatal(err)
	}
	// 自动补后缀
	if _, err := os.Stat(p + ".npz"); err != nil {
		t.Fatalf("应自动补 .npz: %v", err)
	}
	z, err := LoadZ(p + ".npz")
	if err != nil {
		t.Fatal(err)
	}
	if z.Len() != 2 {
		t.Fatalf("应有 2 个数组,得 %d", z.Len())
	}
	names := z.Names()
	if len(names) != 2 {
		t.Fatalf("names = %v", names)
	}
	gotA, ok := z.Get("alpha")
	if !ok {
		t.Fatalf("缺少 alpha,有 %v", names)
	}
	bWantF64(t, gotA, 1.5, -2.25, 3)
	gotB, ok := z.Get("beta")
	if !ok {
		t.Fatalf("缺少 beta")
	}
	if gotB.Dtype() != DInt32 {
		t.Fatalf("beta dtype %s", gotB.Dtype())
	}
	if !shapeEqual(gotB.Shape(), []int{2, 2}) {
		t.Fatalf("beta 形状 %v", gotB.Shape())
	}
	wantI32(t, gotB.Flatten(), 7, 8, 9, 10)

	// 压缩版
	p2 := filepath.Join(dir, "arc2.npz")
	if err := SaveZCompressed(p2, []string{"x"}, []*Array{a}); err != nil {
		t.Fatal(err)
	}
	z2, err := LoadZ(p2)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := z2.Get("x"); !ok {
		t.Fatal("压缩版缺少 x")
	} else {
		bWantF64(t, v, 1.5, -2.25, 3)
	}
}

// 错误路径:名字数量不符、重名、非法 zip、空归档、缺条目。
func TestNpzErrors(t *testing.T) {
	dir := t.TempDir()
	a := bArr(t, []float64{1})
	if err := SaveZ(filepath.Join(dir, "x.npz"), []string{"a", "b"}, []*Array{a}); err == nil {
		t.Fatal("名字数量不符应报错")
	}
	if err := SaveZ(filepath.Join(dir, "y.npz"), []string{"a", "a"}, []*Array{a, a}); err == nil {
		t.Fatal("重名应报错")
	}
	// 非 zip 文件
	bad := filepath.Join(dir, "bad.npz")
	if err := os.WriteFile(bad, []byte("not a zip"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadZ(bad); err == nil {
		t.Fatal("非 zip 应报错")
	}
	if _, err := LoadZ(filepath.Join(dir, "nope.npz")); err == nil {
		t.Fatal("文件不存在应报错")
	}
	// 匿名数组 → 自动命名 arr_0
	p := filepath.Join(dir, "anon.npz")
	if err := SaveZ(p, []string{""}, []*Array{a}); err != nil {
		t.Fatal(err)
	}
	z, err := LoadZ(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := z.Get("arr_0"); !ok {
		t.Fatalf("匿名数组应命名为 arr_0,得 %v", z.Names())
	}
}

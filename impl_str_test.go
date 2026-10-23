package num

// 单测(B):S/U 定宽字符串比较。
// 期望值 = numpy 2.5.4 实测探针字面量。

import "testing"

func strArr(t *testing.T, width int, unicode bool, vals ...string) *Array {
	t.Helper()
	dt := DBytes(width)
	if unicode {
		dt = DUnicode(width)
	}
	a, err := New([]int{len(vals)}, dt)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vals {
		if err := a.SetItem([]int{i}, v); err != nil {
			t.Fatalf("SetItem(%d, %q): %v", i, v, err)
		}
	}
	return a
}

// S 比较:尾 NUL 裁剪 + 字节序。
func TestStrCompareS(t *testing.T) {
	a := strArr(t, 3, false, "ab", "abc", "ab", "zz")
	b := strArr(t, 3, false, "ab", "abd", "ab", "zz")

	out, err := Equal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, false, true, true)

	out, _ = Less(a, b)
	bWantBool(t, out, false, true, false, false)
	out, _ = Greater(a, b)
	bWantBool(t, out, false, false, false, false)
	out, _ = NotEqual(a, b)
	bWantBool(t, out, false, true, false, false)

	// 同 kind 不同宽度(S3 vs S4):按值比较,全等
	w := strArr(t, 4, false, "ab", "abc", "ab", "zz")
	out, err = Equal(a, w)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, true, true, true)
}

// U 比较:码点序(UTF-8 保序)。
func TestStrCompareU(t *testing.T) {
	a := strArr(t, 3, true, "ab", "abc", "中", "zz")
	b := strArr(t, 3, true, "ab", "abd", "中", "zz")
	out, err := Equal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false, true, true)
	out, _ = Less(a, b)
	bWantBool(t, out, false, true, false, false)

	// 码点序:大写 < 小写 < 中文
	x := strArr(t, 1, true, "B", "a", "b", "中")
	y := strArr(t, 1, true, "a", "a", "a", "a")
	out, err = Less(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false, false, false)
}

// 尾 NUL 裁剪:空串与零填充相等。
func TestStrEmptyPadding(t *testing.T) {
	a := strArr(t, 2, false, "")
	b := strArr(t, 2, false, "\x00\x00")
	out, err := Equal(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true)
}

// 错误路径:S×U 混合拒;字符串 add 不做。
func TestStrErrors(t *testing.T) {
	s := strArr(t, 2, false, "ab")
	u := strArr(t, 2, true, "ab")
	if _, err := Equal(s, u); rootOf(err) != ErrDtype {
		t.Fatalf("S×U 比较 err = %v, want ErrDtype", err)
	}
	if _, err := Add(s, s); rootOf(err) != ErrDtype {
		t.Fatalf("S add(字符运算不做)err = %v, want ErrDtype", err)
	}
}

// 广播:S 比较也走右对齐广播。
func TestStrBroadcast(t *testing.T) {
	a := strArr(t, 1, false, "a", "b", "c", "d")
	av, _ := a.Reshape([]int{2, 2})
	b := strArr(t, 1, false, "b", "b")
	out, err := Less(av, b)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 2}) {
		t.Fatalf("shape = %v", out.Shape())
	}
	want := []bool{true, false, false, false}
	for i, w := range want {
		if g := getI(t, out, i/2, i%2).(bool); g != w {
			t.Fatalf("[%d] = %v, want %v", i, g, w)
		}
	}
}

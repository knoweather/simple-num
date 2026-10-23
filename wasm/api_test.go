package wasm

import (
	"encoding/binary"
	"math"
	"regexp"
	"strings"
	"testing"

	num "github.com/knoweather/simple-num"
)

// f64Bytes 把 float64 序列打成小端字节(测试里造 f8 数组用)。
func f64Bytes(vals ...float64) []byte {
	buf := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return buf
}

func valsOf(t *testing.T, a *num.Array) []any {
	t.Helper()
	v, err := Values(a)
	if err != nil {
		t.Fatalf("Values: %v", err)
	}
	return v
}

func TestAPI(t *testing.T) {
	t.Run("dtype 双向(17 类)", func(t *testing.T) {
		cases := []struct {
			s  string
			dt num.Dtype
		}{
			{"b", num.DBool},
			{"i1", num.DInt8}, {"i2", num.DInt16}, {"i4", num.DInt32}, {"i8", num.DInt64},
			{"u1", num.DUint8}, {"u2", num.DUint16}, {"u4", num.DUint32}, {"u8", num.DUint64},
			{"f2", num.DFloat16}, {"f4", num.DFloat32}, {"f8", num.DFloat64},
			{"c8", num.DComplex64}, {"c16", num.DComplex128},
			{"M8", num.Dtype{Kind: num.Datetime64, Itemsize: 8}},
			{"M8[ns]", num.DTime("ns")},
			{"S4", num.DBytes(4)}, {"U3", num.DUnicode(3)},
			{"S0", num.DBytes(0)}, {"U0", num.DUnicode(0)},
		}
		for _, c := range cases {
			dt, err := ParseDtype(c.s)
			if err != nil {
				t.Fatalf("ParseDtype(%q): %v", c.s, err)
			}
			if dt != c.dt {
				t.Errorf("ParseDtype(%q) = %+v, want %+v", c.s, dt, c.dt)
			}
			if got := DtypeString(dt); got != c.s {
				t.Errorf("DtypeString(%+v) = %q, want %q", dt, got, c.s)
			}
		}
		if got := DtypeString(num.DBool); got != "b" {
			t.Errorf("DtypeString(DBool) = %q", got)
		}
		if _, err := ParseDtype("?"); err != nil {
			t.Errorf(`ParseDtype("?"): %v`, err)
		}
	})

	t.Run("dtype 非法串", func(t *testing.T) {
		for _, s := range []string{"", "x4", "i3", "i16", "f5", "c4", "M8[xx]", "M8[", "S-1", "U", "s4", "b1"} {
			_, err := ParseDtype(s)
			if err == nil {
				t.Errorf("ParseDtype(%q) 应报错", s)
				continue
			}
			if !strings.HasPrefix(err.Error(), "wasm: ") {
				t.Errorf("ParseDtype(%q) 错误缺 wasm: 前缀:%v", s, err)
			}
		}
	})

	t.Run("FromBytes 长度校验", func(t *testing.T) {
		_, err := FromBytes("f8", []int{2, 2}, make([]byte, 30))
		if err == nil {
			t.Fatal("长度不符应报错")
		}
		if !strings.HasPrefix(err.Error(), "wasm: ") {
			t.Errorf("错误缺 wasm: 前缀:%v", err)
		}
		if !strings.Contains(err.Error(), "32") || !strings.Contains(err.Error(), "30") {
			t.Errorf("错误应含期望/实际字节数:%v", err)
		}
		a, err := FromBytes("i4", []int{2, 2}, make([]byte, 16))
		if err != nil {
			t.Fatalf("FromBytes 合法长度:%v", err)
		}
		if a.Size() != 4 || a.Dtype() != num.DInt32 {
			t.Errorf("FromBytes 形状/dtype 不符:%v %v", a.Shape(), a.Dtype())
		}
	})

	t.Run("Zeros/Arange 形状与值", func(t *testing.T) {
		z, err := num.Zeros([]int{2, 2}, num.DFloat64)
		if err != nil {
			t.Fatalf("Zeros:%v", err)
		}
		if len(z.Shape()) != 2 || z.Shape()[0] != 2 || z.Shape()[1] != 2 {
			t.Errorf("Zeros 形状 %v", z.Shape())
		}
		for i, v := range valsOf(t, z) {
			if v.(float64) != 0 {
				t.Errorf("Zeros[%d] = %v, want 0", i, v)
			}
		}
		a, err := num.Arange(1, 7, 2, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		if a.Size() != 3 {
			t.Fatalf("Arange 长度 %d", a.Size())
		}
		want := []int32{1, 3, 5}
		for i, v := range valsOf(t, a) {
			if v.(int32) != want[i] {
				t.Errorf("Arange[%d] = %v, want %d", i, v, want[i])
			}
		}
	})

	t.Run("Bytes/FromBytes 往返", func(t *testing.T) {
		for _, c := range []struct {
			name string
			dt   num.Dtype
			desc string
		}{{"f8", num.DFloat64, "f8"}, {"i4", num.DInt32, "i4"}} {
			a, err := num.Arange(1, 9, 1, c.dt)
			if err != nil {
				t.Fatalf("%s Arange:%v", c.name, err)
			}
			b, err := Bytes(a)
			if err != nil {
				t.Fatalf("%s Bytes:%v", c.name, err)
			}
			if len(b) != a.Size()*c.dt.Itemsize {
				t.Fatalf("%s 数据区 %dB,期望 %dB", c.name, len(b), a.Size()*c.dt.Itemsize)
			}
			back, err := FromBytes(c.desc, a.Shape(), b)
			if err != nil {
				t.Fatalf("%s FromBytes:%v", c.name, err)
			}
			if back.Dtype() != c.dt {
				t.Errorf("%s 回读 dtype %v", c.name, back.Dtype())
			}
			got := valsOf(t, back)
			if len(got) != a.Size() {
				t.Fatalf("%s 回读值数 %d", c.name, len(got))
			}
			for i, v := range valsOf(t, a) {
				if got[i] != v {
					t.Errorf("%s 往返 [%d] = %v, want %v", c.name, i, got[i], v)
				}
			}
		}
	})

	t.Run("Values C 序与 0 维", func(t *testing.T) {
		base, err := num.Arange(1, 7, 1, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		m, err := base.Reshape([]int{2, 3})
		if err != nil {
			t.Fatalf("Reshape:%v", err)
		}
		got := valsOf(t, m)
		want := []int32{1, 2, 3, 4, 5, 6} // 行主序
		if len(got) != len(want) {
			t.Fatalf("Values 长度 %d", len(got))
		}
		for i, v := range want {
			if got[i].(int32) != v {
				t.Errorf("Values[%d] = %v, want %d", i, got[i], v)
			}
		}
		s0, err := num.Full([]int{}, int64(42), num.DInt64)
		if err != nil {
			t.Fatalf("Full 0 维:%v", err)
		}
		g0 := valsOf(t, s0)
		if len(g0) != 1 || g0[0].(int64) != 42 {
			t.Errorf("0 维 Values = %v", g0)
		}
		sb, err := num.Full([]int{2}, "ab", num.DBytes(4))
		if err != nil {
			t.Fatalf("Full S4:%v", err)
		}
		gb := valsOf(t, sb)
		if len(gb) != 2 || gb[0] != "ab" || gb[1] != "ab" {
			t.Errorf("S4 Values = %v, want [ab ab]", gb)
		}
	})

	t.Run("Op add 广播", func(t *testing.T) {
		base, err := num.Arange(1, 7, 1, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		x, err := base.Reshape([]int{2, 3})
		if err != nil {
			t.Fatalf("Reshape:%v", err)
		}
		y, err := num.Arange(10, 13, 1, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		r, err := Op("add", x, y)
		if err != nil {
			t.Fatalf("Op add:%v", err)
		}
		if r.Size() != 6 || len(r.Shape()) != 2 || r.Shape()[0] != 2 || r.Shape()[1] != 3 {
			t.Fatalf("add 广播形状 %v", r.Shape())
		}
		want := []int32{11, 13, 15, 14, 16, 18}
		for i, v := range valsOf(t, r) {
			if v.(int32) != want[i] {
				t.Errorf("add[%d] = %v, want %d", i, v, want[i])
			}
		}
	})

	t.Run("Op equal 输出 bool", func(t *testing.T) {
		base, err := num.Arange(1, 7, 1, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		x, err := base.Reshape([]int{2, 3})
		if err != nil {
			t.Fatalf("Reshape:%v", err)
		}
		y, err := num.Arange(1, 4, 1, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		r, err := Op("equal", x, y)
		if err != nil {
			t.Fatalf("Op equal:%v", err)
		}
		if r.Dtype() != num.DBool {
			t.Fatalf("equal dtype = %v, want bool", r.Dtype())
		}
		want := []bool{true, true, true, false, false, false}
		for i, v := range valsOf(t, r) {
			if v.(bool) != want[i] {
				t.Errorf("equal[%d] = %v, want %v", i, v, want[i])
			}
		}
	})

	t.Run("Unary sqrt", func(t *testing.T) {
		x, err := num.Full([]int{2, 2}, 9.0, num.DFloat64)
		if err != nil {
			t.Fatalf("Full:%v", err)
		}
		r, err := Unary("sqrt", x)
		if err != nil {
			t.Fatalf("Unary sqrt:%v", err)
		}
		if r.Dtype() != num.DFloat64 || r.Size() != 4 {
			t.Fatalf("sqrt 结果 %v %d", r.Dtype(), r.Size())
		}
		for i, v := range valsOf(t, r) {
			if v.(float64) != 3.0 {
				t.Errorf("sqrt[%d] = %v, want 3", i, v)
			}
		}
	})

	t.Run("Reduce sum", func(t *testing.T) {
		base, err := num.Arange(1, 7, 1, num.DInt32)
		if err != nil {
			t.Fatalf("Arange:%v", err)
		}
		a, err := base.Reshape([]int{2, 3})
		if err != nil {
			t.Fatalf("Reshape:%v", err)
		}
		all, err := Reduce("sum", a, nil, false)
		if err != nil {
			t.Fatalf("Reduce sum(axis nil):%v", err)
		}
		if len(all.Shape()) != 0 || all.Size() != 1 {
			t.Fatalf("全归约形状 %v", all.Shape())
		}
		if v := valsOf(t, all)[0].(int64); v != 21 {
			t.Errorf("全归约 = %d, want 21", v)
		}
		ax0, err := Reduce("sum", a, []int{0}, true)
		if err != nil {
			t.Fatalf("Reduce sum(axis [0]):%v", err)
		}
		if len(ax0.Shape()) != 2 || ax0.Shape()[0] != 1 || ax0.Shape()[1] != 3 {
			t.Fatalf("keepdims 形状 %v", ax0.Shape())
		}
		want := []int64{5, 7, 9}
		for i, v := range valsOf(t, ax0) {
			if v.(int64) != want[i] {
				t.Errorf("sum axis0[%d] = %d, want %d", i, v, want[i])
			}
		}
	})

	t.Run("未知算子与 LastErr", func(t *testing.T) {
		ClearErr()
		if LastErr() != "" {
			t.Fatalf("ClearErr 后 LastErr = %q", LastErr())
		}
		x, err := num.Zeros([]int{2}, num.DFloat64)
		if err != nil {
			t.Fatalf("Zeros:%v", err)
		}
		_, err = Op("foo", x, x)
		if err == nil || !strings.Contains(err.Error(), `未知算子 "foo"`) {
			t.Fatalf("Op 未知名错误:%v", err)
		}
		if LastErr() != err.Error() {
			t.Errorf("LastErr = %q, want %q", LastErr(), err.Error())
		}
		_, err = Unary("zzz", x)
		if err == nil || !strings.HasPrefix(err.Error(), "wasm: ") {
			t.Fatalf("Unary 未知名错误:%v", err)
		}
		if LastErr() != err.Error() {
			t.Errorf("LastErr = %q, want %q", LastErr(), err.Error())
		}
		_, err = Reduce("qqq", x, nil, false)
		if err == nil || !strings.HasPrefix(err.Error(), "wasm: ") {
			t.Fatalf("Reduce 未知名错误:%v", err)
		}
		if LastErr() != err.Error() {
			t.Errorf("LastErr = %q, want %q", LastErr(), err.Error())
		}
		_, err = FromBytes("i9", []int{1}, []byte{0})
		if err == nil {
			t.Fatal("FromBytes 非法 dtype 应报错")
		}
		if LastErr() != err.Error() {
			t.Errorf("LastErr = %q, want %q", LastErr(), err.Error())
		}
		ClearErr()
		if LastErr() != "" {
			t.Errorf("ClearErr 后 LastErr = %q", LastErr())
		}
		if !semverPrefix.MatchString(Version) {
			t.Errorf("Version = %q, 期望形如 0.1.5 或 0.1.5-3-gabc1234", Version)
		}
	})
}

// 新增算子映射与两个分发器(与核心库对齐)。
func TestExtendedOps(t *testing.T) {
	a, err := FromBytes("f8", []int{3}, f64Bytes(1, 4, 9))
	if err != nil {
		t.Fatal(err)
	}
	// 一元扩充
	for _, name := range []string{"tanh", "sinh", "cosh", "sinc", "sign", "fabs",
		"deg2rad", "rad2deg", "isnan", "isinf", "isfinite", "logical_not", "round1"} {
		if _, err := Unary(name, a); err != nil {
			t.Fatalf("unary %q: %v", name, err)
		}
	}
	// 二元扩充
	b, err := FromBytes("f8", []int{3}, f64Bytes(2, 2, 2))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"maximum", "minimum", "arctan2",
		"logical_and", "logical_or", "logical_xor"} {
		if _, err := Op(name, a, b); err != nil {
			t.Fatalf("op %q: %v", name, err)
		}
	}
	// 归约扩充
	for _, name := range []string{"prod", "median", "nansum", "nanmean", "nanvar",
		"nanstd", "nanmax", "nanmin", "nanmedian"} {
		if _, err := Reduce(name, a, nil, false); err != nil {
			t.Fatalf("reduce %q: %v", name, err)
		}
	}
	// 线代:dot / matmul / solve / det / inv / norm
	m, err := FromBytes("f8", []int{2, 2}, f64Bytes(1, 2, 3, 4))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"dot", "matmul", "inner", "outer"} {
		if _, err := Linalg(name, m, m); err != nil {
			t.Fatalf("linalg %q: %v", name, err)
		}
	}
	for _, name := range []string{"det", "inv", "eigvalsh", "svdvals", "norm_fro", "norm_2"} {
		if _, err := Linalg(name, m, nil); err != nil {
			t.Fatalf("linalg %q: %v", name, err)
		}
	}
	sol, err := Linalg("solve", m, mustArr(t, "f8", []int{2}, 1, 2))
	if err != nil {
		t.Fatal(err)
	}
	if sol == nil {
		t.Fatal("solve 返回 nil")
	}
	// 前缀累积
	for _, name := range []string{"cumsum", "cumprod"} {
		if _, err := Cumulative(name, a, nil); err != nil {
			t.Fatalf("cumulative %q: %v", name, err)
		}
		if _, err := Cumulative(name, a, []int{0}); err != nil {
			t.Fatalf("cumulative %q axis=0: %v", name, err)
		}
	}
	// 未知名字要报错且写入 LastErr
	if _, err := Linalg("bogus", m, nil); err == nil {
		t.Fatal("未知线代算子应报错")
	}
	if LastErr() == "" {
		t.Fatal("LastErr 应记录错误")
	}
	ClearErr()
	if _, err := Cumulative("bogus", a, nil); err == nil {
		t.Fatal("未知累积算子应报错")
	}
	ClearErr()
}

// mustArr 测试辅助:造指定 dtype/形状的数组。
func mustArr(t *testing.T, dt string, shape []int, vals ...float64) *num.Array {
	t.Helper()
	buf := make([]byte, 0, len(vals)*8)
	for _, v := range vals {
		buf = append(buf, f64Bytes(v)...)
	}
	a, err := FromBytes(dt, shape, buf)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// semverPrefix 匹配 semver 前缀(构建脚本可能附加 git describe 后缀)。
var semverPrefix = regexp.MustCompile(`^\d+\.\d+\.\d+`)

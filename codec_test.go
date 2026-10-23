package num

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// —— 编解码:reference(codec_vectors_test.go)+ 语义 ——

// f16 位型全表对拍:向量由 numpy 2.5.4 现场生成(含上溢/次正规/NaN/双舍入样本)。
func TestF16VectorRef(t *testing.T) {
	for i, c := range f32f16Cases {
		if got := f32ToF16(math.Float32frombits(c.f32)); got != c.f16 {
			t.Errorf("[%d] f32ToF16(%#08x) = %#04x, want %#04x", i, c.f32, got, c.f16)
		}
	}
	for i, c := range f64f16Cases {
		if got := f64ToF16(math.Float64frombits(c.f64)); got != c.f16 {
			t.Errorf("[%d] f64ToF16(%#016x) = %#04x, want %#04x", i, c.f64, got, c.f16)
		}
	}
	for i, c := range f16f32Cases {
		if got := math.Float32bits(f16ToF32(c.f16)); got != c.f32 {
			t.Errorf("[%d] f16ToF32(%#04x) = %#08x, want %#08x", i, c.f16, got, c.f32)
		}
	}
}

// f64 → f16 必须单次舍入(不经 f32):两个实测分歧样本 direct ≠ via-f32,
// 若实现被改成"先落 f32",该测试即红( 的回归锚)。
func TestF16SingleRoundingNotViaF32(t *testing.T) {
	cases := []struct {
		bits   uint64
		direct uint16 // numpy np.float16(f64) 实测
		viaF32 uint16 // np.float16(np.float32(f64)) 实测(双舍入歧路)
	}{
		{0xC06A4A0000AD8669, 0xDA93, 0xDA92}, // -210.31250032321557
		{0x40EDA5FFFF1C7FE5, 0x7B69, 0x7B6A}, // 60719.99989151935
	}
	for _, c := range cases {
		f := math.Float64frombits(c.bits)
		if got := f64ToF16(f); got != c.direct {
			t.Fatalf("f64ToF16(%#016x) = %#04x, want %#04x(numpy 直转)", c.bits, got, c.direct)
		}
		if c.direct == c.viaF32 {
			t.Fatalf("样本已失效(直转与经 f32 同值),需重挑")
		}
		if got := f32ToF16(float32(f)); got != c.viaF32 {
			t.Fatalf("f32ToF16(float32(%#016x)) = %#04x, want %#04x", c.bits, got, c.viaF32)
		}
	}
}

// Python str/repr 文本口径全表对拍(numpy 现场生成)。
func TestPyTextVectorRef(t *testing.T) {
	for i, c := range floatTextCases {
		f := math.Float64frombits(c.bits)
		if c.size == 32 {
			f = float64(math.Float32frombits(uint32(c.bits)))
		}
		if got := pyFloatStr(f, c.size); got != c.want {
			t.Errorf("[%d] pyFloatStr(%#x, %d) = %q, want %q", i, c.bits, c.size, got, c.want)
		}
	}
	for i, c := range complexTextCases {
		re, im := math.Float64frombits(c.re), math.Float64frombits(c.im)
		if c.size == 32 {
			re = float64(math.Float32frombits(uint32(c.re)))
			im = float64(math.Float32frombits(uint32(c.im)))
		}
		if got := pyComplexStr(re, im, c.size); got != c.want {
			t.Errorf("[%d] pyComplexStr(%#x, %#x, %d) = %q, want %q", i, c.re, c.im, c.size, got, c.want)
		}
	}
}

// 数值/复数/布尔写进 S/U:Python str 口径 + 按字节(S)/码点(U)截断(实测 numpy 2.5.4)。
func TestTextScalarEncode(t *testing.T) {
	cases := []struct {
		v    any
		dt   Dtype
		want string // 读回文本(截断/补零后的可见值)
	}{
		{"abcde", DBytes(4), "abcd"},               // S 按字节截断
		{"xy", DBytes(4), "xy"},                    // 尾部补 NUL
		{[]byte("ab"), DBytes(3), "ab"},            // []byte 入参
		{"abcde", DUnicode(3), "abc"},              // U 按码点截断
		{"\U0001F600x", DUnicode(1), "\U0001F600"}, // emoji 1 码点占 1 位
		{"\u00e9x", DUnicode(1), "\u00e9"},         // 非 ASCII 按码点
		{true, DBytes(6), "True"},                  // bool → Python str
		{false, DUnicode(6), "False"},
		{int64(-7), DBytes(4), "-7"},
		{uint64(18446744073709551615), DUnicode(21), "18446744073709551615"},
		{100.0, DBytes(8), "100.0"},                   // Python str(100.0)
		{float32(1.0 / 3), DBytes(12), "0.33333334"},  // 按输入标量精度(f4)
		{1.0 / 3, DUnicode(20), "0.3333333333333333"}, // f8 最短
		{complex(1, 2), DBytes(8), "(1+2j)"},          // Python complex repr
		{complex64(complex(0.1, -0.2)), DUnicode(14), "(0.1-0.2j)"},
	}
	for _, c := range cases {
		a, err := New([]int{1}, c.dt)
		if err != nil {
			t.Fatalf("New(%s): %v", c.dt, err)
		}
		// 先写长串再写目标值:余位必须整块清零(numpy 补 NUL,不残留旧字节)
		if err := a.SetItem([]int{0}, "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"); err != nil {
			t.Fatalf("SetItem(%s 预置长串): %v", c.dt, err)
		}
		if err := a.SetItem([]int{0}, c.v); err != nil {
			t.Fatalf("SetItem(%s ← %T): %v", c.dt, c.v, err)
		}
		got, err := a.Item(0)
		if err != nil {
			t.Fatalf("Item(%s): %v", c.dt, err)
		}
		if s, ok := got.(string); !ok || s != c.want {
			t.Errorf("%s ← %#v 读回 = %#v, want %q", c.dt, c.v, got, c.want)
		}
	}
}

// S/U 原始字节:截断/补齐/UTF-32LE 码元布局(十六进制值逐条来自 numpy 实测)。
func TestTextScalarEncodeRaw(t *testing.T) {
	cases := []struct {
		v    any
		dt   Dtype
		want string
	}{
		{"abcde", DBytes(4), "abcd"},
		{"xy", DBytes(4), "xy\x00\x00"},
		{int64(-7), DBytes(4), "-7\x00\x00"},
		{"\U0001F600", DUnicode(2), "\x00\xf6\x01\x00\x00\x00\x00\x00"},
		{"1", DUnicode(3), "1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
		{"\u00e9\u00e9\u00e9", DUnicode(2), "\xe9\x00\x00\x00\xe9\x00\x00\x00"},
	}
	for _, c := range cases {
		a, err := New([]int{1}, c.dt)
		if err != nil {
			t.Fatalf("New(%s): %v", c.dt, err)
		}
		if err := a.SetItem([]int{0}, c.v); err != nil {
			t.Fatalf("SetItem(%s): %v", c.dt, err)
		}
		if got := string(a.buf[:c.dt.Itemsize]); got != c.want {
			t.Errorf("%s ← %#v = % x, want % x", c.dt, c.v, got, c.want)
		}
	}
}

// S/U 读回:去尾零、内嵌 NUL 保留、UTF-32 解码、非法码点降级。
func TestTextDecode(t *testing.T) {
	s, _ := FromBytes([]byte("ab\x00\x00c\x00\x00\x00"), DBytes(4), []int{2})
	if v, _ := s.Item(0); v.(string) != "ab" {
		t.Errorf("S4[0] = %#v, want \"ab\"", v)
	}
	if v, _ := s.Item(1); v.(string) != "c" {
		t.Errorf("S4[1] = %#v, want \"c\"", v)
	}
	emb, _ := FromBytes([]byte("a\x00b\x00"), DBytes(4), []int{1})
	if v, _ := emb.Item(0); v.(string) != "a\x00b" {
		t.Errorf("S4 内嵌 NUL = %q, want %q", v, "a\x00b")
	}

	u, err := New([]int{1}, DUnicode(3))
	if err != nil {
		t.Fatalf("New(U3): %v", err)
	}
	if err := u.SetItem([]int{0}, "a\u0000b"); err != nil {
		t.Fatalf("SetItem(U3): %v", err)
	}
	if v, _ := u.Item(0); v.(string) != "a\u0000b" {
		t.Errorf("U3 内嵌 NUL = %q, want %q", v, "a\u0000b")
	}
	if err := u.SetItem([]int{0}, "\U0001F600"); err != nil {
		t.Fatalf("SetItem(U3 emoji): %v", err)
	}
	if v, _ := u.Item(0); v.(string) != "\U0001F600" {
		t.Errorf("U3 emoji = %q", v)
	}
	// 非法码点(代理对)读出降级 U+FFFD,不 panic
	bad, _ := FromBytes([]byte{0x00, 0xd8, 0x00, 0x00}, DUnicode(1), []int{1})
	if v, err := bad.Item(0); err != nil || v.(string) != "\uFFFD" {
		t.Errorf("U 非法码点 = %#v err=%v, want U+FFFD", v, err)
	}
	// 不支持写文本的类型 → ErrDtype
	z, _ := New([]int{1}, DBytes(4))
	if err := z.SetItem([]int{0}, struct{}{}); !errors.Is(err, ErrDtype) {
		t.Errorf("S4 ← struct{} err = %v, want ErrDtype", err)
	}
}

// f16 → 文本:位型 reference(numpy 抽样锚;规则本身已在 Python 侧穷举 65536 位型验证)。
// 注意 0x2400(= 0.015625)是 tie 样本:numpy 取 "0.01563"(远离零),RNE 会给 "0.015625"。
func TestF16TextRef(t *testing.T) {
	for i, c := range f16TextCases {
		if got := pyFloatStr(float64(f16ToF32(c.f16)), 16); got != c.want {
			t.Errorf("[%d] f16 %#04x → %q, want %q", i, c.f16, got, c.want)
		}
	}
}

// f16 数组 → S/U(Cast):文本按 f16 精度渲染;同一值按 f32 精度会多出位数。
func TestCastF16ToText(t *testing.T) {
	a, err := New([]int{3}, DFloat16)
	if err != nil {
		t.Fatalf("New(f2): %v", err)
	}
	for i, v := range []float64{0.1, 0.015625, 12345} {
		if err := a.SetItem([]int{i}, v); err != nil {
			t.Fatalf("SetItem(f2 ← %v): %v", v, err)
		}
	}
	s, err := Cast(a, DBytes(12))
	if err != nil {
		t.Fatalf("Cast f2→S12: %v", err)
	}
	want := []string{"0.1", "0.01563", "1.2344e+04"} // numpy: np.array([...],dtype="f2").astype("U12")
	for i, w := range want {
		if v, _ := s.Item(i); v.(string) != w {
			t.Errorf("f2→S12[%d] = %#v, want %q", i, v, w)
		}
	}
	// 同值按 f32 精度落字(SetItem 的输入标量精度口径)与 f16 输出不同
	u, err := Cast(a, DUnicode(12))
	if err != nil {
		t.Fatalf("Cast f2→U12: %v", err)
	}
	u0, _ := u.Item(0)
	f2v, _ := a.Item(0)
	f32Text := pyFloatStr(float64(f2v.(float32)), 32)
	if u0.(string) != "0.1" || f32Text == "0.1" {
		t.Errorf("f16 精度文本 = %#v(f32 精度会是 %q)", u0, f32Text)
	}
}

// :数值/S → S/U(same_kind 实测为真)的文本结果;U→S 与 M8→文本仍拒。
func TestCastNumericToText(t *testing.T) {
	cases := []struct {
		build func() (*Array, error)
		to    Dtype
		want  string
	}{
		{func() (*Array, error) { return FromSlice([]int64{123}) }, DBytes(8), "123"},
		{func() (*Array, error) { return FromSlice([]bool{true}) }, DBytes(8), "True"},
		{func() (*Array, error) { return FromSlice([]float64{0.1}) }, DBytes(8), "0.1"},
		{func() (*Array, error) { return FromSlice([]float32{1.0 / 3}) }, DBytes(16), "0.33333334"},
		{func() (*Array, error) { return FromSlice([]complex128{complex(1, 2)}) }, DBytes(8), "(1+2j)"},
		{func() (*Array, error) { return FromSlice([]float64{1e16}) }, DBytes(8), "1e+16"},
		{func() (*Array, error) { return FromSlice([]uint8{7}) }, DUnicode(4), "7"},
	}
	for _, c := range cases {
		src, err := c.build()
		if err != nil {
			t.Fatalf("建源数组: %v", err)
		}
		got, err := Cast(src, c.to)
		if err != nil {
			t.Fatalf("Cast %s → %s: %v", src.Dtype(), c.to, err)
		}
		if v, _ := got.Item(0); v.(string) != c.want {
			t.Errorf("Cast %s → %s [0] = %#v, want %q", src.Dtype(), c.to, v, c.want)
		}
	}
	// S/U 互转(S→U 允、U→S 拒)与 M8 → 文本拒
	sb, err := New([]int{1}, DBytes(3))
	if err != nil {
		t.Fatal(err)
	}
	if err := sb.SetItem([]int{0}, "abc"); err != nil {
		t.Fatal(err)
	}
	su, err := Cast(sb, DUnicode(5))
	if err != nil {
		t.Fatalf("Cast S3 → U5: %v", err)
	}
	if v, _ := su.Item(0); v.(string) != "abc" {
		t.Errorf("S3 → U5 = %#v, want \"abc\"", v)
	}
	if _, err := Cast(su, DBytes(5)); !errors.Is(err, ErrDtype) {
		t.Errorf("U → S 应 ErrDtype(numpy same_kind 判否),得 %v", err)
	}
	m8, err := New([]int{1}, DTime("us"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Cast(m8, DBytes(20)); !errors.Is(err, ErrDtype) {
		t.Errorf("M8 → S 应 ErrDtype(numpy 仅 unsafe),得 %v", err)
	}
}

// f16 数组写入 S/U 视图(赋值 = numpy unsafe 口径):文本同样按 f16 精度。
func TestSetSubF16ToText(t *testing.T) {
	src, err := New([]int{2}, DFloat16)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []float64{0.1, 12345} {
		if err := src.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	dst, err := New([]int{4}, DBytes(12))
	if err != nil {
		t.Fatal(err)
	}
	// numpy: a = np.zeros(4, dtype="S12"); a[:2] = np.array([0.1, 12345.0], dtype="f2")
	view, err := dst.Sub(Slice(0, 2))
	if err != nil {
		t.Fatalf("Sub(Slice(0,2)): %v", err)
	}
	if err := view.SetSub(src); err != nil {
		t.Fatalf("SetSub(f2 → S12 视图): %v", err)
	}
	for i, want := range []string{"0.1", "1.2344e+04"} {
		if v, _ := dst.Item(i); v.(string) != want {
			t.Errorf("S12[%d] = %#v, want %q(f16 精度落字)", i, v, want)
		}
	}
}

// f16 存取:f16 ← 各数值类型(单次舍入)、Item 升 float32。
func TestF16Access(t *testing.T) {
	a, err := New([]int{6}, DFloat16)
	if err != nil {
		t.Fatalf("New(f2): %v", err)
	}
	vals := []any{float32(0.1), float64(0.1), int64(1), true, float32(65519), float64(65520)}
	want := []uint16{f32ToF16(0.1), f64ToF16(0.1), 0x3C00, 0x3C00, 0x7BFF, 0x7C00}
	for i, v := range vals {
		if err := a.SetItem([]int{i}, v); err != nil {
			t.Fatalf("SetItem(f2 ← %T): %v", v, err)
		}
		if got := binary.LittleEndian.Uint16(a.buf[i*2:]); got != want[i] {
			t.Errorf("f2 ← %T 位型 = %#04x, want %#04x", v, got, want[i])
		}
		it, _ := a.Item(i)
		if it.(float32) != f16ToF32(want[i]) {
			t.Errorf("f2[%d] = %v, want %v", i, it, f16ToF32(want[i]))
		}
	}
	if err := a.SetItem([]int{0}, "x"); !errors.Is(err, ErrDtype) {
		t.Errorf("f2 ← string err = %v, want ErrDtype", err)
	}
	if err := a.SetItem([]int{0}, complex(1, 2)); !errors.Is(err, ErrDtype) {
		t.Errorf("f2 ← complex err = %v, want ErrDtype", err)
	}
}

// complex 存取:实数值 imag=0、c8 收窄、读出类型、数值 kind 拒复数。
func TestComplexAccess(t *testing.T) {
	c16, _ := New([]int{1}, DComplex128)
	if err := c16.SetItem([]int{0}, complex(1.5, -2.5)); err != nil {
		t.Fatalf("SetItem(c16): %v", err)
	}
	if v, _ := c16.Item(0); v.(complex128) != complex(1.5, -2.5) {
		t.Errorf("c16 = %v", v)
	}
	if err := c16.SetItem([]int{0}, int64(5)); err != nil { // 实数 → imag 0
		t.Fatalf("SetItem(c16 ← int64): %v", err)
	}
	if v, _ := c16.Item(0); v.(complex128) != complex(5, 0) {
		t.Errorf("c16 ← 5 = %v, want (5+0i)", v)
	}
	c8, _ := New([]int{1}, DComplex64)
	if err := c8.SetItem([]int{0}, complex(1.0/3, -2.0/3)); err != nil {
		t.Fatalf("SetItem(c8): %v", err)
	}
	want := complex(float32(1.0/3), float32(-2.0/3))
	if v, _ := c8.Item(0); v.(complex64) != want {
		t.Errorf("c8 收窄 = %v, want %v", v, want)
	}
	if err := c8.SetItem([]int{0}, float32(1.5)); err != nil { // 浮点 → imag 0
		t.Fatalf("SetItem(c8 ← float32): %v", err)
	}
	if v, _ := c8.Item(0); v.(complex64) != 1.5 {
		t.Errorf("c8 ← 1.5 = %v", v)
	}
	f, _ := New([]int{1}, DFloat64)
	if err := f.SetItem([]int{0}, complex(1, 2)); !errors.Is(err, ErrDtype) {
		t.Errorf("f8 ← complex err = %v, want ErrDtype", err)
	}
	i, _ := New([]int{1}, DInt64)
	if err := i.SetItem([]int{0}, complex(1, 2)); !errors.Is(err, ErrDtype) {
		t.Errorf("i8 ← complex err = %v, want ErrDtype", err)
	}
}

// datetime64 存取:整值 ↔ int64 计数、NaT=MinInt64、浮点/文本拒绝。
func TestDateTimeAccess(t *testing.T) {
	a, _ := New([]int{3}, DTime("s"))
	for i, v := range []any{int64(5), 7, math.MinInt64} {
		if err := a.SetItem([]int{i}, v); err != nil {
			t.Fatalf("SetItem(M8 ← %T): %v", v, err)
		}
	}
	for i, w := range []int64{5, 7, math.MinInt64} {
		if v, _ := a.Item(i); v.(int64) != w {
			t.Errorf("M8[%d] = %v, want %d", i, v, w)
		}
	}
	for _, bad := range []any{1.5, float32(2), "2020-01-01"} {
		if err := a.SetItem([]int{0}, bad); !errors.Is(err, ErrDtype) {
			t.Errorf("M8 ← %#v err = %v, want ErrDtype", bad, err)
		}
	}
}

// Ones/Full 全 dtype(numpy 实测口径:S/U = "1"、complex = 1+0j、M8 = 1、f2 = 0x3C00)。
func TestOnesFullAllDtypes(t *testing.T) {
	cases := []struct {
		dt   Dtype
		want string
	}{
		{DBool, "\x01"},
		{DInt16, "\x01\x00"},
		{DUint64, "\x01\x00\x00\x00\x00\x00\x00\x00"},
		{DFloat16, "\x00\x3c"},
		{DFloat32, "\x00\x00\x80\x3f"},
		{DComplex64, "\x00\x00\x80\x3f\x00\x00\x00\x00"},
		{DComplex128, "\x00\x00\x00\x00\x00\x00\xf0\x3f\x00\x00\x00\x00\x00\x00\x00\x00"},
		{DTime("s"), "\x01\x00\x00\x00\x00\x00\x00\x00"},
		{DBytes(4), "1\x00\x00\x00"},
		{DUnicode(3), "1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
	}
	for _, c := range cases {
		a, err := Ones([]int{2}, c.dt)
		if err != nil {
			t.Fatalf("Ones(%s): %v", c.dt, err)
		}
		for i := 0; i < 2; i++ {
			if got := string(a.buf[i*c.dt.Itemsize : (i+1)*c.dt.Itemsize]); got != c.want {
				t.Errorf("Ones(%s)[%d] = % x, want % x", c.dt, i, got, c.want)
			}
		}
	}
	f, err := Full([]int{2}, 0.1, DFloat16)
	if err != nil {
		t.Fatalf("Full(f2, 0.1): %v", err)
	}
	if got := binary.LittleEndian.Uint16(f.buf); got != f64ToF16(0.1) {
		t.Errorf("Full(f2, 0.1) = %#04x, want %#04x", got, f64ToF16(0.1))
	}
	u, err := Full([]int{1}, "ab", DUnicode(4))
	if err != nil {
		t.Fatalf("Full(U4, \"ab\"): %v", err)
	}
	if v, _ := u.Item(0); v.(string) != "ab" {
		t.Errorf("Full(U4) = %#v", v)
	}
	c, err := Full([]int{1}, complex(1, 2), DComplex64)
	if err != nil {
		t.Fatalf("Full(c8, 1+2i): %v", err)
	}
	if v, _ := c.Item(0); v.(complex64) != complex(float32(1), float32(2)) {
		t.Errorf("Full(c8) = %#v", v)
	}
	if _, err := Full([]int{1}, "x", DFloat64); !errors.Is(err, ErrDtype) {
		t.Errorf("Full(f8, \"x\") err = %v, want ErrDtype", err)
	}
}

// Cast 覆盖新 dtype:f8→f2→f8、数值→c16→c8、S 宽度、M8 单位守卫、promote 宽度。
func TestCastNewDtypes(t *testing.T) {
	f8, _ := FromSlice([]float64{0.1, 65519, 6e-8, math.Inf(1), math.NaN()})
	f2, err := Cast(f8, DFloat16)
	if err != nil {
		t.Fatalf("Cast f8→f2: %v", err)
	}
	for i := 0; i < f8.Size(); i++ {
		v, _ := f8.Item(i)
		got, _ := f2.Item(i)
		want := f16ToF32(f64ToF16(v.(float64)))
		if math.Float32bits(got.(float32)) != math.Float32bits(want) {
			t.Errorf("Cast f8→f2[%d] = %#08x, want %#08x", i, math.Float32bits(got.(float32)), math.Float32bits(want))
		}
	}
	back, err := Cast(f2, DFloat64)
	if err != nil {
		t.Fatalf("Cast f2→f8: %v", err)
	}
	for i := 0; i < f2.Size(); i++ {
		a, _ := f2.Item(i)
		b, _ := back.Item(i)
		if math.Float32bits(float32(b.(float64))) != math.Float32bits(a.(float32)) {
			t.Errorf("f2→f8 往返[%d]: %v → %v", i, a, b)
		}
	}
	b, err := Cast(f8, DFloat16) // bool/f8→f2 之外的 same_kind 组合
	if err != nil {
		t.Fatalf("Cast f8→f2(第二遍): %v", err)
	}
	if b.Dtype() != DFloat16 {
		t.Errorf("Cast f8→f2 dtype = %s", b.Dtype())
	}

	i32, _ := FromSlice([]int32{3, -4})
	c16, err := Cast(i32, DComplex128)
	if err != nil {
		t.Fatalf("Cast i4→c16: %v", err)
	}
	if v, _ := c16.Item(1); v.(complex128) != complex(-4, 0) {
		t.Errorf("i4→c16[1] = %v", v)
	}
	c8, err := Cast(c16, DComplex64)
	if err != nil {
		t.Fatalf("Cast c16→c8: %v", err)
	}
	if v, _ := c8.Item(0); v.(complex64) != 3 {
		t.Errorf("c16→c8[0] = %v", v)
	}
	// f8 → c16 允许;c16 → f8 拒绝(same_kind)
	if _, err := Cast(f8, DComplex128); err != nil {
		t.Errorf("Cast f8→c16: %v", err)
	}
	if _, err := Cast(c16, DFloat64); !errors.Is(err, ErrDtype) {
		t.Errorf("Cast c16→f8 err = %v, want ErrDtype", err)
	}
	// f8 → f2 → b 拒绝(f→i/u/b 否)
	f2b, _ := Cast(f8, DFloat16)
	if _, err := Cast(f2b, DBool); !errors.Is(err, ErrDtype) {
		t.Errorf("Cast f2→b err = %v, want ErrDtype", err)
	}
	// b → f2 允许
	bv, _ := FromSlice([]bool{true, false})
	if _, err := Cast(bv, DFloat16); err != nil {
		t.Errorf("Cast b→f2: %v", err)
	}

	s, _ := New([]int{2}, DBytes(4))
	for i, v := range []string{"abcdef", "x"} {
		if err := s.SetItem([]int{i}, v); err != nil {
			t.Fatalf("SetItem(S4): %v", err)
		}
	}
	s2, err := Cast(s, DBytes(2))
	if err != nil {
		t.Fatalf("Cast S4→S2: %v", err)
	}
	if v, _ := s2.Item(0); v.(string) != "ab" {
		t.Errorf("S4→S2[0] = %#v, want \"ab\"", v)
	}
	u, _ := New([]int{1}, DUnicode(2))
	if err := u.SetItem([]int{0}, "abcd"); err != nil {
		t.Fatalf("SetItem(U2): %v", err)
	}
	u1, err := Cast(u, DUnicode(1))
	if err != nil {
		t.Fatalf("Cast U2→U1: %v", err)
	}
	if v, _ := u1.Item(0); v.(string) != "a" {
		t.Errorf("U2→U1 = %#v, want \"a\"", v)
	}
	if _, err := Cast(u, DBytes(8)); !errors.Is(err, ErrDtype) {
		t.Errorf("Cast U→S err = %v, want ErrDtype", err)
	}

	m, _ := New([]int{1}, DTime("s"))
	if err := m.SetItem([]int{0}, int64(3)); err != nil {
		t.Fatalf("SetItem(M8[s]): %v", err)
	}
	if _, err := Cast(m, DTime("ms")); !errors.Is(err, ErrDtype) {
		t.Errorf("Cast M8[s]→M8[ms] err = %v, want ErrDtype(单位换算未实现)", err)
	}
	if _, err := Cast(m, DTime("s")); err != nil {
		t.Errorf("Cast M8[s]→M8[s]: %v", err)
	}
	// promote:同 kind 定宽字符串取最大宽度(numpy 口径)
	if got, err := promoteTypes(DBytes(3), DBytes(5)); err != nil || got != DBytes(5) {
		t.Errorf("promote(S3,S5) = %v err=%v, want |S5", got, err)
	}
	if got, err := promoteTypes(DUnicode(2), DUnicode(7)); err != nil || got != DUnicode(7) {
		t.Errorf("promote(U2,U7) = %v err=%v, want <U7", got, err)
	}
	if got, err := promoteTypes(DUnicode(2), DBytes(4)); err == nil {
		t.Errorf("promote(U2,S4) = %v, want ErrDtype", got)
	}
}

// 全 dtype Item/SetItem 可达(– 后不再有 ErrDtype 空档)。
func TestCodecAllDtypesReachable(t *testing.T) {
	vals := map[Kind]any{
		Bool: true, Int8: int8(1), Int16: int16(2), Int32: int32(3), Int64: int64(4),
		Uint8: uint8(5), Uint16: uint16(6), Uint32: uint32(7), Uint64: uint64(8),
		Float16: float32(1.5), Float32: float32(1.5), Float64: 1.5,
		Complex64: complex64(1 + 2i), Complex128: complex(1, 2),
		Datetime64: int64(9), Bytes: "s", Unicode: "u",
	}
	for _, dt := range []Dtype{DBool, DInt8, DInt16, DInt32, DInt64, DUint8, DUint16, DUint32,
		DUint64, DFloat16, DFloat32, DFloat64, DComplex64, DComplex128, DTime("us"),
		DBytes(4), DUnicode(2)} {
		a, err := New([]int{2}, dt)
		if err != nil {
			t.Fatalf("New(%s): %v", dt, err)
		}
		if err := a.SetItem([]int{0}, vals[dt.Kind]); err != nil {
			t.Fatalf("SetItem(%s ← %T): %v", dt, vals[dt.Kind], err)
		}
		if v, err := a.Item(0); err != nil || v == nil {
			t.Fatalf("%s Item = %#v err=%v", dt, v, err)
		}
		if _, err := a.Item(1); err != nil { // 未写入元素 = 零值,可读
			t.Fatalf("%s Item(1): %v", dt, err)
		}
	}
}

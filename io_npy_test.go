package num

import (
	"encoding/binary"
	"errors"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// npy 读写单测。真 numpy 2.5.4 产出的字节在 io_npy_fixtures_test.go
// (gen_npy_go.py 生成),这里只做无 python 依赖的装载/往返验证。

func mustLoadBytes(t *testing.T, b []byte) *Array {
	t.Helper()
	a, err := LoadNPYBytes(b)
	if err != nil {
		t.Fatalf("LoadNPYBytes: %v", err)
	}
	return a
}

func wantShapeEq(t *testing.T, a *Array, want ...int) {
	t.Helper()
	if !shapeEqual(a.Shape(), want) {
		t.Fatalf("shape = %v, want %v", a.Shape(), want)
	}
}

func TestLoadNPYNumpyC64(t *testing.T) {
	a := mustLoadBytes(t, npyCF64)
	wantShapeEq(t, a, 2, 3)
	if a.Dtype() != DFloat64 {
		t.Fatalf("dtype = %s", a.Dtype())
	}
	if !a.isCContig() {
		t.Fatalf("C 序文件应装载为 C 连续")
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v, _ := a.Item(i, j)
			if v.(float64) != float64(i*3+j) {
				t.Fatalf("[%d,%d] = %v, want %d", i, j, v, i*3+j)
			}
		}
	}
}

// F 序装载:F 步长零拷贝视图,值语义与 C 序一致。
func TestLoadNPYNumpyFortranOrder(t *testing.T) {
	a := mustLoadBytes(t, npyFF64)
	wantShapeEq(t, a, 2, 3)
	if a.isCContig() {
		t.Fatalf("F 序文件不应判为 C 连续")
	}
	if got := a.st; got[0] != 8 || got[1] != 16 {
		t.Fatalf("F 步长 = %v, want [8 16]", got)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			v, _ := a.Item(i, j)
			if v.(float64) != float64(i*3+j) {
				t.Fatalf("[%d,%d] = %v, want %d", i, j, v, i*3+j)
			}
		}
	}
}

// 大端翻转:>i2 文件装载为小端内部表示。
func TestLoadNPYNumpyBigEndian(t *testing.T) {
	a := mustLoadBytes(t, npyBEI2)
	wantShapeEq(t, a, 3)
	if a.Dtype() != DInt16 {
		t.Fatalf("dtype = %s", a.Dtype())
	}
	for i, w := range []int16{1, 2, 3} {
		v, _ := a.Item(i)
		if v.(int16) != w {
			t.Fatalf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestLoadNPYNumpyF32Bool(t *testing.T) {
	a := mustLoadBytes(t, npyF4)
	wantShapeEq(t, a, 1, 2)
	if a.Dtype() != DFloat32 {
		t.Fatalf("dtype = %s", a.Dtype())
	}
	v0, _ := a.Item(0, 0)
	v1, _ := a.Item(0, 1)
	if v0.(float32) != 1.5 || v1.(float32) != 2.5 {
		t.Fatalf("f4 = %v %v", v0, v1)
	}

	b := mustLoadBytes(t, npyBool)
	wantShapeEq(t, b, 3)
	if b.Dtype() != DBool {
		t.Fatalf("dtype = %s", b.Dtype())
	}
	for i, w := range []bool{true, false, true} {
		v, _ := b.Item(i)
		if v.(bool) != w {
			t.Fatalf("bool[%d] = %v, want %v", i, v, w)
		}
	}
}

// v2.0 头(4 字节头长)。
func TestLoadNPYNumpyVersion2(t *testing.T) {
	a := mustLoadBytes(t, npyV2i4)
	if npyV2i4[6] != 2 {
		t.Fatalf("fixture 版本 = %d, want 2", npyV2i4[6])
	}
	wantShapeEq(t, a, 3)
	if a.Dtype() != DInt32 {
		t.Fatalf("dtype = %s", a.Dtype())
	}
	for i, w := range []int32{0, 1, 2} {
		v, _ := a.Item(i)
		if v.(int32) != w {
			t.Fatalf("[%d] = %v, want %v", i, v, w)
		}
	}
}

// 非数值 dtype:S/U/M8/c16 的编解码(– 到位后读回值逐条对拍)。
func TestLoadNPYNumpyStringTimeComplex(t *testing.T) {
	s := mustLoadBytes(t, npyS4)
	wantShapeEq(t, s, 2)
	if s.Dtype() != DBytes(4) {
		t.Fatalf("S4 dtype = %s", s.Dtype())
	}
	if got := string(s.buf[:8]); got != "xy\x00\x00z\x00\x00\x00" {
		t.Fatalf("S4 raw = %q", got)
	}
	for i, w := range []string{"xy", "z"} {
		if v, err := s.Item(i); err != nil || v.(string) != w {
			t.Fatalf("S4[%d] = %#v err=%v, want %q", i, v, err, w)
		}
	}

	u := mustLoadBytes(t, npyU3)
	wantShapeEq(t, u, 2)
	if u.Dtype() != DUnicode(3) {
		t.Fatalf("U3 dtype = %s (itemsize %d)", u.Dtype(), u.Dtype().Itemsize)
	}
	// U 为 UTF-32LE 码元:"ab" → a,0,0,0,b,0,0,0;第二元素 "c" → c,0,0,0,0,0,0,0
	if got := u.buf[:8]; string(got) != "a\x00\x00\x00b\x00\x00\x00" {
		t.Fatalf("U3 raw = %q", string(got))
	}
	if got := u.buf[12:16]; string(got) != "c\x00\x00\x00" {
		t.Fatalf("U3 元素[1] raw = %q", string(got))
	}
	for i, w := range []string{"ab", "c"} {
		if v, err := u.Item(i); err != nil || v.(string) != w {
			t.Fatalf("U3[%d] = %#v err=%v, want %q", i, v, err, w)
		}
	}

	m := mustLoadBytes(t, npyM8D)
	wantShapeEq(t, m, 2)
	if m.Dtype().Kind != Datetime64 || m.Dtype().Unit != "D" {
		t.Fatalf("M8[D] dtype = %s", m.Dtype())
	}
	if binary.LittleEndian.Uint64(m.buf[0:8]) != 1 || binary.LittleEndian.Uint64(m.buf[8:16]) != 2 {
		t.Fatalf("M8[D] raw = %v %v", binary.LittleEndian.Uint64(m.buf[0:8]), binary.LittleEndian.Uint64(m.buf[8:16]))
	}
	for i, w := range []int64{1, 2} {
		if v, err := m.Item(i); err != nil || v.(int64) != w {
			t.Fatalf("M8[D][%d] = %#v err=%v, want %d", i, v, err, w)
		}
	}

	c := mustLoadBytes(t, npyC16)
	wantShapeEq(t, c, 1)
	if c.Dtype() != DComplex128 {
		t.Fatalf("c16 dtype = %s", c.Dtype())
	}
	re := math.Float64frombits(binary.LittleEndian.Uint64(c.buf[0:8]))
	im := math.Float64frombits(binary.LittleEndian.Uint64(c.buf[8:16]))
	if re != 1 || im != 2 {
		t.Fatalf("c16 raw = %v + %vi, want 1+2i", re, im)
	}
	if v, err := c.Item(0); err != nil || v.(complex128) != complex(1, 2) {
		t.Fatalf("c16[0] = %#v err=%v, want 1+2i", v, err)
	}
}

// f16/c8 fixture(真实 numpy 字节):f16 读回升 float32 位型、c8 读回 complex64。
func TestLoadNPYNumpyHalffloatComplex64(t *testing.T) {
	f := mustLoadBytes(t, npyF2)
	wantShapeEq(t, f, 10)
	if f.Dtype() != DFloat16 {
		t.Fatalf("f2 dtype = %s", f.Dtype())
	}
	for i, w := range npyF2F32Bits {
		v, err := f.Item(i)
		if err != nil {
			t.Fatalf("f2[%d]: %v", i, err)
		}
		if got := math.Float32bits(v.(float32)); got != w {
			t.Fatalf("f2[%d] = %#08x, want %#08x(numpy f2→f4)", i, got, w)
		}
	}

	c := mustLoadBytes(t, npyC8)
	wantShapeEq(t, c, 3)
	if c.Dtype() != DComplex64 {
		t.Fatalf("c8 dtype = %s", c.Dtype())
	}
	want := []complex64{complex(float32(1.5), float32(-0.5)),
		complex(float32(0.1), float32(0.2)), complex(float32(-3), float32(4))}
	for i, w := range want {
		v, err := c.Item(i)
		if err != nil {
			t.Fatalf("c8[%d]: %v", i, err)
		}
		if v.(complex64) != w {
			t.Fatalf("c8[%d] = %v, want %v", i, v, w)
		}
	}
}

// 大端宽类型:>f4 元素翻转、>U2 按 4 字节码元翻转(不是整元素反转)。
func TestLoadNPYNumpyBigEndianWide(t *testing.T) {
	f := mustLoadBytes(t, npyBEF4)
	wantShapeEq(t, f, 2)
	if f.Dtype() != DFloat32 {
		t.Fatalf(">f4 dtype = %s", f.Dtype())
	}
	for i, w := range []float32{1, 2} {
		v, _ := f.Item(i)
		if v.(float32) != w {
			t.Fatalf(">f4[%d] = %v, want %v", i, v, w)
		}
	}

	u := mustLoadBytes(t, npyBEU2)
	wantShapeEq(t, u, 1)
	if u.Dtype() != DUnicode(2) {
		t.Fatalf(">U2 dtype = %s", u.Dtype())
	}
	// 大端码元 00 00 00 61 翻转后应回到小端 61 00 00 00
	if got := u.buf[:8]; string(got) != "a\x00\x00\x00b\x00\x00\x00" {
		t.Fatalf(">U2 raw = %q(码元级翻转有误)", string(got))
	}
}

// 全 dtype 的 SaveNPYBytes 头生成 + 读回(dtype/形状;数据为 0)。
func TestSaveNPYAllDtypesHeader(t *testing.T) {
	dts := []Dtype{DBool, DInt8, DInt16, DInt32, DInt64,
		DUint8, DUint16, DUint32, DUint64, DFloat16, DFloat32, DFloat64,
		DComplex64, DComplex128, DTime("us"), DTime("D"), DBytes(5), DUnicode(3)}
	for _, dt := range dts {
		a, err := New([]int{2, 3}, dt)
		if err != nil {
			t.Fatalf("New(%s): %v", dt, err)
		}
		b, err := SaveNPYBytes(a)
		if err != nil {
			t.Fatalf("SaveNPYBytes(%s): %v", dt, err)
		}
		back, err := LoadNPYBytes(b)
		if err != nil {
			t.Fatalf("LoadNPYBytes(%s): %v", dt, err)
		}
		if back.Dtype() != dt {
			t.Fatalf("round trip dtype = %s, want %s", back.Dtype(), dt)
		}
		wantShapeEq(t, back, 2, 3)
	}
}

func TestSaveNPYWriteError(t *testing.T) {
	dir := t.TempDir()
	a, err := Arange(0, 3, 1, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	// 目标是已存在的目录 → 写失败
	if err := SaveNPY(dir, a); err == nil {
		t.Fatalf("写到目录未报错")
	}
	if _, err := LoadNPY(filepath.Join(dir, "nope.npy")); err == nil {
		t.Fatalf("读缺失文件未报错")
	}
}

// TestLoadNPYErrors 已覆盖头/数据错误;这里补"魔数合法但头字段越界"一例。
func TestLoadNPYHeaderOverrun(t *testing.T) {
	bad := append([]byte(nil), npyCF64...)
	binary.LittleEndian.PutUint16(bad[8:10], 0xFFFF) // 头长 65535 > 文件
	if _, err := LoadNPYBytes(bad); !errors.Is(err, ErrShape) {
		t.Fatalf("头长超界 err = %v, want ErrShape", err)
	}
}

// 写入 → 读回:全 dtype(数值 + f2/c/M8/S/U)、0 维、转置(非连续→写盘前连续化)。
func TestSaveNPYRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dts := []Dtype{DBool, DInt8, DInt16, DInt32, DInt64,
		DUint8, DUint16, DUint32, DUint64, DFloat16, DFloat32, DFloat64,
		DComplex64, DComplex128, DTime("us"), DBytes(5), DUnicode(3)}
	for _, dt := range dts {
		a, err := New([]int{2, 3}, dt)
		if err != nil {
			t.Fatalf("New(%s): %v", dt, err)
		}
		for i := 0; i < 2; i++ {
			for j := 0; j < 3; j++ {
				if err := a.SetItem([]int{i, j}, int64(i*3+j)); err != nil {
					t.Fatalf("SetItem(%s): %v", dt, err)
				}
			}
		}
		// S/U 的 dtype 名含 "|"/"<",不能直接做文件名
		name := strings.NewReplacer("|", "S", "<", "U", ">", "B").Replace(dt.String())
		path := filepath.Join(dir, name+".npy")
		if err := SaveNPY(path, a); err != nil {
			t.Fatalf("SaveNPY(%s): %v", dt, err)
		}
		b, err := LoadNPY(path)
		if err != nil {
			t.Fatalf("LoadNPY(%s): %v", dt, err)
		}
		if b.Dtype() != dt {
			t.Fatalf("round trip dtype = %s, want %s", b.Dtype(), dt)
		}
		wantShapeEq(t, b, 2, 3)
		for i := 0; i < 2; i++ {
			for j := 0; j < 3; j++ {
				va, _ := a.Item(i, j)
				vb, _ := b.Item(i, j)
				if va != vb {
					t.Fatalf("%s [%d,%d]: %v → %v", dt, i, j, va, vb)
				}
			}
		}
	}
}

func TestSaveNPYZeroDimAndNonContig(t *testing.T) {
	dir := t.TempDir()

	// 0 维(shape)
	z, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatalf("New 0-d: %v", err)
	}
	if err := z.SetItem([]int{}, 2.5); err != nil {
		t.Fatalf("SetItem 0-d: %v", err)
	}
	pz := filepath.Join(dir, "z.npy")
	if err := SaveNPY(pz, z); err != nil {
		t.Fatalf("SaveNPY 0-d: %v", err)
	}
	zb, err := LoadNPY(pz)
	if err != nil {
		t.Fatalf("LoadNPY 0-d: %v", err)
	}
	if zb.NDim() != 0 || zb.Size() != 1 {
		t.Fatalf("0-d round trip: ndim=%d size=%d", zb.NDim(), zb.Size())
	}
	if v, _ := zb.Item(); v.(float64) != 2.5 {
		t.Fatalf("0-d value = %v", v)
	}

	// 转置视图(非 C 连续)→ 写盘连续化,值语义保持
	a, err := New([]int{2, 3}, DFloat64)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			_ = a.SetItem([]int{i, j}, int64(i*10+j))
		}
	}
	tr, err := a.Transpose()
	if err != nil {
		t.Fatalf("Transpose: %v", err)
	}
	pt := filepath.Join(dir, "t.npy")
	if err := SaveNPY(pt, tr); err != nil {
		t.Fatalf("SaveNPY transposed: %v", err)
	}
	tb, err := LoadNPY(pt)
	if err != nil {
		t.Fatalf("LoadNPY transposed: %v", err)
	}
	wantShapeEq(t, tb, 3, 2)
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			v, _ := tb.Item(i, j)
			if v.(float64) != float64(j*10+i) {
				t.Fatalf("transposed [%d,%d] = %v, want %v", i, j, v, float64(j*10+i))
			}
		}
	}
}

// 写头合规:v1、总长 64 字节对齐(numpy 同)。
func TestSaveNPYBytesHeader(t *testing.T) {
	a, err := Arange(0, 3, 1, DFloat64)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	b, err := SaveNPYBytes(a)
	if err != nil {
		t.Fatalf("SaveNPYBytes: %v", err)
	}
	if string(b[:6]) != "\x93NUMPY" {
		t.Fatalf("magic = %q", b[:6])
	}
	if b[6] != 1 || b[7] != 0 {
		t.Fatalf("version = %d.%d, want 1.0", b[6], b[7])
	}
	hlen := int(binary.LittleEndian.Uint16(b[8:10]))
	if len(b) != 10+hlen+24 {
		t.Fatalf("总长 %d ≠ 头 %d + 数据 24", len(b), 10+hlen)
	}
	if (10+hlen)%64 != 0 {
		t.Fatalf("头区 %d 未 64 字节对齐(numpy 同)", 10+hlen)
	}
	hdr := string(b[10 : 10+hlen])
	if want := "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }"; hdr[:len(want)] != want {
		t.Fatalf("header = %q", hdr)
	}
}

func TestLoadNPYErrors(t *testing.T) {
	if _, err := LoadNPYBytes([]byte("not a npy")); !errors.Is(err, ErrDtype) {
		t.Fatalf("bad magic err = %v, want ErrDtype", err)
	}

	bad := append([]byte(nil), npyCF64...)
	bad[0] = 0x00
	if _, err := LoadNPYBytes(bad); !errors.Is(err, ErrDtype) {
		t.Fatalf("corrupt magic err = %v, want ErrDtype", err)
	}

	badVer := append([]byte(nil), npyCF64...)
	badVer[6] = 9
	if _, err := LoadNPYBytes(badVer); !errors.Is(err, ErrDtype) {
		t.Fatalf("bad version err = %v, want ErrDtype", err)
	}

	if _, err := LoadNPYBytes(npyCF64[:20]); err == nil {
		t.Fatalf("truncated 未报错")
	}

	// 头合法但数据截断
	if _, err := LoadNPYBytes(npyCF64[:len(npyCF64)-8]); !errors.Is(err, ErrShape) {
		t.Fatalf("截断数据 err = %v, want ErrShape", err)
	}

	// descr 非法
	if _, err := LoadNPYBytes(npyWithHeader("{'descr': '<v9', 'fortran_order': False, 'shape': (1,), }")); !errors.Is(err, ErrDtype) {
		t.Fatalf("bad descr err = %v, want ErrDtype", err)
	}
	// 缺 shape 字段
	if _, err := LoadNPYBytes(npyWithHeader("{'descr': '<f8', 'fortran_order': False, }")); !errors.Is(err, ErrDtype) {
		t.Fatalf("missing shape err = %v, want ErrDtype", err)
	}
	if _, err := LoadNPY(pathMissing(t)); err == nil {
		t.Fatalf("缺失文件未报错")
	}
}

// npyWithHeader 造一个 v1 头 + 8 字节数据的文件(仅供错误路径测试)。
func npyWithHeader(header string) []byte {
	h := header + "\n"
	for (10+len(h))%64 != 0 {
		h += " "
	}
	out := []byte("\x93NUMPY\x01\x00")
	var l [2]byte
	binary.LittleEndian.PutUint16(l[:], uint16(len(h)))
	out = append(out, l[:]...)
	out = append(out, h...)
	return append(out, 0, 0, 0, 0, 0, 0, 0, 0)
}

func pathMissing(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "nope.npy")
}

// —— Day10 收尾"io_npy 边角":v3 头、0-d/1-d F 序、S/U 宽度边界 ——
// 全部用手工拼的字节(零 python 依赖),期望值可手推。

// npyV1 拼 v1 文件:header 文本(以 \n 结尾、头区 64 对齐)+ 指定数据区。
func npyV1(header string, data []byte) []byte {
	h := header + "\n"
	for (10+len(h))%64 != 0 {
		h += " "
	}
	out := []byte("\x93NUMPY\x01\x00")
	var l [2]byte
	binary.LittleEndian.PutUint16(l[:], uint16(len(h)))
	out = append(out, l[:]...)
	out = append(out, h...)
	return append(out, data...)
}

// npyV3 拼 v3 文件:版本 (3,0) + 4 字节小端头长 + UTF-8 header(numpy v2/v3 布局)。
func npyV3(header string, data []byte) []byte {
	h := header + "\n"
	for (12+len(h))%64 != 0 {
		h += " "
	}
	out := []byte("\x93NUMPY\x03\x00")
	var l [4]byte
	binary.LittleEndian.PutUint32(l[:], uint32(len(h)))
	out = append(out, l[:]...)
	out = append(out, h...)
	return append(out, data...)
}

// f64LE 打包小端 f64 序列。
func f64LE(vals ...float64) []byte {
	out := make([]byte, 0, 8*len(vals))
	for _, v := range vals {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
		out = append(out, b[:]...)
	}
	return out
}

// v3 头:4 字节头长 + UTF-8 header 的读取路径(「读 v1/v2/v3」的 v3 支线)。
func TestLoadNPYV3Header(t *testing.T) {
	a := mustLoadBytes(t, npyV3("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }", f64LE(1.5, -2.25)))
	wantShapeEq(t, a, 2)
	if a.Dtype() != DFloat64 {
		t.Fatalf("v3 f8 dtype = %s", a.Dtype())
	}
	for i, w := range []float64{1.5, -2.25} {
		v, err := a.Item(i)
		if err != nil || v.(float64) != w {
			t.Fatalf("v3 f8[%d] = %v(%v), want %v", i, v, err, w)
		}
	}
	// U3 与 UTF-32LE 码元数据:"ab" 与 "中"(U+4E2D)
	data := []byte{
		0x61, 0, 0, 0, 0x62, 0, 0, 0, 0, 0, 0, 0,
		0x2d, 0x4e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	u := mustLoadBytes(t, npyV3("{'descr': '<U3', 'fortran_order': False, 'shape': (2,), }", data))
	if u.Dtype() != DUnicode(3) {
		t.Fatalf("v3 U3 dtype = %s", u.Dtype())
	}
	for i, w := range []string{"ab", "中"} {
		v, err := u.Item(i)
		if err != nil || v.(string) != w {
			t.Fatalf("v3 U3[%d] = %q(%v), want %q", i, v, err, w)
		}
	}
	// 声称 64 字节头、实际只给 16 字节 → 头长超界必须报错(fail-loud)
	bad := []byte("\x93NUMPY\x03\x00")
	var l [4]byte
	binary.LittleEndian.PutUint32(l[:], 64)
	bad = append(bad, l[:]...)
	bad = append(bad, make([]byte, 16)...)
	if _, err := LoadNPYBytes(bad); err == nil {
		t.Fatal("v3 头长超界未报错")
	}
}

// 0 维与 1 维的 fortran_order=True(现有 fixture 只覆盖 2-D F 序)。
func TestNPY0dAnd1dFortranOrder(t *testing.T) {
	a := mustLoadBytes(t, npyV1("{'descr': '<f8', 'fortran_order': True, 'shape': (), }", f64LE(3.5)))
	if a.NDim() != 0 || a.Size() != 1 {
		t.Fatalf("0-d shape=%v size=%d", a.Shape(), a.Size())
	}
	v, err := a.Item()
	if err != nil || v.(float64) != 3.5 {
		t.Fatalf("0-d 值 = %v(%v), want 3.5", v, err)
	}
	b := mustLoadBytes(t, npyV1("{'descr': '<f8', 'fortran_order': True, 'shape': (3,), }", f64LE(1, 2, 3)))
	wantShapeEq(t, b, 3)
	if s := b.Strides(); len(s) != 1 || s[0] != 8 {
		t.Fatalf("1-d F strides = %v, want [8]", s)
	}
	for i := 0; i < 3; i++ {
		v, err := b.Item(i)
		if err != nil || v.(float64) != float64(i+1) {
			t.Fatalf("1-d F[%d] = %v(%v), want %d", i, v, err, i+1)
		}
	}
}

// S/U 宽度边界:S0/U0 读入口径(:宽度按 1 计)+ 定宽截断/补 NUL 的 npy 往返。
func TestNPYSWidthBoundaries(t *testing.T) {
	s0 := mustLoadBytes(t, npyV1("{'descr': 'S0', 'fortran_order': False, 'shape': (2,), }", []byte{'a', 'b'}))
	if dt := s0.Dtype(); dt.Kind != Bytes || dt.Itemsize != 1 {
		t.Fatalf("S0 → %v(itemsize %d), want Bytes/1", dt, dt.Itemsize)
	}
	if v, _ := s0.Item(0); v.(string) != "a" {
		t.Fatalf("S0[0] = %q, want \"a\"", v)
	}
	u0 := mustLoadBytes(t, npyV1("{'descr': 'U0', 'fortran_order': False, 'shape': (1,), }", []byte{0x61, 0, 0, 0}))
	if dt := u0.Dtype(); dt.Kind != Unicode || dt.Itemsize != 4 {
		t.Fatalf("U0 → %v(itemsize %d), want Unicode/4(宽度计 1 码元)", dt, dt.Itemsize)
	}
	if v, _ := u0.Item(0); v.(string) != "a" {
		t.Fatalf("U0[0] = %q, want \"a\"", v)
	}

	// S4 往返:超宽截断、内嵌 NUL 保留、尾 NUL 剔除
	s, err := New([]int{4}, DBytes(4))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []string{"ab", "abcd", "abcdef", "a\x00b"} {
		if err := s.SetItem([]int{i}, v); err != nil {
			t.Fatalf("S4 SetItem[%d]: %v", i, err)
		}
	}
	raw, err := SaveNPYBytes(s)
	if err != nil {
		t.Fatalf("S4 SaveNPYBytes: %v", err)
	}
	got := mustLoadBytes(t, raw)
	if got.Dtype() != DBytes(4) {
		t.Fatalf("S4 往返 dtype = %s", got.Dtype())
	}
	for i, w := range []string{"ab", "abcd", "abcd", "a\x00b"} {
		v, err := got.Item(i)
		if err != nil || v.(string) != w {
			t.Fatalf("S4 往返[%d] = %q(%v), want %q", i, v, err, w)
		}
	}

	// U2 往返:码点截断("abc" → "ab")与空串
	u, err := New([]int{4}, DUnicode(2))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range []string{"中", "ab", "abc", ""} {
		if err := u.SetItem([]int{i}, v); err != nil {
			t.Fatalf("U2 SetItem[%d]: %v", i, err)
		}
	}
	rawU, err := SaveNPYBytes(u)
	if err != nil {
		t.Fatalf("U2 SaveNPYBytes: %v", err)
	}
	gotU := mustLoadBytes(t, rawU)
	if gotU.Dtype() != DUnicode(2) {
		t.Fatalf("U2 往返 dtype = %s", gotU.Dtype())
	}
	for i, w := range []string{"中", "ab", "ab", ""} {
		v, err := gotU.Item(i)
		if err != nil || v.(string) != w {
			t.Fatalf("U2 往返[%d] = %q(%v), want %q", i, v, err, w)
		}
	}
}

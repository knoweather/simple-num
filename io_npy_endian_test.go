package num

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// toBigEndian 把 SaveNPYBytes 产出的 v1 小端字节流转成大端(descr '>')版:
// 头里第一个 '<'(单字节 dtype 写作 '|',同样换成 '>')替换,数据区按元素内
// unit 字节段做段内反转(unit=1 不动)。数据区起点取头长字段 raw[8:10]:头文本
// 的 '\n' 之后还有 64 字节对齐 padding,用 IndexByte 直接定位会落进 padding,
// 所以这里断言头以 '\n' 结尾、再按头长取数据区。
func toBigEndian(t *testing.T, raw []byte, unit, itemsize int) []byte {
	t.Helper()
	if len(raw) < 10 || string(raw[:6]) != "\x93NUMPY" {
		t.Fatalf("npy 魔数不符: %q", raw)
	}
	if raw[6] != 1 {
		t.Fatalf("预期 v1, got %d", raw[6])
	}
	hlen := int(binary.LittleEndian.Uint16(raw[8:10]))
	hdr := raw[10 : 10+hlen]
	if bytes.IndexByte(hdr, '\n') < 0 {
		t.Fatal("header 未以换行结尾")
	}
	i := bytes.IndexAny(hdr, "<|")
	if i < 0 {
		t.Fatalf("header 无 '<' 可换 '>': %q", hdr)
	}
	out := append([]byte(nil), raw...)
	out[10+i] = '>'
	dataOff := 10 + hlen
	for off := dataOff; off+itemsize <= len(out); off += itemsize {
		for j := 0; j+unit <= itemsize; j += unit {
			seg := out[off+j : off+j+unit]
			for a, b := 0, unit-1; a < b; a, b = a+1, b-1 {
				seg[a], seg[b] = seg[b], seg[a]
			}
		}
	}
	return out
}

// 大端(descr '>')读取路径 × 全部 v0 dtype:逐 dtype 造小端字节、翻成大端、
// 装载后与存入值逐元素对拍(复数按分量翻,不整段翻;U 按 4 字节码元翻)。
func TestLoadNPYBigEndianAllDtypes(t *testing.T) {
	iv := []any{-300, 0, 7, 123456789}
	uv := []any{uint64(0), uint64(1), uint64(65535), uint64(4000000000)}
	fv := []any{1.5, -2.25, 0.0, 3.75}
	cv := []any{complex(1, 2), complex(-3, 4), complex(0, -1), complex(5, 0)}
	cases := []struct {
		name string
		dt   Dtype
		unit int
		vals []any
	}{
		{"i2", DInt16, 2, iv},
		{"i4", DInt32, 4, iv},
		{"i8", DInt64, 8, iv},
		{"i1", DInt8, 1, []any{-128, -1, 0, 127}},
		{"u2", DUint16, 2, uv},
		{"u4", DUint32, 4, uv},
		{"u8", DUint64, 8, uv},
		{"u1", DUint8, 1, []any{0, 1, 127, 255}},
		{"f2", DFloat16, 2, []any{float32(1.5), float32(-2.25), float32(0.5), float32(10.0)}},
		{"f4", DFloat32, 4, fv},
		{"f8", DFloat64, 8, fv},
		{"c8", DComplex64, 4, cv},
		{"c16", DComplex128, 8, cv},
		{"b1", DBool, 1, []any{true, false, true, true}},
		{"M8[D]", DTime("D"), 8, []any{int64(0), int64(18262), int64(-1), int64(100)}},
		{"S4", DBytes(4), 1, []any{"ab", "", "xyz", "01234567"}},
		{"U2", DUnicode(2), 4, []any{"a", "中", "ab", ""}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a, err := New([]int{4}, c.dt)
			if err != nil {
				t.Fatalf("New(%s): %v", c.name, err)
			}
			for i, v := range c.vals {
				if err := a.SetItem([]int{i}, v); err != nil {
					t.Fatalf("%s SetItem[%d] %v: %v", c.name, i, v, err)
				}
			}
			raw, err := SaveNPYBytes(a)
			if err != nil {
				t.Fatalf("%s SaveNPYBytes: %v", c.name, err)
			}
			got := mustLoadBytes(t, toBigEndian(t, raw, c.unit, a.Dtype().Itemsize))
			if got.Dtype() != c.dt {
				t.Fatalf("%s dtype = %s, want %s", c.name, got.Dtype(), c.dt)
			}
			wantShapeEq(t, got, 4)
			for i := 0; i < 4; i++ {
				want, err := a.Item(i)
				if err != nil {
					t.Fatalf("%s stored[%d]: %v", c.name, i, err)
				}
				v, err := got.Item(i)
				if err != nil {
					t.Fatalf("%s BE[%d]: %v", c.name, i, err)
				}
				if v != want {
					t.Fatalf("%s[%d] BE=%v want %v(存入 %v)", c.name, i, v, want, c.vals[i])
				}
			}
		})
	}
}

package num

import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// —— npy 读写(nc/sat 桥接与 参考数据对拍入口)——
//
// 读:支持 v1.0/v2.0/v3.0(头长 2/4 字节)、大小端(异端按元素/4 字节码元翻转)、
// C/F 序(F 序直接给 F 步长,零拷贝语义,与 numpy asfortranarray 装载一致),
// 全 dtype:bool/i/u/f2/f4/f8/c8/c16/M8[unit]/S/U。
// 装载缓冲一律新分配(对齐分配,),不改写调用方字节。
// 写:v1.0(头超 65535 自动升 v2.0),ASCII 头,64 字节对齐 padding(numpy 同构);
// 非 C 序数组先连续化(值语义不变)。
// 追加符号(A 面):LoadNPYBytes/SaveNPYBytes 内存形态,xcheck 消费端复用。

// LoadNPY 载入.npy 文件。
func LoadNPY(path string) (*Array, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("num: LoadNPY %s: %w", path, err)
	}
	a, err := LoadNPYBytes(data)
	if err != nil {
		return nil, fmt.Errorf("num: LoadNPY %s: %w", path, err)
	}
	return a, nil
}

// LoadNPYBytes 载入.npy 字节流。
func LoadNPYBytes(data []byte) (*Array, error) {
	if len(data) < 10 || string(data[:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("num: npy 魔数不符: %w", ErrDtype)
	}
	var hlen, hoff int
	switch major := data[6]; major {
	case 1:
		hlen = int(binary.LittleEndian.Uint16(data[8:10]))
		hoff = 10
	case 2, 3:
		if len(data) < 12 {
			return nil, fmt.Errorf("num: npy v%d 头截断: %w", major, ErrShape)
		}
		hlen = int(binary.LittleEndian.Uint32(data[8:12]))
		hoff = 12
	default:
		return nil, fmt.Errorf("num: npy 版本 %d 未支持: %w", data[6], ErrDtype)
	}
	if hoff+hlen > len(data) {
		return nil, fmt.Errorf("num: npy 头长度 %d 超界: %w", hlen, ErrShape)
	}
	descr, fortran, shape, err := parseNPYHeader(string(data[hoff : hoff+hlen]))
	if err != nil {
		return nil, err
	}
	dt, swap, err := parseDescr(descr)
	if err != nil {
		return nil, err
	}
	need := prodShape(shape) * dt.Itemsize
	body := data[hoff+hlen:]
	if len(body) < need {
		return nil, fmt.Errorf("num: npy 数据 %dB < 需要 %dB: %w", len(body), need, ErrShape)
	}
	buf := alignedAlloc(need)
	copy(buf, body[:need])
	if swap {
		swapElemBytes(buf, dt)
	}
	st := cStrides(shape, dt.Itemsize)
	if fortran {
		st = fStrides(shape, dt.Itemsize)
	}
	return &Array{buf: buf, shape: shape, st: st, dtype: dt}, nil
}

// SaveNPY 存.npy(v1.0;头超限自动 v2.0)。
func SaveNPY(path string, a *Array) error {
	b, err := SaveNPYBytes(a)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("num: SaveNPY %s: %w", path, err)
	}
	return nil
}

// SaveNPYBytes 序列化为.npy 字节流。
func SaveNPYBytes(a *Array) ([]byte, error) {
	descr, err := descrOf(a.Dtype())
	if err != nil {
		return nil, err
	}
	src := a
	if !a.isCContig() {
		src = a.AsContiguous()
	}
	// numpy 的头部规范:dict 字面量 + **空格填充** + **末尾单个 '\n'**,
	// 且 (magic6 + ver2 + lenN + header) 总长按 64 对齐。
	// 注意填充必须在 '\n' 之前 —— 把 '\n' 写在前面会让 numpy 的
	// ast.literal_eval 看到"第二行空格"而报 IndentationError(已实测踩过)。
	body := "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
		shapeRepr(src.shape) + ", }"
	magic := "\x93NUMPY"
	ver := byte(1)
	hlenSize := 2
	if 10+len(body)+1 > 65535 {
		ver = 2
		hlenSize = 4
	}
	base := 6 + 2 + hlenSize + len(body) + 1 // 含末尾 '\n'
	pad := (64 - base%64) % 64
	header := body + strings.Repeat(" ", pad) + "\n"
	out := make([]byte, 0, 6+2+hlenSize+len(header)+src.Size()*src.dtype.Itemsize)
	out = append(out, magic...)
	out = append(out, ver, 0)
	if hlenSize == 2 {
		var b2 [2]byte
		binary.LittleEndian.PutUint16(b2[:], uint16(len(header)))
		out = append(out, b2[:]...)
	} else {
		var b4 [4]byte
		binary.LittleEndian.PutUint32(b4[:], uint32(len(header)))
		out = append(out, b4[:]...)
	}
	out = append(out, header...)
	n := src.Size() * src.dtype.Itemsize
	if n > 0 {
		if w := src.fastWindow(); w != nil {
			out = append(out, w...)
		} else {
			c := src.AsContiguous()
			out = append(out, c.buf[:n]...)
		}
	}
	return out, nil
}

// —— 头解析(受限 Python dict 字面量;numpy 写出的形)——

func parseNPYHeader(hdr string) (descr string, fortran bool, shape []int, err error) {
	rest, err := valueAfterKey(hdr, "descr")
	if err != nil {
		return "", false, nil, err
	}
	descr, err = pyStrAt(rest)
	if err != nil {
		return "", false, nil, fmt.Errorf("num: npy descr 解析: %w", err)
	}
	rest, err = valueAfterKey(hdr, "fortran_order")
	if err != nil {
		return "", false, nil, err
	}
	rest = strings.TrimLeft(rest, " \t")
	switch {
	case strings.HasPrefix(rest, "True"):
		fortran = true
	case strings.HasPrefix(rest, "False"):
	default:
		return "", false, nil, fmt.Errorf("num: npy fortran_order 非法: %w", ErrDtype)
	}
	rest, err = valueAfterKey(hdr, "shape")
	if err != nil {
		return "", false, nil, err
	}
	shape, err = pyShapeAt(rest)
	if err != nil {
		return "", false, nil, fmt.Errorf("num: npy shape 解析: %w", err)
	}
	return descr, fortran, shape, nil
}

// valueAfterKey 定位 'key' 之后的 ':' 并返回其右侧(容忍单双引号键)。
func valueAfterKey(h, key string) (string, error) {
	idx := -1
	for _, q := range []string{"'", `"`} {
		if i := strings.Index(h, q+key+q); i >= 0 {
			idx = i + len(key) + 2
			break
		}
	}
	if idx < 0 {
		return "", fmt.Errorf("num: npy 头缺字段 %q: %w", key, ErrDtype)
	}
	c := strings.IndexByte(h[idx:], ':')
	if c < 0 {
		return "", fmt.Errorf("num: npy 字段 %q 缺冒号: %w", key, ErrDtype)
	}
	return h[idx+c+1:], nil
}

// pyStrAt 解析前导空白后的 Python 字符串字面量。
func pyStrAt(s string) (string, error) {
	s = strings.TrimLeft(s, " \t")
	if len(s) == 0 || (s[0] != '\'' && s[0] != '"') {
		return "", fmt.Errorf("num: 期望引号字符串: %w", ErrDtype)
	}
	q := s[0]
	end := strings.IndexByte(s[1:], q)
	if end < 0 {
		return "", fmt.Errorf("num: 字符串未闭合: %w", ErrDtype)
	}
	return s[1 : 1+end], nil
}

// pyShapeAt 解析 "(2, 3)" / "(6,)" / ""。
func pyShapeAt(s string) ([]int, error) {
	s = strings.TrimLeft(s, " \t")
	if len(s) == 0 || s[0] != '(' {
		return nil, fmt.Errorf("num: 期望形状元组: %w", ErrDtype)
	}
	s = s[1:]
	out := []int{}
	for {
		s = strings.TrimLeft(s, " \t")
		if s == "" {
			return nil, fmt.Errorf("num: 形状元组未闭合: %w", ErrDtype)
		}
		if s[0] == ')' {
			break
		}
		j := 0
		for j < len(s) && s[j] >= '0' && s[j] <= '9' {
			j++
		}
		if j == 0 {
			return nil, fmt.Errorf("num: 形状元素非法: %w", ErrDtype)
		}
		v, err := strconv.Atoi(s[:j])
		if err != nil {
			return nil, fmt.Errorf("num: 形状元素 %q: %w", s[:j], ErrDtype)
		}
		out = append(out, v)
		s = strings.TrimLeft(s[j:], " \t")
		if s == "" {
			return nil, fmt.Errorf("num: 形状元组未闭合: %w", ErrDtype)
		}
		switch s[0] {
		case ',':
			s = s[1:]
		case ')':
		default:
			return nil, fmt.Errorf("num: 形状分隔符 %q: %w", s[0:1], ErrDtype)
		}
	}
	return out, nil
}

// parseDescr numpy 描述符 → Dtype + 是否需要翻转字节。
func parseDescr(desc string) (Dtype, bool, error) {
	if desc == "" {
		return Dtype{}, false, fmt.Errorf("num: 空 descr: %w", ErrDtype)
	}
	rest := desc
	swap := false
	switch desc[0] {
	case '<':
		rest = desc[1:]
	case '>':
		rest = desc[1:]
		swap = true
	case '|', '=':
		rest = desc[1:]
	}
	switch {
	case rest == "b1" || rest == "?":
		return DBool, false, nil
	case strings.HasPrefix(rest, "M8"):
		unit, err := m8Unit(rest)
		if err != nil {
			return Dtype{}, false, err
		}
		return Dtype{Kind: Datetime64, Itemsize: 8, Unit: unit}, swap, nil
	case rest[0] == 'S':
		w, err := strconv.Atoi(rest[1:])
		if err != nil || w < 0 {
			return Dtype{}, false, fmt.Errorf("num: descr %q 宽度: %w", desc, ErrDtype)
		}
		if w == 0 {
			w = 1 // numpy S0 → 1 字节(实测口径,记)
		}
		return Dtype{Kind: Bytes, Itemsize: w}, false, nil
	case rest[0] == 'U':
		w, err := strconv.Atoi(rest[1:])
		if err != nil || w < 0 {
			return Dtype{}, false, fmt.Errorf("num: descr %q 宽度: %w", desc, ErrDtype)
		}
		if w == 0 {
			w = 1
		}
		return Dtype{Kind: Unicode, Itemsize: 4 * w}, swap, nil
	}
	if len(rest) < 2 {
		return Dtype{}, false, fmt.Errorf("num: descr %q 无法识别: %w", desc, ErrDtype)
	}
	w, err := strconv.Atoi(rest[1:])
	if err != nil {
		return Dtype{}, false, fmt.Errorf("num: descr %q 宽度: %w", desc, ErrDtype)
	}
	var k Kind
	switch rest[0] {
	case 'i':
		switch w {
		case 1:
			k = Int8
		case 2:
			k = Int16
		case 4:
			k = Int32
		case 8:
			k = Int64
		}
	case 'u':
		switch w {
		case 1:
			k = Uint8
		case 2:
			k = Uint16
		case 4:
			k = Uint32
		case 8:
			k = Uint64
		}
	case 'f':
		switch w {
		case 2:
			k = Float16
		case 4:
			k = Float32
		case 8:
			k = Float64
		}
	case 'c':
		switch w {
		case 8:
			k = Complex64
		case 16:
			k = Complex128
		}
	default:
		return Dtype{}, false, fmt.Errorf("num: descr 类型 %q 未支持: %w", desc, ErrDtype)
	}
	if k == 0 {
		return Dtype{}, false, fmt.Errorf("num: descr %q 宽度未支持: %w", desc, ErrDtype)
	}
	return Dtype{Kind: k, Itemsize: w}, swap, nil
}

// m8Unit 从 "M8[us]" 取单位;无单位按 numpy 默认 "us"(本库 DTime 同)。
func m8Unit(s string) (string, error) {
	i := strings.IndexByte(s, '[')
	if i < 0 {
		return "us", nil
	}
	j := strings.IndexByte(s[i:], ']')
	if j < 0 {
		return "", fmt.Errorf("num: descr %q 未闭合: %w", s, ErrDtype)
	}
	unit := s[i+1 : i+j]
	switch unit {
	case "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns":
		return unit, nil
	}
	return "", fmt.Errorf("num: datetime64 单位 %q 未支持: %w", unit, ErrDtype)
}

// descrOf Dtype → numpy 描述符(小端;S/U 用 |S / <U 形)。
func descrOf(dt Dtype) (string, error) {
	switch dt.Kind {
	case Bool:
		return "|b1", nil
	case Int8:
		return "|i1", nil
	case Uint8:
		return "|u1", nil
	case Int16:
		return "<i2", nil
	case Int32:
		return "<i4", nil
	case Int64:
		return "<i8", nil
	case Uint16:
		return "<u2", nil
	case Uint32:
		return "<u4", nil
	case Uint64:
		return "<u8", nil
	case Float16:
		return "<f2", nil
	case Float32:
		return "<f4", nil
	case Float64:
		return "<f8", nil
	case Complex64:
		return "<c8", nil
	case Complex128:
		return "<c16", nil
	case Datetime64:
		return "<M8[" + dt.Unit + "]", nil
	case Bytes:
		return "|S" + strconv.Itoa(dt.Itemsize), nil
	case Unicode:
		return "<U" + strconv.Itoa(dt.Itemsize/4), nil
	}
	return "", fmt.Errorf("num: dtype %s 无 npy 描述符: %w", dt, ErrDtype)
}

// shapeRepr Go 形状 → numpy 元组字面("(2, 3)" / "(6,)" / "")。
func shapeRepr(sh []int) string {
	if len(sh) == 0 {
		return "()"
	}
	var b strings.Builder
	b.WriteByte('(')
	for i, d := range sh {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Itoa(d))
	}
	if len(sh) == 1 {
		b.WriteByte(',')
	}
	b.WriteByte(')')
	return b.String()
}

// fStrides Fortran 序字节步长。
func fStrides(sh []int, itemsize int) []int {
	st := make([]int, len(sh))
	s := itemsize
	for i := range sh {
		st[i] = s
		s *= sh[i]
	}
	return st
}

// swapElemBytes 异端翻转:Unicode 按 4 字节码元;复数按浮点分量(整元素翻会把
// 实虚部互换,见);S/bool/1 字节不动;其余按元素。
func swapElemBytes(buf []byte, dt Dtype) {
	switch dt.Kind {
	case Bool, Bytes:
		return
	case Unicode:
		for off := 0; off+4 <= len(buf); off += 4 {
			swapN(buf[off : off+4])
		}
		return
	case Complex64, Complex128:
		comp := dt.Itemsize / 2
		for off := 0; off+dt.Itemsize <= len(buf); off += dt.Itemsize {
			swapN(buf[off : off+comp])
			swapN(buf[off+comp : off+dt.Itemsize])
		}
		return
	}
	it := dt.Itemsize
	if it <= 1 {
		return
	}
	for off := 0; off+it <= len(buf); off += it {
		swapN(buf[off : off+it])
	}
}

func swapN(b []byte) {
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
}

package num

import "testing"

// promote 表的认证测试:期望值来自 numpy 2.5.4 np.promote_types 实测
// (转录见.md;表本身在本文件独立成行,防转录错误自证)。

// spot 对拍:每个关键格 = numpy 实测结果(注释里是 numpy dtype 名)。
var promoteSpot = []struct {
	a, b  Dtype
	want  Kind
	model string
}{
	// 同 kind 阶梯
	{DInt8, DInt8, Int8, "int8"},
	{DInt16, DInt32, Int32, "int32"},
	{DUint32, DUint8, Uint32, "uint32"},
	{DFloat64, DFloat32, Float64, "float64"},
	// bool ≈ 1 位有符号整数(numpy:b×u1→i2、b×u8→f8)
	{DBool, DInt8, Int8, "int8"},
	{DBool, DUint8, Int16, "int16"}, // numpy: promote(bool,uint8)==int16
	{DBool, DUint64, Float64, "float64"},
	{DBool, DFloat64, Float64, "float64"},
	// u×i"可容纳匹配"阶梯
	{DInt8, DUint8, Int16, "int16"},
	{DInt8, DUint16, Int32, "int32"},
	{DUint8, DInt16, Int16, "int16"},
	{DUint8, DInt32, Int32, "int32"},
	{DUint32, DInt32, Int64, "int64"},
	{DInt64, DUint64, Float64, "float64"},
	{DInt16, DUint16, Int32, "int32"},
	// i×f 按尾数位宽(f2=11 位,f4=24,f8=53)
	{DInt16, DFloat32, Float32, "float32"},
	{DInt32, DFloat32, Float64, "float64"},
	{DUint32, DFloat32, Float64, "float64"},
	{DInt64, DFloat32, Float64, "float64"},
	{DUint64, DFloat32, Float64, "float64"},
	{DInt8, DFloat16, Float16, "float16"}, // numpy: promote(i1,f2)==float16
	{DUint8, DFloat16, Float16, "float16"},
	{DInt16, DFloat16, Float32, "float32"},
	// c 阶梯
	{DFloat32, DComplex64, Complex64, "complex64"},
	{DFloat64, DComplex64, Complex128, "complex128"},
	{DInt16, DComplex64, Complex64, "complex64"},
	{DInt32, DComplex64, Complex128, "complex128"},
	{DUint64, DComplex64, Complex128, "complex128"},
	{DComplex64, DComplex128, Complex128, "complex128"},
}

func TestPromoteSpot(t *testing.T) {
	for _, s := range promoteSpot {
		got, err := promoteTypes(s.a, s.b)
		if err != nil {
			t.Fatalf("promote(%s,%s): %v", s.a, s.b, err)
		}
		if got.Kind != s.want {
			t.Errorf("promote(%s,%s) = %s, want %s (numpy %s)", s.a, s.b, got, s.want, s.model)
		}
		if got.Itemsize != kindItemsize(got.Kind) {
			t.Errorf("promote(%s,%s) kind=%s itemsize %d", s.a, s.b, got, got.Itemsize)
		}
	}
}

// 不变量一:对称性(全数值对)。
func TestPromoteSymmetric(t *testing.T) {
	kinds := []Kind{Bool, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64,
		Float16, Float32, Float64, Complex64, Complex128}
	for _, ka := range kinds {
		for _, kb := range kinds {
			da := Dtype{Kind: ka, Itemsize: kindItemsize(ka)}
			db := Dtype{Kind: kb, Itemsize: kindItemsize(kb)}
			r1, err1 := promoteTypes(da, db)
			r2, err2 := promoteTypes(db, da)
			if (err1 == nil) != (err2 == nil) {
				t.Fatalf("promote(%s,%s) err 对称破坏", ka, kb)
			}
			if err1 != nil {
				continue
			}
			if r1 != r2 {
				t.Errorf("promote 非对称 (%s,%s): %s ≠ %s", ka, kb, r1, r2)
			}
		}
	}
}

// 不变量二:同 dtype 恒等(含宽度/单位字段)。
func TestPromoteIdentity(t *testing.T) {
	cases := []Dtype{DBool, DInt64, DFloat32, {Kind: Bytes, Itemsize: 7},
		{Kind: Datetime64, Itemsize: 8, Unit: "D"}, DTime("D")}
	for _, d := range cases {
		got, err := promoteTypes(d, d)
		if err != nil {
			t.Fatalf("promote(%s,%s): %v", d, d, err)
		}
		if got != d {
			t.Errorf("同 dtype 恒等破坏:%s → %s", d, got)
		}
	}
}

// 不变量三:出版位面(字符串/时间 × 数值、datetime 单位冲突)一律 ErrDtype。
func TestPromoteUnsupported(t *testing.T) {
	bad := [][2]Dtype{
		{DInt64, DTime("D")},
		{DTime("D"), DInt64},
		{DTime("D"), DTime("h")}, // 单位归一属,先报错
		{DInt8, Dtype{Kind: Bytes, Itemsize: 3}},
		{DBool, Dtype{Kind: Unicode, Itemsize: 8}},
		{DComplex128, DTime("s")},
	}
	for _, c := range bad {
		if _, err := promoteTypes(c[0], c[1]); err == nil {
			t.Errorf("promote(%s,%s) 应报 ErrDtype", c[0], c[1])
		} else if !isErrDtype(err) {
			t.Errorf("promote(%s,%s) 错误未包 ErrDtype:%v", c[0], c[1], err)
		}
	}
}

// Itemsize 与 String 口径(numpy 同名)。
func TestDtypeString(t *testing.T) {
	cases := []struct {
		d    Dtype
		want string
	}{
		{DBool, "bool"},
		{DInt16, "int16"},
		{DUint8, "uint8"},
		{DFloat16, "float16"},
		{DFloat64, "float64"},
		{DComplex64, "complex64"},
		{DTime("D"), "datetime64[D]"},
		{DTime(""), "datetime64[us]"},
		{DBytes(3), "|S3"},
		{DUnicode(4), "<U4"},
	}
	for _, c := range cases {
		if got := c.d.String(); got != c.want {
			t.Errorf("String(%v) = %q, want %q", c.d, got, c.want)
		}
		if c.d.Itemsize != 0 && c.d.Itemsize != kindItemsize(c.d.Kind) &&
			c.d.Kind != Bytes && c.d.Kind != Unicode && c.d.Kind != Datetime64 {
			t.Errorf("Itemsize 与 Kind 不一致:%v", c.d)
		}
	}
}

func isErrDtype(err error) bool {
	for err != nil {
		if err == ErrDtype {
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

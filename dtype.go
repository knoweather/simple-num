package num

import (
	"errors"
	"fmt"
	"strconv"
)

// —— 错误哨兵(全部 fail-loud,禁 panic; §2.1 冻结)——
// 起加第五哨兵 ErrValue(值域错误,如 pow 整型负指数):与 ErrDtype 并列
// 出现在错误链上(多 %w),errors.Is 两者皆真;rootOf 单链口径不变。
var (
	ErrShape   = errors.New("num: shape mismatch")
	ErrDtype   = errors.New("num: dtype mismatch")
	ErrIndex   = errors.New("num: index out of range")
	ErrZeroDiv = errors.New("num: integer divide by zero")
	ErrValue   = errors.New("num: value domain error")
)

// Kind 枚举元素类型(numpy dtype 分类)。
type Kind uint8

const (
	Bool Kind = iota + 1
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float32
	Float64
	Complex64
	Complex128 //
	Datetime64 // (naive int64 + Unit)
	Bytes      // "S" 定宽字节
	Unicode    // "U" 定宽 4 字节码元
	Float16    // 仅存取与转换(算时升 f32,numpy 同策略)
)

// Dtype 描述元素类型。字符串/时间等在 Kind+Itemsize(+Unit)内表达。
type Dtype struct {
	Kind     Kind
	Itemsize int    // 字节宽;S/U 为定宽容量的字节数
	Unit     string // 仅 Datetime64:"Y","M","W","D","h","m","s","ms","us","ns"
}

// 预定义 dtype(数值全集;S/U 用构造器带宽度,DTime 带单位)。
var (
	DBool       = Dtype{Kind: Bool, Itemsize: 1}
	DInt8       = Dtype{Kind: Int8, Itemsize: 1}
	DInt16      = Dtype{Kind: Int16, Itemsize: 2}
	DInt32      = Dtype{Kind: Int32, Itemsize: 4}
	DInt64      = Dtype{Kind: Int64, Itemsize: 8}
	DUint8      = Dtype{Kind: Uint8, Itemsize: 1}
	DUint16     = Dtype{Kind: Uint16, Itemsize: 2}
	DUint32     = Dtype{Kind: Uint32, Itemsize: 4}
	DUint64     = Dtype{Kind: Uint64, Itemsize: 8}
	DFloat16    = Dtype{Kind: Float16, Itemsize: 2}
	DFloat32    = Dtype{Kind: Float32, Itemsize: 4}
	DFloat64    = Dtype{Kind: Float64, Itemsize: 8}
	DComplex64  = Dtype{Kind: Complex64, Itemsize: 8}
	DComplex128 = Dtype{Kind: Complex128, Itemsize: 16}
)

// DTime 构造 datetime64 dtype(unit 缺省记为通用 M8,算术语义 实现)。
func DTime(unit string) Dtype {
	if unit == "" {
		unit = "us" // numpy 语义中无单位 M8 的常用缺省;NPY-NOTES 考据
	}
	return Dtype{Kind: Datetime64, Itemsize: 8, Unit: unit}
}

// DBytes / DUnicode 构造定宽字符串 dtype(width=字节宽 / 码元数)。
func DBytes(width int) Dtype { return Dtype{Kind: Bytes, Itemsize: width} }
func DUnicode(width int) Dtype {
	return Dtype{Kind: Unicode, Itemsize: 4 * width}
}

// kindItemsize 表:数值与时间类的固定字节宽。
func kindItemsize(k Kind) int {
	switch k {
	case Bool, Int8, Uint8:
		return 1
	case Int16, Uint16, Float16:
		return 2
	case Int32, Uint32, Float32:
		return 4
	case Int64, Uint64, Float64, Datetime64:
		return 8
	case Complex64:
		return 8
	case Complex128:
		return 16
	}
	return 0
}

func (k Kind) String() string {
	switch k {
	case Bool:
		return "bool"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint8:
		return "uint8"
	case Uint16:
		return "uint16"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Float16:
		return "float16"
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Complex64:
		return "complex64"
	case Complex128:
		return "complex128"
	case Datetime64:
		return "datetime64"
	case Bytes:
		return "S"
	case Unicode:
		return "U"
	}
	return fmt.Sprintf("Kind(%d)", uint8(k))
}

func (d Dtype) String() string {
	switch d.Kind {
	case Bytes:
		return "|S" + strconv.Itoa(d.Itemsize)
	case Unicode:
		return "<U" + strconv.Itoa(d.Itemsize/4)
	case Datetime64:
		return "datetime64[" + d.Unit + "]"
	}
	return d.Kind.String()
}

// promote/cast 语义( §2.2):
// - promoteTypes 逐格照 numpy 2.x promote_types(实测 numpy 2.5.4 转录,
// 含 bool(视作 1 位有符号整数)与 u×i"可容纳匹配"阶梯);
// 表格逐格与 numpy 对齐,单测矩阵在 promote_test 锁定。
// - Cast:同 kind 直通 + 数值提升阶梯;不支持 → ErrDtype。

func init() {
	promoteIdx = make(map[Kind]int, len(promoteOrder))
	for i, k := range promoteOrder {
		promoteIdx[k] = i
	}
}

// promoteOrder:numpy 表行列序(b,i1..i8,u1..u8,f2,f4,f8,c8,c16)。
var promoteOrder = [14]Kind{
	Bool, Int8, Int16, Int32, Int64,
	Uint8, Uint16, Uint32, Uint64,
	Float16, Float32, Float64,
	Complex64, Complex128,
}

var promoteIdx map[Kind]int

// promoteTable[i][j] = promote(行 kind, 列 kind);0 表示不支持(错误)。
// 逐格转录自 numpy 2.5.4 np.promote_types 实测(见 dtype_test.go 矩阵)。
var promoteTable = [14][14]Kind{
	// b, i1, i2, i4, i8, u1, u2, u4, u8, f2, f4, f8, c8, c16
	{Int8, Int8, Int16, Int32, Int64, Int16, Int32, Int64, Float64, Float16, Float32, Float64, Complex64, Complex128},                                                        // bool
	{Int8, Int8, Int16, Int32, Int64, Int16, Int32, Int64, Float64, Float16, Float32, Float64, Complex64, Complex128},                                                        // int8
	{Int16, Int16, Int16, Int32, Int64, Int16, Int32, Int64, Float64, Float32, Float32, Float64, Complex64, Complex128},                                                      // int16
	{Int32, Int32, Int32, Int32, Int64, Int32, Int32, Int64, Float64, Float64, Float64, Float64, Complex128, Complex128},                                                     // int32
	{Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Complex128, Complex128},                                                     // int64
	{Int16, Int16, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Float16, Float32, Float64, Complex64, Complex128},                                                     // uint8
	{Int32, Int32, Int32, Int32, Int64, Uint16, Uint16, Uint32, Uint64, Float32, Float32, Float64, Complex64, Complex128},                                                    // uint16
	{Int64, Int64, Int64, Int64, Int64, Uint32, Uint32, Uint32, Uint64, Float64, Float64, Float64, Complex128, Complex128},                                                   // uint32
	{Float64, Float64, Float64, Float64, Float64, Uint64, Uint64, Uint64, Uint64, Float64, Float64, Float64, Complex128, Complex128},                                         // uint64
	{Float16, Float16, Float32, Float64, Float64, Float16, Float32, Float64, Float64, Float16, Float32, Float64, Complex64, Complex128},                                      // float16
	{Float32, Float32, Float32, Float64, Float64, Float32, Float32, Float64, Float64, Float32, Float32, Float64, Complex64, Complex128},                                      // float32
	{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Complex128, Complex128},                                     // float64
	{Complex64, Complex64, Complex64, Complex128, Complex128, Complex64, Complex64, Complex128, Complex128, Complex64, Complex64, Complex128, Complex64, Complex128},         // complex64
	{Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128, Complex128}, // complex128
}

// promoteTypes 两 dtype 的 numpy 同款共同类型(NEP 50,数值全集;对称)。
// 不支持(字符串/时间混数值等)→ ErrDtype 包装错误;datetime 规则 补。
func promoteTypes(a, b Dtype) (Dtype, error) {
	if a == b {
		return a, nil // 同 dtype 快路径(datetime 同单位、S/U 同宽度直通)
	}
	if a.Kind == b.Kind && (a.Kind == Bytes || a.Kind == Unicode) {
		// numpy:同 kind 定宽字符串取最大宽度(promote_types('S3','S4') = 'S4')
		if b.Itemsize > a.Itemsize {
			return b, nil
		}
		return a, nil
	}
	ia, ok1 := promoteIdx[a.Kind]
	ib, ok2 := promoteIdx[b.Kind]
	if !ok1 || !ok2 {
		return Dtype{}, fmt.Errorf("num: promote %s × %s: %w", a, b, ErrDtype)
	}
	rk := promoteTable[ia][ib]
	if rk == 0 {
		return Dtype{}, fmt.Errorf("num: promote %s × %s: %w", a, b, ErrDtype)
	}
	return Dtype{Kind: rk, Itemsize: kindItemsize(rk)}, nil
}

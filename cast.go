package num

import (
	"fmt"
	"math"
	"strings"
)

// —— T5 cast(numpy can_cast 口径,实测 numpy 2.5.4 转录)——
//
// same_kind 权限矩阵(行 from kind → 列 to kind;行/列序 = promoteOrder):
// 转录自 np.can_cast(mode="same_kind") 14×14 实测(两次独立核对一致)。要点:
// - b→i*、i↔i、u→*、i/u/b→f/c 全真;
// - b→u 否;i→u 否;f→i/u/b 否;f→b 否;
// - 同 kind 收窄允许(经典口径:f8→f4、f4→f2、c16→c8 均 Y)。
// f16 的算子内核挂载后可用;Cast 的转码路径天然支持 f2↔f4。
//
// v0 补充:矩阵之外还有"数值/S/U → S/U"的 same_kind 许可(numpy 实测
// 全部为真,唯一例外 U→S),见 castAllowed;M8/m8 → 文本在 numpy 属 unsafe。

var castSameKind = [14][14]bool{
	// b i1 i2 i4 i8 u1 u2 u4 u8 f2 f4 f8 c8 c16
	{true, true, true, true, true, false, false, false, false, true, true, true, true, true},         // b
	{true, true, true, true, true, false, false, false, false, true, true, true, true, true},         // i1
	{true, true, true, true, true, false, false, false, false, true, true, true, true, true},         // i2
	{true, true, true, true, true, false, false, false, false, true, true, true, true, true},         // i4
	{true, true, true, true, true, false, false, false, false, true, true, true, true, true},         // i8
	{true, true, true, true, true, true, true, true, true, true, true, true, true, true},             // u1
	{true, true, true, true, true, true, true, true, true, true, true, true, true, true},             // u2
	{true, true, true, true, true, true, true, true, true, true, true, true, true, true},             // u4
	{true, true, true, true, true, true, true, true, true, true, true, true, true, true},             // u8
	{false, false, false, false, false, false, false, false, false, true, true, true, true, true},    // f2
	{false, false, false, false, false, false, false, false, false, true, true, true, true, true},    // f4
	{false, false, false, false, false, false, false, false, false, true, true, true, true, true},    // f8
	{false, false, false, false, false, false, false, false, false, false, false, false, true, true}, // c8
	{false, false, false, false, false, false, false, false, false, false, false, false, true, true}, // c16
}

// castAllowed 权限查询(numpy can_cast same_kind 口径)。
//
// 数值全集查 castSameKind;数值/S/U → S/U 是 numpy same_kind 实测的额外许可
// :bool/int/uint/float/complex/S/U 源都可落到 S/U 文本,唯一例外是
// U→S(unicode 降 ASCII 可能失真,numpy 判否);M8/m8 → S/U 在 numpy 也是否
// (datetime 转文本属 unsafe,仅 astype 默认口径可用,本库 Cast 不开)。
func castAllowed(from, to Kind) bool {
	if fi, ok1 := promoteIdx[from]; ok1 {
		if ti, ok2 := promoteIdx[to]; ok2 {
			return castSameKind[fi][ti]
		}
		if to == Bytes || to == Unicode {
			return true // 数值 → 文本
		}
	}
	if to == Bytes || to == Unicode {
		switch from {
		case Bytes:
			return true
		case Unicode:
			return to != Bytes
		}
	}
	return from == to
}

// castUnitErr 单位换算拒绝文案(Cast 与 CastUnsafe 逐字同款,不静默搬计数)。
func castUnitErr(from, to Dtype) error {
	return fmt.Errorf("num: Cast %s → %s 单位换算未实现: %w", from, to, ErrDtype)
}

// Cast 显式转换(numpy 同名;same_kind 口径,浮点收窄等 unsafe 组合 → ErrDtype;
// unsafe 另开口子待 前定,见)。
// datetime 单位换算未实现,单位不同 → ErrDtype(不静默搬计数)。
func Cast(a *Array, dt Dtype) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	if a.Dtype() == dt {
		return a, nil
	}
	if a.Dtype().Kind == Datetime64 && dt.Kind == Datetime64 {
		return nil, castUnitErr(a.Dtype(), dt)
	}
	if !castAllowed(a.Dtype().Kind, dt.Kind) {
		return nil, fmt.Errorf("num: Cast %s → %s: %w", a.Dtype(), dt, ErrDtype)
	}
	return transformCopy(a, a.Shape(), dt)
}

// CastUnsafe 显式转换的 unsafe 口径(numpy astype(casting="unsafe") 权限口径)。
//
// 与 Cast 完全同入口、同转码路径,唯一差别:跳过 castAllowed 权限查询。
// 入口逐条对齐 Cast:checkDt → 同 dtype 返回原对象 → M8→M8 单位不同仍 ErrDtype
// (未实现,文案与 Cast 同款)。
//
// 值语义按本机 numpy 2.2.1 实测锁死(amd64 Go 原生数值转换与 numpy 同硬件指令语义):
// 1. 浮点 → 整型:向零截断;NaN/±Inf/超界 → 目标位宽硬件不定值(该宽度符号位置
// 1,即该宽度最小值)。Cast 路径不经此处(f→i/u 在 same_kind 矩阵全拒),
// 故 Cast 既有行为零变化。
// 2. 复数 → 浮点:取实部(numpy 实测;Cast 路径 c→f 亦拒,不受影响)。
// 3. 复数 → bool:按任务口径只看实部(给定实测值 [0+0j,1+2j] 两种口径同结果,
// 见偏差清单)。
// 4. U → S:numpy 抛 UnicodeEncodeError;本库不复刻 Python 异常,锁死为
// "非 ASCII 码元替换为 ?"(见偏差清单)。
// 5. M8 → S/U:numpy unsafe 口径放行(渲染 ISO 串);本库未实现 datetime 文本
// 渲染,宁拒不冒充 → 仍 ErrDtype(偏差)。
func CastUnsafe(a *Array, dt Dtype) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	if a.Dtype() == dt {
		return a, nil
	}
	if a.Dtype().Kind == Datetime64 && dt.Kind == Datetime64 {
		return nil, castUnitErr(a.Dtype(), dt)
	}
	if a.Dtype().Kind == Datetime64 && (dt.Kind == Bytes || dt.Kind == Unicode) {
		return nil, fmt.Errorf("num: CastUnsafe %s → %s datetime 文本渲染未实现: %w", a.Dtype(), dt, ErrDtype)
	}
	return transformCopyMode(a, a.Shape(), dt, true)
}

// transformCopy 引擎归一核:广播到 outShape + 转 dt + materialize 连续对齐。
// unsafe=false 即 Cast 口径(既有行为逐字不变);unsafe=true 走 CastUnsafe 口径。
// - 快路径:同 dtype 同 shape → copyStrided(带后缀连续整块 memcpy);
// - 逐行:外层 odometer 遍历前 ndim-1 维;内层行三路:
// 1) 同 dtype 且内层源连续 → 整行 memcpy;
// 2) 同 dtype 且内层广播 → 单元素重复填充;
// 3) 其余 → 逐元素(decode → coerce → encode)。
func transformCopy(src *Array, outShape []int, dt Dtype) (*Array, error) {
	return transformCopyMode(src, outShape, dt, false)
}

func transformCopyMode(src *Array, outShape []int, dt Dtype, unsafe bool) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	if !broadcastOk(src.shape, outShape) {
		return nil, fmt.Errorf("num: 归一 %v(源 %v)不相容: %w", outShape, src.shape, ErrShape)
	}
	out, err := New(outShape, dt)
	if err != nil {
		return nil, err
	}
	same := src.dtype == dt
	it := src.dtype.Itemsize
	ot := dt.Itemsize

	if same && shapeEqual(src.shape, outShape) {
		copyStrided(out.buf, src.buf, src.shape, src.st, src.off, it)
		return out, nil
	}
	// 快速路径:同形状 + 源连续 + 常见数值组合 → 类型化批量转换(免 any 装箱)。
	// 语义与通用路径逐位等价(见 impl_castfast.go 的支持范围说明);
	// 未特化的组合照旧走下面的通用循环。
	if !same && shapeEqual(src.shape, outShape) {
		if castFastConv(src, out) {
			return out, nil
		}
	}
	steps := elemStepsBroadcast(src.shape, src.st, outShape)
	nd := len(outShape)
	if nd == 0 {
		if src.Size() == 0 {
			return out, nil
		}
		return out, copyElemMode(out.buf, 0, dt, src, src.off, same, unsafe)
	}
	innerN := outShape[nd-1]
	innerStep := steps[nd-1] // 0 = 内层广播;it = 内层源连续;其他 = 内层跨步
	outerSteps := steps[:nd-1]
	o := 0
	var lerr error
	odometerVisit(outShape[:nd-1], func(idx []int) bool {
		cur := src.off
		for k, x := range idx {
			cur += x * outerSteps[k]
		}
		switch {
		case same && innerStep == it: // 内层源连续:整行 memcpy
			copy(out.buf[o:o+innerN*it], src.buf[cur:cur+innerN*it])
		case same && innerStep == 0: // 内层广播:单元素重复
			for k := 0; k < innerN; k++ {
				copy(out.buf[o+k*it:o+(k+1)*it], src.buf[cur:cur+it])
			}
		default: // 逐元素(dtype 转换或内层跨步)
			for k := 0; k < innerN; k++ {
				if err := copyElemMode(out.buf, o+k*ot, dt, src, cur+k*innerStep, same, unsafe); err != nil {
					lerr = err
					return false
				}
			}
		}
		o += innerN * ot
		return true
	})
	if lerr != nil {
		return nil, lerr
	}
	return out, nil
}

// copyElem 搬 src.buf[se] 一个元素到 dst[de];same=true 时字节直拷。
func copyElem(dst []byte, de int, dt Dtype, src *Array, se int, same bool) error {
	return copyElemMode(dst, de, dt, src, se, same, false)
}

func copyElemMode(dst []byte, de int, dt Dtype, src *Array, se int, same, unsafe bool) error {
	it := src.dtype.Itemsize
	if same {
		copy(dst[de:de+it], src.buf[se:se+it])
		return nil
	}
	v, err := decodeElemMode(src, se, dt, unsafe)
	if err != nil {
		return err
	}
	return encodeOneMode(dt, de, dst, v, unsafe)
}

// decodeElem 读出源元素,并按目标 dtype 校正文本精度。
//
// numpy 的数值 → 文本按**源标量/源数组自身 dtype** 的精度渲染:f16 数组 → S/U
// 得 f16 口径(0.0999755859375 → "0.1"),f32 数组 → 得 f32 口径(→ "0.099975586")。
// Go 侧 f16 读出统一是 float32(无 f16 原语),故此处显式带上源精度
// (/;Cast 与赋值 SetSub 两条转码路径共用)。
func decodeElem(src *Array, off int, dst Dtype) (any, error) {
	return decodeElemMode(src, off, dst, false)
}

func decodeElemMode(src *Array, off int, dst Dtype, unsafe bool) (any, error) {
	v, err := decodeOne(src.dtype, off, src.buf)
	if err != nil {
		return nil, err
	}
	if src.dtype.Kind == Float16 && (dst.Kind == Bytes || dst.Kind == Unicode) {
		return pyFloatStr(float64(v.(float32)), 16), nil
	}
	// unsafe 口径独有:U → S。numpy 对非 ASCII 码元抛 UnicodeEncodeError;
	// 本库不抛 Python 异常,锁死口径 = 非 ASCII 码元替换为 '?'(偏差,见报告)。
	if unsafe && src.dtype.Kind == Unicode && dst.Kind == Bytes {
		return asciiFold(v.(string)), nil
	}
	return v, nil
}

// encodeOneMode encodeOne(array.go,逐字不改)的 unsafe 解锁面:
// 仅 CastUnsafe 路径(unsafe=true)生效,Cast/SetSub/工厂路径仍走 encodeOne。
func encodeOneMode(dt Dtype, off int, buf []byte, v any, unsafe bool) error {
	if !unsafe {
		return encodeOne(dt, off, buf, v)
	}
	switch dt.Kind {
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
		if c, ok := complexOf(v); ok {
			v = real(c) // numpy:复数 → 整型取实部(ComplexWarning)
		}
		if f, ok := floatOf(v); ok {
			storeI64(dt.Kind, buf[off:off+dt.Itemsize], int64(unsafeFloatToIntBits(dt.Kind, f)))
			return nil
		}
	case Float16, Float32, Float64:
		if c, ok := complexOf(v); ok {
			v = real(c) // numpy:复数 → 浮点取实部
		}
	case Bool:
		if c, ok := complexOf(v); ok {
			v = real(c) != 0 // 任务口径:只看实部
		}
	}
	return encodeOne(dt, off, buf, v)
}

// unsafeFloatToIntBits 浮点 → 目标整型位宽(amd64 硬件口径):
// - 先向零截断;源为 NaN/±Inf 或截断值超出该位宽有符号范围 → 该宽度硬件不定值
// (CVTTSD2SI = 符号位置 1);
// - 负值按补码落位(与 numpy "先有符号转换再按目标宽度落位" 一致,故 u4 目标
// 收 -0.5 → 0、-1.5 → 4294967295)。
//
// 偏差:位宽 1/2 时 numpy 走 double→int64 再截低位,NaN/超界落 0(本口径按目标位宽
// 符号位;给定实测值不覆盖该情形)。
func unsafeFloatToIntBits(k Kind, f float64) uint64 {
	bits := kindItemsize(k) * 8
	undef := uint64(1) << (bits - 1)
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return undef
	}
	t := math.Trunc(f) // 先截断再判界:127.5 → 127 在 i1 内,128.0 才超界
	if t < -float64(undef) || t > float64(undef)-1 {
		return undef
	}
	return uint64(int64(t))
}

func floatOf(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	}
	return 0, false
}

func complexOf(v any) (complex128, bool) {
	switch x := v.(type) {
	case complex128:
		return x, true
	case complex64:
		return complex128(x), true
	}
	return 0, false
}

// asciiFold U → S 的锁死口径:非 ASCII 码元 → '?'(不抛 Python 异常)。
func asciiFold(s string) string {
	var sb strings.Builder
	sb.Grow(len(s))
	for _, r := range s {
		if r > 127 {
			sb.WriteByte('?')
			continue
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

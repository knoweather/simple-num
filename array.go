package num

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
	"unsafe"
)

// Array:多维数组,底层字节缓冲 + dtype + shape + strides(字节)+ offset。
// 零拷贝视图共享 buf;读可并发共享,写须独占(约定,不强管)。
//
// 内存模型:
// - buf 为扁平字节缓冲;元素 i(i_k…i_{ndim-1}) 位于 off + Σ i_k·st_k;C 序缺省。
// - 本库新建缓冲基址 8 字节对齐(alignedAlloc)——x86 上这是 win[T] 型视窗
// 对所有内部 Kind(含 c128,Go Alignof(complex128)==8)的安全性前提;
// 外来缓冲(FromBytes/npy)不保证,快速路径先 isFastContig,不过则整备。
// - Views 的 off 恒为元素边界(off % Itemsize == 0,由步长倍增性保证)。
type Array struct {
	buf   []byte
	shape []int
	st    []int // strides 字节
	off   int   // 字节偏移
	dtype Dtype
}

// —— 内部工具 ——

// alignedAlloc n 字节缓冲,基址 8 字节对齐(n=0 → nil)。长度冗余 ≤8B。
func alignedAlloc(n int) []byte {
	if n <= 0 {
		return nil
	}
	raw := make([]byte, n+8)
	p := uintptr(unsafe.Pointer(unsafe.SliceData(raw)))
	d := (8 - (int(p) & 7)) & 7
	return raw[d : d+n : d+n]
}

// normShapeCopy 形状规整:复制去别名;负维/溢出熔断。
func normShapeCopy(shape []int) ([]int, int, error) {
	sh := make([]int, len(shape))
	total := 1
	for i, d := range shape {
		if d < 0 {
			return nil, 0, fmt.Errorf("num: 负维 shape[%d]=%d: %w", i, d, ErrShape)
		}
		sh[i] = d
		total *= d
		if d != 0 && total < 0 { // int 原= 64 位;溢出回绕为负
			return nil, 0, fmt.Errorf("num: shape %v 累积溢出: %w", shape, ErrShape)
		}
	}
	return sh, total, nil
}

// prodShape 空维乘积(0-d = 1)。
func prodShape(sh []int) int {
	n := 1
	for _, d := range sh {
		n *= d
	}
	return n
}

// cStrides C 序字节步长。
func cStrides(sh []int, itemsize int) []int {
	st := make([]int, len(sh))
	s := itemsize
	for i := len(sh) - 1; i >= 0; i-- {
		st[i] = s
		s *= sh[i]
	}
	return st
}

// isCContig C 序连续判定(1 维/0 维/size≤1 恒真;shape 0 → 真)。
func (a *Array) isCContig() bool {
	if a.Size() <= 1 {
		return true
	}
	it := a.dtype.Itemsize
	s := it
	for i := len(a.shape) - 1; i >= 0; i-- {
		if a.shape[i] == 1 { // 步长不可观察
			continue
		}
		if a.st[i] != s {
			return false
		}
		s *= a.shape[i]
	}
	return true
}

// isAligned8 窗口对齐:缓冲基址与 offset 都在 8 字节边界(win[T] 前提)。
func (a *Array) isAligned8() bool {
	if a.buf == nil {
		return true
	}
	p := uintptr(unsafe.Pointer(unsafe.SliceData(a.buf)))
	if p&7 != 0 {
		return false
	}
	return a.off&7 == 0
}

// elemOff 展开多维索引到字节偏移;维度数/界校验。
func (a *Array) elemOff(i []int) (int, error) {
	if len(i) != len(a.shape) {
		return 0, fmt.Errorf("num: index %v 维度数 %d ≠ %d: %w", i, len(i), len(a.shape), ErrIndex)
	}
	o := a.off
	for k, ix := range i {
		if ix < 0 || ix >= a.shape[k] {
			return 0, fmt.Errorf("num: index[%d]=%d 越界 shape[%d]=%d: %w", k, ix, k, a.shape[k], ErrIndex)
		}
		o += ix * a.st[k]
	}
	return o, nil
}

// —— 工厂 ——

func checkDt(dt Dtype) error {
	switch dt.Kind {
	case Bool, Int8, Int16, Int32, Int64,
		Uint8, Uint16, Uint32, Uint64,
		Float16, Float32, Float64, Complex64, Complex128:
		if kindItemsize(dt.Kind) != dt.Itemsize {
			return fmt.Errorf("num: dtype %s Itemsize=%d 不符: %w", dt, dt.Itemsize, ErrDtype)
		}
	case Datetime64:
		if dt.Itemsize != 8 {
			return fmt.Errorf("num: %s Itemsize≠8: %w", dt, ErrDtype)
		}
		switch dt.Unit {
		case "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns":
		default:
			return fmt.Errorf("num: datetime64 单位 %q 非法: %w", dt.Unit, ErrDtype)
		}
	case Bytes, Unicode:
		if dt.Itemsize <= 0 {
			return fmt.Errorf("num: %s 宽度必须 >0: %w", dt, ErrDtype)
		}
	default:
		return fmt.Errorf("num: 未知 Kind(%d): %w", dt.Kind, ErrDtype)
	}
	return nil
}

// New 未初始化数组。
func New(shape []int, dt Dtype) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	sh, total, err := normShapeCopy(shape)
	if err != nil {
		return nil, err
	}
	return &Array{
		buf:   alignedAlloc(total * dt.Itemsize),
		shape: sh,
		st:    cStrides(sh, dt.Itemsize),
		dtype: dt,
	}, nil
}

// Zeros 全 0(make 底本即 0)。
func Zeros(shape []int, dt Dtype) (*Array, error) { return New(shape, dt) }

// Ones 全 1(numpy 同名口径,实测 numpy 2.5.4):
// bool→true、整型/浮点/f16→1、complex→1+0j、datetime→1(单位内计数)、S/U→"1"。
func Ones(shape []int, dt Dtype) (*Array, error) {
	a, err := New(shape, dt)
	if err != nil {
		return nil, err
	}
	var v any
	switch dt.Kind {
	case Bool:
		v = true
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Datetime64:
		v = int64(1)
	case Float16, Float32, Float64:
		v = float64(1)
	case Complex64, Complex128:
		v = complex128(1)
	case Bytes, Unicode:
		v = "1" // 实测 np.ones(2,dtype="S4")=b'1\0\0\0'、U3='1'
	}
	return fillAny(a, v)
}

// fillAny 把数组每个元素编码为 v(One/Full 共用填充核)。
func fillAny(a *Array, v any) (*Array, error) {
	it := a.dtype.Itemsize
	for i := 0; i < a.Size(); i++ {
		if err := encodeOne(a.dtype, i*it, a.buf, v); err != nil {
			return nil, err
		}
	}
	return a, nil
}

// Full 全 v 数组(v 按元素编码;失败 → ErrDtype 整批)。
func Full(shape []int, v any, dt Dtype) (*Array, error) {
	a, err := New(shape, dt)
	if err != nil {
		return nil, err
	}
	return fillAny(a, v)
}

// FromBytes 已知字节(nc 桥接);buf 被接管引用。
func FromBytes(buf []byte, dt Dtype, shape []int) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	sh, _, err := normShapeCopy(shape)
	if err != nil {
		return nil, err
	}
	need := prodShape(sh) * dt.Itemsize
	if len(buf) < need {
		return nil, fmt.Errorf("num: FromBytes 需 %dB,得 %dB: %w", need, len(buf), ErrShape)
	}
	return &Array{buf: buf, shape: sh, st: cStrides(sh, dt.Itemsize), dtype: dt}, nil
}

// FromFloat64s 把 []float64 零拷贝包装成一维 f64 数组(shape=[len(x)])。
// 数组与 x 共享底层内存:任一方的写入对另一方可见;调用方需保证 x 的生命周期
// 覆盖数组的使用期。
func FromFloat64s(x []float64) (*Array, error) {
	if len(x) == 0 {
		return &Array{shape: []int{0}, st: cStrides([]int{0}, 8), dtype: DFloat64}, nil
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(x))), len(x)*8)
	return FromBytes(buf, DFloat64, []int{len(x)})
}

// Float64s 返回底层 f64 数据的零拷贝一维视图(行主序)。
// 仅当 dtype 为 float64、数组 C 连续且偏移按元素对齐时 ok=true;否则返回
// nil,false,调用方可退回 Item/SetItem 等通用路径。写入视图会直接改写数组,
// 反之亦然。
func (a *Array) Float64s() ([]float64, bool) {
	if a == nil || a.dtype.Kind != Float64 || a.dtype.Itemsize != 8 || !a.isCContig() || a.off%8 != 0 {
		return nil, false
	}
	n := a.Size()
	if n == 0 {
		return []float64{}, true
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(&a.buf[a.off])), n), true
}

// FromSlice 泛型构造:[]float64→f64、[]int64→i64、[]bool、[]int→i64 等。
func FromSlice[T any](x []T) (*Array, error) {
	var dt Dtype
	var zero T
	switch any(zero).(type) {
	case bool:
		dt = DBool
	case int8:
		dt = DInt8
	case int16:
		dt = DInt16
	case int32:
		dt = DInt32
	case int64:
		dt = DInt64
	case int:
		dt = DInt64
	case uint8:
		dt = DUint8
	case uint16:
		dt = DUint16
	case uint32:
		dt = DUint32
	case uint64:
		dt = DUint64
	case uint:
		dt = DUint64
	case float32:
		dt = DFloat32
	case float64:
		dt = DFloat64
	case complex64:
		dt = DComplex64
	case complex128:
		dt = DComplex128
	default:
		return nil, fmt.Errorf("num: FromSlice[%T] 不支持: %w", zero, ErrDtype)
	}
	sh := []int{len(x)}
	a, err := New(sh, dt)
	if err != nil {
		return nil, err
	}
	for i := range x {
		if err := encodeOne(dt, i*dt.Itemsize, a.buf, x[i]); err != nil {
			return nil, err
		}
	}
	return a, nil
}

// —— 访问器 ——

func (a *Array) Shape() []int {
	s := make([]int, len(a.shape))
	copy(s, a.shape)
	return s
}

func (a *Array) Dtype() Dtype { return a.dtype }

func (a *Array) Strides() []int {
	s := make([]int, len(a.st))
	copy(s, a.st)
	return s
}

func (a *Array) Offset() int { return a.off }
func (a *Array) Size() int   { return prodShape(a.shape) }
func (a *Array) NDim() int   { return len(a.shape) }

// Item / SetItem 标量取放(any 编码;越界 ErrIndex、类型不符 ErrDtype)。
func (a *Array) Item(i ...int) (any, error) {
	o, err := a.elemOff(i)
	if err != nil {
		return nil, err
	}
	return decodeOne(a.dtype, o, a.buf)
}

func (a *Array) SetItem(i []int, v any) error {
	o, err := a.elemOff(i)
	if err != nil {
		return err
	}
	return encodeOne(a.dtype, o, a.buf, v)
}

// —— 视图(shape 变换;numpy 同款:view-when-possible)——

// Reshape 尺寸不变(ErrShape);支持单个 -1 维推断(numpy 同款);可零拷贝则零拷贝,
// 非连续源返回拷贝(numpy 同款)。
func (a *Array) Reshape(shape []int) (*Array, error) {
	sh := make([]int, len(shape))
	copy(sh, shape)
	total := a.Size()
	neg := -1
	for i, d := range sh {
		switch {
		case d > 0:
		case d == -1:
			if neg >= 0 {
				return nil, fmt.Errorf("num: Reshape 两个 -1 维: %w", ErrShape)
			}
			neg = i
		default:
			return nil, fmt.Errorf("num: Reshape shape[%d]=%d 非法: %w", i, d, ErrShape)
		}
	}
	prod := 1
	for _, d := range sh {
		if d == -1 {
			continue
		}
		prod *= d
		if prod != 0 && prod < 0 {
			return nil, fmt.Errorf("num: Reshape %v 累积溢出: %w", shape, ErrShape)
		}
	}
	if neg >= 0 {
		if prod == 0 {
			sh[neg] = 0 // numpy: 0 大小数组里 -1 → 0
		} else if total%prod != 0 {
			return nil, fmt.Errorf("num: Reshape %v 与 size %d 不容: %w", shape, total, ErrShape)
		} else {
			sh[neg] = total / prod
		}
	}
	if prodShape(sh) != total {
		return nil, fmt.Errorf("num: Reshape %v 与 size %d 不容: %w", shape, total, ErrShape)
	}
	if a.isCContig() {
		v := *a
		v.shape = sh
		v.st = cStrides(sh, a.dtype.Itemsize)
		return &v, nil
	}
	return a.AsContiguous().Reshape(sh)
}

// Transpose 轴重排(缺省逆序);零拷贝视图。轴越界/重复 → ErrIndex/ErrShape。
func (a *Array) Transpose(axes ...int) (*Array, error) {
	n := a.NDim()
	ax := make([]int, n)
	if len(axes) == 0 {
		for i := 0; i < n; i++ {
			ax[i] = n - 1 - i
		}
	} else {
		if len(axes) != n {
			return nil, fmt.Errorf("num: Transpose 轴数 %d ≠ %d: %w", len(axes), n, ErrShape)
		}
		copy(ax, axes)
	}
	seen := make([]bool, n)
	for i, k := range ax {
		if k < 0 || k >= n {
			return nil, fmt.Errorf("num: Transpose 轴 [%d]=%d 越界: %w", i, k, ErrIndex)
		}
		if seen[k] {
			return nil, fmt.Errorf("num: Transpose 轴重复 %d: %w", k, ErrShape)
		}
		seen[k] = true
	}
	sh2 := make([]int, n)
	st2 := make([]int, n)
	for i, k := range ax {
		sh2[i] = a.shape[k]
		st2[i] = a.st[k]
	}
	return &Array{buf: a.buf, shape: sh2, st: st2, off: a.off, dtype: a.dtype}, nil
}

// AsContiguous C 序连续化 + 对齐保证(基址/偏移 8 字节);已达标返回共享视图。
func (a *Array) AsContiguous() *Array {
	if a.isCContig() && a.isAligned8() {
		v := *a
		v.shape = append([]int(nil), a.shape...)
		v.st = append([]int(nil), a.st...)
		return &v
	}
	out, err := New(a.shape, a.dtype)
	if err != nil {
		return nil // checkDt 已在 New 侧保证;防御分支不可达
	}
	copyStrided(out.buf, a.buf, a.shape, a.st, a.off, a.dtype.Itemsize)
	return out
}

// —— 元素编解码(value ↔ 字节;– 全 dtype 到位,口径见 /)——
//
// 读出面类型(Go 侧无对应原语者按最近类型升格):
// bool / int8..int64 / uint8..uint64 / float32 / float64 / complex64 / complex128 原样;
// Float16 → float32(升格);Datetime64 → int64(单位内计数,NaT = MinInt64);
// Bytes → string(去尾零、内嵌 NUL 保留);Unicode → string(UTF-32LE 码元解码)。
// 写入口:numpy 赋值 cast 口径(bool/整值/浮点互赋、实数值 → 复数 imag=0;
// datetime 仅收整值,浮点拒绝 —— numpy 实测 ValueError 同)。

func decodeOne(dt Dtype, off int, buf []byte) (any, error) {
	b := buf[off : off+dt.Itemsize]
	switch dt.Kind {
	case Bool:
		return b[0] != 0, nil
	case Int8:
		return int8(b[0]), nil
	case Uint8:
		return b[0], nil
	case Int16:
		return int16(binary.LittleEndian.Uint16(b)), nil
	case Uint16:
		return binary.LittleEndian.Uint16(b), nil
	case Int32:
		return int32(binary.LittleEndian.Uint32(b)), nil
	case Uint32:
		return binary.LittleEndian.Uint32(b), nil
	case Int64:
		return int64(binary.LittleEndian.Uint64(b)), nil
	case Uint64:
		return binary.LittleEndian.Uint64(b), nil
	case Float16:
		return f16ToF32(binary.LittleEndian.Uint16(b)), nil
	case Float32:
		return math.Float32frombits(binary.LittleEndian.Uint32(b)), nil
	case Float64:
		return math.Float64frombits(binary.LittleEndian.Uint64(b)), nil
	case Complex64:
		re := math.Float32frombits(binary.LittleEndian.Uint32(b))
		im := math.Float32frombits(binary.LittleEndian.Uint32(b[4:]))
		return complex(re, im), nil
	case Complex128:
		re := math.Float64frombits(binary.LittleEndian.Uint64(b))
		im := math.Float64frombits(binary.LittleEndian.Uint64(b[8:]))
		return complex(re, im), nil
	case Datetime64:
		return int64(binary.LittleEndian.Uint64(b)), nil
	case Bytes:
		return bytesToText(b), nil
	case Unicode:
		return unicodeToText(b), nil
	}
	return nil, fmt.Errorf("num: decode 未知 Kind(%d): %w", dt.Kind, ErrDtype)
}

func encodeOne(dt Dtype, off int, buf []byte, v any) error {
	b := buf[off : off+dt.Itemsize]
	switch dt.Kind {
	case Bool:
		x, err := coerceBool(v)
		if err != nil {
			return err
		}
		if x {
			b[0] = 1
		} else {
			b[0] = 0
		}
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
		i64, err := coerceI64(v)
		if err != nil {
			return err
		}
		storeI64(dt.Kind, b, i64) // 位保持:负值按 C 铸造截低位(回绕,numpy 同)
	case Float16:
		u, err := coerceF16(v)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint16(b, u)
	case Float32, Float64:
		f, err := coerceF64(v)
		if err != nil {
			return err
		}
		storeF64(dt.Kind, b, f)
	case Complex64, Complex128:
		c, err := coerceComplex(v)
		if err != nil {
			return err
		}
		storeComplex(dt.Kind, b, c)
	case Datetime64:
		i64, err := coerceDateTime(v)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(b, uint64(i64))
	case Bytes:
		s, err := textOfScalar(v)
		if err != nil {
			return err
		}
		n := copy(b, s) // 超出宽度静默截断(numpy 同),余位补零
		for i := n; i < len(b); i++ {
			b[i] = 0
		}
	case Unicode:
		s, err := textOfScalar(v)
		if err != nil {
			return err
		}
		w := len(b) / 4 // 码元数(按码点截断,numpy 同)
		i := 0
		for _, r := range s {
			if i >= w {
				break
			}
			if !utf8.ValidRune(r) {
				r = utf8.RuneError // 代理对/超界码点:Go string 不能承载
			}
			binary.LittleEndian.PutUint32(b[i*4:], uint32(r))
			i++
		}
		for j := i * 4; j < len(b); j++ {
			b[j] = 0
		}
	default:
		return fmt.Errorf("num: encode 未知 Kind(%d): %w", dt.Kind, ErrDtype)
	}
	return nil
}

// bytesToText |S 元素 → string:去尾零(内嵌 NUL 保留:numpy 只裁尾部)。
func bytesToText(b []byte) string {
	n := len(b)
	for n > 0 && b[n-1] == 0 {
		n--
	}
	return string(b[:n])
}

// unicodeToText <U 元素(UTF-32LE 码元)→ string:去尾 0 码元。
func unicodeToText(b []byte) string {
	n := len(b) / 4
	rs := make([]rune, n)
	for i := 0; i < n; i++ {
		rs[i] = rune(binary.LittleEndian.Uint32(b[i*4:]))
	}
	for n > 0 && rs[n-1] == 0 {
		n--
	}
	var sb strings.Builder
	for _, r := range rs[:n] {
		if !utf8.ValidRune(r) {
			r = utf8.RuneError
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

// storeI64 / storeF64 数值元素直存(kind 已由调用方校验;供 encodeOne 与
// 工厂/内核接线共用,免 any 装箱)。整型按 C 铸造语义截低位(回绕,numpy 同)。
func storeI64(k Kind, b []byte, i64 int64) {
	u64 := uint64(i64)
	switch k {
	case Int8:
		b[0] = byte(int8(i64))
	case Int16:
		binary.LittleEndian.PutUint16(b, uint16(int16(i64)))
	case Int32:
		binary.LittleEndian.PutUint32(b, uint32(int32(i64)))
	case Int64:
		binary.LittleEndian.PutUint64(b, uint64(i64))
	case Uint8:
		b[0] = byte(u64)
	case Uint16:
		binary.LittleEndian.PutUint16(b, uint16(u64))
	case Uint32:
		binary.LittleEndian.PutUint32(b, uint32(u64))
	case Uint64:
		binary.LittleEndian.PutUint64(b, u64)
	}
}

func storeF64(k Kind, b []byte, f float64) {
	if k == Float32 {
		binary.LittleEndian.PutUint32(b, math.Float32bits(float32(f)))
		return
	}
	binary.LittleEndian.PutUint64(b, math.Float64bits(f))
}

// storeComplex 复数元素直存(实部在前;c8 收窄按 float32 RNE,numpy 同)。
func storeComplex(k Kind, b []byte, c complex128) {
	if k == Complex64 {
		binary.LittleEndian.PutUint32(b, math.Float32bits(float32(real(c))))
		binary.LittleEndian.PutUint32(b[4:], math.Float32bits(float32(imag(c))))
		return
	}
	binary.LittleEndian.PutUint64(b, math.Float64bits(real(c)))
	binary.LittleEndian.PutUint64(b[8:], math.Float64bits(imag(c)))
}

// coerce 工具:Go 数值 → i64 / f64 / bool(ErrDtype reject)。

func coerceI64(v any) (int64, error) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	case int:
		return int64(x), nil
	case int8:
		return int64(x), nil
	case int16:
		return int64(x), nil
	case int32:
		return int64(x), nil
	case int64:
		return x, nil
	case uint8:
		return int64(x), nil
	case uint16:
		return int64(x), nil
	case uint32:
		return int64(x), nil
	case uint64:
		return int64(x), nil // 位语义交 encodeOne 回绕
	case uint:
		return int64(x), nil
	case float32:
		return int64(x), nil // 向零截断(numpy 赋值 cast 同)
	case float64:
		return int64(x), nil
	}
	return 0, fmt.Errorf("num: %T ↔ 整型不支持: %w", v, ErrDtype)
}

func coerceF64(v any) (float64, error) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	case int:
		return float64(x), nil
	case int8:
		return float64(x), nil
	case int16:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case int64:
		return float64(x), nil
	case uint8:
		return float64(x), nil
	case uint16:
		return float64(x), nil
	case uint32:
		return float64(x), nil
	case uint64:
		return float64(x), nil
	case uint:
		return float64(x), nil
	case float32:
		return float64(x), nil
	case float64:
		return x, nil
	}
	return 0, fmt.Errorf("num: %T ↔ 浮点不支持: %w", v, ErrDtype)
}

func coerceBool(v any) (bool, error) {
	switch x := v.(type) {
	case bool:
		return x, nil
	case int:
		return x != 0, nil
	case int8:
		return x != 0, nil
	case int16:
		return x != 0, nil
	case int32:
		return x != 0, nil
	case int64:
		return x != 0, nil
	case uint8:
		return x != 0, nil
	case uint16:
		return x != 0, nil
	case uint32:
		return x != 0, nil
	case uint64:
		return x != 0, nil
	case float32:
		return x != 0, nil
	case float64:
		return x != 0, nil
	}
	return false, fmt.Errorf("num: %T ↔ bool 不支持: %w", v, ErrDtype)
}

// coerceF16 数值 → f16 位型。float32/float64 走本精度单次 RNE(避免双舍入,
// 见);其余数值经 f64 精确提升后单次舍入(整值 ≤2^53 精确)。
func coerceF16(v any) (uint16, error) {
	switch x := v.(type) {
	case float32:
		return f32ToF16(x), nil
	case float64:
		return f64ToF16(x), nil
	}
	f, err := coerceF64(v)
	if err != nil {
		return 0, fmt.Errorf("num: %T ↔ float16 不支持: %w", v, ErrDtype)
	}
	return f64ToF16(f), nil
}

// coerceComplex 复数值/实数值 → complex128(实数值 imag=0,numpy 赋值 cast 同)。
func coerceComplex(v any) (complex128, error) {
	switch x := v.(type) {
	case complex64:
		return complex128(x), nil
	case complex128:
		return x, nil
	}
	f, err := coerceF64(v)
	if err != nil {
		return 0, fmt.Errorf("num: %T ↔ 复数不支持: %w", v, ErrDtype)
	}
	return complex(f, 0), nil
}

// coerceDateTime 整值 → datetime64 单位内计数(浮点/文本拒绝:numpy 实测
// np.zeros(1,dtype='M8[s]')[0] = 1.5 抛 ValueError;NaT 用 MinInt64 表达)。
func coerceDateTime(v any) (int64, error) {
	switch v.(type) {
	case float32, float64:
		return 0, fmt.Errorf("num: %T ↔ datetime64 不支持(须整值): %w", v, ErrDtype)
	}
	i, err := coerceI64(v)
	if err != nil {
		return 0, fmt.Errorf("num: %T ↔ datetime64 不支持: %w", v, ErrDtype)
	}
	return i, nil
}

// textOfScalar 标量 → 文本(S/U 写入用)。数值按 Python str 口径落字,
// 浮点/复数按输入标量自身精度取位数(float32→f32 最短、complex64→f32 分量),
// 与 numpy "按输入 dtype 转文本" 一致;bool → True/False;[]byte 按逐字节文本。
func textOfScalar(v any) (string, error) {
	switch x := v.(type) {
	case string:
		return x, nil
	case []byte:
		return string(x), nil
	case bool:
		if x {
			return "True", nil
		}
		return "False", nil
	case int:
		return strconv.FormatInt(int64(x), 10), nil
	case int8:
		return strconv.FormatInt(int64(x), 10), nil
	case int16:
		return strconv.FormatInt(int64(x), 10), nil
	case int32:
		return strconv.FormatInt(int64(x), 10), nil
	case int64:
		return strconv.FormatInt(x, 10), nil
	case uint:
		return strconv.FormatUint(uint64(x), 10), nil
	case uint8:
		return strconv.FormatUint(uint64(x), 10), nil
	case uint16:
		return strconv.FormatUint(uint64(x), 10), nil
	case uint32:
		return strconv.FormatUint(uint64(x), 10), nil
	case uint64:
		return strconv.FormatUint(x, 10), nil
	case float32:
		return pyFloatStr(float64(x), 32), nil
	case float64:
		return pyFloatStr(x, 64), nil
	case complex64:
		return pyComplexStr(float64(real(x)), float64(imag(x)), 32), nil
	case complex128:
		return pyComplexStr(real(x), imag(x), 64), nil
	}
	return "", fmt.Errorf("num: %T ↔ S/U 文本不支持: %w", v, ErrDtype)
}

// shapeEqual 长度与逐元素相等。
func shapeEqual(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

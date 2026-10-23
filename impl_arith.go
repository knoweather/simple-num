package num

// T7/T8:四则算子内核与适配(B 拥有)。
//
// 语义基准:numpy 2.5.4 实测 + npy_math 源码考据,逐条记 §2。
// 挂载:A 引擎的 dispatchBinary(ufunc.go)经 typed 注册面
// (registerBinaryTyped / registerBinarySpec);入口函数(Add/Sub/...)是 B 的
// 适配层,负责 numpy 类型解析在引擎单值 ForceOut 表达不了的部分:
// - add/multiply 的 bool×bool:numpy 保 bool → ForceOut=Bool 特种算子;
// - subtract 的 bool×bool:numpy 拒 → 入口层 ErrDtype( 同构);
// - divide:numpy"输入含整型 → float64" → 入口层预 Cast。
// 内核吃引擎保证的同 dtype、等长、C 序连续、8 字节对齐窗(/)。

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// errPowNegExp 整型负指数幂(numpy 2.x 同拒:ValueError"Integers to negative
// integer powers are not allowed.";四哨兵无对应项,暂包 ErrDtype,)。
var errPowNegExp = errors.New("num: integer to negative integer powers")

// —— 内核类型约束(与 kindOfVal 的绑定由 A 的 typed 注册面完成)——
type (
	Ints interface {
		int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
	}
	FloatKinds interface{ float32 | float64 }
)

// —— 整型内核(add/sub/mul:Go 原生环绕 = numpy 回绕语义,)——
// 逐元素口径(E):切片内核与标量快速路径共用同一表达式,保证两条路径逐位一致。

func addIntE[T Ints](x, y T) T { return x + y }

func subIntE[T Ints](x, y T) T { return x - y }

func mulIntE[T Ints](x, y T) T { return x * y }

// floorDivIntE 整型 floor_divide( 口径):
// - y==0 → 0(numpy 2.5.4 实测:整型零除返回 0 + warning,不 raise;
// 与 DESIGN ErrZeroDiv 口径的冲突登记);
// - 否则 Go 截断商 q=x/y,余数符号与除数不同时 q--(floor 语义);
// MinInt/-1 由 Go 规范定义为回绕,与 numpy 同值。
func floorDivIntE[T Ints](x, y T) T {
	if y == 0 {
		return 0
	}
	q := x / y
	if r := x % y; r != 0 && ((r < 0) != (y < 0)) {
		q--
	}
	return q
}

// remainderIntE 整型 remainder(符号随除数,numpy 同;y==0 → 0,同)。
func remainderIntE[T Ints](x, y T) T {
	if y == 0 {
		return 0
	}
	r := x % y
	if r != 0 && ((r < 0) != (y < 0)) {
		r += y
	}
	return r
}

// powIntE 整型幂(平方求幂,与 numpy 整型 power helper 同构,:
// 整型乘法精确后回绕,任意等价实现逐位相同)。exp 无符号型别恒 ≥0;负指数
// 先行拦下(有符号型别)——错误链与全量路径逐字同构(errPowNegExp + ErrDtype + ErrValue)。
func powIntE[T Ints](x, y T) (T, error) {
	if y < 0 {
		return 0, fmt.Errorf("num: power %d^%d: %w: %w: %w",
			x, y, errPowNegExp, ErrDtype, ErrValue)
	}
	return ipow(x, y), nil
}

func addInt[T Ints](dst, a, b []T) {
	for i := range dst {
		dst[i] = addIntE(a[i], b[i])
	}
}

func subInt[T Ints](dst, a, b []T) {
	for i := range dst {
		dst[i] = subIntE(a[i], b[i])
	}
}

func mulInt[T Ints](dst, a, b []T) {
	for i := range dst {
		dst[i] = mulIntE(a[i], b[i])
	}
}

func floorDivInt[T Ints](dst, a, b []T) {
	for i := range dst {
		dst[i] = floorDivIntE(a[i], b[i])
	}
}

func remainderInt[T Ints](dst, a, b []T) {
	for i := range dst {
		dst[i] = remainderIntE(a[i], b[i])
	}
}

// ipow 整型幂(平方求幂,与 numpy 整型 power helper 同构,:
// 整型乘法精确后回绕,任意等价实现逐位相同)。exp 无符号型别恒 ≥0;负指数在
// powInt 入口先行拦下(有符号型别)。
func ipow[T Ints](base, exp T) T {
	if exp == 0 || base == 1 {
		return 1
	}
	if exp == 1 {
		return base
	}
	if exp == 2 {
		return base * base
	}
	out := base // first_bit = exp&1
	if exp&1 == 0 {
		out = 1
	}
	e := exp >> 1
	for e > 0 {
		base *= base
		if e&1 == 1 {
			out *= base
		}
		e >>= 1
	}
	return out
}

// powInt 整型 power 内核(签名带 error:负指数 fail-loud,)。
// 无符号型别的 e<0 为编译期恒假,不产生分支成本。
// 错误链:errPowNegExp(文本)+ ErrDtype + ErrValue 双 %w( 落地)——
// ErrDtype 列前保 rootOf 单链首臂口径,errors.Is 两哨兵皆真。
func powInt[T Ints](dst, a, b []T) error {
	for i := range dst {
		v, err := powIntE(a[i], b[i])
		if err != nil {
			return err
		}
		dst[i] = v
	}
	return nil
}

// —— 浮点内核(add/sub/mul/div:IEEE 直通,0/0→NaN、1/0→±Inf,)——
// 逐元素口径与整型同源:切片内核与标量快速路径共用同一表达式。

func addFE[T FloatKinds](x, y T) T { return x + y }

func subFE[T FloatKinds](x, y T) T { return x - y }

func mulFE[T FloatKinds](x, y T) T { return x * y }

func divFE[T FloatKinds](x, y T) T { return x / y }

func addF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = addFE(a[i], b[i])
	}
}

func subF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = subFE(a[i], b[i])
	}
}

func mulF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = mulFE(a[i], b[i])
	}
}

func divF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = divFE(a[i], b[i])
	}
}

// npF32 T 类型判别(编译期常量;npFmod 的 f32/f64 位型分叉用)。
func npF32[T FloatKinds]() bool {
	var z T
	_, ok := any(z).(float32)
	return ok
}

// npQNaN 参考平台(UCRT)静态 NaN 位型:f64 正 qNaN
// 0x7FF8000000000000、负 indefinite 0xFFF8000000000000;f32 0x7FC00000 /
// 0xFFC00000。fixtures 在本机(UCRT)生成,Go 侧逐位复刻才能过 bitwise gate;
// 若 fixtures 换平台重生成,位型需重新转录(记)。
func npQNaN[T FloatKinds](neg bool) T {
	if npF32[T]() {
		if neg {
			return T(math.Float32frombits(0xffc00000))
		}
		return T(math.Float32frombits(0x7fc00000))
	}
	if neg {
		return T(math.Float64frombits(0xfff8000000000000))
	}
	return T(math.Float64frombits(0x7ff8000000000000))
}

// npFmod C fmod 口径:数值路径与 math.Mod 一致(实测与 UCRT 逐一相等),
// 差异全在 NaN 位型 —— Go math.Mod 把所有 NaN/±Inf/±0 入口规范成 Go 的
// NaN=0x…001,而 numpy 使用的 UCRT fa：
// - 输入任一 NaN → 传播该 NaN(payload 原样保留;numpy 实测);
// - a = ±Inf → 静态 NaN:f64 随被除数符号(+inf→0x7FF8…0,−inf→0xFFF8…0),
// f32 两者皆 0xFFC00000;
// - b = ±0 且 a 有限 → 静态 NaN:f64 0x7FF8…0,f32 0xFFC00000。
func npFmod[T FloatKinds](a, b T) T {
	if math.IsNaN(float64(a)) {
		return a
	}
	if math.IsNaN(float64(b)) {
		return b
	}
	if math.IsInf(float64(a), 0) {
		if npF32[T]() {
			return npQNaN[T](true)
		}
		return npQNaN[T](math.Signbit(float64(a)))
	}
	if float64(b) == 0 {
		return npQNaN[T](npF32[T]())
	}
	return T(math.Mod(float64(a), float64(b)))
}

// npFloorDiv 复刻 numpy npy_floor_divide@c@(npy_divmod, 逐行):
//
//	b==0 → a/b(IEEE:±Inf/NaN;7.5/-0.0 → -Inf)
//	否则:mod=fmod(a,b);div=(a-mod)/b;mod≠0 且 (b<0)≠(mod<0) → mod+=b,div--;
//	mod==0 → mod=copysign(0,b);div≠0 → floor(div) 且 div-floor>0.5 时 +1(snap);
//	div==0 → copysign(0, a/b)。
//
// 每步显式 T 回绕:f32 与 numpy f32 运算逐位同构;f64 时 T 为空操作。
// fmod 走 npFmod(NaN 位型按参考平台复刻,);x86 SSE 双 QNaN 取 src1、
// 单 QNaN 取 QNaN 操作数,div=(a-mod)/b 因此原样带走 fmod 的位型。
func npFloorDiv[T FloatKinds](a, b T) T {
	if b == 0 {
		return a / b
	}
	af, bf := float64(a), float64(b)
	mod := T(npFmod(a, b)) // fmod 精确
	div := T((af - float64(mod)) / bf)
	if mod != 0 {
		if (b < 0) != (mod < 0) {
			mod += b
			div = T(float64(div) - 1)
		}
	} else {
		mod = T(math.Copysign(0, bf))
	}
	if div != 0 {
		fl := T(math.Floor(float64(div)))
		if float64(div)-float64(fl) > 0.5 {
			fl = T(float64(fl) + 1)
		}
		return fl
	}
	return T(math.Copysign(0, af/bf))
}

// npRemainder 复刻 numpy npy_remainder@c@(/§2.8):
//
//	b==0 → fmod(a,b)(恒 NaN;位型见 npFmod);否则 divmod 余数:fmod 结果符号与
//	除数不同时 +b。
func npRemainder[T FloatKinds](a, b T) T {
	if b == 0 {
		return T(npFmod(a, b))
	}
	mod := T(npFmod(a, b))
	if mod != 0 {
		if (b < 0) != (mod < 0) {
			mod += b
		}
	} else {
		mod = T(math.Copysign(0, float64(b)))
	}
	return mod
}

// 逐元素口径(与上面浮点内核同源):
func floorDivFE[T FloatKinds](x, y T) T { return npFloorDiv(x, y) }

func remainderFE[T FloatKinds](x, y T) T { return npRemainder(x, y) }

// powFE 浮点幂:T(math.Pow(f64(x), f64(y)))。npy_pow 即 C pow 薄封装,
// Go math.Pow 特殊值表与 C99 一致;门槛 rel(f8 4e-16 / f4 1e-6,)。
func powFE[T FloatKinds](x, y T) T { return T(math.Pow(float64(x), float64(y))) }

func floorDivF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = floorDivFE(a[i], b[i])
	}
}

func remainderF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = remainderFE(a[i], b[i])
	}
}

func powF[T FloatKinds](dst, a, b []T) {
	for i := range dst {
		dst[i] = powFE(a[i], b[i])
	}
}

// —— bool×bool 特种算子(:numpy add/multiply 保 bool;引擎 promote
// (bool,bool)=Int8,用 ForceOut=Bool 特种算子纠正。操作数由引擎归一为
// int8 0/1 窗,OR/AND 即 !=0 合取)——

func boolOrKernel(dst, a, b []byte) error {
	x, y, z := win[bool](dst), win[int8](a), win[int8](b)
	for i := range x {
		x[i] = y[i] != 0 || z[i] != 0
	}
	return nil
}

func boolAndKernel(dst, a, b []byte) error {
	x, y, z := win[bool](dst), win[int8](a), win[int8](b)
	for i := range x {
		x[i] = y[i] != 0 && z[i] != 0
	}
	return nil
}

// f16BinKernel f16 二元内核(,numpy HALF 策略):f16 → f32 → 运算 → f16。
// 实测 0.1+0.2: f32 中间值 0.2999267578125 → f16 0.2998046875(0x34CC)✓。
func f16BinKernel(f func(dst, a, b []float32)) BinaryKernel {
	return func(dst, a, b []byte) error {
		n := len(dst) / 2
		df := make([]float32, n)
		af := make([]float32, n)
		bf := make([]float32, n)
		for i := 0; i < n; i++ {
			af[i] = f16ToF32(binary.LittleEndian.Uint16(a[i*2:]))
			bf[i] = f16ToF32(binary.LittleEndian.Uint16(b[i*2:]))
		}
		f(df, af, bf)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint16(dst[i*2:], f32ToF16(df[i]))
		}
		return nil
	}
}

// —— 标量快速路径注册面(0-stride 逐元素展开;逐元素口径与全量内核同源,
// 保证快速路径与全量路径逐位一致)——

// registerScalarPair 为 op 的 T kind 注册左/右标量内核:
// ScalarL = dst[i] = elem(s, a[i])(左操作数是标量),
// ScalarR = dst[i] = elem(a[i], s)(右操作数是标量)。
func registerScalarPair[T any](op string, elem func(x, y T) T) {
	registerBinaryScalarTyped[T](op,
		func(dst, a []T, s T) {
			for i := range dst {
				dst[i] = elem(s, a[i])
			}
		},
		func(dst, a []T, s T) {
			for i := range dst {
				dst[i] = elem(a[i], s)
			}
		})
}

// registerScalarPairErr 同 registerScalarPair,逐元素带 error(power 整型负指数
// 的 fail-loud 口径与全量内核同源)。
func registerScalarPairErr[T Ints](op string, elem func(x, y T) (T, error)) {
	k := kindOfVal[T]()
	l := map[Kind]BinaryScalarKernel{k: func(dst, norm, scalar []byte) error {
		d, a := win[T](dst), win[T](norm)
		s := win[T](scalar)[0]
		for i := range d {
			v, err := elem(s, a[i])
			if err != nil {
				return err
			}
			d[i] = v
		}
		return nil
	}}
	r := map[Kind]BinaryScalarKernel{k: func(dst, norm, scalar []byte) error {
		d, a := win[T](dst), win[T](norm)
		s := win[T](scalar)[0]
		for i := range d {
			v, err := elem(a[i], s)
			if err != nil {
				return err
			}
			d[i] = v
		}
		return nil
	}}
	registerBinaryScalar(op, l, r)
}

// f16ScalarPair f16 标量快速路径( 同策略:f16→f32 运算→f16;scalar 窗恰 2 字节)。
func f16ScalarPair(op string, elem func(x, y float32) float32) {
	l := map[Kind]BinaryScalarKernel{Float16: func(dst, norm, scalar []byte) error {
		n := len(dst) / 2
		s := f16ToF32(binary.LittleEndian.Uint16(scalar))
		for i := 0; i < n; i++ {
			a := f16ToF32(binary.LittleEndian.Uint16(norm[i*2:]))
			binary.LittleEndian.PutUint16(dst[i*2:], f32ToF16(elem(s, a)))
		}
		return nil
	}}
	r := map[Kind]BinaryScalarKernel{Float16: func(dst, norm, scalar []byte) error {
		n := len(dst) / 2
		s := f16ToF32(binary.LittleEndian.Uint16(scalar))
		for i := 0; i < n; i++ {
			a := f16ToF32(binary.LittleEndian.Uint16(norm[i*2:]))
			binary.LittleEndian.PutUint16(dst[i*2:], f32ToF16(elem(a, s)))
		}
		return nil
	}}
	registerBinaryScalar(op, l, r)
}

// —— 注册(init 分片合入;A 的 _test_ 名与 B 的算子名不冲突,)——

func init() {
	// bool×bool 特种算子(路由见 Add/Multiply 入口)。
	registerBinarySpec("add_bool", BinarySpec{ForceOut: Bool,
		Kernels: map[Kind]BinaryKernel{Bool: boolOrKernel}})
	registerBinarySpec("multiply_bool", BinarySpec{ForceOut: Bool,
		Kernels: map[Kind]BinaryKernel{Bool: boolAndKernel}})

	// 整型:add/sub/mul/floor_divide/remainder 全 dtype。
	for _, reg := range []func(){
		func() { arithInts[int8]() }, func() { arithInts[int16]() },
		func() { arithInts[int32]() }, func() { arithInts[int64]() },
		func() { arithInts[uint8]() }, func() { arithInts[uint16]() },
		func() { arithInts[uint32]() }, func() { arithInts[uint64]() },
	} {
		reg()
	}
	// 整型 power(带 error 签名,单独注册)。
	for _, reg := range []func(){
		func() { arithPow[int8]() }, func() { arithPow[int16]() },
		func() { arithPow[int32]() }, func() { arithPow[int64]() },
		func() { arithPow[uint8]() }, func() { arithPow[uint16]() },
		func() { arithPow[uint32]() }, func() { arithPow[uint64]() },
	} {
		reg()
	}
	// 浮点:全部四则(divide 仅 f32/f64 有内核;整型输入由 Div 入口预 Cast)。
	arithFloats[float32]()
	arithFloats[float64]()

	// f16:numpy HALF 策略 —— f16→f32 运算→f16;promote(f16,f16)=f16。
	registerBinarySpec("add", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(addF[float32])}})
	registerBinarySpec("subtract", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(subF[float32])}})
	registerBinarySpec("multiply", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(mulF[float32])}})
	registerBinarySpec("divide", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(divF[float32])}})
	registerBinarySpec("floor_divide", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(floorDivF[float32])}})
	registerBinarySpec("remainder", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(remainderF[float32])}})
	registerBinarySpec("power", BinarySpec{Kernels: map[Kind]BinaryKernel{
		Float16: f16BinKernel(powF[float32])}})
	f16ScalarPair("add", addFE[float32])
	f16ScalarPair("subtract", subFE[float32])
	f16ScalarPair("multiply", mulFE[float32])
	f16ScalarPair("divide", divFE[float32])
	f16ScalarPair("floor_divide", floorDivFE[float32])
	f16ScalarPair("remainder", remainderFE[float32])
	f16ScalarPair("power", powFE[float32])
}

// arithInts 为一个整型注册 add/subtract/multiply/floor_divide/remainder
// (全量内核 + 同 kind 集合的标量快速路径内核)。
func arithInts[T Ints]() {
	registerBinaryTyped[T]("add", addInt[T])
	registerBinaryTyped[T]("subtract", subInt[T])
	registerBinaryTyped[T]("multiply", mulInt[T])
	registerBinaryTyped[T]("floor_divide", floorDivInt[T])
	registerBinaryTyped[T]("remainder", remainderInt[T])
	registerScalarPair[T]("add", addIntE[T])
	registerScalarPair[T]("subtract", subIntE[T])
	registerScalarPair[T]("multiply", mulIntE[T])
	registerScalarPair[T]("floor_divide", floorDivIntE[T])
	registerScalarPair[T]("remainder", remainderIntE[T])
}

// arithPow 为一个整型注册 power(内核带 error:负指数 fail-loud)。
func arithPow[T Ints]() {
	k := kindOfVal[T]()
	registerBinarySpec("power", BinarySpec{Kernels: map[Kind]BinaryKernel{
		k: func(dst, a, b []byte) error {
			return powInt[T](win[T](dst), win[T](a), win[T](b))
		},
	}})
	registerScalarPairErr[T]("power", powIntE[T])
}

// arithFloats 为一个浮点注册全部四则(含 divide)。
func arithFloats[T FloatKinds]() {
	registerBinaryTyped[T]("add", addF[T])
	registerBinaryTyped[T]("subtract", subF[T])
	registerBinaryTyped[T]("multiply", mulF[T])
	registerBinaryTyped[T]("divide", divF[T])
	registerBinaryTyped[T]("floor_divide", floorDivF[T])
	registerBinaryTyped[T]("remainder", remainderF[T])
	registerBinaryTyped[T]("power", powF[T])
	registerScalarPair[T]("add", addFE[T])
	registerScalarPair[T]("subtract", subFE[T])
	registerScalarPair[T]("multiply", mulFE[T])
	registerScalarPair[T]("divide", divFE[T])
	registerScalarPair[T]("floor_divide", floorDivFE[T])
	registerScalarPair[T]("remainder", remainderFE[T])
	registerScalarPair[T]("power", powFE[T])
}

// —— 适配入口(B 的公共 API 面;算子名 = numpy ufunc 名,Go 侧 PascalCase)——
//
// 语义与错误模型:广播/晋级/零除/边角全部按 §2;错误 fail-loud
// wrap 四哨兵,禁 panic。

// bothBool 两操作数是否都是 bool dtype(bool×bool 需要 numpy 特判路由)。
func bothBool(x, y *Array) bool {
	return x.Dtype().Kind == Bool && y.Dtype().Kind == Bool
}

// isIntegerKind 整型族(bool 视作整型族成员:promote(bool,*)=整型阶梯)。
func isIntegerKind(k Kind) bool {
	switch k {
	case Bool, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
		return true
	}
	return false
}

// asI8IfBoolBoth floor_divide/remainder/power 的 bool×bool 适配:numpy 输出
// int8,而引擎 promote 同 dtype 快路(promoteTypes(bool,bool)=Bool)没有 bool
// 输出内核 —— 入口层预 Cast 到 int8(Cast 权限 b→i1 = Y),与 同思路。
func asI8IfBoolBoth(x, y *Array) (*Array, *Array, error) {
	if !bothBool(x, y) {
		return x, y, nil
	}
	i8 := Dtype{Kind: Int8, Itemsize: 1}
	var err error
	if x, err = Cast(x, i8); err != nil {
		return nil, nil, fmt.Errorf("num: 算子 bool×bool → int8: %w", err)
	}
	if y, err = Cast(y, i8); err != nil {
		return nil, nil, fmt.Errorf("num: 算子 bool×bool → int8: %w", err)
	}
	return x, y, nil
}

// Add x+y(numpy add)。bool×bool → bool(OR);datetime64 ± 整型走 路径。
func Add(x, y *Array) (*Array, error) {
	if x.Dtype().Kind == Datetime64 || y.Dtype().Kind == Datetime64 {
		return dtAdd(x, y, false)
	}
	if bothBool(x, y) {
		return dispatchBinary("add_bool", x, y)
	}
	return dispatchBinary("add", x, y)
}

// Subtract x-y(numpy subtract)。bool×bool → ErrDtype(numpy 同拒,);
// datetime64 - 整型走;datetime64 - datetime64 是 timedelta64(本库不做)。
func Subtract(x, y *Array) (*Array, error) {
	if x.Dtype().Kind == Datetime64 || y.Dtype().Kind == Datetime64 {
		if x.Dtype().Kind == Datetime64 && y.Dtype().Kind == Datetime64 {
			return nil, fmt.Errorf("num: datetime64 - datetime64 → timedelta64(本库不做): %w", ErrDtype)
		}
		return dtAdd(x, y, true)
	}
	if bothBool(x, y) {
		return nil, fmt.Errorf("num: Subtract(bool,bool) 不受支持(numpy boolean subtract 同拒): %w", ErrDtype)
	}
	return dispatchBinary("subtract", x, y)
}

// Multiply x*y(numpy multiply)。bool×bool → bool(AND)。
func Multiply(x, y *Array) (*Array, error) {
	if bothBool(x, y) {
		return dispatchBinary("multiply_bool", x, y)
	}
	return dispatchBinary("multiply", x, y)
}

// Div x/y(numpy divide ≡ true_divide)。numpy 类型解析:输入含整型 → float64,
// 否则 = promote(f4×i4→f8 等);float 侧 IEEE(0/0→NaN,x/0→±Inf)。。
func Div(x, y *Array) (*Array, error) {
	pdt, err := promoteTypes(x.Dtype(), y.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: Div: %w", err)
	}
	if isIntegerKind(pdt.Kind) {
		f8 := Dtype{Kind: Float64, Itemsize: 8}
		if x, err = Cast(x, f8); err != nil {
			return nil, fmt.Errorf("num: Div: %w", err)
		}
		if y, err = Cast(y, f8); err != nil {
			return nil, fmt.Errorf("num: Div: %w", err)
		}
	}
	return dispatchBinary("divide", x, y)
}

// FloorDivide x//y(numpy floor_divide;整型零除 → 0,;float 走 npy_divmod)。
func FloorDivide(x, y *Array) (*Array, error) {
	x, y, err := asI8IfBoolBoth(x, y)
	if err != nil {
		return nil, err
	}
	return dispatchBinary("floor_divide", x, y)
}

// Remainder x%%y(numpy remainder ≡ mod;符号随除数;整型零除 → 0,)。
func Remainder(x, y *Array) (*Array, error) {
	x, y, err := asI8IfBoolBoth(x, y)
	if err != nil {
		return nil, err
	}
	return dispatchBinary("remainder", x, y)
}

// Power x**y(numpy power;整型负指数 → fail-loud,;float 走 math.Pow)。
func Power(x, y *Array) (*Array, error) {
	x, y, err := asI8IfBoolBoth(x, y)
	if err != nil {
		return nil, err
	}
	return dispatchBinary("power", x, y)
}

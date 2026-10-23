package num

import (
	"fmt"
	"unsafe"
)

// ufunc 分发底座(A):算子注册 + 广播/类型归一 + 内核调度。
// 内核签名( §2.3):全部同 dtype、同 shape、C 序连续扁平切片;
// 引擎保证窗基址 8 字节对齐、元素边界、等长(/)。错误只出形状/零除类。
// 内核适配(B,impl_*.go init)走 registerUnarySpec/registerBinarySpec 或
// typed 便捷口(registerUnaryTyped/registerBinaryTyped/registerCompareTyped);
// 晋级策略(unaryFallback/ForceOut)按 numpy 实测在注册时声明:
// - exp/log/sin/... 整型输入 → Float64(numpy ufunc 类型解析,int 无循环);
// - div(True divide)任意整型对 → Float64;
// - cmp 系列(lt/eq/...)→ ForceOut=Bool(registerCompareTyped 自动完成)。
// 注册允许分片合入(先 registerBinaryTyped 再补充 ForceOut),重复内核覆盖。

// UnaryKernel 一元内核:dst/src 等长(elem 数 = len/Itemsize)。
type UnaryKernel func(dst, src []byte) error

// BinaryKernel 二元内核:dst/a/b 等长(cmp 系列的 dst 为 bool 字节窗)。
type BinaryKernel func(dst, a, b []byte) error

// BinaryScalarKernel 广播标量快速路径内核:dst/norm 等长(norm = 与 dst 等长的
// 已归一窗),scalar = 恰 1 元素窗。0-stride 逐元素展开,不物化广播副本。
// ScalarL 语义 dst[i] = op(scalar, norm[i]);ScalarR 语义 dst[i] = op(norm[i], scalar)。
type BinaryScalarKernel func(dst, norm, scalar []byte) error

// UnarySpec 一元算子规格:按输入 Kind 挂内核;Fallback 为整型/未注册输入的
// 晋级链(按序取第一个已注册目标,exp 系 = {Float64},f16 内核 就位后
// 可为 {Float16, Float64} 与 numpy 逐位对齐)。
type UnarySpec struct {
	Kernels  map[Kind]UnaryKernel
	Fallback []Kind
}

// BinarySpec 二元算子规格:Kernels 按输出 Kind 键;ForceOut≠0 时覆盖
// promote 结果(div → Float64;cmp → Bool)。操作数一律先归一到输出 dtype。
// 查找键(/):ForceOut==Bool 的算子(cmp 系列)按输入 Kind(per-idt)查内核
// —— Bool 单键会被全部输入 kind 互相覆盖(最后注册者获胜);其余算子(含 div,
// ForceOut=Float64 或 0)维持 outKind 查找,一字不变。
// ScalarL/ScalarR 为广播标量快速路径内核(按输入 Kind 键):一侧操作数恰
// 1 元素时按方向逐元素展开,不物化整广播副本;未注册 = 走 Kernels 全量路径。
type BinarySpec struct {
	Kernels  map[Kind]BinaryKernel
	ForceOut Kind
	ScalarL  map[Kind]BinaryScalarKernel // 左操作数是标量:dst[i] = op(s, a[i])
	ScalarR  map[Kind]BinaryScalarKernel // 右操作数是标量:dst[i] = op(a[i], s)
}

var (
	unaryReg  = map[string]*UnarySpec{}
	binaryReg = map[string]*BinarySpec{}
)

// registerUnarySpec 合入一元规格(nil Kernels 自动补)。
func registerUnarySpec(op string, s UnarySpec) {
	o := unaryReg[op]
	if o == nil {
		o = &UnarySpec{Kernels: map[Kind]UnaryKernel{}}
	}
	for k, fn := range s.Kernels {
		if fn == nil {
			panic("num registerUnarySpec nil fn") // 内部使用,程序员错误
		}
		o.Kernels[k] = fn
	}
	if len(s.Fallback) > 0 {
		o.Fallback = s.Fallback
	}
	unaryReg[op] = o
}

// registerBinarySpec 合入二元规格(Kernels / ScalarL / ScalarR 同规:
// nil map 跳过,nil fn panic,按键追加不冲既有)。
func registerBinarySpec(op string, s BinarySpec) {
	o := binaryReg[op]
	if o == nil {
		o = &BinarySpec{Kernels: map[Kind]BinaryKernel{}}
	}
	for k, fn := range s.Kernels {
		if fn == nil {
			panic("num registerBinarySpec nil fn") // 内部使用,程序员错误
		}
		o.Kernels[k] = fn
	}
	for k, fn := range s.ScalarL {
		if fn == nil {
			panic("num registerBinarySpec nil fn") // 内部使用,程序员错误
		}
		if o.ScalarL == nil {
			o.ScalarL = map[Kind]BinaryScalarKernel{}
		}
		o.ScalarL[k] = fn
	}
	for k, fn := range s.ScalarR {
		if fn == nil {
			panic("num registerBinarySpec nil fn") // 内部使用,程序员错误
		}
		if o.ScalarR == nil {
			o.ScalarR = map[Kind]BinaryScalarKernel{}
		}
		o.ScalarR[k] = fn
	}
	if s.ForceOut != 0 {
		o.ForceOut = s.ForceOut
	}
	binaryReg[op] = o
}

// registerBinaryScalar 合入标量快速路径内核(nil map 跳过,nil fn panic,
// 与 Kernels 同规)。供 impl_*.go 的适配层使用。
func registerBinaryScalar(op string, left, right map[Kind]BinaryScalarKernel) {
	registerBinarySpec(op, BinarySpec{ScalarL: left, ScalarR: right})
}

// kindOfVal Go 类型 ↔ 内部 Kind(T 型窗口与 dtype 的绑定面)。
func kindOfVal[T any]() Kind {
	var z T
	switch any(z).(type) {
	case bool:
		return Bool
	case int8:
		return Int8
	case int16:
		return Int16
	case int32:
		return Int32
	case int64:
		return Int64
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint64:
		return Uint64
	case float32:
		return Float32
	case float64:
		return Float64
	case complex64:
		return Complex64
	case complex128:
		return Complex128
	}
	return 0
}

// registerUnaryTyped 一元内核 typed 注册(out/in 同 T,kernel 内核类型自动匹配)。
func registerUnaryTyped[T any](op string, fn func(dst, src []T)) {
	k := kindOfVal[T]()
	if k == 0 {
		panic("num registerUnaryTyped unsupported T") // 内部使用,程序员错误
	}
	registerUnarySpec(op, UnarySpec{Kernels: map[Kind]UnaryKernel{k: func(dst, src []byte) error {
		fn(win[T](dst), win[T](src))
		return nil
	}}})
}

// registerBinaryTyped 二元内核 typed 注册(输出与两输入同 T;cmp 系列勿用)。
func registerBinaryTyped[T any](op string, fn func(dst, a, b []T)) {
	k := kindOfVal[T]()
	if k == 0 {
		panic("num registerBinaryTyped unsupported T") // 内部使用,程序员错误
	}
	registerBinarySpec(op, BinarySpec{Kernels: map[Kind]BinaryKernel{k: func(dst, a, b []byte) error {
		fn(win[T](dst), win[T](a), win[T](b))
		return nil
	}}})
}

// registerBinaryScalarTyped 标量内核 typed 注册(输出与数组侧同 T,scalar 窗恰
// 1 元素)。left/right 传 nil = 该方向不注册(回退到全量内核,保证不变)。
func registerBinaryScalarTyped[T any](op string, left, right func(dst, a []T, s T)) {
	k := kindOfVal[T]()
	if k == 0 {
		panic("num registerBinaryScalarTyped unsupported T") // 内部使用,程序员错误
	}
	var l, r map[Kind]BinaryScalarKernel
	if left != nil {
		l = map[Kind]BinaryScalarKernel{k: func(dst, norm, scalar []byte) error {
			left(win[T](dst), win[T](norm), win[T](scalar)[0])
			return nil
		}}
	}
	if right != nil {
		r = map[Kind]BinaryScalarKernel{k: func(dst, norm, scalar []byte) error {
			right(win[T](dst), win[T](norm), win[T](scalar)[0])
			return nil
		}}
	}
	registerBinaryScalar(op, l, r)
}

// registerCompareTyped 比较内核 typed 注册(dst=bool 窗,a/b=T 窗;自动 ForceOut Bool)。
// 键 = 输入 kind(/):Bool 单键会被多 dtype 互相覆盖,故按 T 的 kind 存,
// 与 dispatchBinary 的 per-idt 查找配套。
func registerCompareTyped[T any](op string, fn func(dst []bool, a, b []T)) {
	k := kindOfVal[T]()
	if k == 0 {
		panic("num registerCompareTyped unsupported T") // 内部使用,程序员错误
	}
	registerBinarySpec(op, BinarySpec{ForceOut: Bool, Kernels: map[Kind]BinaryKernel{k: func(dst, a, b []byte) error {
		fn(win[bool](dst), win[T](a), win[T](b))
		return nil
	}}})
}

// win 把 C 序连续、元素边界对齐的字节视窗转成元素切片(仅引擎保证的前提下)。
func win[T any](b []byte) []T {
	var z T
	sz := int(unsafe.Sizeof(z))
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/sz)
}

// engineWindow 取内核直通窗;fastWindow 失败 = 内部不变量破坏(panic 探针,
// 非用户路径:所有引擎入口都经 normalizeOperand 整备)。唯 size-0 例外:
// New(0-size) 给 buf=nil → fastWindow 回 nil,属正常空窗而非不变量破坏
// ( 修:返回空非 nil 切片,size-0 算子不再 panic)。
func engineWindow(a *Array) []byte {
	w := a.fastWindow()
	if w != nil {
		return w
	}
	if a.Size() == 0 {
		return []byte{} // size-0:空窗,非 nil( 修)
	}
	panic("num: engine window invariant broken")
}

// normalizeOperand 引擎操作数归一:同 dtype 且形状已达标且窗口合格 → 零拷贝直通;
// 否则 transformCopy(广播+cast+连续对齐一体的拷贝)。
func normalizeOperand(a *Array, outSh []int, dt Dtype) (*Array, error) {
	if a.Dtype() == dt && shapeEqual(a.Shape(), outSh) {
		if a.fastWindow() != nil {
			return a, nil
		}
	}
	return transformCopy(a, outSh, dt)
}

// dispatchUnary 一元分发。晋级链:Fallback 按序取第一个已注册目标。
func dispatchUnary(op string, a *Array) (*Array, error) {
	spec := unaryReg[op]
	if spec == nil {
		return nil, fmt.Errorf("num: ufunc %q 未注册: %w", op, ErrDtype)
	}
	dt := a.Dtype()
	k, ok := spec.Kernels[dt.Kind]
	if !ok {
		for _, fk := range spec.Fallback {
			if kk, have := spec.Kernels[fk]; have {
				dt = Dtype{Kind: fk, Itemsize: kindItemsize(fk)}
				k = kk
				ok = true
				break
			}
		}
	}
	if !ok {
		return nil, fmt.Errorf("num: ufunc %q 无 %s 内核: %w", op, a.Dtype(), ErrDtype)
	}
	out, err := New(a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	ax, err := normalizeOperand(a, a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	if err := k(engineWindow(out), engineWindow(ax)); err != nil {
		return nil, err
	}
	return out, nil
}

// dispatchBinary 二元分发:广播 → promote → ForceOut 覆盖 → 归一两操作数 → 内核。
func dispatchBinary(op string, x, y *Array) (*Array, error) {
	spec := binaryReg[op]
	if spec == nil {
		return nil, fmt.Errorf("num: ufunc %q 未注册: %w", op, ErrDtype)
	}
	outSh, err := broadcastShapes(x.Shape(), y.Shape())
	if err != nil {
		return nil, fmt.Errorf("num: ufunc %q: %w", op, err)
	}
	pdt, err := promoteTypes(x.Dtype(), y.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: ufunc %q: %w", op, err)
	}
	// 操作数归一 dtype = promote 结果;输出 dtype = ForceOut 覆盖(cmp:入 f64、出 bool)。
	inKind := pdt.Kind
	outKind := pdt.Kind
	if spec.ForceOut != 0 {
		outKind = spec.ForceOut
	}
	idt := Dtype{Kind: inKind, Itemsize: kindItemsize(inKind)}
	odt := Dtype{Kind: outKind, Itemsize: kindItemsize(outKind)}
	// 查找键(/):ForceOut==Bool 的算子按输入 kind(per-idt)查内核;
	// 其余算子(含 div)维持 outKind 查找。lk = 实际查到的 kind(错误文案用)。
	key := outKind
	if spec.ForceOut == Bool {
		key = inKind
	}
	lk := Dtype{Kind: key, Itemsize: kindItemsize(key)}
	k, ok := spec.Kernels[key]
	if !ok {
		return nil, fmt.Errorf("num: ufunc %q 无 %s 输出内核: %w", op, lk, ErrDtype)
	}
	// —— 广播标量快速路径(0-stride 逐元素展开)——
	// 触发条件:outSize>1 且(左操作数恰 1 元素且 ScalarL[inKind] 有内核,
	// 或右操作数恰 1 元素且 ScalarR[inKind] 有内核)。非标量一侧照常
	// normalizeOperand(同 dtype 时零拷贝,维持现状);标量一侧不广播,只做
	// 1 元素 dtype 归一(同一 transformCopy 转码 → 与全量路径逐位一致)。
	// 任何条件不满足(双标量、outSize≤1、无对应 Scalar 内核、未注册 kind)
	// → 落到下面完全不变的既有路径(回退保证)。
	if prodShape(outSh) > 1 {
		var sk BinaryScalarKernel
		var arr, sc *Array
		if x.Size() == 1 && spec.ScalarL[inKind] != nil {
			sk, arr, sc = spec.ScalarL[inKind], y, x
		} else if y.Size() == 1 && spec.ScalarR[inKind] != nil {
			sk, arr, sc = spec.ScalarR[inKind], x, y
		}
		if sk != nil {
			oa, err := normalizeOperand(arr, outSh, idt)
			if err != nil {
				return nil, err
			}
			os, err := transformCopy(sc, sc.Shape(), idt) // 恰 1 元素
			if err != nil {
				return nil, err
			}
			out, err := New(outSh, odt)
			if err != nil {
				return nil, err
			}
			if err := sk(engineWindow(out), engineWindow(oa), engineWindow(os)); err != nil {
				return nil, err
			}
			return out, nil
		}
	}
	ox, err := normalizeOperand(x, outSh, idt)
	if err != nil {
		return nil, err
	}
	oy, err := normalizeOperand(y, outSh, idt)
	if err != nil {
		return nil, err
	}
	out, err := New(outSh, odt)
	if err != nil {
		return nil, err
	}
	if err := k(engineWindow(out), engineWindow(ox), engineWindow(oy)); err != nil {
		return nil, err
	}
	return out, nil
}

// dispatchCompare 比较分发(与二元同构,输出 dtype 已由 ForceOut=Bool 定死)。
func dispatchCompare(op string, x, y *Array) (*Array, error) {
	return dispatchBinary(op, x, y)
}

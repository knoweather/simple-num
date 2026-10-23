package num

import (
	"fmt"
	"math"
)

// —— 常用别名与补缺(Step 25):amax/amin/prod/around/array_equal/argwhere/
// flatnonzero/isreal/identity/empty/tanh/sinh/cosh/sinc/ravel/shape/
// reshape/squeeze/transpose/concat + convolve/correlate ——

func init() {
	registerTrig("tanh", math.Tanh)
	registerTrig("sinh", math.Sinh)
	registerTrig("cosh", math.Cosh)
	// 补充 ufunc:对数/指数/开方/反双曲
	registerTrig("log2", math.Log2)
	registerTrig("expm1", math.Expm1)
	registerTrig("log1p", math.Log1p)
	registerTrig("cbrt", math.Cbrt)
	registerTrig("arcsinh", math.Asinh)
	registerTrig("arccosh", math.Acosh)
	registerTrig("arctanh", math.Atanh)
	// negative:逐 dtype 取负(整型的回绕行为与 numpy 一致;bool 无定义,numpy 同拒)
	registerUnaryTyped[float32]("negative", negate[float32])
	registerUnaryTyped[float64]("negative", negate[float64])
	registerUnaryTyped[int8]("negative", negate[int8])
	registerUnaryTyped[int16]("negative", negate[int16])
	registerUnaryTyped[int32]("negative", negate[int32])
	registerUnaryTyped[int64]("negative", negate[int64])
	registerUnaryTyped[uint8]("negative", negate[uint8])
	registerUnaryTyped[uint16]("negative", negate[uint16])
	registerUnaryTyped[uint32]("negative", negate[uint32])
	registerUnaryTyped[uint64]("negative", negate[uint64])
}

// negate 泛型取负。
func negate[T castableNum](dst, src []T) {
	for i := range src {
		dst[i] = -src[i]
	}
}

// Negative ≡ np.negative:逐元素取负(整型按回绕,与 numpy 的 int 语义一致)。
func Negative(a *Array) (*Array, error) { return dispatchUnary("negative", a) }

// Amax ≡ np.amax(即 np.max)。
func Amax(a *Array, axis []int, keepdims bool) (*Array, error) { return Max(a, axis, keepdims) }

// Amin ≡ np.amin(即 np.min)。
func Amin(a *Array, axis []int, keepdims bool) (*Array, error) { return Min(a, axis, keepdims) }

// Prod ≡ np.prod:连乘。整型输入按 Sum 的累积规则提升(numpy 同:int32 → int64)。
func Prod(a *Array, axis []int, keepdims bool) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: prod: 数组为 nil: %w", ErrDtype)
	}
	acc, err := sumAccDtype(a.Dtype())
	if err != nil {
		return nil, err
	}
	src := a
	if acc != a.Dtype() {
		if src, err = transformCopy(a, a.Shape(), acc); err != nil {
			return nil, err
		}
	}
	return Reduce(Multiply, src, axis, keepdims)
}

// Around ≡ np.around(即 np.round)。
func Around(a *Array, decimals int) (*Array, error) { return Round(a, decimals) }

// ArrayEqual ≡ np.array_equal:形状一致且逐元素相等(NaN 不等于自身,numpy 同)。
func ArrayEqual(a, b *Array) bool {
	if a == nil || b == nil {
		return a == b
	}
	if !shapeEqual(a.Shape(), b.Shape()) {
		return false
	}
	eq, err := Equal(a, b)
	if err != nil {
		return false
	}
	w := win[uint8](engineWindow(eq.AsContiguous()))
	for _, v := range w {
		if v == 0 {
			return false
		}
	}
	return true
}

// Flatnonzero ≡ np.flatnonzero:非零元素的**平坦下标**(1-D int64)。
func Flatnonzero(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: flatnonzero: 数组为 nil: %w", ErrDtype)
	}
	nz, err := Nonzero(a)
	if err != nil {
		return nil, err
	}
	if len(nz) == 0 {
		return New([]int{0}, DInt64)
	}
	// 逐轴下标 → 平坦下标(**元素**步长;cStrides 给的是字节步长,这里必须除 itemsize)
	byteSt := cStrides(a.Shape(), a.Dtype().Itemsize)
	strides := make([]int, len(byteSt))
	for i := range byteSt {
		strides[i] = byteSt[i] / a.Dtype().Itemsize
	}
	cnt := nz[0].Size()
	out, err := New([]int{cnt}, DInt64)
	if err != nil {
		return nil, err
	}
	ow := win[int64](engineWindow(out))
	idx := make([]int64, len(nz))
	for k := 0; k < cnt; k++ {
		flat := int64(0)
		for d := 0; d < len(nz); d++ {
			if idx[d], err = idxInt64At(nz[d], k); err != nil {
				return nil, err
			}
			flat += idx[d] * int64(strides[d])
		}
		ow[k] = flat
	}
	return out, nil
}

// idxInt64At 取整型数组第 k 个元素。
func idxInt64At(a *Array, k int) (int64, error) {
	v, err := a.Item(k)
	if err != nil {
		return 0, err
	}
	switch x := v.(type) {
	case int64:
		return x, nil
	case int32:
		return int64(x), nil
	case int:
		return int64(x), nil
	default:
		return 0, fmt.Errorf("num: 期望整型下标,得 %T: %w", v, ErrDtype)
	}
}

// Argwhere ≡ np.argwhere:非零元素的位置,形状 (N, ndim)(int64)。
func Argwhere(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: argwhere: 数组为 nil: %w", ErrDtype)
	}
	nz, err := Nonzero(a)
	if err != nil {
		return nil, err
	}
	nd := a.NDim()
	if nd == 0 {
		return New([]int{0, 0}, DInt64)
	}
	cnt := 0
	if len(nz) > 0 {
		cnt = nz[0].Size()
	}
	out, err := New([]int{cnt, nd}, DInt64)
	if err != nil {
		return nil, err
	}
	ow := win[int64](engineWindow(out))
	for d := 0; d < nd; d++ {
		for k := 0; k < cnt; k++ {
			v, err := idxInt64At(nz[d], k)
			if err != nil {
				return nil, err
			}
			ow[k*nd+d] = v
		}
	}
	return out, nil
}

// IsReal ≡ np.isreal:实数 dtype 恒真;复数按虚部是否为 0。
func IsReal(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: isreal: 数组为 nil: %w", ErrDtype)
	}
	out, err := New(a.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	ow := win[uint8](engineWindow(out))
	switch a.Dtype().Kind {
	case Complex64:
		for i, v := range win[complex64](engineWindow(a.AsContiguous())) {
			if imag(v) == 0 {
				ow[i] = 1
			}
		}
	case Complex128:
		for i, v := range win[complex128](engineWindow(a.AsContiguous())) {
			if imag(v) == 0 {
				ow[i] = 1
			}
		}
	default:
		for i := range ow {
			ow[i] = 1
		}
	}
	return out, nil
}

// Identity ≡ np.identity(n):n×n 单位阵(float64)。
func Identity(n int) (*Array, error) { return Eye(n, DFloat64) }

// Empty ≡ np.empty:分配未初始化数组。
func Empty(shape []int, dt Dtype) (*Array, error) { return New(shape, dt) }

// Tanh 双曲正切。
func Tanh(x *Array) (*Array, error) { return trigDispatch("tanh", x) }

// Sinh 双曲正弦。
func Sinh(x *Array) (*Array, error) { return trigDispatch("sinh", x) }

// Cosh 双曲余弦。
func Cosh(x *Array) (*Array, error) { return trigDispatch("cosh", x) }

// Sinc ≡ np.sinc(x) = sin(πx)/(πx),x=0 处取 1。
func Sinc(x *Array) (*Array, error) {
	sc, err := unaryScaled(x, math.Pi)
	if err != nil {
		return nil, err
	}
	sinv, err := Sin(sc)
	if err != nil {
		return nil, err
	}
	out, err := transformCopy(sinv, sinv.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	xw, err := transformCopy(x.AsContiguous(), x.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	ow := win[float64](engineWindow(out))
	xv := win[float64](engineWindow(xw))
	for i := range ow {
		if xv[i] == 0 {
			ow[i] = 1
			continue
		}
		ow[i] = ow[i] / (math.Pi * xv[i])
	}
	return out, nil
}

// unaryScaled 把数组乘上标量(内部辅助,避免引入新算子)。
func unaryScaled(x *Array, s float64) (*Array, error) {
	f, err := transformCopy(x.AsContiguous(), x.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	for i := range w {
		w[i] *= s
	}
	return f, nil
}

// Ravel ≡ np.ravel:展平为 1-D。
func Ravel(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: ravel: 数组为 nil: %w", ErrDtype)
	}
	return a.AsContiguous().Reshape([]int{a.Size()})
}

// Shape ≡ np.shape:返回形状副本。
func Shape(a *Array) []int {
	if a == nil {
		return nil
	}
	return append([]int(nil), a.Shape()...)
}

// Reshape ≡ np.reshape(包一层的便捷形式)。
func Reshape(a *Array, shape []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: reshape: 数组为 nil: %w", ErrDtype)
	}
	return a.Reshape(shape)
}

// Squeeze ≡ np.squeeze:去掉长度为 1 的轴。
func Squeeze(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: squeeze: 数组为 nil: %w", ErrDtype)
	}
	return a.Squeeze()
}

// Transpose ≡ np.transpose:默认反转全部轴。
func Transpose(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: transpose: 数组为 nil: %w", ErrDtype)
	}
	return a.Transpose()
}

// Concat ≡ np.concatenate 的简写(默认 axis=0)。
func Concat(as ...*Array) (*Array, error) { return Concatenate(0, as...) }

// Convolve ≡ np.convolve(a, v, mode):1-D 离散卷积,mode 为 "full"/"same"/"valid"。
func Convolve(a, v *Array, mode string) (*Array, error) {
	full, n, m, err := convCore(a, v, false, "convolve")
	if err != nil {
		return nil, err
	}
	return convSlice(full, n, m, mode, "convolve")
}

// Correlate ≡ np.correlate(a, v, mode):1-D 互相关(等价于卷积时反转第二个序列)。
func Correlate(a, v *Array, mode string) (*Array, error) {
	full, n, m, err := convCore(a, v, true, "correlate")
	if err != nil {
		return nil, err
	}
	return convSlice(full, n, m, mode, "correlate")
}

// convCore 计算 full 卷积(或互相关),返回结果与两个输入长度。
func convCore(a, v *Array, reverse bool, op string) (*Array, int, int, error) {
	if a == nil || v == nil {
		return nil, 0, 0, fmt.Errorf("num: %s: 入参为 nil: %w", op, ErrDtype)
	}
	if a.NDim() != 1 || v.NDim() != 1 {
		return nil, 0, 0, fmt.Errorf("num: %s: 只支持 1-D(得 %d/%d 维): %w", op, a.NDim(), v.NDim(), ErrShape)
	}
	n, m := a.Size(), v.Size()
	if n == 0 || m == 0 {
		return nil, 0, 0, fmt.Errorf("num: %s: 输入不能为空: %w", op, ErrShape)
	}
	af, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, 0, 0, err
	}
	vf, err := transformCopy(v.AsContiguous(), v.Shape(), DFloat64)
	if err != nil {
		return nil, 0, 0, err
	}
	aw := win[float64](engineWindow(af))
	vw := win[float64](engineWindow(vf))
	if reverse {
		// 互相关 = 与"反转后的 v"卷积
		rev := make([]float64, m)
		for i := range vw {
			rev[i] = vw[m-1-i]
		}
		vw = rev
	}
	full, err := New([]int{n + m - 1}, DFloat64)
	if err != nil {
		return nil, 0, 0, err
	}
	fw := win[float64](engineWindow(full))
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			fw[i+j] += aw[i] * vw[j]
		}
	}
	return full, n, m, nil
}

// convSlice 按 mode 截取卷积结果。
func convSlice(full *Array, n, m int, mode, op string) (*Array, error) {
	switch mode {
	case "", "full":
		return full, nil
	case "same":
		want := n
		if m > want {
			want = m
		}
		start := (full.Size() - want) / 2
		return sliceAlong(full, 0, start, start+want)
	case "valid":
		// numpy:长度 = max - min + 1,起点 = min - 1
		lo, hi := n, m
		if lo > hi {
			lo, hi = hi, lo
		}
		want := hi - lo + 1
		start := lo - 1
		return sliceAlong(full, 0, start, start+want)
	default:
		return nil, fmt.Errorf("num: %s: 未知 mode %q(应为 full/same/valid): %w", op, mode, ErrValue)
	}
}

// Log2 ≡ np.log2。
func Log2(x *Array) (*Array, error) { return trigDispatch("log2", x) }

// Expm1 ≡ np.expm1(exp(x)-1,小 x 时更精确)。
func Expm1(x *Array) (*Array, error) { return trigDispatch("expm1", x) }

// Log1p ≡ np.log1p(log(1+x),小 x 时更精确)。
func Log1p(x *Array) (*Array, error) { return trigDispatch("log1p", x) }

// Cbrt ≡ np.cbrt(保留负号,与 pow(x,1/3) 不同)。
func Cbrt(x *Array) (*Array, error) { return trigDispatch("cbrt", x) }

// Arcsinh ≡ np.arcsinh。
func Arcsinh(x *Array) (*Array, error) { return trigDispatch("arcsinh", x) }

// Arccosh ≡ np.arccosh。
func Arccosh(x *Array) (*Array, error) { return trigDispatch("arccosh", x) }

// Arctanh ≡ np.arctanh。
func Arctanh(x *Array) (*Array, error) { return trigDispatch("arctanh", x) }

// Square ≡ np.square:逐元素平方,**保持 dtype**(整型按回绕)。
func Square(x *Array) (*Array, error) { return Multiply(x, x) }

// Reciprocal ≡ np.reciprocal:1/x。整型输入在 numpy 里会报错,这里按浮点计算。
func Reciprocal(x *Array) (*Array, error) {
	f, err := transformCopy(x, x.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	return Div(bArrOf(1.0, f.Dtype()), f)
}

// bArrOf 造一个标量数组(内部辅助)。
func bArrOf(v float64, dt Dtype) *Array {
	a, err := New([]int{}, dt)
	if err != nil {
		return nil
	}
	if err := a.SetItem([]int{}, v); err != nil {
		return nil
	}
	return a
}

// Degrees ≡ np.degrees(与 rad2deg 同义)。
func Degrees(x *Array) (*Array, error) { return Rad2Deg(x) }

// Radians ≡ np.radians(与 deg2rad 同义)。
func Radians(x *Array) (*Array, error) { return Deg2Rad(x) }

// Positive ≡ np.positive:返回副本(不做取负)。
func Positive(x *Array) (*Array, error) {
	c, err := transformCopy(x, x.Shape(), x.Dtype())
	if err != nil {
		return nil, err
	}
	return c, nil
}

// Real ≡ np.real:复数取实部(实数输入返回副本)。
func Real(x *Array) (*Array, error) {
	switch x.Dtype().Kind {
	case Complex64:
		out, err := New(x.Shape(), DFloat32)
		if err != nil {
			return nil, err
		}
		src := win[complex64](engineWindow(x.AsContiguous()))
		dst := win[float32](engineWindow(out))
		for i := range src {
			dst[i] = real(src[i])
		}
		return out, nil
	case Complex128:
		out, err := New(x.Shape(), DFloat64)
		if err != nil {
			return nil, err
		}
		src := win[complex128](engineWindow(x.AsContiguous()))
		dst := win[float64](engineWindow(out))
		for i := range src {
			dst[i] = real(src[i])
		}
		return out, nil
	default:
		return Positive(x)
	}
}

// Imag ≡ np.imag:复数取虚部(实数输入返回全 0)。
func Imag(x *Array) (*Array, error) {
	switch x.Dtype().Kind {
	case Complex64:
		out, err := New(x.Shape(), DFloat32)
		if err != nil {
			return nil, err
		}
		src := win[complex64](engineWindow(x.AsContiguous()))
		dst := win[float32](engineWindow(out))
		for i := range src {
			dst[i] = imag(src[i])
		}
		return out, nil
	case Complex128:
		out, err := New(x.Shape(), DFloat64)
		if err != nil {
			return nil, err
		}
		src := win[complex128](engineWindow(x.AsContiguous()))
		dst := win[float64](engineWindow(out))
		for i := range src {
			dst[i] = imag(src[i])
		}
		return out, nil
	default:
		f, err := transformCopy(x, x.Shape(), DFloat64)
		if err != nil {
			return nil, err
		}
		w := win[float64](engineWindow(f))
		for i := range w {
			w[i] = 0
		}
		return f, nil
	}
}

package num

import "fmt"

// —— Must 风格 API:出错直接 panic,成功返回 *Array,便于链式调用 ——
//
// 设计取舍:标准 API 全部返回 (值, error),安全但啰嗦;Must* 是"脚本/测试/一次性管道"
// 用的简写 —— 一旦出错就 panic(值是 error,可用 recover 取回原始错误并 errors.Is 判定)。
// 两者并存,互不影响:
//
//	s := num.MustFromSlice([]float64{1, 2, 3, 4, 5, 6}).
//		MustReshape(2, 3).
//		MustTranspose().
//		MustSumAll()
//	fmt.Println(s.MustFloat()) // 21

// mustPanic 统一的 panic 形态:带操作名,可用 errors.Is 匹配原始哨兵错误。
func mustPanic(op string, err error) {
	if err != nil {
		panic(fmt.Errorf("num: %s: %w", op, err))
	}
}

// —— 构造入口 ——

// MustFromSlice 同 FromSlice,出错 panic。
func MustFromSlice[T any](x []T) *Array {
	a, err := FromSlice(x)
	mustPanic("FromSlice", err)
	return a
}

// MustZeros 同 Zeros。
func MustZeros(shape []int, dt Dtype) *Array {
	a, err := Zeros(shape, dt)
	mustPanic("Zeros", err)
	return a
}

// MustOnes 同 Ones。
func MustOnes(shape []int, dt Dtype) *Array {
	a, err := Ones(shape, dt)
	mustPanic("Ones", err)
	return a
}

// MustEmpty 同 Empty。
func MustEmpty(shape []int, dt Dtype) *Array {
	a, err := Empty(shape, dt)
	mustPanic("Empty", err)
	return a
}

// MustFull 同 Full。
func MustFull(shape []int, v any, dt Dtype) *Array {
	a, err := Full(shape, v, dt)
	mustPanic("Full", err)
	return a
}

// MustArange 同 Arange。
func MustArange(start, stop, step float64, dt Dtype) *Array {
	a, err := Arange(start, stop, step, dt)
	mustPanic("Arange", err)
	return a
}

// MustLinSpace 同 LinSpace。
func MustLinSpace(start, stop float64, n int, endpoint bool) *Array {
	a, err := LinSpace(start, stop, n, endpoint)
	mustPanic("LinSpace", err)
	return a
}

// MustEye 同 Eye。
func MustEye(n int, dt Dtype) *Array {
	a, err := Eye(n, dt)
	mustPanic("Eye", err)
	return a
}

// MustIdentity 同 Identity。
func MustIdentity(n int) *Array {
	a, err := Identity(n)
	mustPanic("Identity", err)
	return a
}

// —— 形状与视图(链式) ——

// MustReshape 同 Reshape;支持变参写形状:a.MustReshape(2, 3)。
func (a *Array) MustReshape(shape ...int) *Array {
	r, err := a.Reshape(shape)
	mustPanic("Reshape", err)
	return r
}

// MustTranspose 同 Transpose。
func (a *Array) MustTranspose(axes ...int) *Array {
	r, err := a.Transpose(axes...)
	mustPanic("Transpose", err)
	return r
}

// MustSqueeze 同 Squeeze。
func (a *Array) MustSqueeze(axes ...int) *Array {
	r, err := a.Squeeze(axes...)
	mustPanic("Squeeze", err)
	return r
}

// MustExpandDims 同 ExpandDims。
func (a *Array) MustExpandDims(axis int) *Array {
	r, err := a.ExpandDims(axis)
	mustPanic("ExpandDims", err)
	return r
}

// MustSwapAxes 同 SwapAxes。
func (a *Array) MustSwapAxes(i, j int) *Array {
	r, err := a.SwapAxes(i, j)
	mustPanic("SwapAxes", err)
	return r
}

// MustSub 同 Sub(索引/切片)。
func (a *Array) MustSub(idx ...Indexer) *Array {
	r, err := a.Sub(idx...)
	mustPanic("Sub", err)
	return r
}

// MustBroadcastTo 同 BroadcastTo;支持变参写形状。
func (a *Array) MustBroadcastTo(shape ...int) *Array {
	r, err := a.BroadcastTo(shape)
	mustPanic("BroadcastTo", err)
	return r
}

// MustCopy 返回独立副本同 Copy。
func (a *Array) MustCopy() *Array { return Copy(a) }

// MustCast 同 Cast。
func (a *Array) MustCast(dt Dtype) *Array {
	r, err := Cast(a, dt)
	mustPanic("Cast", err)
	return r
}

// MustCastUnsafe 同 CastUnsafe(允许跨 kind,例如 float→int)。
func (a *Array) MustCastUnsafe(dt Dtype) *Array {
	r, err := CastUnsafe(a, dt)
	mustPanic("CastUnsafe", err)
	return r
}

// —— 算子(链式) ——

// MustAdd 同 Add。
func (a *Array) MustAdd(b *Array) *Array {
	r, err := Add(a, b)
	mustPanic("Add", err)
	return r
}

// MustSubtract 同 Subtract。
func (a *Array) MustSubtract(b *Array) *Array {
	r, err := Subtract(a, b)
	mustPanic("Subtract", err)
	return r
}

// MustMultiply 同 Multiply。
func (a *Array) MustMultiply(b *Array) *Array {
	r, err := Multiply(a, b)
	mustPanic("Multiply", err)
	return r
}

// MustDivide 同 Div。
func (a *Array) MustDivide(b *Array) *Array {
	r, err := Div(a, b)
	mustPanic("Divide", err)
	return r
}

// MustMatMul 同 MatMul。
func (a *Array) MustMatMul(b *Array) *Array {
	r, err := MatMul(a, b)
	mustPanic("MatMul", err)
	return r
}

// MustDot 同 Dot。
func (a *Array) MustDot(b *Array) *Array {
	r, err := Dot(a, b)
	mustPanic("Dot", err)
	return r
}

// MustAbs 同 Abs。
func (a *Array) MustAbs() *Array {
	r, err := Abs(a)
	mustPanic("Abs", err)
	return r
}

// MustNegative 同 Negative。
func (a *Array) MustNegative() *Array {
	r, err := Negative(a)
	mustPanic("Negative", err)
	return r
}

// MustExp 同 Exp。
func (a *Array) MustExp() *Array {
	r, err := Exp(a)
	mustPanic("Exp", err)
	return r
}

// MustLog 同 Log。
func (a *Array) MustLog() *Array {
	r, err := Log(a)
	mustPanic("Log", err)
	return r
}

// MustSqrt 同 Sqrt。
func (a *Array) MustSqrt() *Array {
	r, err := Sqrt(a)
	mustPanic("Sqrt", err)
	return r
}

// MustSin 同 Sin。
func (a *Array) MustSin() *Array {
	r, err := Sin(a)
	mustPanic("Sin", err)
	return r
}

// MustCos 同 Cos。
func (a *Array) MustCos() *Array {
	r, err := Cos(a)
	mustPanic("Cos", err)
	return r
}

// MustFloor 同 Floor。
func (a *Array) MustFloor() *Array {
	r, err := Floor(a)
	mustPanic("Floor", err)
	return r
}

// MustCeil 同 Ceil。
func (a *Array) MustCeil() *Array {
	r, err := Ceil(a)
	mustPanic("Ceil", err)
	return r
}

// MustRound 同 Round;小数位省略时为 0。
func (a *Array) MustRound(decimals ...int) *Array {
	d := 0
	if len(decimals) > 0 {
		d = decimals[0]
	}
	r, err := Round(a, d)
	mustPanic("Round", err)
	return r
}

// MustClip 同 Clip。
func (a *Array) MustClip(lo, hi any) *Array {
	r, err := Clip(a, lo, hi)
	mustPanic("Clip", err)
	return r
}

// MustSort 同 Sort。
func (a *Array) MustSort(axis ...int) *Array {
	r, err := Sort(a, axis)
	mustPanic("Sort", err)
	return r
}

// MustDiag 同 Diag。
func (a *Array) MustDiag(k int) *Array {
	r, err := Diag(a, k)
	mustPanic("Diag", err)
	return r
}

// —— 归约(链式) ——

// MustSum 同 Sum。
func (a *Array) MustSum(axis []int, keepdims bool) *Array {
	r, err := Sum(a, axis, keepdims)
	mustPanic("Sum", err)
	return r
}

// MustMean 同 Mean。
func (a *Array) MustMean(axis []int, keepdims bool) *Array {
	r, err := Mean(a, axis, keepdims)
	mustPanic("Mean", err)
	return r
}

// MustProd 同 Prod。
func (a *Array) MustProd(axis []int, keepdims bool) *Array {
	r, err := Prod(a, axis, keepdims)
	mustPanic("Prod", err)
	return r
}

// MustMax 同 Max。
func (a *Array) MustMax(axis []int, keepdims bool) *Array {
	r, err := Max(a, axis, keepdims)
	mustPanic("Max", err)
	return r
}

// MustMin 同 Min。
func (a *Array) MustMin(axis []int, keepdims bool) *Array {
	r, err := Min(a, axis, keepdims)
	mustPanic("Min", err)
	return r
}

// MustStd 同 Std。
func (a *Array) MustStd(axis []int, keepdims bool) *Array {
	r, err := Std(a, axis, keepdims)
	mustPanic("Std", err)
	return r
}

// MustVar 同 Var。
func (a *Array) MustVar(axis []int, keepdims bool) *Array {
	r, err := Var(a, axis, keepdims)
	mustPanic("Var", err)
	return r
}

// MustArgMax 同 ArgMax。
func (a *Array) MustArgMax(axis []int, keepdims bool) *Array {
	r, err := ArgMax(a, axis, keepdims)
	mustPanic("ArgMax", err)
	return r
}

// MustArgMin 同 ArgMin。
func (a *Array) MustArgMin(axis []int, keepdims bool) *Array {
	r, err := ArgMin(a, axis, keepdims)
	mustPanic("ArgMin", err)
	return r
}

// MustCumsum 同 Cumsum。
func (a *Array) MustCumsum(axis []int) *Array {
	r, err := Cumsum(a, axis)
	mustPanic("Cumsum", err)
	return r
}

// MustCumprod 同 Cumprod。
func (a *Array) MustCumprod(axis []int) *Array {
	r, err := Cumprod(a, axis)
	mustPanic("Cumprod", err)
	return r
}

// MustSumAll 全归约求和(axis=nil、keepdims=false)的简写。
func (a *Array) MustSumAll() *Array { return a.MustSum(nil, false) }

// MustMeanAll 全归约求平均。
func (a *Array) MustMeanAll() *Array { return a.MustMean(nil, false) }

// MustMaxAll 全归约求最大。
func (a *Array) MustMaxAll() *Array { return a.MustMax(nil, false) }

// MustMinAll 全归约求最小。
func (a *Array) MustMinAll() *Array { return a.MustMin(nil, false) }

// —— 终结取值 ——

// MustItem 同 Item,出错 panic。
func (a *Array) MustItem(i ...int) any {
	v, err := a.Item(i...)
	mustPanic("Item", err)
	return v
}

// MustFloat 取单个元素并转成 float64(非数值类型 panic)。
func (a *Array) MustFloat(i ...int) float64 {
	switch x := a.MustItem(i...).(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int64:
		return float64(x)
	case int32:
		return float64(x)
	case int16:
		return float64(x)
	case int8:
		return float64(x)
	case uint64:
		return float64(x)
	case uint32:
		return float64(x)
	case uint16:
		return float64(x)
	case uint8:
		return float64(x)
	case bool:
		if x {
			return 1
		}
		return 0
	default:
		mustPanic("Float", fmt.Errorf("无法转为 float64: %w", ErrDtype))
		return 0
	}
}

// MustInt 取单个元素并转成 int64。
func (a *Array) MustInt(i ...int) int64 {
	v := a.MustItem(i...)
	switch x := v.(type) {
	case int64:
		return x
	case int32:
		return int64(x)
	case int16:
		return int64(x)
	case int8:
		return int64(x)
	case uint64:
		return int64(x)
	case uint32:
		return int64(x)
	case uint16:
		return int64(x)
	case uint8:
		return int64(x)
	case float64:
		return int64(x)
	case float32:
		return int64(x)
	case bool:
		if x {
			return 1
		}
		return 0
	default:
		mustPanic("Int", fmt.Errorf("无法转为 int64: %w", ErrDtype))
		return 0
	}
}

// MustDet 同 Det(返回 float64,不是数组)。
func (a *Array) MustDet() float64 {
	v, err := Det(a)
	mustPanic("Det", err)
	return v
}

// MustInv 同 Inv。
func (a *Array) MustInv() *Array {
	r, err := Inv(a)
	mustPanic("Inv", err)
	return r
}

// MustSolve 同 Solve。
func (a *Array) MustSolve(b *Array) *Array {
	r, err := Solve(a, b)
	mustPanic("Solve", err)
	return r
}

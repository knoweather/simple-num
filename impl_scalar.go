package num

import "fmt"

// Number 是标量运算可接受的 Go 数值类型集合。
//
// 注意与 numpy 的差别:numpy 的 Python 标量是"弱标量"(int8 数组 + 1 仍是 int8);
// 这里标量的 dtype 由 **Go 类型**决定 —— 想保持 dtype 就传同类型标量:
//
//	num.AddScalar(int8Arr, int8(1)) // → int8
//	num.AddScalar(int8Arr, 1)       // → int64(因为 Go 的 1 是 int)
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64 | ~complex64 | ~complex128
}

// scalarArr 把 Go 标量包成长度 1 的数组(dtype 由 Go 类型决定,可参与广播)。
func scalarArr(v any) (*Array, error) {
	switch x := v.(type) {
	case int:
		return FromSlice([]int{x})
	case int8:
		return FromSlice([]int8{x})
	case int16:
		return FromSlice([]int16{x})
	case int32:
		return FromSlice([]int32{x})
	case int64:
		return FromSlice([]int64{x})
	case uint:
		return FromSlice([]uint{x})
	case uint8:
		return FromSlice([]uint8{x})
	case uint16:
		return FromSlice([]uint16{x})
	case uint32:
		return FromSlice([]uint32{x})
	case uint64:
		return FromSlice([]uint64{x})
	case float32:
		return FromSlice([]float32{x})
	case float64:
		return FromSlice([]float64{x})
	case complex64:
		return FromSlice([]complex64{x})
	case complex128:
		return FromSlice([]complex128{x})
	case bool:
		return FromSlice([]bool{x})
	default:
		return nil, fmt.Errorf("num: 标量: 不支持的 Go 类型 %T: %w", v, ErrDtype)
	}
}

// —— 包级泛型标量运算(类型安全,dtype 由 Go 类型决定) ——

// AddScalar 计算 a + v(标量按 Go 类型决定 dtype)。
func AddScalar[T Number](a *Array, v T) (*Array, error) {
	s, err := FromSlice([]T{v})
	if err != nil {
		return nil, err
	}
	return Add(a, s)
}

// SubScalar 计算 a - v。
func SubScalar[T Number](a *Array, v T) (*Array, error) {
	s, err := FromSlice([]T{v})
	if err != nil {
		return nil, err
	}
	return Subtract(a, s)
}

// MulScalar 计算 a * v。
func MulScalar[T Number](a *Array, v T) (*Array, error) {
	s, err := FromSlice([]T{v})
	if err != nil {
		return nil, err
	}
	return Multiply(a, s)
}

// DivScalar 计算 a / v。
func DivScalar[T Number](a *Array, v T) (*Array, error) {
	s, err := FromSlice([]T{v})
	if err != nil {
		return nil, err
	}
	return Div(a, s)
}

// PowScalar 计算 a ** v。
func PowScalar[T Number](a *Array, v T) (*Array, error) {
	s, err := FromSlice([]T{v})
	if err != nil {
		return nil, err
	}
	return Power(a, s)
}

// —— 方法形式(接受 any,便于链式;dtype 同样由 Go 类型决定) ——

// AddScalar 计算 a + v。
func (a *Array) AddScalar(v any) (*Array, error) {
	s, err := scalarArr(v)
	if err != nil {
		return nil, err
	}
	return Add(a, s)
}

// SubScalar 计算 a - v。
func (a *Array) SubScalar(v any) (*Array, error) {
	s, err := scalarArr(v)
	if err != nil {
		return nil, err
	}
	return Subtract(a, s)
}

// MulScalar 计算 a * v。
func (a *Array) MulScalar(v any) (*Array, error) {
	s, err := scalarArr(v)
	if err != nil {
		return nil, err
	}
	return Multiply(a, s)
}

// DivScalar 计算 a / v。
func (a *Array) DivScalar(v any) (*Array, error) {
	s, err := scalarArr(v)
	if err != nil {
		return nil, err
	}
	return Div(a, s)
}

// PowScalar 计算 a ** v。
func (a *Array) PowScalar(v any) (*Array, error) {
	s, err := scalarArr(v)
	if err != nil {
		return nil, err
	}
	return Power(a, s)
}

// —— Must 版本(链式) ——

// MustAddScalar 计算 a + v,出错 panic。
func (a *Array) MustAddScalar(v any) *Array {
	r, err := a.AddScalar(v)
	mustPanic("AddScalar", err)
	return r
}

// MustSubScalar 计算 a - v,出错 panic。
func (a *Array) MustSubScalar(v any) *Array {
	r, err := a.SubScalar(v)
	mustPanic("SubScalar", err)
	return r
}

// MustMulScalar 计算 a * v,出错 panic。
func (a *Array) MustMulScalar(v any) *Array {
	r, err := a.MulScalar(v)
	mustPanic("MulScalar", err)
	return r
}

// MustDivScalar 计算 a / v,出错 panic。
func (a *Array) MustDivScalar(v any) *Array {
	r, err := a.DivScalar(v)
	mustPanic("DivScalar", err)
	return r
}

// MustPowScalar 计算 a ** v,出错 panic。
func (a *Array) MustPowScalar(v any) *Array {
	r, err := a.PowScalar(v)
	mustPanic("PowScalar", err)
	return r
}

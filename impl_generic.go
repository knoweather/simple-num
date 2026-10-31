package num

import "fmt"

// —— 泛型方法(Go 1.27 起方法可以自带类型参数)——
//
// 注意 Go 的类型推断只从**实参**推断,不从返回值上下文推断,所以调用时要显式写类型:
//
//	xs, err := a.As[float32]()   // ✓
//	xs, err := a.As()            // ✗ 无法推断

// dtypeOfNumber 把 Go 数值类型映射到 Dtype。
func dtypeOfNumber[T Number]() (Dtype, error) {
	var zero T
	switch any(zero).(type) {
	case int, int64:
		return DInt64, nil
	case int8:
		return DInt8, nil
	case int16:
		return DInt16, nil
	case int32:
		return DInt32, nil
	case uint, uint64:
		return DUint64, nil
	case uint8:
		return DUint8, nil
	case uint16:
		return DUint16, nil
	case uint32:
		return DUint32, nil
	case float32:
		return DFloat32, nil
	case float64:
		return DFloat64, nil
	case complex64:
		return DComplex64, nil
	case complex128:
		return DComplex128, nil
	default:
		return Dtype{}, fmt.Errorf("num: 不支持的数值类型 %T: %w", zero, ErrDtype)
	}
}

// As 把数组按行主序展开成 []T(必要时逐元素转换 dtype)。
func (a *Array) As[T Number]() ([]T, error) {
	if a == nil {
		return nil, fmt.Errorf("num: As: 数组为 nil: %w", ErrDtype)
	}
	dt, err := dtypeOfNumber[T]()
	if err != nil {
		return nil, err
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), dt)
	if err != nil {
		// 与 numpy 同口径:复数 → 实数会丢弃虚部(相当于 numpy 的 ComplexWarning),
		// 其余跨类转换仍然报错。
		if isCplxKind(a.Dtype().Kind) && !isCplxKind(dt.Kind) {
			f, err = transformCopyMode(a.AsContiguous(), a.Shape(), dt, true)
		}
		if err != nil {
			return nil, err
		}
	}
	out := make([]T, f.Size())
	copy(out, win[T](engineWindow(f)))
	return out, nil
}

// MustAs 同 As,出错 panic。
func (a *Array) MustAs[T Number]() []T {
	v, err := a.As[T]()
	mustPanic("As", err)
	return v
}

// At 取单个元素并转成 T。
func (a *Array) At[T Number](i ...int) (T, error) {
	var zero T
	if a == nil {
		return zero, fmt.Errorf("num: At: 数组为 nil: %w", ErrDtype)
	}
	v, err := a.Item(i...)
	if err != nil {
		return zero, err
	}
	return convToNumber[T](v)
}

// MustAt 同 At,出错 panic。
func (a *Array) MustAt[T Number](i ...int) T {
	v, err := a.At[T](i...)
	mustPanic("At", err)
	return v
}

// convToNumber 把 Item 返回的 any 转成目标数值类型。
//
// Go 的泛型转换不允许在约束内部跨"复数 ↔ 实数"互转,所以先归一化成
// (float64, complex128) 两种形态,再按目标类型的具体指针写入。
func convToNumber[T Number](v any) (T, error) {
	var out T
	c, err := toComplex128Value(v)
	if err != nil {
		return out, err
	}
	switch p := any(&out).(type) {
	case *int:
		*p = int(real(c))
	case *int8:
		*p = int8(real(c))
	case *int16:
		*p = int16(real(c))
	case *int32:
		*p = int32(real(c))
	case *int64:
		*p = int64(real(c))
	case *uint:
		*p = uint(real(c))
	case *uint8:
		*p = uint8(real(c))
	case *uint16:
		*p = uint16(real(c))
	case *uint32:
		*p = uint32(real(c))
	case *uint64:
		*p = uint64(real(c))
	case *float32:
		*p = float32(real(c))
	case *float64:
		*p = real(c)
	case *complex64:
		*p = complex64(c)
	case *complex128:
		*p = c
	default:
		return out, fmt.Errorf("num: 不支持的目标类型 %T: %w", out, ErrDtype)
	}
	return out, nil
}

// toComplex128Value 把任意标量归一化成 complex128(实数取实部)。
func toComplex128Value(v any) (complex128, error) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	case int:
		return complex(float64(x), 0), nil
	case int8:
		return complex(float64(x), 0), nil
	case int16:
		return complex(float64(x), 0), nil
	case int32:
		return complex(float64(x), 0), nil
	case int64:
		return complex(float64(x), 0), nil
	case uint:
		return complex(float64(x), 0), nil
	case uint8:
		return complex(float64(x), 0), nil
	case uint16:
		return complex(float64(x), 0), nil
	case uint32:
		return complex(float64(x), 0), nil
	case uint64:
		return complex(float64(x), 0), nil
	case float32:
		return complex(float64(x), 0), nil
	case float64:
		return complex(x, 0), nil
	case complex64:
		return complex128(x), nil
	case complex128:
		return x, nil
	default:
		return 0, fmt.Errorf("num: 无法把 %T 转为数值: %w", v, ErrDtype)
	}
}

// FromRows 从 [][]T 构造 2-D 数组(参差行报错,与 numpy 对不规则嵌套的态度一致)。
func FromRows[T any](rows [][]T) (*Array, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("num: FromRows: 空输入: %w", ErrShape)
	}
	cols := len(rows[0])
	flat := make([]T, 0, len(rows)*cols)
	for i, r := range rows {
		if len(r) != cols {
			return nil, fmt.Errorf("num: FromRows: 第 %d 行有 %d 列,首行是 %d 列: %w", i, len(r), cols, ErrShape)
		}
		flat = append(flat, r...)
	}
	a, err := FromSlice(flat)
	if err != nil {
		return nil, err
	}
	return a.Reshape([]int{len(rows), cols})
}

// MustFromRows 同 FromRows,出错 panic。
func MustFromRows[T any](rows [][]T) *Array {
	a, err := FromRows(rows)
	mustPanic("FromRows", err)
	return a
}

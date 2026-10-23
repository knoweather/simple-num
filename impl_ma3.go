package num

import "fmt"

// —— 掩码数组谓词构造器(Step 21):masked_equal / less / greater / inside / outside ——
//
// 与 numpy 的 np.ma.masked_* 一致:按"比较条件为真"的位置构造掩码。
// 这些构造器**只设置掩码**,数据原样保留(与 numpy 相同)。

// scalarArrOfDtype 把标量按 dtype 转成 0-d 数组(供比较算子使用)。
func scalarArrOfDtype(dt Dtype, v any) (*Array, error) {
	a, err := New([]int{}, dt)
	if err != nil {
		return nil, fmt.Errorf("num: ma: 标量构造失败: %w", err)
	}
	if err := a.SetItem([]int{}, v); err != nil {
		return nil, fmt.Errorf("num: ma: 标量 %v 无法写入 %s: %w", v, dt, err)
	}
	return a, nil
}

// maskedFromCond 用条件数组构造 MaskedArray。
func maskedFromCond(a, cond *Array) (*MaskedArray, error) {
	return NewMasked(a, cond)
}

// MaskedEqual ≡ np.ma.masked_equal(a, v):等于 v 的位置标为无效。
func MaskedEqual(a *Array, v any) (*MaskedArray, error) {
	s, err := scalarArrOfDtype(a.Dtype(), v)
	if err != nil {
		return nil, err
	}
	c, err := Equal(a, s)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_equal: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedNotEqual ≡ np.ma.masked_not_equal(a, v)。
func MaskedNotEqual(a *Array, v any) (*MaskedArray, error) {
	s, err := scalarArrOfDtype(a.Dtype(), v)
	if err != nil {
		return nil, err
	}
	c, err := NotEqual(a, s)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_not_equal: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedLess ≡ np.ma.masked_less(a, v):小于 v 的位置标为无效。
func MaskedLess(a *Array, v any) (*MaskedArray, error) {
	s, err := scalarArrOfDtype(a.Dtype(), v)
	if err != nil {
		return nil, err
	}
	c, err := Less(a, s)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_less: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedLessEqual ≡ np.ma.masked_less_equal(a, v)。
func MaskedLessEqual(a *Array, v any) (*MaskedArray, error) {
	s, err := scalarArrOfDtype(a.Dtype(), v)
	if err != nil {
		return nil, err
	}
	c, err := LessEqual(a, s)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_less_equal: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedGreater ≡ np.ma.masked_greater(a, v)。
func MaskedGreater(a *Array, v any) (*MaskedArray, error) {
	s, err := scalarArrOfDtype(a.Dtype(), v)
	if err != nil {
		return nil, err
	}
	c, err := Greater(a, s)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_greater: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedGreaterEqual ≡ np.ma.masked_greater_equal(a, v)。
func MaskedGreaterEqual(a *Array, v any) (*MaskedArray, error) {
	s, err := scalarArrOfDtype(a.Dtype(), v)
	if err != nil {
		return nil, err
	}
	c, err := GreaterEqual(a, s)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_greater_equal: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedInside ≡ np.ma.masked_inside(a, v1, v2):闭区间 [v1, v2] 内标为无效。
func MaskedInside(a *Array, v1, v2 any) (*MaskedArray, error) {
	lo, err := scalarArrOfDtype(a.Dtype(), v1)
	if err != nil {
		return nil, err
	}
	hi, err := scalarArrOfDtype(a.Dtype(), v2)
	if err != nil {
		return nil, err
	}
	ge, err := GreaterEqual(a, lo)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_inside: %w", err)
	}
	le, err := LessEqual(a, hi)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_inside: %w", err)
	}
	c, err := LogicalAnd(ge, le)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_inside: %w", err)
	}
	return maskedFromCond(a, c)
}

// MaskedOutside ≡ np.ma.masked_outside(a, v1, v2):区间外标为无效。
func MaskedOutside(a *Array, v1, v2 any) (*MaskedArray, error) {
	lo, err := scalarArrOfDtype(a.Dtype(), v1)
	if err != nil {
		return nil, err
	}
	hi, err := scalarArrOfDtype(a.Dtype(), v2)
	if err != nil {
		return nil, err
	}
	lt, err := Less(a, lo)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_outside: %w", err)
	}
	gt, err := Greater(a, hi)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_outside: %w", err)
	}
	c, err := LogicalOr(lt, gt)
	if err != nil {
		return nil, fmt.Errorf("num: ma.masked_outside: %w", err)
	}
	return maskedFromCond(a, c)
}

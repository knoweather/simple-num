package num

import (
	"fmt"
	"sort"
)

// —— ufunc 协议层(Step 10):out= / where= / reduce / accumulate / outer ——
//
// numpy 的 ufunc 关键字在 Go 里没有语法对应,故用**显式变体 + 算子函数参数**表达:
//
//	out= → BinaryOut(fn, dst, a, b) / UnaryOut(fn, dst, a)
//	where= → BinaryWhere(fn, dst, a, b, cond) / UnaryWhere(fn, dst, a, cond)
//	reduce → Reduce(fn, a, axis, keepdims)
//	accumulate → Accumulate(fn, a, axis)
//	ufunc.outer → UfuncOuter(fn, a, b)
//
// 语义与 numpy 一致:where= 只写条件为真的位置(其余保持 dst 原值),
// reduce 沿轴逐元素折叠(要求该轴非空,否则 ErrShape —— numpy 无 identity 时同)。

// BinaryFunc 二元算子签名(与 Add/Subtract/Multiply/Div/... 一致)。
type BinaryFunc func(a, b *Array) (*Array, error)

// UnaryFunc 一元算子签名(与 Abs/Exp/Sin/... 一致)。
type UnaryFunc func(a *Array) (*Array, error)

// BinaryOut ≡ np.add(a, b, out=dst):结果写回已有的 dst。
func BinaryOut(fn BinaryFunc, dst, a, b *Array) error {
	res, err := fn(a, b)
	if err != nil {
		return err
	}
	return copyResultInto(dst, res)
}

// UnaryOut ≡ np.abs(a, out=dst)。
func UnaryOut(fn UnaryFunc, dst, a *Array) error {
	res, err := fn(a)
	if err != nil {
		return err
	}
	return copyResultInto(dst, res)
}

// BinaryWhere ≡ np.add(a, b, out=dst, where=cond):只写条件为真的位置。
func BinaryWhere(fn BinaryFunc, dst, a, b, cond *Array) error {
	res, err := fn(a, b)
	if err != nil {
		return err
	}
	return copyWhereInto(dst, res, cond)
}

// UnaryWhere ≡ np.abs(a, out=dst, where=cond)。
func UnaryWhere(fn UnaryFunc, dst, a, cond *Array) error {
	res, err := fn(a)
	if err != nil {
		return err
	}
	return copyWhereInto(dst, res, cond)
}

// copyResultInto 把结果整块写进 dst(形状必须一致;dtype 不一致时按显式转换口径落值)。
func copyResultInto(dst, src *Array) error {
	if dst == nil || src == nil {
		return fmt.Errorf("num: out=: 入参为 nil: %w", ErrDtype)
	}
	if !shapeEqual(dst.Shape(), src.Shape()) {
		return fmt.Errorf("num: out=: dst 形状 %v ≠ 结果形状 %v: %w", dst.Shape(), src.Shape(), ErrShape)
	}
	s := src
	if dst.Dtype() != src.Dtype() {
		c, err := transformCopy(src.AsContiguous(), src.Shape(), dst.Dtype())
		if err != nil {
			return fmt.Errorf("num: out=: dtype 转换失败: %w", err)
		}
		s = c
	}
	copy(engineWindow(dst), engineWindow(s.AsContiguous()))
	return nil
}

// copyWhereInto 只把 cond 为真的位置写进 dst,其余保持 dst 原值。
func copyWhereInto(dst, src, cond *Array) error {
	if dst == nil || src == nil || cond == nil {
		return fmt.Errorf("num: where=: 入参为 nil: %w", ErrDtype)
	}
	if !shapeEqual(dst.Shape(), src.Shape()) {
		return fmt.Errorf("num: where=: dst 形状 %v ≠ 结果形状 %v: %w", dst.Shape(), src.Shape(), ErrShape)
	}
	nc, err := normalizeOperand(cond, dst.Shape(), cond.Dtype())
	if err != nil {
		return fmt.Errorf("num: where=: cond 归一失败: %w", err)
	}
	mask, err := truthyMask(nc)
	if err != nil {
		return err
	}
	s := src
	if dst.Dtype() != src.Dtype() {
		c, err := transformCopy(src.AsContiguous(), src.Shape(), dst.Dtype())
		if err != nil {
			return fmt.Errorf("num: where=: dtype 转换失败: %w", err)
		}
		s = c
	}
	es := dst.Dtype().Itemsize
	db := engineWindow(dst)
	sb := engineWindow(s.AsContiguous())
	for i, m := range mask {
		if !m {
			continue
		}
		copy(db[i*es:(i+1)*es], sb[i*es:(i+1)*es])
	}
	return nil
}

// Reduce ≡ ufunc.reduce(a, axis, keepdims):沿给定轴折叠。
//
// axis 为空表示所有轴(numpy 的 axis=None,结果 0-d);收缩轴长度为 0 → ErrShape
// (numpy 在无 identity 时同样报错)。
func Reduce(fn BinaryFunc, a *Array, axis []int, keepdims bool) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: reduce: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() == 0 {
		return transformCopy(a, a.Shape(), a.Dtype())
	}
	var axes []int
	if len(axis) == 0 {
		for i := 0; i < a.NDim(); i++ {
			axes = append(axes, i)
		}
	} else {
		seen := map[int]bool{}
		for _, ax := range axis {
			d, err := normalizeAxis(a, ax)
			if err != nil {
				return nil, fmt.Errorf("num: reduce: %w", err)
			}
			if !seen[d] {
				seen[d] = true
				axes = append(axes, d)
			}
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(axes)))
	kept := append([]int{}, a.Shape()...)
	cur := a
	for _, d := range axes {
		nxt, err := reduceAxis(fn, cur, d)
		if err != nil {
			return nil, err
		}
		cur = nxt
		kept[d] = 1
	}
	if keepdims {
		return cur.Reshape(kept)
	}
	return cur, nil
}

// reduceAxis 沿单轴折叠(每次用 1 厚切片调 fn,正确性优先)。
func reduceAxis(fn BinaryFunc, a *Array, d int) (*Array, error) {
	n := a.Shape()[d]
	if n == 0 {
		return nil, fmt.Errorf("num: reduce: 轴 %d 长度为 0,无 identity 可用: %w", d, ErrShape)
	}
	sub := func(i int) (*Array, error) {
		idx := make([]Indexer, a.NDim())
		for j := range idx {
			idx[j] = All()
		}
		idx[d] = Slice(i, i+1)
		return a.Sub(idx...)
	}
	acc, err := sub(0)
	if err != nil {
		return nil, err
	}
	for i := 1; i < n; i++ {
		cur, err := sub(i)
		if err != nil {
			return nil, err
		}
		if acc, err = fn(acc, cur); err != nil {
			return nil, err
		}
	}
	sh := append([]int{}, acc.Shape()...)
	sh = append(sh[:d], sh[d+1:]...)
	return acc.Reshape(sh)
}

// Accumulate ≡ ufunc.accumulate(a, axis):前缀累积(结果与输入同形)。
// axis 为空表示最后一轴(numpy 默认 axis=0;这里按 numpy 的语义取 0)。
func Accumulate(fn BinaryFunc, a *Array, axis []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: accumulate: 数组为 nil: %w", ErrDtype)
	}
	if a.NDim() == 0 {
		return transformCopy(a, a.Shape(), a.Dtype())
	}
	d := 0
	if len(axis) > 0 {
		var err error
		if d, err = normalizeAxis(a, axis[0]); err != nil {
			return nil, fmt.Errorf("num: accumulate: %w", err)
		}
	}
	out, err := transformCopy(a, a.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	n := out.Shape()[d]
	if n <= 1 {
		return out, nil
	}
	sl := func(i int) (*Array, error) {
		idx := make([]Indexer, out.NDim())
		for j := range idx {
			idx[j] = All()
		}
		idx[d] = Slice(i, i+1)
		return out.Sub(idx...)
	}
	for i := 1; i < n; i++ {
		prev, err := sl(i - 1)
		if err != nil {
			return nil, err
		}
		cur, err := sl(i)
		if err != nil {
			return nil, err
		}
		nxt, err := fn(prev, cur)
		if err != nil {
			return nil, err
		}
		// 写回第 i 片
		idx := make([]Indexer, out.NDim())
		for j := range idx {
			idx[j] = All()
		}
		idx[d] = Slice(i, i+1)
		if err := out.SetSub(nxt, idx...); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// UfuncOuter ≡ ufunc.outer(a, b):拉平后逐对应用 fn,结果 (a.Size, b.Size)。
func UfuncOuter(fn BinaryFunc, a, b *Array) (*Array, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("num: outer: 入参为 nil: %w", ErrDtype)
	}
	af := a.AsContiguous().Flatten()
	bf := b.AsContiguous().Flatten()
	out, err := New([]int{af.Size(), bf.Size()}, af.Dtype())
	if err != nil {
		return nil, err
	}
	// 逐行:整行用 fn(标量广播)算
	for i := 0; i < af.Size(); i++ {
		scalar, err := sliceScalar(af, i)
		if err != nil {
			return nil, err
		}
		row, err := fn(scalar, bf)
		if err != nil {
			return nil, err
		}
		rr, err := row.Reshape([]int{1, bf.Size()})
		if err != nil {
			return nil, err
		}
		idx := make([]Indexer, 2)
		idx[0] = Slice(i, i+1)
		idx[1] = All()
		if err := out.SetSub(rr, idx...); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// sliceScalar 取 1-D 数组第 i 个元素的 0-d 数组。
func sliceScalar(a *Array, i int) (*Array, error) {
	v, err := a.Item(i)
	if err != nil {
		return nil, err
	}
	sc, err := New([]int{}, a.Dtype())
	if err != nil {
		return nil, err
	}
	if err := sc.SetItem([]int{}, v); err != nil {
		return nil, err
	}
	return sc, nil
}

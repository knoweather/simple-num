package num

import (
	"fmt"
	"math"
)

// —— NaN 族补齐(Step 18):nanmedian / nanargmin / nanargmax ——
//
// 语义与 numpy 一致:
// - nanmedian:跳过 NaN 求中位数;全 NaN 的线返回 NaN(并发出 RuntimeWarning)。
// - nanargmin/nanargmax:跳过 NaN 取极值的**下标**(int64);全 NaN 的线
// numpy 抛 ValueError,本库 fail-loud 返回错误(而非静默给 0)。

// NanMedian ≡ np.nanmedian(a, axis):跳过 NaN 的中位数。
func NanMedian(a *Array, axis []int, keepdims bool) (*Array, error) {
	return NanPercentile(a, 50, axis, keepdims)
}

// NanArgMin ≡ np.nanargmin(a, axis):跳过 NaN 的最小值下标(int64)。
func NanArgMin(a *Array, axis []int, keepdims bool) (*Array, error) {
	return nanArg(a, axis, keepdims, false)
}

// NanArgMax ≡ np.nanargmax(a, axis):跳过 NaN 的最大值下标(int64)。
func NanArgMax(a *Array, axis []int, keepdims bool) (*Array, error) {
	return nanArg(a, axis, keepdims, true)
}

// nanArg 公共实现。
func nanArg(a *Array, axis []int, keepdims bool, wantMax bool) (*Array, error) {
	op := "nanargmin"
	if wantMax {
		op = "nanargmax"
	}
	if a == nil {
		return nil, fmt.Errorf("num: %s: 数组为 nil: %w", op, ErrDtype)
	}
	if len(axis) == 0 {
		v, err := nanArgOf(a.AsContiguous().Flatten(), wantMax)
		if err != nil {
			return nil, fmt.Errorf("num: %s: %w", op, err)
		}
		var sh []int
		if keepdims {
			sh = make([]int, a.NDim())
			for i := range sh {
				sh[i] = 1
			}
		}
		out, err := New(sh, DInt64)
		if err != nil {
			return nil, err
		}
		return out, setScalarInt64(out, v)
	}
	if a.NDim() == 0 {
		return nil, fmt.Errorf("num: %s: 0-d 输入不支持: %w", op, ErrShape)
	}
	d, err := normalizeAxis(a, axis[0])
	if err != nil {
		return nil, fmt.Errorf("num: %s: %w", op, err)
	}
	rows, m, n, err := axisRows(a, d)
	if err != nil {
		return nil, err
	}
	outSh := make([]int, a.NDim()-1)
	k := 0
	for i := 0; i < a.NDim(); i++ {
		if i == d {
			continue
		}
		outSh[k] = a.Shape()[i]
		k++
	}
	if keepdims {
		outSh = append([]int(nil), a.Shape()...)
		outSh[d] = 1
	}
	out, err := New(outSh, DInt64)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, fmt.Errorf("num: %s: 归约轴长度为 0: %w", op, ErrShape)
	}
	if m == 0 {
		return out, nil
	}
	es := rows.Dtype().Itemsize
	sb := engineWindow(rows)
	ow := win[int64](engineWindow(out))
	for r := 0; r < m; r++ {
		sub := &Array{buf: sb[r*n*es : (r+1)*n*es], shape: []int{n},
			st: cStrides([]int{n}, es), dtype: rows.Dtype()}
		v, err := nanArgOf(sub, wantMax)
		if err != nil {
			return nil, fmt.Errorf("num: %s: 第 %d 条线: %w", op, r, err)
		}
		ow[r] = v
	}
	return out, nil
}

// nanArgOf 在 1-D 数组里找极值下标(跳过 NaN;全 NaN → 错误)。
func nanArgOf(a *Array, wantMax bool) (int64, error) {
	n := a.Size()
	if n == 0 {
		return 0, fmt.Errorf("空数组无极值下标: %w", ErrShape)
	}
	// 统一转 float64 比较(NaN 判定与 numpy 的 nan* 口径一致)
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return 0, err
	}
	w := win[float64](engineWindow(f))
	best := int64(-1)
	var bv float64
	for i, v := range w {
		if v != v { // NaN 跳过
			continue
		}
		if best < 0 {
			best, bv = int64(i), v
			continue
		}
		if wantMax {
			if v > bv {
				best, bv = int64(i), v
			}
		} else {
			if v < bv {
				best, bv = int64(i), v
			}
		}
	}
	if best < 0 {
		return 0, fmt.Errorf("全 NaN 的线没有极值下标(numpy 抛 ValueError): %w", ErrValue)
	}
	return best, nil
}

// setScalarInt64 写入单元素 int64 结果(支持 0-d 与 keepdims 形状)。
func setScalarInt64(out *Array, v int64) error {
	if out.Size() == 0 {
		return nil
	}
	if out.Dtype().Kind != Int64 {
		return fmt.Errorf("num: 期望 int64 目标,得 %s: %w", out.Dtype(), ErrDtype)
	}
	buf := engineWindow(out)
	if len(buf) < 8 {
		return fmt.Errorf("num: 标量写入缓冲不足: %w", ErrShape)
	}
	binaryPutInt64(buf[:8], v)
	return nil
}

// binaryPutInt64 小端写 int64(避免额外 import 顺序问题)。
func binaryPutInt64(b []byte, v int64) {
	u := uint64(v)
	b[0] = byte(u)
	b[1] = byte(u >> 8)
	b[2] = byte(u >> 16)
	b[3] = byte(u >> 24)
	b[4] = byte(u >> 32)
	b[5] = byte(u >> 40)
	b[6] = byte(u >> 48)
	b[7] = byte(u >> 56)
}

var _ = math.NaN

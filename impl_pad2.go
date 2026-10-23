package num

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
)

// —— pad 模式扩展(Step 14):linear_ramp / maximum / minimum / mean / median ——
//
// 语义与 numpy 一致:
// - maximum/minimum/mean/median:用**沿该轴的统计值**填充(逐"线"计算,
// 即固定其它轴下标后对这条线求统计);输出 dtype 仍是输入 dtype
// (整数按截断落值,numpy 同)。
// - linear_ramp:从边缘值线性过渡到 end_values(默认 0),同样按线计算。

// PadOpts pad 的参数集合(*_like 之外的模式需要额外参数)。
type PadOpts struct {
	Mode      string
	CVal      float64
	EndBefore float64
	EndAfter  float64
	// StatLength > 0 时统计类模式只用**边缘 N 个元素**计算统计量(numpy 的 stat_length);
	// 0 表示用整条"线"。前侧取前 N 个、后侧取后 N 个,两侧统计量可以不同。
	StatLength int
}

// PadWith ≡ np.pad(a, widths, mode, constant_values, end_values)。
func PadWith(a *Array, widths [][2]int, opt PadOpts) (*Array, error) {
	switch opt.Mode {
	case "", "constant", "edge", "reflect", "symmetric", "wrap":
		return Pad(a, widths, opt.Mode, opt.CVal)
	case "linear_ramp", "maximum", "minimum", "mean", "median":
		return padStat(a, widths, opt)
	default:
		return nil, fmt.Errorf("num: pad: 不支持的 mode %q: %w", opt.Mode, ErrValue)
	}
}

// padStat 统计/线性过渡类模式的实现。
func padStat(a *Array, widths [][2]int, opt PadOpts) (*Array, error) {
	nd := a.NDim()
	if nd == 0 {
		return nil, fmt.Errorf("num: pad: 0-d 输入不支持: %w", ErrShape)
	}
	pw, err := normalizePadWidths(widths, nd)
	if err != nil {
		return nil, err
	}
	// 先用 edge 模式把内部与边缘填好(保证内部区域来自源数组)
	out, err := Pad(a, widths, "edge", 0)
	if err != nil {
		return nil, err
	}
	for d := 0; d < nd; d++ {
		pb, pa := pw[d][0], pw[d][1]
		if pb == 0 && pa == 0 {
			continue
		}
		n := a.Shape()[d]
		if n == 0 {
			return nil, fmt.Errorf("num: pad: 轴 %d 长度为 0 无法求统计: %w", d, ErrShape)
		}
		switch opt.Mode {
		case "linear_ramp":
			if err := padRampAxis(out, a, d, pb, pa, opt.EndBefore, opt.EndAfter); err != nil {
				return nil, err
			}
		default:
			nOut := out.Shape()[d]
			if opt.StatLength > 0 {
				// 前侧用前 N 个、后侧用后 N 个,分别求统计量(numpy 的 stat_length)
				sl := opt.StatLength
				if sl > n {
					sl = n
				}
				if pb > 0 {
					sub, err := sliceRangeAlongAxis(a, d, 0, sl)
					if err != nil {
						return nil, err
					}
					st, err := padStatAxis(sub, d, opt.Mode)
					if err != nil {
						return nil, err
					}
					if err := fillPadRegion(out, d, 0, pb, st); err != nil {
						return nil, err
					}
				}
				if pa > 0 {
					sub, err := sliceRangeAlongAxis(a, d, n-sl, n)
					if err != nil {
						return nil, err
					}
					st, err := padStatAxis(sub, d, opt.Mode)
					if err != nil {
						return nil, err
					}
					if err := fillPadRegion(out, d, nOut-pa, nOut, st); err != nil {
						return nil, err
					}
				}
				continue
			}
			stat, err := padStatAxis(a, d, opt.Mode)
			if err != nil {
				return nil, err
			}
			if err := fillPadRegion(out, d, 0, pb, stat); err != nil {
				return nil, err
			}
			if err := fillPadRegion(out, d, nOut-pa, nOut, stat); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// padStatAxis 计算沿轴 d 的统计量,返回形状与 a 相同但轴 d 长度为 1 的数组。
func padStatAxis(a *Array, d int, mode string) (*Array, error) {
	var (
		res *Array
		err error
	)
	switch mode {
	case "maximum":
		res, err = Max(a, []int{d}, true)
	case "minimum":
		res, err = Min(a, []int{d}, true)
	case "mean":
		res, err = Mean(a, []int{d}, true)
	case "median":
		res, err = Median(a, []int{d}, true)
	default:
		return nil, fmt.Errorf("num: pad: 未实现的统计模式 %q: %w", mode, ErrValue)
	}
	if err != nil {
		return nil, err
	}
	// 落回输入 dtype(numpy 对整型输入返回整型,按截断)
	if res.Dtype() != a.Dtype() {
		c, err := transformCopy(res, res.Shape(), a.Dtype())
		if err != nil {
			return nil, err
		}
		res = c
	}
	return res, nil
}

// fillPadRegion 把 vals(轴 d 长度为 1)广播写进 out 在轴 d 上的 [lo,hi) 区间。
func fillPadRegion(out *Array, d, lo, hi int, vals *Array) error {
	if lo >= hi {
		return nil
	}
	es := out.Dtype().Itemsize
	db := engineWindow(out)
	vb := engineWindow(vals.AsContiguous())
	sh := out.Shape()
	vs := vals.Shape()
	// vals 的各维步长(轴 d 长度为 1,跳过即可)
	vst := make([]int, len(vs))
	acc := 1
	for i := len(vs) - 1; i >= 0; i-- {
		vst[i] = acc
		acc *= vs[i]
	}
	idx := make([]int, len(sh))
	size := out.Size()
	for fi := 0; fi < size; fi++ {
		rem := fi
		for k := len(sh) - 1; k >= 0; k-- {
			idx[k] = rem % sh[k]
			rem /= sh[k]
		}
		if idx[d] < lo || idx[d] >= hi {
			continue
		}
		vi := 0
		for k := 0; k < len(vs); k++ {
			if k == d {
				continue
			}
			vi += idx[k] * vst[k]
		}
		copy(db[fi*es:(fi+1)*es], vb[vi*es:(vi+1)*es])
	}
	return nil
}

// padRampAxis 沿轴 d 做线性过渡填充。
func padRampAxis(out, a *Array, d, pb, pa int, endBefore, endAfter float64) error {
	nd := a.NDim()
	n := a.Shape()[d]
	if pb > 0 {
		edge, err := sliceAlongAxis(a, d, 0)
		if err != nil {
			return err
		}
		if err := fillRampRegion(out, d, 0, pb, edge, endBefore, pb, true); err != nil {
			return err
		}
	}
	if pa > 0 {
		edge, err := sliceAlongAxis(a, d, n-1)
		if err != nil {
			return err
		}
		if err := fillRampRegion(out, d, out.Shape()[d]-pa, out.Shape()[d], edge, endAfter, pa, false); err != nil {
			return err
		}
	}
	_ = nd
	return nil
}

// sliceAlongAxis 取 a 沿轴 d 的第 i 个切片,结果形状中该轴长度为 1。
func sliceAlongAxis(a *Array, d, i int) (*Array, error) {
	idx := make([]Indexer, a.NDim())
	for k := range idx {
		idx[k] = All()
	}
	idx[d] = Slice(i, i+1)
	return a.Sub(idx...)
}

// fillRampRegion 把斜坡写进 out 的 [lo,hi) 区间。
// before=true 表示这是"前侧"填充(从外向内靠近边缘)。
func fillRampRegion(out *Array, d, lo, hi int, edge *Array, endVal float64, p int, before bool) error {
	if lo >= hi || p <= 0 {
		return nil
	}
	es := out.Dtype().Itemsize
	db := engineWindow(out)
	ew := win[float64](engineWindow(mustF64(edge)))
	sh := out.Shape()
	idx := make([]int, len(sh))
	size := out.Size()
	for fi := 0; fi < size; fi++ {
		rem := fi
		for k := len(sh) - 1; k >= 0; k-- {
			idx[k] = rem % sh[k]
			rem /= sh[k]
		}
		if idx[d] < lo || idx[d] >= hi {
			continue
		}
		// 该线在 edge 里的位置
		vi := 0
		stride := 1
		for k := len(sh) - 1; k >= 0; k-- {
			if k == d {
				continue
			}
			vi += idx[k] * stride
			_ = k
			// 简化:edge 形状与 out 相同但轴 d 为 1,故其步长与 out 一致
			stride *= sh[k]
		}
		// 上面 stride 计算依赖轴序,改用直接构造 edge 的平坦下标
		vi = flatIndexSkipping(sh, idx, d)
		e := ew[vi]
		var t float64
		if before {
			t = float64(p-(idx[d]-lo)) / float64(p)
		} else {
			t = float64(idx[d]-lo+1) / float64(p)
		}
		v := e + (endVal-e)*t
		if err := writeScalarToBytes(db[fi*es:(fi+1)*es], out.Dtype(), v); err != nil {
			return err
		}
	}
	return nil
}

// flatIndexSkipping 计算"去掉轴 skip 后"的平坦下标(用于访问轴长约 1 的广播数组)。
func flatIndexSkipping(sh, idx []int, skip int) int {
	vi := 0
	acc := 1
	for k := len(sh) - 1; k >= 0; k-- {
		if k == skip {
			acc *= 1
			continue
		}
		vi += idx[k] * acc
		acc *= sh[k]
	}
	return vi
}

// mustF64 把数组转成 float64 视图(内部用,失败时返回空数组)。
func mustF64(a *Array) *Array {
	c, err := transformCopy(a, a.Shape(), DFloat64)
	if err != nil {
		z, _ := New([]int{0}, DFloat64)
		return z
	}
	return c
}

// writeScalarToBytes 把 float64 值按目标 dtype 落进字节片。
// 整型按 **截断向零** 落值(Go 的 float→int 转换即此语义,与 numpy 的 astype 一致)。
func writeScalarToBytes(dst []byte, dt Dtype, v float64) error {
	switch dt.Kind {
	case Float64:
		binary.LittleEndian.PutUint64(dst, math.Float64bits(v))
	case Float32:
		binary.LittleEndian.PutUint32(dst, math.Float32bits(float32(v)))
	case Int8:
		dst[0] = byte(int8(v))
	case Int16:
		binary.LittleEndian.PutUint16(dst, uint16(int16(v)))
	case Int32:
		binary.LittleEndian.PutUint32(dst, uint32(int32(v)))
	case Int64:
		binary.LittleEndian.PutUint64(dst, uint64(int64(v)))
	case Uint8:
		dst[0] = byte(uint8(v))
	case Uint16:
		binary.LittleEndian.PutUint16(dst, uint16(v))
	case Uint32:
		binary.LittleEndian.PutUint32(dst, uint32(v))
	case Uint64:
		binary.LittleEndian.PutUint64(dst, uint64(v))
	default:
		return fmt.Errorf("num: pad linear_ramp 无 %s 内核: %w", dt, ErrDtype)
	}
	return nil
}

// UniqueAxis ≡ np.unique(a, axis=k):按"沿该轴的切片"去重(按字典序排序)。
//
// 1-D 输入等价于普通 unique(返回排序去重后的一维)。
// 比较键用 float64(对 |v| < 2^53 的整型/浮点精确;更大的整型会有精度限制,已注明)。
func UniqueAxis(a *Array, axis int) (*Array, error) {
	if a.NDim() == 1 {
		return Unique(a)
	}
	if a.NDim() == 0 {
		return nil, fmt.Errorf("num: unique(axis): 0-d 输入不支持: %w", ErrShape)
	}
	d, err := normalizeAxis(a, axis)
	if err != nil {
		return nil, fmt.Errorf("num: unique(axis): %w", err)
	}
	// 把轴移到最前,得到 (n, M) 的行视角
	nd := a.NDim()
	perm := append([]int{d}, notAxes(nd, []int{d})...)
	tr, err := a.Transpose(perm...)
	if err != nil {
		return nil, err
	}
	n := a.Shape()[d]
	m := a.Size() / n
	rows, err := tr.AsContiguous().Reshape([]int{n, m})
	if err != nil {
		return nil, err
	}
	keys, err := transformCopy(rows, []int{n, m}, DFloat64)
	if err != nil {
		return nil, err
	}
	kw := win[float64](engineWindow(keys))
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(x, y int) bool {
		ra := kw[idx[x]*m : (idx[x]+1)*m]
		rb := kw[idx[y]*m : (idx[y]+1)*m]
		for j := 0; j < m; j++ {
			if ra[j] != rb[j] {
				// NaN 排在最后(numpy 的排序语义)
				if ra[j] != ra[j] {
					return false
				}
				if rb[j] != rb[j] {
					return true
				}
				return ra[j] < rb[j]
			}
		}
		return false
	})
	// 去重
	keep := make([]int, 0, n)
	for i, k := range idx {
		if i == 0 {
			keep = append(keep, k)
			continue
		}
		prev := idx[i-1]
		same := true
		for j := 0; j < m; j++ {
			pa, pb := kw[prev*m+j], kw[k*m+j]
			if pa != pb && !(pa != pa && pb != pb) {
				same = false
				break
			}
		}
		if !same {
			keep = append(keep, k)
		}
	}
	// 组装输出:取 keep 这些行,再逆置换回原轴序
	outRows, err := New([]int{len(keep), m}, a.Dtype())
	if err != nil {
		return nil, err
	}
	es := a.Dtype().Itemsize
	sb := engineWindow(rows)
	db := engineWindow(outRows)
	for i, k := range keep {
		copy(db[i*m*es:(i+1)*m*es], sb[k*m*es:(k+1)*m*es])
	}
	// (len(keep), m) → 置换形状 → 逆置换
	permShape := make([]int, nd)
	permShape[0] = len(keep)
	for i, ax := range perm[1:] {
		permShape[i+1] = a.Shape()[ax]
	}
	reshaped, err := outRows.Reshape(permShape)
	if err != nil {
		return nil, err
	}
	inv := make([]int, nd)
	for i, p := range perm {
		inv[p] = i
	}
	return reshaped.Transpose(inv...)
}

// sliceRangeAlongAxis 取 a 沿轴 d 的 [lo,hi) 区间(其余轴全取),用于 stat_length。
func sliceRangeAlongAxis(a *Array, d, lo, hi int) (*Array, error) {
	idx := make([]Indexer, a.NDim())
	for k := range idx {
		idx[k] = All()
	}
	idx[d] = Slice(lo, hi)
	return a.Sub(idx...)
}

// Cumsum ≡ np.cumsum(a, axis):前缀和。
//
// 与 Accumulate 的差异(与 numpy 一致):axis 省略时**先拉平**;dtype 按 sum 的
// 累积规则提升(有符号整型/bool → int64,无符号 → uint64,浮点保持)。
func Cumsum(a *Array, axis []int) (*Array, error) {
	return cumLike(a, axis, Add)
}

// Cumprod ≡ np.cumprod(a, axis):前缀积(dtype 规则与 Cumsum 同)。
func Cumprod(a *Array, axis []int) (*Array, error) {
	return cumLike(a, axis, Multiply)
}

func cumLike(a *Array, axis []int, fn BinaryFunc) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: cumsum/cumprod: 数组为 nil: %w", ErrDtype)
	}
	acc, err := sumAccDtype(a.Dtype())
	if err != nil {
		return nil, err
	}
	src := a
	if acc != a.Dtype() {
		src, err = transformCopy(a, a.Shape(), acc)
		if err != nil {
			return nil, err
		}
	}
	if len(axis) == 0 {
		// ndarray 语义:默认展平
		return Accumulate(fn, src.AsContiguous().Flatten(), nil)
	}
	return Accumulate(fn, src, axis)
}

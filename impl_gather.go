package num

// :gather/scatter(B 拥有)。
//
// 语义基准(numpy 2.5.4 实测,):
// - Take(平铺,≡ np.take(axis=None)):负索引归一(+= n),越界 → ErrIndex;
// 输出形状 = 索引数组形状,dtype 同源;索引须为整型(浮点 → ErrDtype)。
// - TakeAlongAxis(≡ np.take_along_axis):索引 ndim 须等于源 ndim;非轴维
// 允许 1 → 广播;负索引归一;越界 → ErrIndex;输出形状 = 索引形状。
// - Put(平铺 scatter,≡ np.put,原地):负索引归一、重复 last-write-wins、
// 值数组按 k % len(vals) 循环取用;值按目标 dtype **不安全转换**
// (float → int 向零截断,numpy put 同)。
//
// 实现:纯字节级拷贝(itemsize 步进),与 dtype 无关 —— f16/complex/S/U 也能
// gather(不经过 decode/encode);Put 的取值/写回走 decodeOne/encodeOne
// (复用 A 的编解码,含截断与回绕语义)。

import "fmt"

// idxInts 读整数索引数组为 []int64(任何整型 dtype;u64 超 MaxInt64 → ErrIndex)。
func idxInts(idx *Array) ([]int64, error) {
	n := idx.Size()
	out := make([]int64, n)
	if n == 0 {
		return out, nil
	}
	c := idx.AsContiguous()
	switch idx.Dtype().Kind {
	case Int8:
		for i, v := range win[int8](engineWindow(c)) {
			out[i] = int64(v)
		}
	case Int16:
		for i, v := range win[int16](engineWindow(c)) {
			out[i] = int64(v)
		}
	case Int32:
		for i, v := range win[int32](engineWindow(c)) {
			out[i] = int64(v)
		}
	case Int64:
		copy(out, win[int64](engineWindow(c)))
	case Uint8:
		for i, v := range win[uint8](engineWindow(c)) {
			out[i] = int64(v)
		}
	case Uint16:
		for i, v := range win[uint16](engineWindow(c)) {
			out[i] = int64(v)
		}
	case Uint32:
		for i, v := range win[uint32](engineWindow(c)) {
			out[i] = int64(v)
		}
	case Uint64:
		for i, v := range win[uint64](engineWindow(c)) {
			if v > uint64(1<<63-1) {
				return nil, fmt.Errorf("num: 索引 %d 超出 int64: %w", v, ErrIndex)
			}
			out[i] = int64(v)
		}
	default:
		return nil, fmt.Errorf("num: 索引 dtype %s 不支持(numpy 同拒浮点索引): %w", idx.Dtype(), ErrDtype)
	}
	return out, nil
}

// flatOff C 序第 k 个元素的字节偏移(支持任意步长布局;越界调用方保证)。
func (a *Array) flatOff(k int) int {
	off := a.off
	for d := len(a.shape) - 1; d >= 0; d-- {
		off += (k % a.shape[d]) * a.st[d]
		k /= a.shape[d]
	}
	return off
}

// Take 平铺 gather(≡ np.take(axis=None))。负索引归一,越界 → ErrIndex。
func Take(a, idx *Array) (*Array, error) {
	n := a.Size()
	ii, err := idxInts(idx)
	if err != nil {
		return nil, err
	}
	out, err := New(idx.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	for k, v := range ii {
		if v < 0 {
			v += int64(n)
		}
		if v < 0 || v >= int64(n) {
			return nil, fmt.Errorf("num: Take 索引 %d 越界(size=%d): %w", ii[k], n, ErrIndex)
		}
		ii[k] = v
	}
	src := a.AsContiguous()
	it := a.Dtype().Itemsize
	dst := engineWindow(out)
	sw := engineWindow(src)
	for k, v := range ii {
		copy(dst[k*it:(k+1)*it], sw[int(v)*it:int(v+1)*it])
	}
	return out, nil
}

// TakeAlongAxis ≡ np.take_along_axis:索引 ndim = 源 ndim,非轴维 1 可广播。
func TakeAlongAxis(a, idx *Array, axis int) (*Array, error) {
	nd := a.NDim()
	if nd == 0 {
		return nil, fmt.Errorf("num: TakeAlongAxis 源为 0-d: %w", ErrShape)
	}
	if idx.NDim() != nd {
		return nil, fmt.Errorf("num: TakeAlongAxis 索引 ndim=%d ≠ 源 ndim=%d: %w", idx.NDim(), nd, ErrShape)
	}
	ax := axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return nil, fmt.Errorf("num: TakeAlongAxis axis %d 越界(ndim=%d): %w", axis, nd, ErrIndex)
	}
	aSh, iSh := a.Shape(), idx.Shape()
	for d := 0; d < nd; d++ {
		if d == ax {
			continue
		}
		if iSh[d] != aSh[d] && iSh[d] != 1 {
			return nil, fmt.Errorf("num: TakeAlongAxis 非轴维 %d:%d 与源 %d 不容: %w", d, iSh[d], aSh[d], ErrShape)
		}
	}
	ii, err := idxInts(idx)
	if err != nil {
		return nil, err
	}
	out, err := New(iSh, a.Dtype())
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	src := a.AsContiguous()
	it := a.Dtype().Itemsize
	st := cStrides(aSh, it)
	sw, dw := engineWindow(src), engineWindow(out)
	pos := make([]int, nd)
	for k := 0; k < out.Size(); k++ {
		off := 0
		for d := 0; d < nd; d++ {
			var ix int
			if d == ax {
				v := ii[k]
				if v < 0 {
					v += int64(aSh[d])
				}
				if v < 0 || v >= int64(aSh[d]) {
					return nil, fmt.Errorf("num: TakeAlongAxis 索引 %d 越界(轴 %d 大小 %d): %w", ii[k], d, aSh[d], ErrIndex)
				}
				ix = int(v)
			} else if iSh[d] != 1 {
				ix = pos[d]
			}
			off += ix * st[d]
		}
		copy(dw[k*it:(k+1)*it], sw[off:off+it])
		for d := nd - 1; d >= 0; d-- {
			pos[d]++
			if pos[d] < iSh[d] {
				break
			}
			pos[d] = 0
		}
	}
	return out, nil
}

// Put 平铺 scatter(≡ np.put,原地):负索引归一、重复 last-write-wins、
// 值循环取用、值按目标 dtype 不安全转换。
func Put(a, idx, vals *Array) error {
	n := a.Size()
	ii, err := idxInts(idx)
	if err != nil {
		return err
	}
	if len(ii) == 0 {
		return nil
	}
	vn := vals.Size()
	if vn == 0 {
		return fmt.Errorf("num: Put 值数组为空: %w", ErrShape)
	}
	vc := vals.AsContiguous()
	for k, v := range ii {
		if v < 0 {
			v += int64(n)
		}
		if v < 0 || v >= int64(n) {
			return fmt.Errorf("num: Put 索引 %d 越界(size=%d): %w", ii[k], n, ErrIndex)
		}
		val, err := decodeOne(vals.Dtype(), vc.flatOff(k%vn), vc.buf)
		if err != nil {
			return fmt.Errorf("num: Put 取值: %w", err)
		}
		if err := encodeOne(a.Dtype(), a.flatOff(int(v)), a.buf, val); err != nil {
			return fmt.Errorf("num: Put 写回: %w", err)
		}
	}
	return nil
}

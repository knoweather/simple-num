package num

import (
	"fmt"
	"unsafe"
)

// —— 基本索引(numpy 基础索引子集)——
//
// 支持:整点(含负索引)、切片 start:stop:step(step≥1)、省略号(至多一个)、
// 新轴 None;缺省尾部维按整档(:)补齐(故 a[1] 合法,numpy 同)。
// 语义 = numpy 基础索引:切片端点 None → 0/n,负值加 n 后 clamp 到 [0,n];
// 越界整点 / 索引项多于维度 / 多个省略号 → ErrIndex;负步长与 step=0 → ErrIndex
// ( §2.1 ErrIndex 注释里"negative step 未支持期"即此)。
// Sub 返回**零拷贝视图**(共享 buf,shape/st/off 重排)。
//
// 赋值口径(实测 numpy 2.5.4,):np 的**赋值** cast 是 unsafe 语义——
// 浮点→整型截断、整数回绕都允许(`a[:2]=np.array([1.5,2.5])` → [1,2]),
// 这与显式 Cast(same_kind)不同;因此 SetSub 走 decode→encode 的数值转换,
// 非数值 dtype(complex/datetime/S/U)在 – 前照旧报 ErrDtype。
// 重叠赋值(如 a[1:]=a[:-1])与 numpy 同:先复制源再写(实测 numpy 得 [0,0,1,2,3,4])。

// SliceNone 切片端点"无"标记(对应 numpy slice 的 None)。取 int32 下界,
// 以便 32 位目标(交叉编译)也能通过编译。
const SliceNone = -1 << 31

type idxKind uint8

const (
	idxInt idxKind = iota + 1
	idxSlice
	idxEllipsis
	idxNewAxis
)

// Indexer 基本索引项(由 Idx/Slice/SliceStep/All 与 Ellipsis/NewAxis 构造;
// 零值为非法项,Sub 会报 ErrIndex)。
type Indexer struct {
	kind       idxKind
	i          int
	start, end int
	step       int
}

// Idx 整点索引(负值加维度长;越界 → ErrIndex)。
func Idx(i int) Indexer { return Indexer{kind: idxInt, i: i} }

// Slice start:stop(step=1;端点用 SliceNone 表示缺省)。
func Slice(start, stop int) Indexer {
	return Indexer{kind: idxSlice, start: start, end: stop, step: 1}
}

// SliceStep start:stop:step(step 必须 ≥1)。
func SliceStep(start, stop, step int) Indexer {
	return Indexer{kind: idxSlice, start: start, end: stop, step: step}
}

// All 整档切片 ":"。
func All() Indexer { return Slice(SliceNone, SliceNone) }

// Ellipsis 省略号 "..."(每个索引式最多一个,展开为若干整档)。
func Ellipsis() Indexer { return Indexer{kind: idxEllipsis} }

// NewAxis 新轴 None(插入长度 1 的轴,步长 0)。
func NewAxis() Indexer { return Indexer{kind: idxNewAxis} }

// Sub 基本索引取视图(零拷贝)。
func (a *Array) Sub(idx ...Indexer) (*Array, error) {
	items, err := a.expandIndexers(idx)
	if err != nil {
		return nil, err
	}
	off := a.off
	sh := make([]int, 0, len(items))
	st := make([]int, 0, len(items))
	dim := 0
	for k, it := range items {
		switch it.kind {
		case idxNewAxis:
			sh = append(sh, 1)
			st = append(st, 0)
		case idxInt:
			if dim >= len(a.shape) {
				return nil, fmt.Errorf("num: 索引项 %d: 整点索引多于维度 %d: %w", k, len(a.shape), ErrIndex)
			}
			n := a.shape[dim]
			i := it.i
			if i < 0 {
				i += n
			}
			if i < 0 || i >= n {
				return nil, fmt.Errorf("num: 索引项 %d: 整点 %d 越界(维长 %d): %w", k, it.i, n, ErrIndex)
			}
			off += i * a.st[dim]
			dim++
		case idxSlice:
			if dim >= len(a.shape) {
				return nil, fmt.Errorf("num: 索引项 %d: 切片多于维度 %d: %w", k, len(a.shape), ErrIndex)
			}
			if it.step < 1 {
				return nil, fmt.Errorf("num: 索引项 %d: 步长 %d 不支持(v0 仅 step≥1): %w", k, it.step, ErrIndex)
			}
			n := a.shape[dim]
			start, stop := sliceBounds(it.start, it.end, n)
			length := 0
			if stop > start {
				length = (stop - start + it.step - 1) / it.step
			}
			sh = append(sh, length)
			st = append(st, a.st[dim]*it.step)
			off += start * a.st[dim]
			dim++
		default:
			return nil, fmt.Errorf("num: 索引项 %d: 非法(numpy 空索引项): %w", k, ErrIndex)
		}
	}
	// 尾部未覆盖的维度按整档补齐(numpy 同:a[1] 于 3 维 → 形状 (3,4))
	for ; dim < len(a.shape); dim++ {
		sh = append(sh, a.shape[dim])
		st = append(st, a.st[dim])
	}
	return &Array{buf: a.buf, shape: sh, st: st, off: off, dtype: a.dtype}, nil
}

// expandIndexers 展开省略号(至多一个;补 ndim-非省略号消耗项 个整档)。
// 新轴(None)不消耗维度,故计数时排除(实测 numpy:`a[1:2,...,None]` 于 3 维
// 得 (1,3,4,1) —— 省略号吃掉全部剩余 2 维,None 追加在最后)。
func (a *Array) expandIndexers(idx []Indexer) ([]Indexer, error) {
	nEll := 0
	pos := -1
	nNew := 0
	for i, it := range idx {
		switch it.kind {
		case idxEllipsis:
			nEll++
			pos = i
		case idxNewAxis:
			nNew++
		}
	}
	if nEll > 1 {
		return nil, fmt.Errorf("num: %d 个省略号(至多 1 个): %w", nEll, ErrIndex)
	}
	if nEll == 0 {
		return idx, nil
	}
	fill := len(a.shape) - (len(idx) - 1 - nNew)
	if fill < 0 {
		return nil, fmt.Errorf("num: 索引项消耗 %d 维 > %d: %w", len(idx)-1-nNew, len(a.shape), ErrIndex)
	}
	out := make([]Indexer, 0, len(idx)-1+fill)
	out = append(out, idx[:pos]...)
	for i := 0; i < fill; i++ {
		out = append(out, All())
	}
	out = append(out, idx[pos+1:]...)
	return out, nil
}

// sliceBounds numpy/python 切片端点归一(step≥1):None → 0/n;负值加 n;
// 再 clamp 到 [0,n]。
func sliceBounds(start, stop, n int) (int, int) {
	if start == SliceNone {
		start = 0
	} else if start < 0 {
		if start += n; start < 0 {
			start = 0
		}
	} else if start > n {
		start = n
	}
	if stop == SliceNone {
		stop = n
	} else if stop < 0 {
		if stop += n; stop < 0 {
			stop = 0
		}
	} else if stop > n {
		stop = n
	}
	return start, stop
}

// SetSubScalar 把标量 v 写入索引视图的每个元素(v 按目标 dtype 编码一次;
// 编码失败 → ErrDtype,与 Full 同口径)。
func (a *Array) SetSubScalar(v any, idx ...Indexer) error {
	sub, err := a.Sub(idx...)
	if err != nil {
		return err
	}
	if sub.Size() == 0 {
		return nil
	}
	it := sub.dtype.Itemsize
	tmp := make([]byte, it)
	if err := encodeOne(sub.dtype, 0, tmp, v); err != nil {
		return err
	}
	sub.forEachElemOff(func(off int) bool {
		copy(sub.buf[off:off+it], tmp)
		return true
	})
	return nil
}

// SetSub 把 src 广播写入索引视图(形状须能广播到视图;dtype 不同按 numpy
// 赋值 unsafe 口径数值转换;重叠先复制源)。
func (a *Array) SetSub(src *Array, idx ...Indexer) error {
	sub, err := a.Sub(idx...)
	if err != nil {
		return err
	}
	if src == nil {
		return fmt.Errorf("num: SetSub nil 源: %w", ErrDtype)
	}
	if !broadcastOk(src.Shape(), sub.Shape()) {
		return fmt.Errorf("num: SetSub 源形状 %v 不能广播到视图 %v: %w", src.Shape(), sub.Shape(), ErrShape)
	}
	if sub.Size() == 0 {
		return nil
	}
	if overlaps(sub.buf, src.buf) { // numpy 同:重叠赋值先取源快照(恒拷贝)
		src = src.copyOf()
	}
	steps := elemStepsBroadcast(src.shape, src.st, sub.shape)
	it := sub.dtype.Itemsize
	if src.dtype == sub.dtype {
		walkPair(sub, src, steps, func(dstOff, srcOff int) bool {
			copy(sub.buf[dstOff:dstOff+it], src.buf[srcOff:srcOff+it])
			return true
		})
		return nil
	}
	var perr error
	walkPair(sub, src, steps, func(dstOff, srcOff int) bool {
		v, err := decodeElem(src, srcOff, sub.dtype)
		if err == nil {
			err = encodeOne(sub.dtype, dstOff, sub.buf, v)
		}
		if err != nil {
			perr = err
			return false
		}
		return true
	})
	return perr
}

// forEachElemOff 按 C 序遍访视图元素字节偏移(空数组不访问)。
func (a *Array) forEachElemOff(visit func(off int) bool) {
	if a.Size() == 0 {
		return
	}
	sh := a.shape
	idx := make([]int, len(sh))
	off := a.off
	for {
		if !visit(off) {
			return
		}
		k := len(sh) - 1
		for {
			if k < 0 {
				return
			}
			idx[k]++
			off += a.st[k]
			if idx[k] < sh[k] {
				break
			}
			off -= idx[k] * a.st[k]
			idx[k] = 0
			k--
		}
	}
}

// walkPair 按 out 形状 C 序同时遍访 out(真步长)与 src(广播步长 steps)。
func walkPair(out, src *Array, steps []int, visit func(dstOff, srcOff int) bool) {
	if out.Size() == 0 {
		return
	}
	sh := out.shape
	idx := make([]int, len(sh))
	dstOff := out.off
	srcOff := src.off
	for {
		if !visit(dstOff, srcOff) {
			return
		}
		k := len(sh) - 1
		for {
			if k < 0 {
				return
			}
			idx[k]++
			dstOff += out.st[k]
			srcOff += steps[k]
			if idx[k] < sh[k] {
				break
			}
			dstOff -= idx[k] * out.st[k]
			srcOff -= idx[k] * steps[k]
			idx[k] = 0
			k--
		}
	}
}

// copyOf 恒拷贝为独立缓冲的同形 C 连续数组(重叠赋值的源快照)。
// 不能复用 AsContiguous:已连续时它返回共享 buf 的视图,断不开别名。
func (a *Array) copyOf() *Array {
	out, err := New(a.shape, a.dtype)
	if err != nil {
		return nil // dtype 已校验;防御分支不可达
	}
	if w := a.fastWindow(); w != nil {
		copy(out.buf, w)
		return out
	}
	copyStrided(out.buf, a.buf, a.shape, a.st, a.off, a.dtype.Itemsize)
	return out
}

// overlaps 两缓冲区间是否重叠(同一底层数组的潜在别名检测)。
func overlaps(x, y []byte) bool {
	if len(x) == 0 || len(y) == 0 {
		return false
	}
	xs := uintptr(unsafe.Pointer(unsafe.SliceData(x)))
	ys := uintptr(unsafe.Pointer(unsafe.SliceData(y)))
	return xs < ys+uintptr(len(y)) && ys < xs+uintptr(len(x))
}

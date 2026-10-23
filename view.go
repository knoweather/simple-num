package num

// 视图机制(A):零拷贝子视图、广播步长、快速窗判定、strided 拷贝。
// 内存模型与对齐纪律见 array.go 顶部注释。

// newViewInternal 共用视图构造(共享 buf,shape/st/off 重排)。
func newViewInternal(a *Array, shape, st []int, off int) *Array {
	return &Array{buf: a.buf, shape: shape, st: st, off: off, dtype: a.dtype}
}

// fastWindow 引擎直通窗:C 序连续 + 基址/偏移 8 字节对齐;返回
// buf[off : off+Size*Itemsize]。不满足返回 nil(调用方走整备 transformCopy)。
func (a *Array) fastWindow() []byte {
	if !a.isCContig() || !a.isAligned8() {
		return nil
	}
	n := a.Size() * a.dtype.Itemsize
	if n == 0 {
		return a.buf[:0:0]
	}
	return a.buf[a.off : a.off+n : a.off+n]
}

// broadcastOk 源形状能否广播到 out(右对齐;每维 =1 或相等)。
func broadcastOk(src, out []int) bool {
	for k := 1; k <= len(out); k++ {
		sd := 1 // 缺失头部维 = 1
		if k <= len(src) {
			sd = src[len(src)-k]
		}
		od := out[len(out)-k]
		if sd != od && sd != 1 {
			return false
		}
	}
	return true
}

// elemStepsBroadcast 逐元素源步长:src 维=1(或缺失头部维)→ 步长 0;相等 → 实步长。
func elemStepsBroadcast(src, srcSt, out []int) []int {
	steps := make([]int, len(out))
	for k := 1; k <= len(out); k++ {
		var sd, ss int
		if k <= len(src) {
			sd = src[len(src)-k]
			ss = srcSt[len(srcSt)-k]
		}
		od := out[len(out)-k]
		if sd == od {
			steps[len(out)-k] = ss
		} else {
			steps[len(out)-k] = 0
		}
	}
	return steps
}

// odometerVisit 按 C 序遍访 shape 的每个元素下标组合;visit 返回 false 提前终止。
func odometerVisit(sh []int, visit func(idx []int) bool) bool {
	n := prodShape(sh)
	idx := make([]int, len(sh))
	for m := 0; m < n; m++ {
		if !visit(idx) {
			return false
		}
		for k := len(sh) - 1; k >= 0; k-- {
			idx[k]++
			if idx[k] < sh[k] {
				break
			}
			idx[k] = 0
		}
	}
	return true
}

// copyStrided dst C 序连续起步,src 按 shape/st(off 起步)逐元素搬 it 字节。
// 带后缀连续优化:剩余 shape 的源段若本身 C 序连续,整块 memcpy。
func copyStrided(dst []byte, src []byte, shape []int, st []int, off int, it int) {
	// 后缀连续检测:从最内维起 st 呈 C 序(itemsize 递增)。
	suffix := 0
	s := it
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] == 1 {
			suffix++ // stride 不可观察,并入连续段判断无副作用
			s *= 1
			continue
		}
		if shape[i] == 0 {
			break
		}
		if st[i] != s {
			break
		}
		suffix++
		s *= shape[i]
	}
	if suffix == len(shape) {
		n := prodShape(shape) * it
		if n > 0 {
			copy(dst[:n], src[off:off+n])
		}
		return
	}
	n := shape[0]
	step := st[0]
	restSt := st[1:]
	restSize := prodShape(shape[1:]) * it
	dst0 := dst
	src0 := src
	for j := 0; j < n; j++ {
		copyStrided(dst0, src0, shape[1:], restSt, off, it)
		dst0 = dst0[restSize:]
		off += step
	}
}

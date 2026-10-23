package num

import (
	"fmt"
	"runtime"
	"sync"
)

// —— 线代核心(Step 7):dot / matmul / inner / outer ——
//
// 口径与 numpy 一致:
// - np.dot :a 的最后一维与 b 的倒数第二维做内积;1-D 参与时按 numpy 的形状规则
// 降维(1-D·1-D → 0-d 标量;N-D·1-D → a.shape[:-1];1-D·M-D →
// b.shape[:-2]+(b.shape[-1],));任一侧 0-d 退化为逐元素乘法(numpy 同)。
// - np.matmul(@):按 numpy 的"1-D 先提升为 (1,K)/(K,1),算完再降回"的规则,
// 并对 batch 维做右对齐广播。
// - np.inner :两侧最后一维内积,结果形状 = a.shape[:-1] + b.shape[:-1]。
// - np.outer :两侧拉平后做外积,形状 (m, n)。
//
// 累加与结果 dtype 均按 promote 后的类型进行(整型按同宽累加,与 numpy 的溢出行为一致)。
// 性能:朴素三重循环(未做分块/SIMD);1e6 级矩阵乘法属后续优化项。

// Dot ≡ np.dot。
func Dot(a, b *Array) (*Array, error) {
	if a.NDim() == 0 || b.NDim() == 0 {
		return Multiply(a, b) // numpy:标量与数组的 dot 即逐元素乘
	}
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: dot: dtype 不可提升: %w", err)
	}
	if isCplxKind(dt.Kind) {
		return dotComplex(a, b)
	}
	ac, err := transformCopy(a, a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	bc, err := transformCopy(b, b.Shape(), dt)
	if err != nil {
		return nil, err
	}
	as, bs := ac.Shape(), bc.Shape()
	ad, bd := len(as), len(bs)
	if bd >= 2 {
		if as[ad-1] != bs[bd-2] {
			return nil, fmt.Errorf("num: dot: a 末维 %d ≠ b 倒数第二维 %d: %w", as[ad-1], bs[bd-2], ErrShape)
		}
	} else if as[ad-1] != bs[0] { // b 为 1-D
		return nil, fmt.Errorf("num: dot: a 末维 %d ≠ b 长度 %d: %w", as[ad-1], bs[0], ErrShape)
	}
	p := prodShape(as[:ad-1])
	k := as[ad-1]
	// b 的 (Q, K, N) 视角
	var q, n int
	if bd == 1 {
		q, n = 1, 1
	} else {
		q = prodShape(bs[:bd-2])
		n = bs[bd-1]
	}
	var outSh []int
	switch {
	case ad == 1 && bd == 1: // 0-d 标量
		outSh = []int{}
	case ad == 1: // (1,) × (Q, K, N) → (Q, N)... numpy: b.shape[:-2]+(N,)
		outSh = append(append([]int{}, bs[:bd-2]...), n)
	case bd == 1: // (P, K) × (K,) → (P,)
		outSh = append([]int{}, as[:ad-1]...)
	default:
		outSh = append(append([]int{}, as[:ad-1]...), bs[:bd-2]...)
		outSh = append(outSh, n)
	}
	out, err := New(outSh, dt)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 || k == 0 {
		return out, nil
	}
	if !linalgDispatch(dt, engineWindow(out), engineWindow(ac), engineWindow(bc), [4]int{p, q, k, n}, modeDot) {
		return nil, fmt.Errorf("num: dot 无 %s 内核: %w", dt, ErrDtype)
	}
	return out, nil
}

// MatMul ≡ np.matmul(a, b)(即 Python 的 @)。
func MatMul(a, b *Array) (*Array, error) {
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: matmul: dtype 不可提升: %w", err)
	}
	if isCplxKind(dt.Kind) {
		return matmulComplex(a, b)
	}
	ac, err := transformCopy(a, a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	bc, err := transformCopy(b, b.Shape(), dt)
	if err != nil {
		return nil, err
	}
	as, bs := ac.Shape(), bc.Shape()
	// 1-D 提升
	aWas1 := len(as) == 1
	bWas1 := len(bs) == 1
	if aWas1 {
		as = []int{1, as[0]}
	}
	if bWas1 {
		bs = []int{bs[0], 1}
	}
	if len(as) < 2 || len(bs) < 2 {
		return nil, fmt.Errorf("num: matmul: 维度不足: %w", ErrShape)
	}
	m, k := as[len(as)-2], as[len(as)-1]
	k2, n := bs[len(bs)-2], bs[len(bs)-1]
	if k != k2 {
		return nil, fmt.Errorf("num: matmul: 内维 %d ≠ %d: %w", k, k2, ErrShape)
	}
	aBatch := as[:len(as)-2]
	bBatch := bs[:len(bs)-2]
	batch, err := broadcastShapes(aBatch, bBatch)
	if err != nil {
		return nil, fmt.Errorf("num: matmul: batch 维不可广播: %w", err)
	}
	outSh := append(append([]int{}, batch...), m, n)
	out, err := New(outSh, dt)
	if err != nil {
		return nil, err
	}
	nb := prodShape(batch)
	if nb == 0 || m == 0 || n == 0 || k == 0 {
		return squeezeMatMul(out, aWas1, bWas1)
	}
	// 逐 batch 计算;广播维的源下标按 0 取。
	aw := engineWindow(ac)
	bw := engineWindow(bc)
	ow := engineWindow(out)
	edt := dt.Itemsize
	_ = edt
	idx := make([]int, len(batch))
	for bi := 0; bi < nb; bi++ {
		// 把 bi 拆成 batch 多下标
		rem := bi
		for d := len(batch) - 1; d >= 0; d-- {
			idx[d] = rem % batch[d]
			rem /= batch[d]
		}
		// a 的 batch 偏移
		ao := 0
		for d := 0; d < len(aBatch); d++ {
			src := len(batch) - len(aBatch) + d
			v := 0
			if aBatch[d] != 1 {
				v = idx[src]
			}
			ao = ao*aBatch[d] + v
		}
		bo := 0
		for d := 0; d < len(bBatch); d++ {
			src := len(batch) - len(bBatch) + d
			v := 0
			if bBatch[d] != 1 {
				v = idx[src]
			}
			bo = bo*bBatch[d] + v
		}
		aoff := ao * m * k
		boff := bo * k * n
		ooff := bi * m * n
		if !linalgDispatch(dt, ow[ooff*dt.Itemsize:(ooff+m*n)*dt.Itemsize],
			aw[aoff*dt.Itemsize:(aoff+m*k)*dt.Itemsize],
			bw[boff*dt.Itemsize:(boff+k*n)*dt.Itemsize],
			[4]int{m, k, n, 0}, modeGemm) {
			return nil, fmt.Errorf("num: matmul 无 %s 内核: %w", dt, ErrDtype)
		}
	}
	return squeezeMatMul(out, aWas1, bWas1)
}

// squeezeMatMul 按 numpy 规则把 1-D 参与产生的长度 1 维度去掉。
func squeezeMatMul(out *Array, aWas1, bWas1 bool) (*Array, error) {
	sh := out.Shape()
	if !aWas1 && !bWas1 {
		return out, nil
	}
	var ns []int
	if aWas1 && bWas1 {
		ns = []int{}
	} else if aWas1 {
		ns = append(ns, sh[:len(sh)-2]...)
		ns = append(ns, sh[len(sh)-1])
	} else {
		ns = append(ns, sh[:len(sh)-1]...)
	}
	r, err := out.Reshape(ns)
	if err != nil {
		return nil, err
	}
	return r, nil
}

// Inner ≡ np.inner:两侧最后一维内积。
func Inner(a, b *Array) (*Array, error) {
	if a.NDim() == 0 || b.NDim() == 0 {
		return Multiply(a, b)
	}
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: inner: dtype 不可提升: %w", err)
	}
	ac, err := transformCopy(a, a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	bc, err := transformCopy(b, b.Shape(), dt)
	if err != nil {
		return nil, err
	}
	as, bs := ac.Shape(), bc.Shape()
	k := as[len(as)-1]
	if k != bs[len(bs)-1] {
		return nil, fmt.Errorf("num: inner: 末维 %d ≠ %d: %w", k, bs[len(bs)-1], ErrShape)
	}
	p := prodShape(as[:len(as)-1])
	q := prodShape(bs[:len(bs)-1])
	outSh := append(append([]int{}, as[:len(as)-1]...), bs[:len(bs)-1]...)
	out, err := New(outSh, dt)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 || k == 0 {
		return out, nil
	}
	if !linalgDispatch(dt, engineWindow(out), engineWindow(ac), engineWindow(bc), [4]int{p, q, k, 0}, modeInner) {
		return nil, fmt.Errorf("num: inner 无 %s 内核: %w", dt, ErrDtype)
	}
	return out, nil
}

// Outer ≡ np.outer:两侧拉平后外积。
func Outer(a, b *Array) (*Array, error) {
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: outer: dtype 不可提升: %w", err)
	}
	ac, err := transformCopy(a.AsContiguous().Flatten(), []int{a.Size()}, dt)
	if err != nil {
		return nil, err
	}
	bc, err := transformCopy(b.AsContiguous().Flatten(), []int{b.Size()}, dt)
	if err != nil {
		return nil, err
	}
	m, n := ac.Size(), bc.Size()
	out, err := New([]int{m, n}, dt)
	if err != nil {
		return nil, err
	}
	if m == 0 || n == 0 {
		return out, nil
	}
	if !linalgDispatch(dt, engineWindow(out), engineWindow(ac), engineWindow(bc), [4]int{m, n, 0, 0}, modeOuter) {
		return nil, fmt.Errorf("num: outer 无 %s 内核: %w", dt, ErrDtype)
	}
	return out, nil
}

// —— 内核分派 ——

const (
	modeGemm = iota
	modeDot
	modeInner
	modeOuter
)

// linalgDispatch 按 dtype 分派到类型化内核;返回 false 表示该 dtype 未支持。
func linalgDispatch(dt Dtype, out, a, b []byte, d [4]int, mode int) bool {
	switch dt.Kind {
	case Float64:
		linalgTyped(win[float64](out), win[float64](a), win[float64](b), d, mode)
	case Float32:
		linalgTyped(win[float32](out), win[float32](a), win[float32](b), d, mode)
	case Int8:
		linalgTyped(win[int8](out), win[int8](a), win[int8](b), d, mode)
	case Int16:
		linalgTyped(win[int16](out), win[int16](a), win[int16](b), d, mode)
	case Int32:
		linalgTyped(win[int32](out), win[int32](a), win[int32](b), d, mode)
	case Int64:
		linalgTyped(win[int64](out), win[int64](a), win[int64](b), d, mode)
	case Uint8:
		linalgTyped(win[uint8](out), win[uint8](a), win[uint8](b), d, mode)
	case Uint16:
		linalgTyped(win[uint16](out), win[uint16](a), win[uint16](b), d, mode)
	case Uint32:
		linalgTyped(win[uint32](out), win[uint32](a), win[uint32](b), d, mode)
	case Uint64:
		linalgTyped(win[uint64](out), win[uint64](a), win[uint64](b), d, mode)
	default:
		return false
	}
	return true
}

// linalgTyped 类型化内核(numpy 的累加语义:按结果 dtype 累加,整型自然溢出回绕)。
func linalgTyped[T castableNum](out, a, b []T, d [4]int, mode int) {
	switch mode {
	case modeGemm: // out(m,n) = a(m,k) · b(k,n)
		m, k, n := d[0], d[1], d[2]
		// 大矩阵按行切分到多个 goroutine(每行只由一个 worker 写,无数据竞争)。
		// 纯 Go 无 SIMD 内建,并行是这里最有效的一档加速。
		if int64(m)*int64(k)*int64(n) >= 1<<21 { // ≈ 2M MAC 以上才值得并行
			workers := runtime.GOMAXPROCS(0)
			if workers > m {
				workers = m
			}
			if workers > 1 {
				var wg sync.WaitGroup
				chunk := (m + workers - 1) / workers
				for w := 0; w < workers; w++ {
					lo := w * chunk
					hi := lo + chunk
					if hi > m {
						hi = m
					}
					if lo >= hi {
						break
					}
					wg.Add(1)
					go func(i0, i1 int) {
						defer wg.Done()
						gemmRows(out, a, b, i0, i1, k, n)
					}(lo, hi)
				}
				wg.Wait()
				return
			}
		}
		gemmRows(out, a, b, 0, m, k, n)
	case modeDot: // out(p,q,n) = Σ_k a(p,k)·b(q,k,n)
		p, q, k, n := d[0], d[1], d[2], d[3]
		for pi := 0; pi < p; pi++ {
			for qi := 0; qi < q; qi++ {
				obase := (pi*q + qi) * n
				abas := pi * k
				bbase := qi * k * n
				for kk := 0; kk < k; kk++ {
					av := a[abas+kk]
					brow := b[bbase+kk*n : bbase+(kk+1)*n]
					for j := 0; j < n; j++ {
						out[obase+j] += av * brow[j]
					}
				}
			}
		}
	case modeInner: // out(p,q) = Σ_k a(p,k)·b(q,k)
		p, q, k := d[0], d[1], d[2]
		for pi := 0; pi < p; pi++ {
			for qi := 0; qi < q; qi++ {
				var s T
				ab, bb := pi*k, qi*k
				for kk := 0; kk < k; kk++ {
					s += a[ab+kk] * b[bb+kk]
				}
				out[pi*q+qi] = s
			}
		}
	case modeOuter: // out(m,n) = a(m)·b(n)
		m, n := d[0], d[1]
		for i := 0; i < m; i++ {
			av := a[i]
			row := out[i*n : (i+1)*n]
			for j := 0; j < n; j++ {
				row[j] = av * b[j]
			}
		}
	}
}

// gemmRows 计算 out[i0:i1, :] = a[i0:i1, :] · b(供并行 worker 调用;4 行展开)。
func gemmRows[T castableNum](out, a, b []T, i0, i1, k, n int) {
	i := i0
	for ; i+3 < i1; i += 4 {
		r0 := out[i*n : (i+1)*n]
		r1 := out[(i+1)*n : (i+2)*n]
		r2 := out[(i+2)*n : (i+3)*n]
		r3 := out[(i+3)*n : (i+4)*n]
		a0 := a[i*k : (i+1)*k]
		a1 := a[(i+1)*k : (i+2)*k]
		a2 := a[(i+2)*k : (i+3)*k]
		a3 := a[(i+3)*k : (i+4)*k]
		for p := 0; p < k; p++ {
			v0, v1, v2, v3 := a0[p], a1[p], a2[p], a3[p]
			bp := b[p*n : (p+1)*n]
			for j := 0; j < n; j++ {
				bv := bp[j]
				r0[j] += v0 * bv
				r1[j] += v1 * bv
				r2[j] += v2 * bv
				r3[j] += v3 * bv
			}
		}
	}
	for ; i < i1; i++ {
		row := out[i*n : (i+1)*n]
		ai := a[i*k : (i+1)*k]
		for p := 0; p < k; p++ {
			av := ai[p]
			bp := b[p*n : (p+1)*n]
			for j := 0; j < n; j++ {
				row[j] += av * bp[j]
			}
		}
	}
}

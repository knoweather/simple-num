package num

import "fmt"

// matmulComplex 复数矩阵乘法:与 MatMul 同语义(1-D 提升、batch 广播),
// 只是内核换成复数运算(numpy 的 matmul 支持复数是常见用法,这里补齐)。
func matmulComplex(a, b *Array) (*Array, error) {
	dt, err := promoteTypes(a.Dtype(), b.Dtype())
	if err != nil {
		return nil, fmt.Errorf("num: matmul: dtype 不可提升: %w", err)
	}
	ac, err := transformCopy(a.AsContiguous(), a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	bc, err := transformCopy(b.AsContiguous(), b.Shape(), dt)
	if err != nil {
		return nil, err
	}
	as, bs := ac.Shape(), bc.Shape()
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
	batch, err := broadcastShapes(as[:len(as)-2], bs[:len(bs)-2])
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
	aw := win[complex128](engineWindow(ac))
	bw := win[complex128](engineWindow(bc))
	ow := win[complex128](engineWindow(out))
	aBatch := as[:len(as)-2]
	bBatch := bs[:len(bs)-2]
	aStride := m * k
	bStride := k * n
	oStride := m * n
	for bi := 0; bi < nb; bi++ {
		rem := bi
		aOff, bOff := 0, 0
		aMul, bMul := 1, 1
		for d := len(batch) - 1; d >= 0; d-- {
			pos := 0
			if batch[d] > 0 {
				pos = rem % batch[d]
				rem /= batch[d]
			}
			aDim, bDim := 1, 1
			if d >= len(batch)-len(aBatch) {
				aDim = aBatch[d-(len(batch)-len(aBatch))]
			}
			if d >= len(batch)-len(bBatch) {
				bDim = bBatch[d-(len(batch)-len(bBatch))]
			}
			aIdx, bIdx := 0, 0
			if aDim > 1 {
				aIdx = pos
			}
			if bDim > 1 {
				bIdx = pos
			}
			aOff += aIdx * aMul
			bOff += bIdx * bMul
			aMul *= aDim
			bMul *= bDim
		}
		abase := aOff * aStride
		bbase := bOff * bStride
		obase := bi * oStride
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s complex128
				for t := 0; t < k; t++ {
					s += aw[abase+i*k+t] * bw[bbase+t*n+j]
				}
				ow[obase+i*n+j] = s
			}
		}
	}
	return squeezeMatMul(out, aWas1, bWas1)
}

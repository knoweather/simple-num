package num

import (
	"fmt"
	"sort"
	"strings"
)

// —— 线代长尾(Step 9):einsum / tensordot / unique(多输出) ——

// Tensordot ≡ np.tensordot(a, b, axes=(axesA, axesB)):把 a 的 axesA 与 b 的 axesB
// 对应收缩,结果形状 = a 去掉 axesA 的轴 + b 去掉 axesB 的轴。
//
// 实现:先转置成 (M,K) 与 (K,N) 再走矩阵乘,复用既有引擎与 dtype 提升。
func Tensordot(a, b *Array, axesA, axesB []int) (*Array, error) {
	if len(axesA) != len(axesB) {
		return nil, fmt.Errorf("num: tensordot: 收缩轴数 %d ≠ %d: %w", len(axesA), len(axesB), ErrShape)
	}
	na := make([]int, len(axesA))
	for i, ax := range axesA {
		d, err := normalizeAxis(a, ax)
		if err != nil {
			return nil, fmt.Errorf("num: tensordot: a 轴: %w", err)
		}
		na[i] = d
	}
	nb := make([]int, len(axesB))
	for i, ax := range axesB {
		d, err := normalizeAxis(b, ax)
		if err != nil {
			return nil, fmt.Errorf("num: tensordot: b 轴: %w", err)
		}
		nb[i] = d
	}
	for i := range na {
		if a.Shape()[na[i]] != b.Shape()[nb[i]] {
			return nil, fmt.Errorf("num: tensordot: 收缩维长度 %d ≠ %d: %w",
				a.Shape()[na[i]], b.Shape()[nb[i]], ErrShape)
		}
	}
	aFree, bFree := notAxes(a.NDim(), na), notAxes(b.NDim(), nb)
	aPerm := append(append([]int{}, aFree...), na...)
	bPerm := append(append([]int{}, nb...), bFree...)
	at, err := a.Transpose(aPerm...)
	if err != nil {
		return nil, err
	}
	bt, err := b.Transpose(bPerm...)
	if err != nil {
		return nil, err
	}
	m, k, n := 1, 1, 1
	for _, ax := range aFree {
		m *= a.Shape()[ax]
	}
	for _, ax := range na {
		k *= a.Shape()[ax]
	}
	for _, ax := range bFree {
		n *= b.Shape()[ax]
	}
	am, err := at.AsContiguous().Reshape([]int{m, k})
	if err != nil {
		return nil, err
	}
	bm, err := bt.AsContiguous().Reshape([]int{k, n})
	if err != nil {
		return nil, err
	}
	prod, err := MatMul(am, bm)
	if err != nil {
		return nil, err
	}
	outSh := make([]int, 0, len(aFree)+len(bFree))
	for _, ax := range aFree {
		outSh = append(outSh, a.Shape()[ax])
	}
	for _, ax := range bFree {
		outSh = append(outSh, b.Shape()[ax])
	}
	return prod.Reshape(outSh)
}

// notAxes 返回 0..nd-1 中不在 drop 里的轴(升序)。
func notAxes(nd int, drop []int) []int {
	dropped := make([]bool, nd)
	for _, d := range drop {
		if d >= 0 && d < nd {
			dropped[d] = true
		}
	}
	out := make([]int, 0, nd-len(drop))
	for i := 0; i < nd; i++ {
		if !dropped[i] {
			out = append(out, i)
		}
	}
	return out
}

// Einsum ≡ np.einsum(subscripts, operands...)。
//
// 支持:逗号分隔的输入项、可选 "-> 输出"、隐式输出(只出现一次的标签按字母序)、
// 同一项内重复标签(取对角);不支持 "..."(省略号)。
//
// 实现:把所有标签的下标组合做**完全枚举**并按 numpy 语义累加 —— 重复标签取对角、
// 隐式输出按字母序,都能自然表达。规模是"标签尺寸之积",适合中小张量;
// 大张量需要专门的收缩图与分块优化(留作后续)。
func Einsum(subscripts string, operands ...*Array) (*Array, error) {
	inputs, output, err := parseEinsum(subscripts)
	if err != nil {
		return nil, err
	}
	if len(inputs) != len(operands) {
		return nil, fmt.Errorf("num: einsum: 输入项 %d 个 ≠ 操作数 %d 个: %w",
			len(inputs), len(operands), ErrShape)
	}
	sizes := map[rune]int{}
	for i, term := range inputs {
		if len(term) != operands[i].NDim() {
			return nil, fmt.Errorf("num: einsum: 第 %d 项标签 %q 与 ndim %d 不符: %w",
				i, term, operands[i].NDim(), ErrShape)
		}
		for d, lb := range term {
			n := operands[i].Shape()[d]
			if prev, ok := sizes[lb]; ok {
				if prev != n {
					return nil, fmt.Errorf("num: einsum: 标签 %q 尺寸冲突 %d≠%d: %w", lb, prev, n, ErrShape)
				}
			} else {
				sizes[lb] = n
			}
		}
	}
	if output == nil {
		cnt := map[rune]int{}
		for _, term := range inputs {
			for _, lb := range term {
				cnt[lb]++
			}
		}
		var lbls []rune
		for lb, c := range cnt {
			if c == 1 {
				lbls = append(lbls, lb)
			}
		}
		sort.Slice(lbls, func(i, j int) bool { return lbls[i] < lbls[j] })
		output = lbls
	}
	for _, lb := range output {
		if _, ok := sizes[lb]; !ok {
			return nil, fmt.Errorf("num: einsum: 输出标签 %q 未出现在输入里: %w", lb, ErrDtype)
		}
	}
	dt := operands[0].Dtype()
	for _, a := range operands[1:] {
		if dt, err = promoteTypes(dt, a.Dtype()); err != nil {
			return nil, fmt.Errorf("num: einsum: dtype 不可提升: %w", err)
		}
	}
	// 全部操作数先归一到结果 dtype(整型保持整型累加,避免精度损失)
	cops := make([]*Array, len(operands))
	for i, a := range operands {
		c, err := transformCopy(a.AsContiguous(), a.Shape(), dt)
		if err != nil {
			return nil, err
		}
		cops[i] = c
	}
	outSh := make([]int, len(output))
	for i, lb := range output {
		outSh[i] = sizes[lb]
	}
	// 快路径:多操作数且无重复标签 → 贪心配对收缩(复杂度从 ∏标签尺寸 降到
	// 若干次 MatMul)。含重复标签(取对角)或单操作数时走下面的枚举实现。
	if len(cops) >= 2 && !hasDupLabels(inputs) {
		return einsumFast(inputs, output, cops, sizes, dt)
	}
	out, err := New(outSh, dt)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	return out, einsumRun(out, cops, inputs, output, sizes, dt)
}

// einsumRun 完全枚举标签组合并累加。
func einsumRun(out *Array, ops []*Array, inputs [][]rune, output []rune,
	sizes map[rune]int, dt Dtype,
) error {
	// 全局标签顺序:输出标签在前,其余(收缩标签)在后
	allLabels := append([]rune{}, output...)
	seen := map[rune]bool{}
	for _, lb := range output {
		seen[lb] = true
	}
	var rest []rune
	for lb := range sizes {
		if !seen[lb] {
			rest = append(rest, lb)
		}
	}
	sort.Slice(rest, func(i, j int) bool { return rest[i] < rest[j] })
	allLabels = append(allLabels, rest...)
	ns := make([]int, len(allLabels))
	for i, lb := range allLabels {
		ns[i] = sizes[lb]
	}
	// 标签 → 全局下标位置
	gpos := make([]int, 0, len(allLabels))
	for i, lb := range allLabels {
		_ = lb
		gpos = append(gpos, i)
	}
	// 每个操作数:各维对应的全局标签位置 + 元素步长
	labIdx := make([][]int, len(ops))
	steps := make([][]int, len(ops))
	for i, term := range inputs {
		labIdx[i] = make([]int, len(term))
		steps[i] = make([]int, len(term))
		acc := 1
		for d := len(term) - 1; d >= 0; d-- {
			steps[i][d] = acc
			acc *= ops[i].Shape()[d]
		}
		for d, lb := range term {
			labIdx[i][d] = tagPos(allLabels, lb)
		}
	}
	total := 1
	for _, n := range ns {
		total *= n
	}
	idx := make([]int, len(allLabels))
	lin := make([]int, len(ops))
	outW := engineWindow(out)
	// 输出平坦步长
	outSteps := make([]int, len(output))
	acc := 1
	for d := len(output) - 1; d >= 0; d-- {
		outSteps[d] = acc
		acc *= ns[d]
	}
	for k := 0; k < total; k++ {
		rem := k
		for d := len(ns) - 1; d >= 0; d-- {
			idx[d] = rem % ns[d]
			rem /= ns[d]
		}
		for i := range ops {
			p := 0
			for d := range inputs[i] {
				p += idx[labIdx[i][d]] * steps[i][d]
			}
			lin[i] = p
		}
		oi := 0
		for d := range output {
			oi += idx[d] * outSteps[d]
		}
		if err := einsumAccum(outW, out.Dtype(), ops, lin, oi); err != nil {
			return err
		}
	}
	return nil
}

// tagPos 返回标签在全局标签表中的位置。
func tagPos(all []rune, lb rune) int {
	for i, x := range all {
		if x == lb {
			return i
		}
	}
	return 0
}

// einsumAccum 把一个乘积项累加到输出位置(按 dtype 类型化,整型保持不变)。
func einsumAccum(outW []byte, dt Dtype, ops []*Array, lin []int, oi int) error {
	switch dt.Kind {
	case Float64:
		p := 1.0
		for i := range ops {
			p *= win[float64](engineWindow(ops[i]))[lin[i]]
		}
		win[float64](outW)[oi] += p
	case Float32:
		p := float32(1)
		for i := range ops {
			p *= win[float32](engineWindow(ops[i]))[lin[i]]
		}
		win[float32](outW)[oi] += p
	case Int64:
		p := int64(1)
		for i := range ops {
			p *= win[int64](engineWindow(ops[i]))[lin[i]]
		}
		win[int64](outW)[oi] += p
	case Int32:
		p := int32(1)
		for i := range ops {
			p *= win[int32](engineWindow(ops[i]))[lin[i]]
		}
		win[int32](outW)[oi] += p
	case Int16:
		p := int16(1)
		for i := range ops {
			p *= win[int16](engineWindow(ops[i]))[lin[i]]
		}
		win[int16](outW)[oi] += p
	case Int8:
		p := int8(1)
		for i := range ops {
			p *= win[int8](engineWindow(ops[i]))[lin[i]]
		}
		win[int8](outW)[oi] += p
	case Uint64:
		p := uint64(1)
		for i := range ops {
			p *= win[uint64](engineWindow(ops[i]))[lin[i]]
		}
		win[uint64](outW)[oi] += p
	case Uint32:
		p := uint32(1)
		for i := range ops {
			p *= win[uint32](engineWindow(ops[i]))[lin[i]]
		}
		win[uint32](outW)[oi] += p
	case Uint16:
		p := uint16(1)
		for i := range ops {
			p *= win[uint16](engineWindow(ops[i]))[lin[i]]
		}
		win[uint16](outW)[oi] += p
	case Uint8:
		p := uint8(1)
		for i := range ops {
			p *= win[uint8](engineWindow(ops[i]))[lin[i]]
		}
		win[uint8](outW)[oi] += p
	default:
		return fmt.Errorf("num: einsum 无 %s 内核: %w", dt, ErrDtype)
	}
	return nil
}

// parseEinsum 解析 "ij,jk->ik" 形式。
func parseEinsum(s string) ([][]rune, []rune, error) {
	s = strings.ReplaceAll(s, " ", "")
	parts := strings.SplitN(s, "->", 2)
	inputs, err := parseEinsumTerms(parts[0])
	if err != nil {
		return nil, nil, err
	}
	if len(parts) == 1 {
		return inputs, nil, nil
	}
	out, err := parseEinsumTerms(parts[1])
	if err != nil {
		return nil, nil, err
	}
	if len(out) != 1 {
		return nil, nil, fmt.Errorf("num: einsum: 输出项应恰为一个: %w", ErrShape)
	}
	return inputs, out[0], nil
}

func parseEinsumTerms(s string) ([][]rune, error) {
	terms := strings.Split(s, ",")
	out := make([][]rune, len(terms))
	for i, t := range terms {
		if strings.Contains(t, "...") {
			return nil, fmt.Errorf("num: einsum: 暂不支持省略号: %w", ErrValue)
		}
		r := []rune(t)
		for _, lb := range r {
			if !isEinsumLabel(lb) {
				return nil, fmt.Errorf("num: einsum: 非法标签 %q: %w", lb, ErrValue)
			}
		}
		out[i] = r
	}
	return out, nil
}

func isEinsumLabel(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

// UniqueIdx ≡ np.unique(a, return_index=True, return_counts=True):
// 返回 (values, firstIndex, counts) 三个同长 1-D 数组。
func UniqueIdx(a *Array) ([]*Array, error) {
	flat := a.AsContiguous().Flatten()
	n := flat.Size()
	order, err := Argsort(flat, nil)
	if err != nil {
		return nil, err
	}
	ow := win[int64](engineWindow(order))
	es := a.Dtype().Itemsize
	buf := engineWindow(flat)
	type group struct {
		val   int
		first int
		count int
	}
	groups := make([]group, 0, n)
	for i := 0; i < n; i++ {
		k := int(ow[i])
		if i == 0 || !sameElemBytes(buf[k*es:(k+1)*es], buf[int(ow[i-1])*es:(int(ow[i-1])+1)*es]) {
			groups = append(groups, group{val: k, first: k, count: 1})
			continue
		}
		g := &groups[len(groups)-1]
		g.count++
		if k < g.first {
			g.first = k
		}
	}
	vals, err := New([]int{len(groups)}, a.Dtype())
	if err != nil {
		return nil, err
	}
	idxs, err := New([]int{len(groups)}, DInt64)
	if err != nil {
		return nil, err
	}
	cnts, err := New([]int{len(groups)}, DInt64)
	if err != nil {
		return nil, err
	}
	vb := engineWindow(vals)
	iw := win[int64](engineWindow(idxs))
	cw := win[int64](engineWindow(cnts))
	for i, g := range groups {
		copy(vb[i*es:(i+1)*es], buf[g.val*es:(g.val+1)*es])
		iw[i] = int64(g.first)
		cw[i] = int64(g.count)
	}
	return []*Array{vals, idxs, cnts}, nil
}

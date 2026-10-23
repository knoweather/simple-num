package num

import (
	"fmt"
	"sort"
)

// —— einsum 收缩图(Step 22):贪心配对收缩,把链式张量收缩落到 MatMul 上 ——
//
// 原实现对"所有标签的组合"做完全枚举,复杂度 ∏(各标签尺寸),对矩阵乘式
// (如 "ij,jk->ik")是我方 MatMul 的数百倍。这里改为:
// 1. 每轮挑一对操作数,按"收缩后中间结果最小"的贪心策略合并;
// 2. 合并用 Tensordot 的做法(转置成 (mA,k)·(k,nB) 后走 MatMul);
// 3. 最后把标签排列到输出顺序,并对多余标签求和。
// 含**重复标签**(取对角,如 "ii->i")或**单个操作数**时回退到原枚举实现。

// eterm 一个中间项:标签 + 数组。
type eterm struct {
	labels []rune
	arr    *Array
}

// einsumFast 贪心配对收缩。sizes 为标签→尺寸表。
func einsumFast(terms [][]rune, output []rune, ops []*Array, sizes map[rune]int, dt Dtype) (*Array, error) {
	cur := make([]eterm, len(ops))
	for i := range ops {
		cur[i] = eterm{labels: append([]rune(nil), terms[i]...), arr: ops[i]}
	}
	for len(cur) > 1 {
		bi, bj, best := -1, -1, int64(1)<<62
		for i := 0; i < len(cur); i++ {
			for j := i + 1; j < len(cur); j++ {
				sz := contractSize(cur[i], cur[j], sizes)
				if sz < best {
					best, bi, bj = sz, i, j
				}
			}
		}
		merged, err := contractPair(cur[bi], cur[bj], sizes)
		if err != nil {
			return nil, err
		}
		next := make([]eterm, 0, len(cur)-1)
		for k := range cur {
			switch k {
			case bi:
				next = append(next, merged)
			case bj:
				// 已被合并,跳过
			default:
				next = append(next, cur[k])
			}
		}
		cur = next
	}
	return finishEinsum(cur[0], output, dt)
}

// contractSize 两个项合并后的中间结果元素数(A∪B 的标签尺寸之积)。
func contractSize(a, b eterm, sizes map[rune]int) int64 {
	seen := map[rune]bool{}
	sz := int64(1)
	for _, lb := range a.labels {
		if !seen[lb] {
			seen[lb] = true
			sz *= int64(sizes[lb])
		}
	}
	for _, lb := range b.labels {
		if !seen[lb] {
			seen[lb] = true
			sz *= int64(sizes[lb])
		}
	}
	return sz
}

// contractPair 合并两个项:共享标签收缩,其余作为自由轴。
func contractPair(a, b eterm, sizes map[rune]int) (eterm, error) {
	// 0-d 标量项:直接逐元素乘
	if len(a.labels) == 0 {
		r, err := Multiply(b.arr, a.arr)
		if err != nil {
			return eterm{}, err
		}
		return eterm{labels: b.labels, arr: r}, nil
	}
	if len(b.labels) == 0 {
		r, err := Multiply(a.arr, b.arr)
		if err != nil {
			return eterm{}, err
		}
		return eterm{labels: a.labels, arr: r}, nil
	}
	// 共享标签(A 中的出现顺序)
	var shared []rune
	inB := map[rune]bool{}
	for _, lb := range b.labels {
		inB[lb] = true
	}
	for _, lb := range a.labels {
		if inB[lb] {
			shared = append(shared, lb)
		}
	}
	var freeA, freeB []rune
	shSet := map[rune]bool{}
	for _, lb := range shared {
		shSet[lb] = true
	}
	for _, lb := range a.labels {
		if !shSet[lb] {
			freeA = append(freeA, lb)
		}
	}
	for _, lb := range b.labels {
		if !shSet[lb] {
			freeB = append(freeB, lb)
		}
	}
	// 排列:自由轴在前、共享轴在后
	permA := append([]int{}, indexOfLabels(a.labels, freeA)...)
	permA = append(permA, indexOfLabels(a.labels, shared)...)
	permB := append([]int{}, indexOfLabels(b.labels, shared)...)
	permB = append(permB, indexOfLabels(b.labels, freeB)...)
	at, err := a.arr.Transpose(permA...)
	if err != nil {
		return eterm{}, err
	}
	bt, err := b.arr.Transpose(permB...)
	if err != nil {
		return eterm{}, err
	}
	mA := 1
	for _, lb := range freeA {
		mA *= sizes[lb]
	}
	k := 1
	for _, lb := range shared {
		k *= sizes[lb]
	}
	nB := 1
	for _, lb := range freeB {
		nB *= sizes[lb]
	}
	am, err := at.AsContiguous().Reshape([]int{mA, k})
	if err != nil {
		return eterm{}, err
	}
	bm, err := bt.AsContiguous().Reshape([]int{k, nB})
	if err != nil {
		return eterm{}, err
	}
	pr, err := MatMul(am, bm)
	if err != nil {
		return eterm{}, err
	}
	newLabels := append(append([]rune{}, freeA...), freeB...)
	newShape := make([]int, len(newLabels))
	for i, lb := range newLabels {
		newShape[i] = sizes[lb]
	}
	out, err := pr.Reshape(newShape)
	if err != nil {
		return eterm{}, err
	}
	return eterm{labels: newLabels, arr: out}, nil
}

// indexOfLabels 返回 want 中每个标签在 labels 里的下标。
func indexOfLabels(labels, want []rune) []int {
	pos := make([]int, 0, len(want))
	for _, w := range want {
		for i, lb := range labels {
			if lb == w {
				pos = append(pos, i)
				break
			}
		}
	}
	return pos
}

// finishEinsum 把最终项排列到输出标签顺序,并对多余标签求和。
func finishEinsum(last eterm, output []rune, dt Dtype) (*Array, error) {
	// 输出标签在 last.labels 中的下标
	perm := make([]int, 0, len(last.labels))
	outSet := map[rune]bool{}
	for _, lb := range output {
		outSet[lb] = true
		found := -1
		for i, x := range last.labels {
			if x == lb {
				found = i
				break
			}
		}
		if found < 0 {
			return nil, fmt.Errorf("num: einsum: 输出标签 %q 在收缩结果里不存在: %w", lb, ErrValue)
		}
		perm = append(perm, found)
	}
	var extra []int
	for i, lb := range last.labels {
		if !outSet[lb] {
			extra = append(extra, i)
		}
	}
	perm = append(perm, extra...)
	arr := last.arr
	if len(perm) != len(last.labels) || !isIdentityPerm(perm) {
		t, err := arr.Transpose(perm...)
		if err != nil {
			return nil, err
		}
		arr = t.AsContiguous()
	}
	if len(extra) == 0 {
		return maybeCastTo(arr, dt)
	}
	// 对多余标签求和:reshape 成 (prodOut, prodExtra) 后沿轴 1 求和
	outN := 1
	for _, lb := range output {
		outN *= arr.Shape()[indexInPerm(perm, lb, last.labels)]
	}
	// 更稳妥:直接用输出标签的形状
	outShape := make([]int, len(output))
	permShape := arr.Shape()
	for i := range output {
		outShape[i] = permShape[i] // 排列后输出轴在前
	}
	rest := 1
	for i := len(output); i < len(permShape); i++ {
		rest *= permShape[i]
	}
	_ = outN
	flat, err := arr.Reshape([]int{prodInts(outShape), rest})
	if err != nil {
		return nil, err
	}
	summed, err := Sum(flat, []int{1}, false)
	if err != nil {
		return nil, err
	}
	final, err := summed.Reshape(outShape)
	if err != nil {
		return nil, err
	}
	return maybeCastTo(final, dt)
}

// indexInPerm 返回标签在 lastLabels 中经 perm 排列后的位置。
func indexInPerm(perm []int, lb rune, lastLabels []rune) int {
	for i, p := range perm {
		if lastLabels[p] == lb {
			return i
		}
	}
	return 0
}

func isIdentityPerm(p []int) bool {
	for i, v := range p {
		if i != v {
			return false
		}
	}
	return true
}

func prodInts(s []int) int {
	n := 1
	for _, v := range s {
		n *= v
	}
	return n
}

// maybeCastTo 把结果落回目标 dtype。
func maybeCastTo(a *Array, dt Dtype) (*Array, error) {
	if a.Dtype() == dt {
		return a, nil
	}
	return Cast(a, dt)
}

// hasDupLabels 判断是否有重复标签(需要走枚举实现取对角)。
func hasDupLabels(terms [][]rune) bool {
	for _, t := range terms {
		seen := map[rune]bool{}
		for _, lb := range t {
			if seen[lb] {
				return true
			}
			seen[lb] = true
		}
	}
	return false
}

var _ = sort.Ints

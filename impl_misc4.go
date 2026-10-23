package num

import (
	"fmt"
	"math"
	"sort"
)

// —— 形状/索引/数学小工具(Step 36) ——

// orderable 可比较的数值类型(分区/选择用)。
type orderable interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~float32 | ~float64
}

// isNaNT 判断值是否为 NaN(非浮点恒 false)。
func isNaNT[T orderable](v T) bool {
	switch x := any(v).(type) {
	case float32:
		return x != x
	case float64:
		return x != x
	default:
		return false
	}
}

// lessT 比较:NaN 视为最大(numpy 的 partition 把 NaN 排最后)。
func lessT[T orderable](x, y T) bool {
	if isNaNT(x) {
		return false
	}
	if isNaNT(y) {
		return true
	}
	return x < y
}

// Partition ≡ np.partition(a, kth, axis):返回一份拷贝,使第 kth 个元素处于排序位置,
// 其左不大于、其右不小于它(NaN 排最后)。两侧内部顺序不作保证 —— 与 numpy 一致。
func Partition(a *Array, kth int, axis int) (*Array, error) {
	out, err := partitionCopy(a, axis)
	if err != nil {
		return nil, err
	}
	if err := selectRows(out, kth, false); err != nil {
		return nil, err
	}
	return out, nil
}

// Argpartition ≡ np.argpartition(a, kth, axis):返回分区后的下标(int64)。
func Argpartition(a *Array, kth int, axis int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: argpartition: 数组为 nil: %w", ErrDtype)
	}
	d := 0
	if a.NDim() > 1 {
		var err error
		if d, err = normalizeAxis(a, axis); err != nil {
			return nil, fmt.Errorf("num: argpartition: %w", err)
		}
	}
	rows, m, n, err := axisRows(a.AsContiguous(), d)
	if err != nil {
		return nil, err
	}
	out, err := New(rows.Shape(), DInt64)
	if err != nil {
		return nil, err
	}
	if err := selectRowsInto(rows, out, m, n, kth); err != nil {
		return nil, err
	}
	if a.NDim() > 1 {
		return out, nil
	}
	return out, nil
}

// partitionCopy 复制成"按轴分行可分区"的形态。
func partitionCopy(a *Array, axis int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: partition: 数组为 nil: %w", ErrDtype)
	}
	// 必须真复制:transformCopy 在同 shape/dtype 时返回原缓冲,原地分区会污染入参
	return Copy(a.AsContiguous()), nil
}

// selectRows 对副本的每一行做 quickselect(kth 处就位)。
func selectRows(arr *Array, kth int, _ bool) error {
	if arr.NDim() == 0 {
		if kth != 0 && kth != -1 {
			return fmt.Errorf("num: partition: kth %d 越界(0-d): %w", kth, ErrIndex)
		}
		return nil
	}
	d := 0
	rows, m, n, err := axisRows(arr, d)
	if err != nil {
		return err
	}
	if kth < 0 {
		kth += n
	}
	if kth < 0 || kth >= n {
		return fmt.Errorf("num: partition: kth %d 越界(轴长 %d): %w", kth, n, ErrIndex)
	}
	return applySelect(rows, m, n, kth, nil)
}

// selectRowsInto 与 selectRows 相同,但把下标写进 dst(int64)。
func selectRowsInto(rows, dst *Array, m, n, kth int) error {
	if kth < 0 {
		kth += n
	}
	if n == 0 {
		return nil
	}
	if kth < 0 || kth >= n {
		return fmt.Errorf("num: argpartition: kth %d 越界(轴长 %d): %w", kth, n, ErrIndex)
	}
	return applySelect(rows, m, n, kth, dst)
}

// applySelect 按 dtype 分发到泛型 quickselect;dst 非 nil 时写下标。
func applySelect(rows *Array, m, n, kth int, dst *Array) error {
	dt := rows.Dtype()
	src := rows.AsContiguous()
	sb := engineWindow(src)
	es := dt.Itemsize
	idx := make([]int, n)
	var db []byte
	if dst != nil {
		db = engineWindow(dst)
	}
	writeBack := func(r int) {
		if dst != nil {
			for i := 0; i < n; i++ {
				pos := r*n + i
				putInt64(db[pos*8:], int64(idx[i]))
			}
		}
	}
	// 行快照:原地交换会把后续要读的元素覆盖掉
	row := make([]byte, n*es)
	writePermuted := func(r int) {
		copy(row, sb[r*n*es:(r+1)*n*es])
		for i := 0; i < n; i++ {
			copy(sb[(r*n+i)*es:], row[idx[i]*es:(idx[i]+1)*es])
		}
	}
	run := func(read func(pos int) float64, set func(pos, from int)) error {
		for r := 0; r < m; r++ {
			for i := 0; i < n; i++ {
				idx[i] = i
			}
			vals := make([]float64, n)
			for i := 0; i < n; i++ {
				vals[i] = read(r*n + i)
			}
			// 用 float64 键做选择;整数在 2^53 内精确,超出时退化风险已在文档注明
			quickselectF64(vals, idx, kth)
			writeBack(r)
			if dst == nil {
				writePermuted(r)
			}
		}
		return nil
	}
	// 比较键统一转 f64(整数在 2^53 内精确;分区本身按字节搬运,不改变 dtype)
	keys, err := transformCopy(src, src.Shape(), DFloat64)
	if err != nil {
		return fmt.Errorf("num: partition: 该 dtype 不支持分区(%s): %w", dt, ErrDtype)
	}
	kw := win[float64](engineWindow(keys))
	readF64 := func(pos int) float64 {
		return kw[pos]
	}
	setFrom := func(pos, from int) {
		copy(sb[pos*es:(pos+1)*es], sb[from*es:(from+1)*es])
	}
	if err := run(readF64, setFrom); err != nil {
		return err
	}
	return nil
}

// quickselectF64 用 float64 键在 idx 上做选择:返回后 idx[k] 是第 k 小,
// 其左侧不大于、右侧不小于它(NaN 视为最大,排最后)。
func quickselectF64(vals []float64, idx []int, k int) {
	less := func(x, y float64) bool {
		if x != x {
			return false
		}
		if y != y {
			return true
		}
		return x < y
	}
	lo, hi := 0, len(idx)-1
	for lo < hi {
		// 三数取中,把枢轴换到 hi,避免有序输入退化
		mid := lo + (hi-lo)/2
		if less(vals[idx[mid]], vals[idx[lo]]) {
			idx[lo], idx[mid] = idx[mid], idx[lo]
		}
		if less(vals[idx[hi]], vals[idx[lo]]) {
			idx[lo], idx[hi] = idx[hi], idx[lo]
		}
		if less(vals[idx[hi]], vals[idx[mid]]) {
			idx[mid], idx[hi] = idx[hi], idx[mid]
		}
		idx[mid], idx[hi] = idx[hi], idx[mid]
		// Lomuto 分区
		p := lo
		for j := lo; j < hi; j++ {
			if less(vals[idx[j]], vals[idx[hi]]) {
				idx[p], idx[j] = idx[j], idx[p]
				p++
			}
		}
		idx[p], idx[hi] = idx[hi], idx[p]
		if p == k {
			return
		}
		if p > k {
			hi = p - 1
		} else {
			lo = p + 1
		}
	}
}

// Moveaxis ≡ np.moveaxis(a, source, destination):把若干轴搬到新位置(返回视图)。
func Moveaxis(a *Array, source, destination []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: moveaxis: 数组为 nil: %w", ErrDtype)
	}
	nd := a.NDim()
	if len(source) != len(destination) {
		return nil, fmt.Errorf("num: moveaxis: source/destination 长度不同(%d/%d): %w",
			len(source), len(destination), ErrShape)
	}
	src, err := normalizeAxesAll(a, source)
	if err != nil {
		return nil, fmt.Errorf("num: moveaxis: %w", err)
	}
	dst, err := normalizeAxesAll(a, destination)
	if err != nil {
		return nil, fmt.Errorf("num: moveaxis: %w", err)
	}
	inSrc := make(map[int]bool, len(src))
	for _, s := range src {
		if inSrc[s] {
			return nil, fmt.Errorf("num: moveaxis: source 含重复轴 %d: %w", s, ErrShape)
		}
		inSrc[s] = true
	}
	seenDst := make(map[int]bool, len(dst))
	for _, d := range dst {
		if seenDst[d] {
			return nil, fmt.Errorf("num: moveaxis: destination 含重复轴 %d: %w", d, ErrShape)
		}
		seenDst[d] = true
	}
	// 与 numpy 同序:先取"非 source 轴"(保持原序),再按 destination 升序逐对插回
	order := make([]int, 0, nd)
	for ax := 0; ax < nd; ax++ {
		if !inSrc[ax] {
			order = append(order, ax)
		}
	}
	type pair struct{ d, s int }
	pairs := make([]pair, len(src))
	for i := range src {
		pairs[i] = pair{d: dst[i], s: src[i]}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].d < pairs[j].d })
	for _, pp := range pairs {
		at := pp.d
		if at > len(order) {
			at = len(order)
		}
		order = append(order, 0)
		copy(order[at+1:], order[at:])
		order[at] = pp.s
	}
	return a.Transpose(order...)
}

// normalizeAxesAll 允许负数轴并检查越界。
func normalizeAxesAll(a *Array, axes []int) ([]int, error) {
	out := make([]int, len(axes))
	for i, ax := range axes {
		v, err := normalizeAxis(a, ax)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// Resize ≡ np.resize(a, shape):按展平后的元素循环填充到新形状。
func Resize(a *Array, shape []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: resize: 数组为 nil: %w", ErrDtype)
	}
	shape, n, err := normShapeCopy(shape)
	if err != nil {
		return nil, err
	}
	flat := a.AsContiguous().Flatten()
	if flat.Size() == 0 {
		if n == 0 {
			return New(shape, a.Dtype())
		}
		return nil, fmt.Errorf("num: resize: 空数组不能放大到 %v: %w", shape, ErrShape)
	}
	out, err := New(shape, a.Dtype())
	if err != nil {
		return nil, err
	}
	es := a.Dtype().Itemsize
	sb := engineWindow(flat)
	ob := engineWindow(out)
	src := flat.Size()
	for i := 0; i < n; i++ {
		j := i % src
		copy(ob[i*es:(i+1)*es], sb[j*es:(j+1)*es])
	}
	return out, nil
}

// BroadcastArrays ≡ np.broadcast_arrays:把多个数组广播到同一形状(返回视图)。
func BroadcastArrays(arrays ...*Array) ([]*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("num: broadcast_arrays: 至少一个数组: %w", ErrShape)
	}
	shape := arrays[0].Shape()
	for _, a := range arrays[1:] {
		sh, err := broadcastShapes(shape, a.Shape())
		if err != nil {
			return nil, fmt.Errorf("num: broadcast_arrays: %w", err)
		}
		shape = sh
	}
	out := make([]*Array, len(arrays))
	for i, a := range arrays {
		v, err := a.BroadcastTo(shape)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// Choose ≡ np.choose(idx, choices):结果的第 i 个元素取 choices[idx[i]][i]。
func Choose(idx *Array, choices []*Array) (*Array, error) {
	if idx == nil || len(choices) == 0 {
		return nil, fmt.Errorf("num: choose: 需要索引与至少一个候选: %w", ErrShape)
	}
	if !isIntKind(idx.Dtype().Kind) {
		return nil, fmt.Errorf("num: choose: 索引须为整型(得 %s): %w", idx.Dtype(), ErrDtype)
	}
	all := append([]*Array{idx}, choices...)
	bc, err := BroadcastArrays(all...)
	if err != nil {
		return nil, err
	}
	bidx := bc[0]
	bch := bc[1:]
	out, err := New(bidx.Shape(), bch[0].Dtype())
	if err != nil {
		return nil, err
	}
	// 先按 dtype 提升所有候选
	dt, err := bch[0].Dtype(), error(nil)
	for _, c := range bch[1:] {
		dt, err = promoteTypes(dt, c.Dtype())
		if err != nil {
			return nil, err
		}
	}
	if dt != bch[0].Dtype() {
		out, err = New(bidx.Shape(), dt)
		if err != nil {
			return nil, err
		}
	}
	es := dt.Itemsize
	ob := engineWindow(out)
	iw := win[int64](engineWindow(bidx.AsContiguous()))
	if bidx.Dtype().Itemsize != 8 {
		return nil, fmt.Errorf("num: choose: 索引须为 64 位整型: %w", ErrDtype)
	}
	cbufs := make([][]byte, len(bch))
	for i, c := range bch {
		f, err := transformCopy(c, c.Shape(), dt)
		if err != nil {
			return nil, err
		}
		cbufs[i] = engineWindow(f)
	}
	n := out.Size()
	for i := 0; i < n; i++ {
		k := iw[i]
		if k < 0 || k >= int64(len(bch)) {
			return nil, fmt.Errorf("num: choose: 索引 %d 越界(候选 %d 个): %w", k, len(bch), ErrIndex)
		}
		copy(ob[i*es:(i+1)*es], cbufs[k][i*es:(i+1)*es])
	}
	return out, nil
}

// TrilIndices ≡ np.tril_indices(n, k):下三角(含对角线)的行/列下标。
func TrilIndices(n, k int) ([]*Array, error) { return triangleIndices(n, k, true) }

// TriuIndices ≡ np.triu_indices(n, k):上三角(含对角线)的行/列下标。
func TriuIndices(n, k int) ([]*Array, error) { return triangleIndices(n, k, false) }

func triangleIndices(n, k int, lower bool) ([]*Array, error) {
	if n < 0 {
		return nil, fmt.Errorf("num: tril/triu_indices: n 须非负(得 %d): %w", n, ErrValue)
	}
	rows := make([]int64, 0, n*n)
	cols := make([]int64, 0, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if lower {
				if j <= i+k {
					rows = append(rows, int64(i))
					cols = append(cols, int64(j))
				}
			} else if j >= i+k {
				rows = append(rows, int64(i))
				cols = append(cols, int64(j))
			}
		}
	}
	r, err := FromSlice(rows)
	if err != nil {
		return nil, err
	}
	c, err := FromSlice(cols)
	if err != nil {
		return nil, err
	}
	return []*Array{r, c}, nil
}

// UnravelIndex ≡ np.unravel_index(indices, shape):展平下标 → 各维下标。
func UnravelIndex(indices *Array, shape []int) ([]*Array, error) {
	if indices == nil {
		return nil, fmt.Errorf("num: unravel_index: 下标为 nil: %w", ErrDtype)
	}
	if len(shape) == 0 {
		return nil, fmt.Errorf("num: unravel_index: shape 为空: %w", ErrShape)
	}
	strides := make([]int64, len(shape))
	acc := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = acc
		acc *= int64(shape[i])
	}
	if !isIntKind(indices.Dtype().Kind) {
		return nil, fmt.Errorf("num: unravel_index: 下标须为整型: %w", ErrDtype)
	}
	flat, err := transformCopy(indices.AsContiguous(), indices.Shape(), DInt64)
	if err != nil {
		return nil, err
	}
	iw := win[int64](engineWindow(flat))
	out := make([]*Array, len(shape))
	for d := range shape {
		buf := make([]int64, len(iw))
		for i, v := range iw {
			if v < 0 || v >= acc {
				return nil, fmt.Errorf("num: unravel_index: 下标 %d 越界(共 %d): %w", v, acc, ErrIndex)
			}
			buf[i] = (v / strides[d]) % int64(shape[d])
		}
		arr, err := FromSlice(buf)
		if err != nil {
			return nil, err
		}
		r, err := arr.Reshape(indices.Shape())
		if err != nil {
			return nil, err
		}
		out[d] = r
	}
	return out, nil
}

// FFTShift ≡ np.fft.fftshift:把各轴的前半段与后半段互换。
func FFTShift(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: fftshift: 数组为 nil: %w", ErrDtype)
	}
	cur := a.AsContiguous()
	for ax := 0; ax < a.NDim(); ax++ {
		sh := cur.Shape()[ax]
		if sh <= 1 {
			continue
		}
		r, err := Roll(cur, sh/2, []int{ax})
		if err != nil {
			return nil, err
		}
		cur = r
	}
	return cur, nil
}

// IFFTShift ≡ np.fft.ifftshift:FFTShift 的逆(奇数长度时不同)。
func IFFTShift(a *Array) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: ifftshift: 数组为 nil: %w", ErrDtype)
	}
	cur := a.AsContiguous()
	for ax := 0; ax < a.NDim(); ax++ {
		sh := cur.Shape()[ax]
		if sh <= 1 {
			continue
		}
		r, err := Roll(cur, -(sh / 2), []int{ax})
		if err != nil {
			return nil, err
		}
		cur = r
	}
	return cur, nil
}

// Exp2 ≡ np.exp2:2 的幂。整型输入提升为 float64。
func Exp2(x *Array) (*Array, error) {
	if x == nil {
		return nil, fmt.Errorf("num: exp2: 数组为 nil: %w", ErrDtype)
	}
	dt := DFloat64
	if x.Dtype() == DFloat32 {
		dt = DFloat32
	}
	f, err := transformCopy(x.AsContiguous(), x.Shape(), dt)
	if err != nil {
		return nil, err
	}
	if dt == DFloat32 {
		w := win[float32](engineWindow(f))
		for i := range w {
			w[i] = float32(math.Exp2(float64(w[i])))
		}
		return f, nil
	}
	w := win[float64](engineWindow(f))
	for i := range w {
		w[i] = math.Exp2(w[i])
	}
	return f, nil
}

// NanCumprod ≡ np.nancumprod:把 NaN 当 1 的前缀积。
func NanCumprod(a *Array, axis []int) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: nancumprod: 数组为 nil: %w", ErrDtype)
	}
	f, err := transformCopy(a.AsContiguous(), a.Shape(), DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(f))
	for i := range w {
		if w[i] != w[i] {
			w[i] = 1
		}
	}
	return Cumprod(f, axis)
}

// 小工具:64 位整数写入。
func putInt64(b []byte, v int64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}

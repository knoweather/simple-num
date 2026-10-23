package num

// :聚合 9 算子(B 自建归约层)。
//
// ufunc.go 没有 reduce 注册/分发面,故归约在这里自带分发:
// 适配层自建归约 —— 复用 A 的 Array 访问面、transformCopy 归一原语与 win[T]
// 视窗,语义按 numpy 2.5.4 实测:
//
// - 输出 dtype:sum —— b/i8..i64 → i64、u8..u64 → u64、f4 → f4、f8 → f8;
// mean/var/std —— f4 → f4,f16 → f16(HALF 策略),其余 → f64;min/max → 输入
// dtype;argmin/argmax → i64;any/all → bool。
// - axis:nil = 全维;[]int{...} 支持负数与多轴(重复 → ErrShape,越界 → ErrIndex);
// **多轴按"归约维原序展平为一块"做 pairwise**(numpy 的 transpose+flatten
// 语义;实测 axis=(0,1) 走展平顺序而非逐轴);keepdims 把归约维留成 1。
// - 0-size 归约维:sum → 0(恒等元)、mean/var/std → NaN、min/max/argmin/argmax
// → ErrShape(numpy 抛 ValueError)、any → false、all → true。
// - NaN:sum/mean/var/std/min/max 传播 NaN;argmin/argmax 首个 NaN 位置胜出;
// any/all 里 NaN 为真。
// - sum 每个输出元素 = +0.0 + pairwise(块)( 恒等元);
// mean = sum/n;var = pairwise((x-mean)²)/n;std = sqrt(var)(numpy _var 同构)。
// - complex/datetime/S/U → ErrDtype(待挂载)。

import (
	"fmt"
	"math"
)

// —— 归约计划 ——

// aggPlan 归约计划(输入按连续布局规划,元素步长)。
type aggPlan struct {
	inShape  []int
	redDims  []int // 升序
	redShape []int
	redStep  []int // 元素步长(连续输入)
	keptStep []int
	outKept  []int // 输出形状(不含 keepdims 的 1)
	redN     int
	full     bool // 归约全部维(连续 pairwise 直通)
}

// newAggPlan 校验 axis 并建计划。
func newAggPlan(a *Array, axes []int, keepdims bool) (*aggPlan, error) {
	nd := a.NDim()
	p := &aggPlan{inShape: a.Shape(), redN: 1}
	if nd == 0 {
		if len(axes) != 0 {
			return nil, fmt.Errorf("num: 0-d 数组不接受 axis: %w", ErrIndex)
		}
		p.full = true
		return p, nil
	}
	red := make([]bool, nd)
	if axes == nil {
		for i := range red {
			red[i] = true
		}
	} else {
		seen := make([]bool, nd)
		for _, ax := range axes {
			k := ax
			if k < 0 {
				k += nd
			}
			if k < 0 || k >= nd {
				return nil, fmt.Errorf("num: axis %d 越界(ndim=%d): %w", ax, nd, ErrIndex)
			}
			if seen[k] {
				return nil, fmt.Errorf("num: axis %d 重复: %w", ax, ErrShape)
			}
			seen[k] = true
			red[k] = true
		}
	}
	st := cStrides(p.inShape, 1)
	for i := 0; i < nd; i++ {
		if red[i] {
			p.redDims = append(p.redDims, i)
			p.redShape = append(p.redShape, p.inShape[i])
			p.redStep = append(p.redStep, st[i])
			p.redN *= p.inShape[i]
		} else {
			p.keptStep = append(p.keptStep, st[i])
			p.outKept = append(p.outKept, p.inShape[i])
		}
	}
	p.full = len(p.redDims) == nd
	return p, nil
}

// outShape 输出形状(keepdims 时归约维留 1)。
func (p *aggPlan) outShape(keepdims bool) []int {
	if !keepdims {
		return append([]int{}, p.outKept...)
	}
	sh := make([]int, len(p.inShape))
	for i, d := range p.inShape {
		sh[i] = 1
		if !containsInt(p.redDims, i) {
			sh[i] = d
		}
	}
	return sh
}

func containsInt(xs []int, v int) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

// gatherBlock 按归约维升序(展平序)把块拷进 scratch,返回元素数。
func gatherBlock[T any](src []T, base int, redShape, redStep []int, scratch []T) int {
	if len(redShape) == 0 {
		scratch[0] = src[base]
		return 1
	}
	idx := make([]int, len(redShape))
	n := 0
	for {
		off := base
		for k, ix := range idx {
			off += ix * redStep[k]
		}
		scratch[n] = src[off]
		n++
		k := len(idx) - 1
		for ; k >= 0; k-- {
			idx[k]++
			if idx[k] < redShape[k] {
				break
			}
			idx[k] = 0
		}
		if k < 0 {
			break
		}
	}
	return n
}

// eachOut 遍历每个输出元素,回调 (输出扁平下标, 归约块)。
func eachOut[T any](src []T, p *aggPlan, scratch []T, visit func(outIdx int, block []T)) {
	if p.full {
		visit(0, src)
		return
	}
	n := 1
	for _, d := range p.outKept {
		n *= d
	}
	idx := make([]int, len(p.outKept))
	for o := 0; o < n; o++ {
		base := 0
		for k, ix := range idx {
			base += ix * p.keptStep[k]
		}
		cnt := gatherBlock(src, base, p.redShape, p.redStep, scratch)
		visit(o, scratch[:cnt])
		for k := len(idx) - 1; k >= 0; k-- {
			idx[k]++
			if idx[k] < p.outKept[k] {
				break
			}
			idx[k] = 0
		}
	}
}

// runReduce 同 dtype 归约驱动。
func runReduce[T any](src, out *Array, p *aggPlan, k func([]T) T) {
	s := win[T](engineWindow(src))
	o := win[T](engineWindow(out))
	scratch := make([]T, p.redN)
	eachOut(s, p, scratch, func(i int, b []T) { o[i] = k(b) })
}

// runReduceTo 异 dtype 归约驱动(argmin → i64、any → bool)。
func runReduceTo[T any, R any](src, out *Array, p *aggPlan, k func([]T) R) {
	s := win[T](engineWindow(src))
	o := win[R](engineWindow(out))
	scratch := make([]T, p.redN)
	eachOut(s, p, scratch, func(i int, b []T) { o[i] = k(b) })
}

// —— 内核 ——

// sumF 浮点 pairwise 求和 + +0.0 恒等元。
func sumF[T FloatKinds](b []T) T { return 0 + sumPairwise(b) }

// sumI 整型求和(回绕,顺序无关)。
func sumI[T Ints](b []T) T {
	var s T
	for _, v := range b {
		s += v
	}
	return s
}

func meanF[T FloatKinds](b []T) T {
	n := len(b)
	if n == 0 {
		return T(math.NaN())
	}
	return sumF(b) / T(n)
}

// varF numpy _var 同构:mean → pairwise((x-μ)²) / n。
func varF[T FloatKinds](b []T) T {
	n := len(b)
	if n == 0 {
		return T(math.NaN())
	}
	mu := meanF(b)
	dev := make([]T, n)
	for i, v := range b {
		d := v - mu
		dev[i] = d * d
	}
	return (0 + sumPairwise(dev)) / T(n)
}

func stdF[T FloatKinds](b []T) T { return T(math.Sqrt(float64(varF(b)))) }

// minF/maxF:NaN 传播(numpy 同;返回首个 NaN 本体,保留位型)。
func minF[T FloatKinds](b []T) T {
	m := b[0]
	for _, v := range b[1:] {
		if math.IsNaN(float64(v)) {
			return v
		}
		if v < m {
			m = v
		}
	}
	return m
}

func maxF[T FloatKinds](b []T) T {
	m := b[0]
	for _, v := range b[1:] {
		if math.IsNaN(float64(v)) {
			return v
		}
		if v > m {
			m = v
		}
	}
	return m
}

func minI[T Ints](b []T) T {
	m := b[0]
	for _, v := range b[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxI[T Ints](b []T) T {
	m := b[0]
	for _, v := range b[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// minBool/maxBool:False < True。
func minBool(b []bool) bool {
	for _, v := range b {
		if !v {
			return false
		}
	}
	return true
}

func maxBool(b []bool) bool {
	for _, v := range b {
		if v {
			return true
		}
	}
	return false
}

// argMinF/argMaxF:首个 NaN 位置胜出(numpy 浮点 arg 循环语义),否则首现极值。
func argMinF[T FloatKinds](b []T) int64 {
	best := 0
	for i, v := range b {
		if math.IsNaN(float64(v)) {
			return int64(i)
		}
		if v < b[best] {
			best = i
		}
	}
	return int64(best)
}

func argMaxF[T FloatKinds](b []T) int64 {
	best := 0
	for i, v := range b {
		if math.IsNaN(float64(v)) {
			return int64(i)
		}
		if v > b[best] {
			best = i
		}
	}
	return int64(best)
}

func argMinI[T Ints](b []T) int64 {
	best := 0
	for i, v := range b {
		if v < b[best] {
			best = i
		}
	}
	return int64(best)
}

func argMaxI[T Ints](b []T) int64 {
	best := 0
	for i, v := range b {
		if v > b[best] {
			best = i
		}
	}
	return int64(best)
}

// anyTrue/allTrue:非零为真(NaN 非零 → 真,numpy 同)。
func anyTrue[T EqKinds](b []T) bool {
	var z T
	for _, v := range b {
		if v != z {
			return true
		}
	}
	return false
}

func allTrue[T EqKinds](b []T) bool {
	var z T
	for _, v := range b {
		if v == z {
			return false
		}
	}
	return true
}

// —— dtype 规则( 实测)——

func sumAccDtype(d Dtype) (Dtype, error) {
	switch d.Kind {
	case Bool, Int8, Int16, Int32, Int64:
		return DInt64, nil
	case Uint8, Uint16, Uint32, Uint64:
		return DUint64, nil
	case Float16:
		return DFloat16, nil // numpy:f16 求和用 f32 累加后转回 f16
	case Float32:
		return DFloat32, nil
	case Float64:
		return DFloat64, nil
	}
	return Dtype{}, fmt.Errorf("num: sum 无 %s 内核: %w", d, ErrDtype)
}

func floatAccDtype(d Dtype) (Dtype, error) {
	switch d.Kind {
	case Float32:
		return DFloat32, nil
	case Float64:
		return DFloat64, nil
	case Bool, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
		return DFloat64, nil
	}
	return Dtype{}, fmt.Errorf("num: 归约无 %s 内核: %w", d, ErrDtype)
}

// —— 入口 ——

// Sum 求和(axis:nil = 全维;多轴按归约维原序展平 pairwise)。
func Sum(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, err := newAggPlan(a, axis, keepdims)
	if err != nil {
		return nil, err
	}
	acc, err := sumAccDtype(a.Dtype())
	if err != nil {
		return nil, err
	}
	out, err := New(p.outShape(keepdims), acc)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil // 空输出:无可写元素
	}
	if a.Size() == 0 {
		return out, nil // 恒等元:New 已零填(+0.0 位型即 0x0)
	}
	src, err := transformCopy(a, a.Shape(), acc)
	if err != nil {
		return nil, err
	}
	switch acc.Kind {
	case Int64:
		runReduce[int64](src, out, p, sumI[int64])
	case Uint64:
		runReduce[uint64](src, out, p, sumI[uint64])
	case Float16:
		sumF16(src, out, p)
	case Float32:
		runReduce[float32](src, out, p, sumF[float32])
	case Float64:
		runReduce[float64](src, out, p, sumF[float64])
	}
	return out, nil
}

// sumF16 f16 求和:numpy HALF 策略 —— 元素加宽 f32、f32 pairwise 累加、
// 结果转回 f16(实测 sum(f16) → f16)。
func sumF16(src, out *Array, p *aggPlan) {
	sw := win[uint16](engineWindow(src))
	f32s := make([]float32, len(sw))
	for i, h := range sw {
		f32s[i] = f16ToF32(h)
	}
	scratch := make([]float32, p.redN)
	ow := win[uint16](engineWindow(out))
	eachOut(f32s, p, scratch, func(i int, b []float32) {
		ow[i] = f32ToF16(0 + sumPairwise(b))
	})
}

// meanVarStdF16 f16 的 mean/var/std:HALF 策略 —— 输出 dtype = f16,内部按 f32
// 计算后 cast 回 f16(numpy 实测:mean/var/std(f16) → f16,计算在 f32 完成;
// 与 sumF16 同构)。0-size 归约维 → NaN(f16)。
func meanVarStdF16(a *Array, p *aggPlan, keepdims bool, f32k func([]float32) float32) (*Array, error) {
	out, err := New(p.outShape(keepdims), DFloat16)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	if a.Size() == 0 {
		// 0-size 归约维:mean/var/std → NaN(f16)
		w := win[uint16](engineWindow(out))
		for i := range w {
			w[i] = f32ToF16(float32(math.NaN()))
		}
		return out, nil
	}
	src := a.AsContiguous()
	sw := win[uint16](engineWindow(src))
	f32s := make([]float32, len(sw))
	for i, h := range sw {
		f32s[i] = f16ToF32(h)
	}
	scratch := make([]float32, p.redN)
	ow := win[uint16](engineWindow(out))
	eachOut(f32s, p, scratch, func(i int, b []float32) {
		ow[i] = f32ToF16(f32k(b))
	})
	return out, nil
}

// reduceF16ToF16 f16 归约:元素加宽 f32 → float32 内核 → f32ToF16 写回 f16 输出
// (min/max 用;f16 值可在 f32 精确表示,转回无损失;NaN 保位型)。
func reduceF16ToF16(a *Array, out *Array, p *aggPlan, f32k func([]float32) float32) {
	src := a.AsContiguous()
	sw := win[uint16](engineWindow(src))
	f32s := make([]float32, len(sw))
	for i, h := range sw {
		f32s[i] = f16ToF32(h)
	}
	scratch := make([]float32, p.redN)
	ow := win[uint16](engineWindow(out))
	eachOut(f32s, p, scratch, func(i int, b []float32) { ow[i] = f32ToF16(f32k(b)) })
}

// reduceF16ToT f16 归约:元素加宽 f32 → 内核(返回 T=i64/bool)→ 直接写回 T 输出
// (argmin/argmax → i64、any/all → bool)。
func reduceF16ToT[T any](a *Array, out *Array, p *aggPlan, f32k func([]float32) T) {
	src := a.AsContiguous()
	sw := win[uint16](engineWindow(src))
	f32s := make([]float32, len(sw))
	for i, h := range sw {
		f32s[i] = f16ToF32(h)
	}
	scratch := make([]float32, p.redN)
	ow := win[T](engineWindow(out))
	eachOut(f32s, p, scratch, func(i int, b []float32) { ow[i] = f32k(b) })
}

// meanLike mean/var/std 共用骨架(f4 → f4,其余 → f64;0-size → NaN)。
func meanLike(a *Array, axis []int, keepdims bool,
	f64k func([]float64) float64, f32k func([]float32) float32,
) (*Array, error) {
	p, err := newAggPlan(a, axis, keepdims)
	if err != nil {
		return nil, err
	}
	if a.Dtype().Kind == Float16 {
		return meanVarStdF16(a, p, keepdims, f32k)
	}
	acc, err := floatAccDtype(a.Dtype())
	if err != nil {
		return nil, err
	}
	out, err := New(p.outShape(keepdims), acc)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	if a.Size() == 0 {
		if acc.Kind == Float32 {
			w := win[float32](engineWindow(out))
			for i := range w {
				w[i] = float32(math.NaN())
			}
		} else {
			w := win[float64](engineWindow(out))
			for i := range w {
				w[i] = math.NaN()
			}
		}
		return out, nil
	}
	src, err := transformCopy(a, a.Shape(), acc)
	if err != nil {
		return nil, err
	}
	switch acc.Kind {
	case Float32:
		runReduce[float32](src, out, p, f32k)
	case Float64:
		runReduce[float64](src, out, p, f64k)
	}
	return out, nil
}

// Mean 均值。
func Mean(a *Array, axis []int, keepdims bool) (*Array, error) {
	return meanLike(a, axis, keepdims, meanF[float64], meanF[float32])
}

// Var 方差(ddof=0;numpy _var 同构)。
func Var(a *Array, axis []int, keepdims bool) (*Array, error) {
	return meanLike(a, axis, keepdims, varF[float64], varF[float32])
}

// Std 标准差。
func Std(a *Array, axis []int, keepdims bool) (*Array, error) {
	return meanLike(a, axis, keepdims, stdF[float64], stdF[float32])
}

// minMaxPrep min/max/arg/any/all 共用前置:计划 + 0-size 检查 + 输出。
func aggPrep(a *Array, axis []int, keepdims bool, outDT Dtype, emptyErr bool) (*aggPlan, *Array, bool, error) {
	p, err := newAggPlan(a, axis, keepdims)
	if err != nil {
		return nil, nil, false, err
	}
	if emptyErr && p.redN == 0 {
		return nil, nil, false, fmt.Errorf("num: 0-size 归约维的 min/max/arg(numpy 同 raise): %w", ErrShape)
	}
	out, err := New(p.outShape(keepdims), outDT)
	if err != nil {
		return nil, nil, false, err
	}
	return p, out, out.Size() == 0, nil
}

// Min 最小值(NaN 传播;保输入 dtype)。
func Min(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, a.Dtype(), true)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Dtype().Kind == Float16 {
		reduceF16ToF16(a, out, p, minF[float32])
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Bool:
		runReduce[bool](src, out, p, minBool)
	case Int8:
		runReduce[int8](src, out, p, minI[int8])
	case Int16:
		runReduce[int16](src, out, p, minI[int16])
	case Int32:
		runReduce[int32](src, out, p, minI[int32])
	case Int64:
		runReduce[int64](src, out, p, minI[int64])
	case Uint8:
		runReduce[uint8](src, out, p, minI[uint8])
	case Uint16:
		runReduce[uint16](src, out, p, minI[uint16])
	case Uint32:
		runReduce[uint32](src, out, p, minI[uint32])
	case Uint64:
		runReduce[uint64](src, out, p, minI[uint64])
	case Float32:
		runReduce[float32](src, out, p, minF[float32])
	case Float64:
		runReduce[float64](src, out, p, minF[float64])
	default:
		return nil, fmt.Errorf("num: min 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// Max 最大值(NaN 传播;保输入 dtype)。
func Max(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, a.Dtype(), true)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Dtype().Kind == Float16 {
		reduceF16ToF16(a, out, p, maxF[float32])
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Bool:
		runReduce[bool](src, out, p, maxBool)
	case Int8:
		runReduce[int8](src, out, p, maxI[int8])
	case Int16:
		runReduce[int16](src, out, p, maxI[int16])
	case Int32:
		runReduce[int32](src, out, p, maxI[int32])
	case Int64:
		runReduce[int64](src, out, p, maxI[int64])
	case Uint8:
		runReduce[uint8](src, out, p, maxI[uint8])
	case Uint16:
		runReduce[uint16](src, out, p, maxI[uint16])
	case Uint32:
		runReduce[uint32](src, out, p, maxI[uint32])
	case Uint64:
		runReduce[uint64](src, out, p, maxI[uint64])
	case Float32:
		runReduce[float32](src, out, p, maxF[float32])
	case Float64:
		runReduce[float64](src, out, p, maxF[float64])
	default:
		return nil, fmt.Errorf("num: max 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// ArgMin 最小值位置(块内 0 基;首个 NaN 位置胜出;输出 i64)。
func ArgMin(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, DInt64, true)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Dtype().Kind == Float16 {
		reduceF16ToT(a, out, p, argMinF[float32])
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Bool:
		runReduceTo[bool, int64](src, out, p, func(b []bool) int64 {
			for i, v := range b {
				if !v {
					return int64(i)
				}
			}
			return 0
		})
	case Int8:
		runReduceTo[int8, int64](src, out, p, argMinI[int8])
	case Int16:
		runReduceTo[int16, int64](src, out, p, argMinI[int16])
	case Int32:
		runReduceTo[int32, int64](src, out, p, argMinI[int32])
	case Int64:
		runReduceTo[int64, int64](src, out, p, argMinI[int64])
	case Uint8:
		runReduceTo[uint8, int64](src, out, p, argMinI[uint8])
	case Uint16:
		runReduceTo[uint16, int64](src, out, p, argMinI[uint16])
	case Uint32:
		runReduceTo[uint32, int64](src, out, p, argMinI[uint32])
	case Uint64:
		runReduceTo[uint64, int64](src, out, p, argMinI[uint64])
	case Float32:
		runReduceTo[float32, int64](src, out, p, argMinF[float32])
	case Float64:
		runReduceTo[float64, int64](src, out, p, argMinF[float64])
	default:
		return nil, fmt.Errorf("num: argmin 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// ArgMax 最大值位置(块内 0 基;首个 NaN 位置胜出;输出 i64)。
func ArgMax(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, DInt64, true)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Dtype().Kind == Float16 {
		reduceF16ToT(a, out, p, argMaxF[float32])
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Bool:
		runReduceTo[bool, int64](src, out, p, func(b []bool) int64 {
			for i, v := range b {
				if v {
					return int64(i)
				}
			}
			return 0
		})
	case Int8:
		runReduceTo[int8, int64](src, out, p, argMaxI[int8])
	case Int16:
		runReduceTo[int16, int64](src, out, p, argMaxI[int16])
	case Int32:
		runReduceTo[int32, int64](src, out, p, argMaxI[int32])
	case Int64:
		runReduceTo[int64, int64](src, out, p, argMaxI[int64])
	case Uint8:
		runReduceTo[uint8, int64](src, out, p, argMaxI[uint8])
	case Uint16:
		runReduceTo[uint16, int64](src, out, p, argMaxI[uint16])
	case Uint32:
		runReduceTo[uint32, int64](src, out, p, argMaxI[uint32])
	case Uint64:
		runReduceTo[uint64, int64](src, out, p, argMaxI[uint64])
	case Float32:
		runReduceTo[float32, int64](src, out, p, argMaxF[float32])
	case Float64:
		runReduceTo[float64, int64](src, out, p, argMaxF[float64])
	default:
		return nil, fmt.Errorf("num: argmax 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// AnyTrue 是否存在非零(numpy any;NaN 为真;空归约维 → false)。
func AnyTrue(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, DBool, false)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Size() == 0 {
		return out, nil // 空归约维 → false(New 已零填)
	}
	if a.Dtype().Kind == Float16 {
		reduceF16ToT(a, out, p, anyTrue[float32])
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Bool:
		runReduceTo[bool, bool](src, out, p, anyTrue[bool])
	case Int8:
		runReduceTo[int8, bool](src, out, p, anyTrue[int8])
	case Int16:
		runReduceTo[int16, bool](src, out, p, anyTrue[int16])
	case Int32:
		runReduceTo[int32, bool](src, out, p, anyTrue[int32])
	case Int64:
		runReduceTo[int64, bool](src, out, p, anyTrue[int64])
	case Uint8:
		runReduceTo[uint8, bool](src, out, p, anyTrue[uint8])
	case Uint16:
		runReduceTo[uint16, bool](src, out, p, anyTrue[uint16])
	case Uint32:
		runReduceTo[uint32, bool](src, out, p, anyTrue[uint32])
	case Uint64:
		runReduceTo[uint64, bool](src, out, p, anyTrue[uint64])
	case Float32:
		runReduceTo[float32, bool](src, out, p, anyTrue[float32])
	case Float64:
		runReduceTo[float64, bool](src, out, p, anyTrue[float64])
	default:
		return nil, fmt.Errorf("num: any 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// AllTrue 是否全非零(numpy all;NaN 为真;空归约维 → true)。
func AllTrue(a *Array, axis []int, keepdims bool) (*Array, error) {
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, DBool, false)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Size() == 0 {
		// 0-size:any → false、all → true
		w := win[bool](engineWindow(out))
		for i := range w {
			w[i] = true
		}
		return out, nil
	}
	if a.Dtype().Kind == Float16 {
		reduceF16ToT(a, out, p, allTrue[float32])
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Bool:
		runReduceTo[bool, bool](src, out, p, allTrue[bool])
	case Int8:
		runReduceTo[int8, bool](src, out, p, allTrue[int8])
	case Int16:
		runReduceTo[int16, bool](src, out, p, allTrue[int16])
	case Int32:
		runReduceTo[int32, bool](src, out, p, allTrue[int32])
	case Int64:
		runReduceTo[int64, bool](src, out, p, allTrue[int64])
	case Uint8:
		runReduceTo[uint8, bool](src, out, p, allTrue[uint8])
	case Uint16:
		runReduceTo[uint16, bool](src, out, p, allTrue[uint16])
	case Uint32:
		runReduceTo[uint32, bool](src, out, p, allTrue[uint32])
	case Uint64:
		runReduceTo[uint64, bool](src, out, p, allTrue[uint64])
	case Float32:
		runReduceTo[float32, bool](src, out, p, allTrue[float32])
	case Float64:
		runReduceTo[float64, bool](src, out, p, allTrue[float64])
	default:
		return nil, fmt.Errorf("num: all 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

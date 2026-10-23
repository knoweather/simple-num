package num

import "fmt"

// —— NaN 家族归约(numpy nan* 系列)——
//
// 口径与 numpy 一致:NaN 被跳过(而不是传播);整型/bool 无 NaN 概念,直接委托
// 对应的普通归约。全 NaN 的切片:nanmax/nanmin 返回 NaN(numpy 同时发
// RuntimeWarning,本库无 warning 通道),nansum 得 0,nanmean 得 NaN。

// NanSum ≡ np.nansum:NaN 视作 0。numpy 的实现就是"先替换再求和",故与 Sum
// 的 pairwise 口径逐位一致。
func NanSum(a *Array, axis []int, keepdims bool) (*Array, error) {
	if !kindHasNaN(a.Dtype().Kind) {
		return Sum(a, axis, keepdims)
	}
	r, err := replaceNaNValue(a, 0)
	if err != nil {
		return nil, fmt.Errorf("num: nansum: %w", err)
	}
	return Sum(r, axis, keepdims)
}

// NanMax ≡ np.nanmax:跳过 NaN 取最大;全 NaN 切片返回 NaN。
func NanMax(a *Array, axis []int, keepdims bool) (*Array, error) {
	return nanMinMax(a, axis, keepdims, true)
}

// NanMin ≡ np.nanmin:跳过 NaN 取最小;全 NaN 切片返回 NaN。
func NanMin(a *Array, axis []int, keepdims bool) (*Array, error) {
	return nanMinMax(a, axis, keepdims, false)
}

func nanMinMax(a *Array, axis []int, keepdims bool, isMax bool) (*Array, error) {
	if !kindHasNaN(a.Dtype().Kind) { // 整型/bool:等价普通 min/max
		if isMax {
			return Max(a, axis, keepdims)
		}
		return Min(a, axis, keepdims)
	}
	p, out, emptyOut, err := aggPrep(a, axis, keepdims, a.Dtype(), true)
	if err != nil || emptyOut {
		return out, err
	}
	if a.Dtype().Kind == Float16 {
		if isMax {
			reduceF16ToF16(a, out, p, nanMaxF[float32])
		} else {
			reduceF16ToF16(a, out, p, nanMinF[float32])
		}
		return out, nil
	}
	src := a.AsContiguous()
	switch a.Dtype().Kind {
	case Float32:
		if isMax {
			runReduce[float32](src, out, p, nanMaxF[float32])
		} else {
			runReduce[float32](src, out, p, nanMinF[float32])
		}
	case Float64:
		if isMax {
			runReduce[float64](src, out, p, nanMaxF[float64])
		} else {
			runReduce[float64](src, out, p, nanMinF[float64])
		}
	default:
		return nil, fmt.Errorf("num: nanmax/nanmin 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// kindHasNaN 判定 dtype 是否可能有 NaN(仅浮点;复数暂不支持 nan* 归约)。
func kindHasNaN(k Kind) bool {
	return k == Float16 || k == Float32 || k == Float64
}

// replaceNaNValue 返回 a 的拷贝,其中 NaN 替换为 repl(其余位型原样保留)。
func replaceNaNValue(a *Array, repl float64) (*Array, error) {
	src := a.AsContiguous()
	out, err := New(a.Shape(), a.Dtype())
	if err != nil {
		return nil, err
	}
	n := out.Size()
	if n == 0 {
		return out, nil
	}
	si, di := engineWindow(src), engineWindow(out)
	switch a.Dtype().Kind {
	case Float16:
		s, d := win[uint16](si), win[uint16](di)
		r := f32ToF16(float32(repl))
		for i, v := range s {
			if f16ToF32(v) != f16ToF32(v) {
				d[i] = r
			} else {
				d[i] = v
			}
		}
	case Float32:
		s, d := win[float32](si), win[float32](di)
		r := float32(repl)
		for i, v := range s {
			if v != v {
				d[i] = r
			} else {
				d[i] = v
			}
		}
	case Float64:
		s, d := win[float64](si), win[float64](di)
		for i, v := range s {
			if v != v {
				d[i] = repl
			} else {
				d[i] = v
			}
		}
	default:
		return nil, fmt.Errorf("num: replaceNaN 无 %s 内核: %w", a.Dtype(), ErrDtype)
	}
	return out, nil
}

// nanMaxF 跳过 NaN 取最大;全 NaN 返回首个 NaN 的位型。
func nanMaxF[T float32 | float64](b []T) T {
	var best, nanVal T
	found, hasNaN := false, false
	for _, v := range b {
		if v != v {
			if !hasNaN {
				nanVal = v
				hasNaN = true
			}
			continue
		}
		if !found || v > best {
			best = v
			found = true
		}
	}
	if !found {
		return nanVal
	}
	return best
}

// nanMinF 跳过 NaN 取最小;全 NaN 返回首个 NaN 的位型。
func nanMinF[T float32 | float64](b []T) T {
	var best, nanVal T
	found, hasNaN := false, false
	for _, v := range b {
		if v != v {
			if !hasNaN {
				nanVal = v
				hasNaN = true
			}
			continue
		}
		if !found || v < best {
			best = v
			found = true
		}
	}
	if !found {
		return nanVal
	}
	return best
}

// —— nanmean / nanvar / nanstd ——
//
// 口径:先剔除 NaN 再套用普通算法(等价 numpy 的 where=~isnan 归约:
// nanmean = 非 NaN 元素均值,nanvar = Σ(x-μ)²/n(非 NaN 计数),nanstd = √nanvar)。
// 全 NaN 的切片 → 均值/方差为 NaN(空块 → meanF/varF 本已返回 NaN)。

// filterNaN 返回剔除了 NaN 的子切片(共享底层数组,零拷贝)。
func filterNaN[T float32 | float64](b []T) []T {
	out := make([]T, 0, len(b))
	for _, v := range b {
		if v == v {
			out = append(out, v)
		}
	}
	return out
}

func nanMeanF[T float32 | float64](b []T) T { return meanF(filterNaN(b)) }
func nanVarF[T float32 | float64](b []T) T  { return varF(filterNaN(b)) }
func nanStdF[T float32 | float64](b []T) T  { return stdF(filterNaN(b)) }

// NanMean ≡ np.nanmean:忽略 NaN 求均值;整型输入按 numpy 提升为浮点。
func NanMean(a *Array, axis []int, keepdims bool) (*Array, error) {
	return meanLike(a, axis, keepdims, nanMeanF[float64], nanMeanF[float32])
}

// NanVar ≡ np.nanvar:忽略 NaN 求方差(ddof=0,与 numpy 默认一致)。
func NanVar(a *Array, axis []int, keepdims bool) (*Array, error) {
	return meanLike(a, axis, keepdims, nanVarF[float64], nanVarF[float32])
}

// NanStd ≡ np.nanstd:忽略 NaN 求标准差(ddof=0)。
func NanStd(a *Array, axis []int, keepdims bool) (*Array, error) {
	return meanLike(a, axis, keepdims, nanStdF[float64], nanStdF[float32])
}

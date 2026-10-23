package num

import (
	"fmt"
	"math"
)

// Clip ≡ np.clip(a, lo, hi):逐元素夹取到 [lo, hi]。
//
// - lo/hi 传 nil 表示该侧不夹(numpy 的 None);两者都 nil 等价于拷贝;
// - **NaN 传播**:元素为 NaN 时结果仍是同一 NaN(不参与比较);
// - **lo > hi 时结果恒为 hi**(numpy 实测行为:先夹下界再夹上界,后者胜出);
// - 结果 dtype = promote(a, 非 nil 边界)—— 与 numpy 同:f8 数组配 i4 边界
// 仍是 f8,i4 数组配 i4 边界保持 i4;
// - ±0 位型保留;复数/文本/时间无"大小"语义 → ErrDtype。
func Clip(a *Array, lo, hi any) (*Array, error) {
	if a == nil {
		return nil, fmt.Errorf("num: clip: 数组为 nil: %w", ErrDtype)
	}
	hasLo, hasHi := lo != nil, hi != nil
	if !hasLo && !hasHi {
		return transformCopy(a, a.Shape(), a.Dtype())
	}
	dt := a.Dtype()
	if hasLo {
		sd, err := scalarDtype(lo)
		if err != nil {
			return nil, fmt.Errorf("num: clip: 下界: %w", err)
		}
		if dt, err = promoteTypes(dt, sd); err != nil {
			return nil, fmt.Errorf("num: clip: 下界 dtype 不可提升: %w", err)
		}
	}
	if hasHi {
		sd, err := scalarDtype(hi)
		if err != nil {
			return nil, fmt.Errorf("num: clip: 上界: %w", err)
		}
		if dt, err = promoteTypes(dt, sd); err != nil {
			return nil, fmt.Errorf("num: clip: 上界 dtype 不可提升: %w", err)
		}
	}
	src := a
	if src.Dtype() != dt || src.fastWindow() == nil {
		var err error
		if src, err = transformCopy(a, a.Shape(), dt); err != nil {
			return nil, fmt.Errorf("num: clip: 源数组归一失败: %w", err)
		}
	}
	// 边界值经"0-d 数组 → 目标 dtype"转码,复用既有 cast 语义(与 numpy 的
	// 边界按 promote 后类型参与比较一致)。
	var loAny, hiAny any
	if hasLo {
		v, err := scalarAs(lo, dt)
		if err != nil {
			return nil, fmt.Errorf("num: clip: 下界转 %s 失败: %w", dt, err)
		}
		loAny = v
	}
	if hasHi {
		v, err := scalarAs(hi, dt)
		if err != nil {
			return nil, fmt.Errorf("num: clip: 上界转 %s 失败: %w", dt, err)
		}
		hiAny = v
	}
	out, err := New(a.Shape(), dt)
	if err != nil {
		return nil, err
	}
	n := out.Size()
	if n == 0 {
		return out, nil
	}
	if err := clipInto(engineWindow(out), engineWindow(src), loAny, hiAny, dt); err != nil {
		return nil, err
	}
	return out, nil
}

// scalarDtype 由 Go 标量推断 numpy 口径的 dtype(用于 clip 边界的 promote)。
func scalarDtype(v any) (Dtype, error) {
	switch x := v.(type) {
	case bool:
		return DBool, nil
	case int:
		return DInt64, nil
	case int8:
		return DInt8, nil
	case int16:
		return DInt16, nil
	case int32:
		return DInt32, nil
	case int64:
		return DInt64, nil
	case uint:
		return DUint64, nil
	case uint8:
		return DUint8, nil
	case uint16:
		return DUint16, nil
	case uint32:
		return DUint32, nil
	case uint64:
		return DUint64, nil
	case float32:
		return DFloat32, nil
	case float64:
		return DFloat64, nil
	default:
		_ = x
		return Dtype{}, fmt.Errorf("num: 标量类型 %T 无对应 dtype: %w", v, ErrDtype)
	}
}

// scalarAs 把标量值转成目标 dtype 下的 Go 值(经 0-d 数组 + transformCopy,
// 与显式 cast 同语义;不复刻 numpy 的 ComplexWarning 等告警)。
func scalarAs(v any, dt Dtype) (any, error) {
	sd, err := scalarDtype(v)
	if err != nil {
		return nil, err
	}
	tmp, err := New([]int{}, sd)
	if err != nil {
		return nil, err
	}
	if err := tmp.SetItem([]int{}, v); err != nil {
		return nil, err
	}
	if sd == dt {
		return tmp.Item()
	}
	conv, err := transformCopy(tmp, []int{}, dt)
	if err != nil {
		return nil, err
	}
	return conv.Item()
}

// clipInto 按目标 dtype 做类型化夹取;loAny/hiAny 为 nil 表示该侧不夹。
func clipInto(dst, src []byte, loAny, hiAny any, dt Dtype) error {
	switch dt.Kind {
	case Float64:
		lv, hv := math.Inf(-1), math.Inf(1)
		if loAny != nil {
			lv = loAny.(float64)
		}
		if hiAny != nil {
			hv = hiAny.(float64)
		}
		s, d := win[float64](src), win[float64](dst)
		for i, v := range s {
			if v != v { // NaN 传播
				d[i] = v
				continue
			}
			if v < lv {
				v = lv
			}
			if v > hv {
				v = hv
			}
			d[i] = v
		}
	case Float32:
		lv, hv := float32(math.Inf(-1)), float32(math.Inf(1))
		if loAny != nil {
			lv = loAny.(float32)
		}
		if hiAny != nil {
			hv = hiAny.(float32)
		}
		s, d := win[float32](src), win[float32](dst)
		for i, v := range s {
			if v != v {
				d[i] = v
				continue
			}
			if v < lv {
				v = lv
			}
			if v > hv {
				v = hv
			}
			d[i] = v
		}
	case Float16:
		f := func(v any, def float32) float32 {
			if v == nil {
				return def
			}
			switch x := v.(type) {
			case uint16: // f16 的存储形态
				return f16ToF32(x)
			case float32:
				return x
			}
			return def
		}
		lv, hv := f(loAny, float32(math.Inf(-1))), f(hiAny, float32(math.Inf(1)))
		s, d := win[uint16](src), win[uint16](dst)
		for i, raw := range s {
			v := f16ToF32(raw)
			if v != v {
				d[i] = raw
				continue
			}
			if v < lv {
				v = lv
			}
			if v > hv {
				v = hv
			}
			d[i] = f32ToF16(v)
		}
	case Int8:
		clipInt(win[int8](src), win[int8](dst), loAny, hiAny,
			func(a any) (int64, bool) { x, ok := a.(int8); return int64(x), ok })
	case Int16:
		clipInt(win[int16](src), win[int16](dst), loAny, hiAny,
			func(a any) (int64, bool) { x, ok := a.(int16); return int64(x), ok })
	case Int32:
		clipInt(win[int32](src), win[int32](dst), loAny, hiAny,
			func(a any) (int64, bool) { x, ok := a.(int32); return int64(x), ok })
	case Int64:
		clipInt(win[int64](src), win[int64](dst), loAny, hiAny,
			func(a any) (int64, bool) { x, ok := a.(int64); return x, ok })
	case Uint8:
		clipUint(win[uint8](src), win[uint8](dst), loAny, hiAny,
			func(a any) (uint64, bool) { x, ok := a.(uint8); return uint64(x), ok })
	case Uint16:
		clipUint(win[uint16](src), win[uint16](dst), loAny, hiAny,
			func(a any) (uint64, bool) { x, ok := a.(uint16); return uint64(x), ok })
	case Uint32:
		clipUint(win[uint32](src), win[uint32](dst), loAny, hiAny,
			func(a any) (uint64, bool) { x, ok := a.(uint32); return uint64(x), ok })
	case Uint64:
		clipUint(win[uint64](src), win[uint64](dst), loAny, hiAny,
			func(a any) (uint64, bool) { x, ok := a.(uint64); return x, ok })
	case Bool:
		lv, hv := uint8(0), uint8(1)
		if loAny != nil {
			if b, ok := loAny.(bool); ok && b {
				lv = 1
			}
		}
		if hiAny != nil {
			if b, ok := hiAny.(bool); ok && !b {
				hv = 0
			}
		}
		s, d := win[uint8](src), win[uint8](dst)
		for i, v := range s {
			if v < lv {
				v = lv
			}
			if v > hv {
				v = hv
			}
			d[i] = v
		}
	default:
		return fmt.Errorf("num: clip: dtype %s 无大小语义: %w", dt, ErrDtype)
	}
	return nil
}

// clipInt 有符号整型夹取(用 int64 中转,避免逐类型重复比较逻辑)。
func clipInt[T int8 | int16 | int32 | int64](src, dst []T, loAny, hiAny any,
	conv func(any) (int64, bool)) {
	lo, hi := int64(math.MinInt64), int64(math.MaxInt64)
	if loAny != nil {
		if v, ok := conv(loAny); ok {
			lo = v
		}
	}
	if hiAny != nil {
		if v, ok := conv(hiAny); ok {
			hi = v
		}
	}
	for i, v := range src {
		x := int64(v)
		if x < lo {
			x = lo
		}
		if x > hi {
			x = hi
		}
		dst[i] = T(x)
	}
}

// clipUint 无符号整型夹取。
func clipUint[T uint8 | uint16 | uint32 | uint64](src, dst []T, loAny, hiAny any,
	conv func(any) (uint64, bool)) {
	lo, hi := uint64(0), uint64(math.MaxUint64)
	if loAny != nil {
		if v, ok := conv(loAny); ok {
			lo = v
		}
	}
	if hiAny != nil {
		if v, ok := conv(hiAny); ok {
			hi = v
		}
	}
	for i, v := range src {
		x := uint64(v)
		if x < lo {
			x = lo
		}
		if x > hi {
			x = hi
		}
		dst[i] = T(x)
	}
}

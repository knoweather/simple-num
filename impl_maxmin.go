package num

// Maximum ≡ np.maximum:逐元素取大,**NaN 传播**(任一侧 NaN → 该 NaN 按位返回)。
//
// 与 fmax 的区别:numpy 的 fmax 忽略 NaN,maximum 传播 NaN;本库只提供 maximum 语义。
// 相等时的取舍与 numpy 一致(比较用 `a > b`,故相等取右侧,±0 的符号也照此规则)。
func Maximum(x, y *Array) (*Array, error) { return dispatchBinary("maximum", x, y) }

// Minimum ≡ np.minimum:逐元素取小,NaN 传播,相等取右侧(同 numpy 的 `a < b`)。
func Minimum(x, y *Array) (*Array, error) { return dispatchBinary("minimum", x, y) }

func init() {
	regMaxMinFloat[float64](true)
	regMaxMinFloat[float64](false)
	regMaxMinFloat[float32](true)
	regMaxMinFloat[float32](false)

	regMaxMinInt[int8](true)
	regMaxMinInt[int8](false)
	regMaxMinInt[int16](true)
	regMaxMinInt[int16](false)
	regMaxMinInt[int32](true)
	regMaxMinInt[int32](false)
	regMaxMinInt[int64](true)
	regMaxMinInt[int64](false)
	regMaxMinInt[uint8](true)
	regMaxMinInt[uint8](false)
	regMaxMinInt[uint16](true)
	regMaxMinInt[uint16](false)
	regMaxMinInt[uint32](true)
	regMaxMinInt[uint32](false)
	regMaxMinInt[uint64](true)
	regMaxMinInt[uint64](false)

	regMaxMinF16(true)
	regMaxMinF16(false)
}

// regMaxMinFloat 注册浮点内核(NaN 按位传播:返回原操作数,保 payload)。
func regMaxMinFloat[T FloatKinds](isMax bool) {
	op := "minimum"
	if isMax {
		op = "maximum"
	}
	registerBinaryTyped(op, func(dst, a, b []T) {
		for i := range dst {
			av, bv := a[i], b[i]
			if isNaNFloat(av) {
				dst[i] = av
				continue
			}
			if isNaNFloat(bv) {
				dst[i] = bv
				continue
			}
			if isMax {
				if av > bv {
					dst[i] = av
				} else {
					dst[i] = bv
				}
			} else {
				if av < bv {
					dst[i] = av
				} else {
					dst[i] = bv
				}
			}
		}
	})
}

// isNaNFloat 泛型 NaN 判定(只有浮点实例化)。
func isNaNFloat[T float32 | float64](v T) bool { return v != v }

// regMaxMinInt 注册整型内核(无 NaN 语义)。
func regMaxMinInt[T Ints](isMax bool) {
	op := "minimum"
	if isMax {
		op = "maximum"
	}
	registerBinaryTyped(op, func(dst, a, b []T) {
		for i := range dst {
			if isMax {
				if a[i] > b[i] {
					dst[i] = a[i]
				} else {
					dst[i] = b[i]
				}
			} else {
				if a[i] < b[i] {
					dst[i] = a[i]
				} else {
					dst[i] = b[i]
				}
			}
		}
	})
}

// regMaxMinF16 注册 f16 内核:比较在 f32 域做,写回取原操作数的位型(保 NaN payload)。
func regMaxMinF16(isMax bool) {
	op := "minimum"
	if isMax {
		op = "maximum"
	}
	registerBinaryTyped(op, func(dst, a, b []uint16) {
		for i := range dst {
			av, bv := f16ToF32(a[i]), f16ToF32(b[i])
			switch {
			case av != av:
				dst[i] = a[i]
			case bv != bv:
				dst[i] = b[i]
			case isMax && av > bv, !isMax && av < bv:
				dst[i] = a[i]
			default:
				dst[i] = b[i]
			}
		}
	})
}

package num

// —— 位运算(Step 34):and / or / xor / 左移 / 右移 ——
//
// numpy 口径:只接受整型与 bool(浮点/复数抛 TypeError,这里返回 ErrDtype);
// **移位量 < 0 或 ≥ 位宽 → 0**(numpy 2.x 即如此,不报错);有符号右移是算术右移。

// BitwiseAnd ≡ np.bitwise_and。
func BitwiseAnd(x, y *Array) (*Array, error) { return dispatchBinary("bitwise_and", x, y) }

// BitwiseOr ≡ np.bitwise_or。
func BitwiseOr(x, y *Array) (*Array, error) { return dispatchBinary("bitwise_or", x, y) }

// BitwiseXor ≡ np.bitwise_xor。
func BitwiseXor(x, y *Array) (*Array, error) { return dispatchBinary("bitwise_xor", x, y) }

// LeftShift ≡ np.left_shift:移位量非法时结果为 0。
func LeftShift(x, y *Array) (*Array, error) { return dispatchBinary("left_shift", x, y) }

// RightShift ≡ np.right_shift:有符号走算术右移,移位量非法时结果为 0。
func RightShift(x, y *Array) (*Array, error) { return dispatchBinary("right_shift", x, y) }

// shiftNum 移位内核可用的整型集合(Go 的移位只接受整型,castableNum 含浮点)。
type shiftNum interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// shiftKernel 生成整型的移位内核;非法移位量一律得 0(与 numpy 一致)。
func shiftKernel[T shiftNum](bits int, left bool) func(dst, a, b []T) {
	return func(dst, a, b []T) {
		for i := range dst {
			sh := int64(b[i])
			if sh < 0 || sh >= int64(bits) {
				dst[i] = 0
				continue
			}
			if left {
				dst[i] = a[i] << uint(sh)
			} else {
				dst[i] = a[i] >> uint(sh)
			}
		}
	}
}

func init() {
	// 按位与/或/异或:整型 + bool(bool 以 0/1 存储,按位运算等价逻辑运算)
	registerBinaryTyped[int8]("bitwise_and", func(dst, a, b []int8) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[int16]("bitwise_and", func(dst, a, b []int16) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[int32]("bitwise_and", func(dst, a, b []int32) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[int64]("bitwise_and", func(dst, a, b []int64) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[uint8]("bitwise_and", func(dst, a, b []uint8) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[uint16]("bitwise_and", func(dst, a, b []uint16) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[uint32]("bitwise_and", func(dst, a, b []uint32) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[uint64]("bitwise_and", func(dst, a, b []uint64) {
		for i := range dst {
			dst[i] = a[i] & b[i]
		}
	})
	registerBinaryTyped[bool]("bitwise_and", func(dst, a, b []bool) {
		for i := range dst {
			dst[i] = a[i] && b[i]
		}
	})

	registerBinaryTyped[int8]("bitwise_or", func(dst, a, b []int8) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[int16]("bitwise_or", func(dst, a, b []int16) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[int32]("bitwise_or", func(dst, a, b []int32) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[int64]("bitwise_or", func(dst, a, b []int64) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[uint8]("bitwise_or", func(dst, a, b []uint8) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[uint16]("bitwise_or", func(dst, a, b []uint16) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[uint32]("bitwise_or", func(dst, a, b []uint32) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[uint64]("bitwise_or", func(dst, a, b []uint64) {
		for i := range dst {
			dst[i] = a[i] | b[i]
		}
	})
	registerBinaryTyped[bool]("bitwise_or", func(dst, a, b []bool) {
		for i := range dst {
			dst[i] = a[i] || b[i]
		}
	})

	registerBinaryTyped[int8]("bitwise_xor", func(dst, a, b []int8) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[int16]("bitwise_xor", func(dst, a, b []int16) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[int32]("bitwise_xor", func(dst, a, b []int32) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[int64]("bitwise_xor", func(dst, a, b []int64) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[uint8]("bitwise_xor", func(dst, a, b []uint8) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[uint16]("bitwise_xor", func(dst, a, b []uint16) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[uint32]("bitwise_xor", func(dst, a, b []uint32) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[uint64]("bitwise_xor", func(dst, a, b []uint64) {
		for i := range dst {
			dst[i] = a[i] ^ b[i]
		}
	})
	registerBinaryTyped[bool]("bitwise_xor", func(dst, a, b []bool) {
		for i := range dst {
			dst[i] = a[i] != b[i]
		}
	})

	// 移位:整型;bool 的移位在 numpy 里会提升到整型,本库要求显式转换
	registerBinaryTyped[int8]("left_shift", shiftKernel[int8](8, true))
	registerBinaryTyped[int16]("left_shift", shiftKernel[int16](16, true))
	registerBinaryTyped[int32]("left_shift", shiftKernel[int32](32, true))
	registerBinaryTyped[int64]("left_shift", shiftKernel[int64](64, true))
	registerBinaryTyped[uint8]("left_shift", shiftKernel[uint8](8, true))
	registerBinaryTyped[uint16]("left_shift", shiftKernel[uint16](16, true))
	registerBinaryTyped[uint32]("left_shift", shiftKernel[uint32](32, true))
	registerBinaryTyped[uint64]("left_shift", shiftKernel[uint64](64, true))

	registerBinaryTyped[int8]("right_shift", shiftKernel[int8](8, false))
	registerBinaryTyped[int16]("right_shift", shiftKernel[int16](16, false))
	registerBinaryTyped[int32]("right_shift", shiftKernel[int32](32, false))
	registerBinaryTyped[int64]("right_shift", shiftKernel[int64](64, false))
	registerBinaryTyped[uint8]("right_shift", shiftKernel[uint8](8, false))
	registerBinaryTyped[uint16]("right_shift", shiftKernel[uint16](16, false))
	registerBinaryTyped[uint32]("right_shift", shiftKernel[uint32](32, false))
	registerBinaryTyped[uint64]("right_shift", shiftKernel[uint64](64, false))
}

package num

// —— cast 性能特化(横切):常见数值 dtype 组合走类型化批量转换 ——
//
// 背景:Cast / CastUnsafe 的通用路径逐元素走 decodeElem → any 装箱 → encodeOne,
// 1e6 元素约 19 ms(numpy astype 约 0.9 ms,差 20×)。这里对**语义完全等价**的
// 常见组合(浮点↔浮点、整型↔整型、整型→浮点)做批量转换。
//
// 刻意不纳入的类型:
// - f64 → f16(本库口径是"单次 RNE、不经 f32",批量路径会二次舍入);
// - 浮点 → 整型(unsafe 口径依赖硬件位型,保留原路径以确保与 numpy 逐位一致);
// - 复数 / 文本 / datetime(语义复杂,保留原路径)。
//
// 语义等价性:整型↔整型按 Go 的"截断低位"规则,与 C/numpy 一致;
// 整型→浮点按 IEEE 就近舍入,与 C 一致。

type castableNum interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~float32 | ~float64
}

// convNum 通用数值转换(编译器会为每对具体类型生成无装箱的紧循环)。
func convNum[S castableNum, D castableNum](src []S, dst []D) {
	for i := range src {
		dst[i] = D(src[i])
	}
}

// castFastConv 尝试快速转换;返回 false 表示该组合未特化(调用方回退通用路径)。
func castFastConv(src, out *Array) bool {
	n := out.Size()
	if n == 0 {
		return true // 空数组:无需搬运
	}
	if src.fastWindow() == nil {
		return false
	}
	sb, db := engineWindow(src), engineWindow(out)
	sk, dk := src.dtype.Kind, out.dtype.Kind

	// f16 以 uint16 存储,单独处理
	switch {
	case sk == Float16 && dk == Float32:
		s, d := win[uint16](sb), win[float32](db)
		for i := 0; i < n; i++ {
			d[i] = f16ToF32(s[i])
		}
		return true
	case sk == Float16 && dk == Float64:
		s, d := win[uint16](sb), win[float64](db)
		for i := 0; i < n; i++ {
			d[i] = float64(f16ToF32(s[i]))
		}
		return true
	case sk == Float32 && dk == Float16:
		s, d := win[float32](sb), win[uint16](db)
		for i := 0; i < n; i++ {
			d[i] = f32ToF16(s[i])
		}
		return true
	}

	// 浮点 → 整型(仅 unsafe 口径会走到这里:安全 Cast 的 same_kind 权限表本就拒绝)。
	// 逐元素复用 unsafeFloatToIntBits(纯函数、无装箱),语义与通用路径逐位一致:
	// NaN/±Inf 与超界 → 目标位宽符号位(0x80…),否则先 Trunc 再判界。
	switch sk {
	case Float64:
		if f, ok := unsafeFloatConv(win[float64](sb), dk, db); ok {
			return f
		}
	case Float32:
		w := win[float32](sb)
		tmp := make([]float64, len(w))
		for i, v := range w {
			tmp[i] = float64(v)
		}
		if f, ok := unsafeFloatConv(tmp, dk, db); ok {
			return f
		}
	}

	switch sk {
	case Float64:
		switch dk {
		case Float32:
			convNum(win[float64](sb), win[float32](db))
			return true
		}
	case Float32:
		switch dk {
		case Float64:
			convNum(win[float32](sb), win[float64](db))
			return true
		}
	case Int8:
		switch dk {
		case Int16:
			convNum(win[int8](sb), win[int16](db))
			return true
		case Int32:
			convNum(win[int8](sb), win[int32](db))
			return true
		case Int64:
			convNum(win[int8](sb), win[int64](db))
			return true
		case Float32:
			convNum(win[int8](sb), win[float32](db))
			return true
		case Float64:
			convNum(win[int8](sb), win[float64](db))
			return true
		}
	case Int16:
		switch dk {
		case Int32:
			convNum(win[int16](sb), win[int32](db))
			return true
		case Int64:
			convNum(win[int16](sb), win[int64](db))
			return true
		case Int8:
			convNum(win[int16](sb), win[int8](db))
			return true
		case Float32:
			convNum(win[int16](sb), win[float32](db))
			return true
		case Float64:
			convNum(win[int16](sb), win[float64](db))
			return true
		}
	case Int32:
		switch dk {
		case Int64:
			convNum(win[int32](sb), win[int64](db))
			return true
		case Int16:
			convNum(win[int32](sb), win[int16](db))
			return true
		case Int8:
			convNum(win[int32](sb), win[int8](db))
			return true
		case Uint16:
			convNum(win[int32](sb), win[uint16](db))
			return true
		case Float32:
			convNum(win[int32](sb), win[float32](db))
			return true
		case Float64:
			convNum(win[int32](sb), win[float64](db))
			return true
		}
	case Int64:
		switch dk {
		case Int32:
			convNum(win[int64](sb), win[int32](db))
			return true
		case Int8:
			convNum(win[int64](sb), win[int8](db))
			return true
		case Uint8:
			convNum(win[int64](sb), win[uint8](db))
			return true
		case Float32:
			convNum(win[int64](sb), win[float32](db))
			return true
		case Float64:
			convNum(win[int64](sb), win[float64](db))
			return true
		}
	case Uint8:
		switch dk {
		case Uint16:
			convNum(win[uint8](sb), win[uint16](db))
			return true
		case Uint32:
			convNum(win[uint8](sb), win[uint32](db))
			return true
		case Uint64:
			convNum(win[uint8](sb), win[uint64](db))
			return true
		case Int16:
			convNum(win[uint8](sb), win[int16](db))
			return true
		case Int32:
			convNum(win[uint8](sb), win[int32](db))
			return true
		case Int64:
			convNum(win[uint8](sb), win[int64](db))
			return true
		case Float32:
			convNum(win[uint8](sb), win[float32](db))
			return true
		case Float64:
			convNum(win[uint8](sb), win[float64](db))
			return true
		}
	case Uint16:
		switch dk {
		case Uint32:
			convNum(win[uint16](sb), win[uint32](db))
			return true
		case Uint64:
			convNum(win[uint16](sb), win[uint64](db))
			return true
		case Uint8:
			convNum(win[uint16](sb), win[uint8](db))
			return true
		case Int32:
			convNum(win[uint16](sb), win[int32](db))
			return true
		case Float32:
			convNum(win[uint16](sb), win[float32](db))
			return true
		case Float64:
			convNum(win[uint16](sb), win[float64](db))
			return true
		}
	case Uint32:
		switch dk {
		case Uint64:
			convNum(win[uint32](sb), win[uint64](db))
			return true
		case Uint16:
			convNum(win[uint32](sb), win[uint16](db))
			return true
		case Uint8:
			convNum(win[uint32](sb), win[uint8](db))
			return true
		case Float64:
			convNum(win[uint32](sb), win[float64](db))
			return true
		}
	case Uint64:
		switch dk {
		case Uint32:
			convNum(win[uint64](sb), win[uint32](db))
			return true
		case Uint8:
			convNum(win[uint64](sb), win[uint8](db))
			return true
		case Float64:
			convNum(win[uint64](sb), win[float64](db))
			return true
		}
	}
	return false
}

// unsafeFloatConv 浮点 → 整型(unsafe 口径);返回 ok=false 表示目标类型未特化。
//
// 快慢两路:绝大多数值落在目标位宽内 → 直接截断转换(硬件级,无分支预测惩罚);
// 越界 / NaN / ±Inf 才走 unsafeFloatToIntBits 拿"符号位"口径的位型。
// 判界用半开区间 `lo <= v < hi`(hi 取 2^k,可被 float64 精确表示)。
func unsafeFloatConv(src []float64, dk Kind, db []byte) (bool, bool) {
	switch dk {
	case Int8:
		d := win[int8](db)
		for i, v := range src {
			if v >= -128 && v < 128 {
				d[i] = int8(v)
			} else {
				d[i] = int8(unsafeFloatToIntBits(Int8, v))
			}
		}
		return true, true
	case Int16:
		d := win[int16](db)
		for i, v := range src {
			if v >= -32768 && v < 32768 {
				d[i] = int16(v)
			} else {
				d[i] = int16(unsafeFloatToIntBits(Int16, v))
			}
		}
		return true, true
	case Int32:
		d := win[int32](db)
		for i, v := range src {
			if v >= -2147483648 && v < 2147483648 {
				d[i] = int32(v)
			} else {
				d[i] = int32(unsafeFloatToIntBits(Int32, v))
			}
		}
		return true, true
	case Int64:
		d := win[int64](db)
		for i, v := range src {
			if v >= -9223372036854775808.0 && v < 9223372036854775808.0 {
				d[i] = int64(v)
			} else {
				d[i] = int64(unsafeFloatToIntBits(Int64, v))
			}
		}
		return true, true
	case Uint8:
		d := win[uint8](db)
		for i, v := range src {
			if v >= 0 && v < 256 {
				d[i] = uint8(v)
			} else {
				d[i] = uint8(unsafeFloatToIntBits(Uint8, v))
			}
		}
		return true, true
	case Uint16:
		d := win[uint16](db)
		for i, v := range src {
			if v >= 0 && v < 65536 {
				d[i] = uint16(v)
			} else {
				d[i] = uint16(unsafeFloatToIntBits(Uint16, v))
			}
		}
		return true, true
	case Uint32:
		d := win[uint32](db)
		for i, v := range src {
			if v >= 0 && v < 4294967296 {
				d[i] = uint32(v)
			} else {
				d[i] = uint32(unsafeFloatToIntBits(Uint32, v))
			}
		}
		return true, true
	case Uint64:
		d := win[uint64](db)
		for i, v := range src {
			if v >= 0 && v < 18446744073709551616.0 {
				d[i] = uint64(v)
			} else {
				d[i] = unsafeFloatToIntBits(Uint64, v)
			}
		}
		return true, true
	}
	return false, false
}

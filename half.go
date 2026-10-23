package num

import "math"

// —— float16 位型转换(IEEE 754 binary16)——
//
// 所有口径由 numpy 2.5.4 实测锁定(codec_vectors_test.go 为现场生成的位型 reference):
//
// - 舍入 = round-to-nearest-even(RNE),且 **单次舍入**:float64 → float16 不经
// float32(实测 200 万个随机 double 中有 2 例双舍入分歧,如
// -210.31250032321557 直转 0xDA93、经 f32 得 0xDA92),故 f64/f32 各走本精度位算法。
// - 上溢:≥ 65520(= f16 最大值 65504 与 2^16 正中,tie-to-even 归 2^16)→ ±Inf;
// 65519 → 0x7BFF(numpy 实测同)。
// - NaN:sign | 0x7C00 | (源尾数高 10 位),该 10 位为 0 时置 1(不退化为 Inf)。
// 实测 0x7FF8000000000000 → 0x7E00、0x7FF0000000000001 → 0x7C01 ✓。
// - 次正规:用 24/53 位有效数 M 右移 s 位 + sticky 位 RNE;低于半格 → ±0
// (f32:s≥25 → 0;f64:s≥54 → 0)。
//
// 存取方向:f16 读出统一升 float32(Go 无 f16 原语),f16 → f32 精确无损。

// f16ToF32 binary16 位型 → float32(精确;含次正规规整化)。
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	e := uint32(h>>10) & 0x1F
	m := uint32(h & 0x3FF)
	switch e {
	case 0x1F: // Inf / NaN:Inf 平移指数;NaN 另置 quiet 位(0x400000,numpy 实测)
		if m == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7FC00000 | m<<13)
	case 0: // ±0 或次正规:值 = m·2^-24
		if m == 0 {
			return math.Float32frombits(sign)
		}
		k := uint(0) // m 最高位下标(0..9)
		for x := m; x > 1; x >>= 1 {
			k++
		}
		// 值 = 2^(k-24)·(1+f) → f32 指数 k-24+127 = k+103,尾数 (m-2^k)·2^(23-k)
		return math.Float32frombits(sign | uint32(k+103)<<23 | (m-(1<<k))<<(23-k))
	}
	return math.Float32frombits(sign | (e+112)<<23 | m<<13) // 112 = 127-15
}

// f32ToF16 float32 → binary16 位型(RNE)。
func f32ToF16(f float32) uint16 {
	u := math.Float32bits(f)
	sign := uint16(u>>16) & 0x8000
	if u&0x7FFFFFFF == 0 { // ±0
		return sign
	}
	e := int32(u>>23&0xFF) - 127
	m := u & 0x7FFFFF
	if e == 128 { // Inf / NaN
		if m == 0 {
			return sign | 0x7C00
		}
		n := uint16(m >> 13)
		if n == 0 {
			n = 1 // 防退化为 Inf(numpy 同款兜底)
		}
		return sign | 0x7C00 | n
	}
	if e > 15 { // 上溢(含舍入进位越界者见下)
		return sign | 0x7C00
	}
	if e >= -14 { // 正常数:取高 10 位尾数,低 13 位 RNE
		mt := m >> 13
		lo := m & 0x1FFF
		if lo > 0x1000 || (lo == 0x1000 && mt&1 == 1) {
			mt++
		}
		exp := uint32(e+15) << 10
		if mt>>10 != 0 { // 进位:尾数溢出 → 指数 +1
			exp += 1 << 10
			mt = 0
		}
		if exp >= 0x7C00 { // 进位至 Inf(如 65520 的 tie-to-even)
			return sign | 0x7C00
		}
		return sign | uint16(exp) | uint16(mt)
	}
	// 次正规:值 = M·2^(e-23),M = 1.m(24 位);n = 值/2^-24 = M·2^(e+1) → 右移 s 位
	M := uint32(1)<<23 | m
	s := uint32(-(e + 1))
	if s >= 25 { // 低于次正规半格 → ±0
		return sign
	}
	mt := M >> s
	lo := M & (uint32(1)<<s - 1)
	half := uint32(1) << (s - 1)
	if lo > half || (lo == half && mt&1 == 1) {
		mt++
	}
	if mt >= 0x400 { // 进位到最小正常数 2^-14(位型 exp=1,mant=0)
		return sign | 0x0400
	}
	return sign | uint16(mt)
}

// f64ToF16 float64 → binary16 位型(RNE;不经 float32,避免双舍入)。
func f64ToF16(f float64) uint16 {
	u := math.Float64bits(f)
	sign := uint16(u>>48) & 0x8000
	if u&0x7FFFFFFFFFFFFFFF == 0 { // ±0
		return sign
	}
	e := int32(u>>52&0x7FF) - 1023
	m := u & 0xFFFFFFFFFFFFF
	if e == 1024 { // Inf / NaN
		if m == 0 {
			return sign | 0x7C00
		}
		n := uint16(m >> 42)
		if n == 0 {
			n = 1
		}
		return sign | 0x7C00 | n
	}
	if e > 15 {
		return sign | 0x7C00
	}
	if e >= -14 { // 正常数:低 42 位 RNE
		mt := uint32(m >> 42)
		lo := m & (1<<42 - 1)
		if lo > 1<<41 || (lo == 1<<41 && mt&1 == 1) {
			mt++
		}
		exp := uint32(e+15) << 10
		if mt>>10 != 0 {
			exp += 1 << 10
			mt = 0
		}
		if exp >= 0x7C00 {
			return sign | 0x7C00
		}
		return sign | uint16(exp) | uint16(mt)
	}
	// 次正规:M 为 53 位有效数,值 = M·2^(e-52) → n = M·2^(e-28) → 右移 s = 28-e 位
	M := uint64(1)<<52 | m
	s := uint32(28 - e)
	if s >= 54 {
		return sign
	}
	mt := M >> s
	lo := M & (uint64(1)<<s - 1)
	half := uint64(1) << (s - 1)
	if lo > half || (lo == half && mt&1 == 1) {
		mt++
	}
	if mt >= 0x400 {
		return sign | 0x0400
	}
	return sign | uint16(mt)
}

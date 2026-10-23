package num

import (
	"math"
	"strconv"
	"strings"
)

// —— 数值 → 文本(numpy 2.5.4 实测口径;)——
//
// numpy 把数值写进 S/U 元素时按 Python str 口径落字,本库复刻该口径
// (codec_vectors_test.go 的 floatTextCases/complexTextCases 为 numpy 现场 reference):
//
//	100.0→"100.0" 1e16→"1e+16" 0.0001→"0.0001" 1e-05→"1e-05"
//	-0.0→"-0.0" inf→"inf" nan→"nan" True→"True" 1+2j→"(1+2j)"
//
// 与 Go 的 fmt %g / strconv 'g' 的三点差异,故自实现:
// 1. 定点/科学记数的切换线:Python/numpy 用"十进制指数 ∈ [-4, 阈值) 走定点",
// Go 的 'g' 在指数 ≥ 有效位数时就转科学记数(如 1e6 得 "1e+06",numpy 得 "1000000.0");
// 2. 定点整值补 ".0"(Python str(100.0) = "100.0";Go 给 "100");
// 3. inf/nan 小写、复数按 Python complex repr(纯虚省括号、分量不舍 ".0")。
//
// 记法判定分两档(穷举 f16 全部 65536 位型、f32 抽样 20 万位型实测一致):
// - f64(=CPython 口径):用最短位数的十进制指数 dec;科学 iff dec ≤ -5 || dec ≥ 16;
// - f16/f32(numpy dragon4 口径):记法用**值量级** floor(log10|v|)。二者只在一个
// 场景分叉:最短位数向上进位为 10^dec(位数 "1")而值略低于 10^dec —— 此时
// f16/f32 的量级比位数低一档(如 f32 0.0001 = 9.9999997e-05 → numpy "1e-04",
// 位数法会错判为定点 "0.0001"),故 dec 减 1;渲染仍用位数的指数。
// 阈值:f16 ≥ 3、f32 ≥ 6、f64 ≥ 16;下界统一 ≤ -5。
//
// 位数本身取"最短可回读"串:32/64 位由 strconv.FormatFloat(…, 'e', -1, bits) 给出
// (与 CPython 同为 shortest 算法);16 位 Go 无原生格式化,f16Shortest 按 p=1..5
// 搜索并处理 tie(见其注释)。精度按输入标量自身 dtype 取(f16 须与 f32 区分)。

// textSciThr 科学记数的上界阈值(f16/f32 值量级口径、f64 位数口径;下界统一 -5)。
func textSciThr(bits int) int {
	switch bits {
	case 16:
		return 3
	case 32:
		return 6
	}
	return 16
}

// pyFloatStr 数值 → Python str(float) 同构文本;bits ∈ {16,32,64} 表示输入精度。
func pyFloatStr(v float64, bits int) string {
	switch {
	case math.IsNaN(v):
		return "nan"
	case math.IsInf(v, 1):
		return "inf"
	case math.IsInf(v, -1):
		return "-inf"
	}
	neg := math.Signbit(v)
	a := math.Abs(v)

	var digits string
	var exp int
	if bits == 16 {
		digits, exp = f16Shortest(float32(a))
	} else {
		digits, exp = splitEDigits(strconv.FormatFloat(a, 'e', -1, bits))
	}

	dec := exp // 记法判定指数
	if bits != 64 && digits == "1" && a < math.Pow(10, float64(exp)) {
		dec = exp - 1 // 值量级比十进制进位低一档(见文件头说明)
	}
	negStr := ""
	if neg {
		negStr = "-"
	}
	if dec <= -5 || dec >= textSciThr(bits) { // 科学记数(指数用位数口径 exp)
		var b strings.Builder
		b.WriteString(negStr)
		b.WriteString(digits[:1])
		if len(digits) > 1 {
			b.WriteByte('.')
			b.WriteString(digits[1:])
		}
		b.WriteByte('e')
		b.WriteString(pyExp(exp))
		return b.String()
	}
	var b strings.Builder
	b.WriteString(negStr)
	switch {
	case exp >= 0:
		if len(digits) > exp+1 {
			b.WriteString(digits[:exp+1])
			b.WriteByte('.')
			b.WriteString(digits[exp+1:])
		} else {
			b.WriteString(digits)
			b.WriteString(strings.Repeat("0", exp+1-len(digits)))
			b.WriteString(".0")
		}
	default: // -4 ≤ exp ≤ -1
		b.WriteString("0.")
		b.WriteString(strings.Repeat("0", -exp-1))
		b.WriteString(digits)
	}
	return b.String()
}

// splitEDigits "1.2345e+07" → ("12345", 7);同时去掉尾随零(最短串本身不含)。
func splitEDigits(s string) (string, int) {
	mant, expPart, _ := strings.Cut(s, "e")
	e, _ := strconv.Atoi(expPart) // 不可达错误:FormatFloat 输出恒合法
	d := strings.Replace(mant, ".", "", 1)
	if t := strings.TrimRight(d, "0"); t != "" {
		d = t
	}
	return d, e
}

// parseF32 按 f32 舍入解析十进制串(与 strconv.ParseFloat 的 bitSize=32 同)。
func parseF32(s string) float32 {
	f, _ := strconv.ParseFloat(s, 32) // 不可达错误
	return float32(f)
}

// f16Shortest f16 值(经 float32 传入,精确)→ 最短可回读十进制位数与指数。
//
// 16 位精度 Go 无原生格式化,按位数 p=1..5 搜索:先取 RNE 候选,不可回读时再试
// ±1 末位(numpy 的 dragon4 在该 tie 场景朝远离零方向取位:f16 0.015625 →
// "1.563e-02",而 RNE 给 "1.562e-02";穷举 65536 个 f16 位型实测一致)。
func f16Shortest(v float32) (string, int) {
	h := f32ToF16(v)
	for p := 1; p <= 5; p++ {
		s := strconv.FormatFloat(float64(v), 'e', p-1, 32)
		digits, exp := splitEDigits(s)
		if f32ToF16(parseF32(s)) == h {
			return digits, exp
		}
		// p 位小数的粒度:10^(exp-p+1);在**舍入后的小数**上加/减一粒度
		step := math.Pow(10, float64(exp-len(digits)+1))
		base := parseF32(s)
		for _, sgn := range []float64{1, -1} {
			t := strconv.FormatFloat(float64(base)+sgn*step, 'e', p-1, 32)
			if f32ToF16(parseF32(t)) == h {
				d, e := splitEDigits(t)
				return d, e
			}
		}
	}
	// 不可达:f16 至多 5 位十进制即可唯一回读
	return splitEDigits(strconv.FormatFloat(float64(v), 'f', -1, 32))
}

// pyExp Python 指数后缀:带符号、至少两位(e+16 / e-05 / e+308)。
func pyExp(e int) string {
	sign := "+"
	if e < 0 {
		sign, e = "-", -e
	}
	d := strconv.Itoa(e)
	if len(d) < 2 {
		d = "0" + d
	}
	return sign + d
}

// pyFloatShort 同 pyFloatStr,但不补 ".0"(Python 复数分量的口径)。
func pyFloatShort(v float64, bits int) string {
	s := pyFloatStr(v, bits)
	if strings.HasSuffix(s, ".0") {
		return s[:len(s)-2]
	}
	return s
}

// pyComplexStr 复数 → Python complex repr 同构文本。
// 纯虚(re 为正零)省括号 → "1j";否则 "(re±imj)";分量按 pyFloatShort 落字。
func pyComplexStr(re, im float64, bits int) string {
	if re == 0 && !math.Signbit(re) {
		return pyFloatShort(im, bits) + "j"
	}
	sep := "+"
	if math.Signbit(im) {
		sep = "-"
	}
	return "(" + pyFloatShort(re, bits) + sep + pyFloatShort(math.Abs(im), bits) + "j)"
}

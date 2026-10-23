package num

// :datetime64(M8)算子与单位换算(B 拥有)。
//
// 语义基准(numpy 2.5.4 实测,):
// - M8 内部一律 int64 计数(单位内),NaT = MinInt64;
// - 算术:M8 + int → M8、int + M8 → M8、M8 - int → M8(单位不变);
// M8 - M8 → timedelta64(**DESIGN 明确不做** → ErrDtype);M8 + M8 / M8 ± float
// → numpy TypeError → ErrDtype;int - M8 → TypeError → ErrDtype;
// **NaT 参与算术 → NaT**(MinInt64 保持);
// - 比较:仅 datetime × datetime(单位不同按日历/比例归一到较细单位);
// NaT 参与有序比较恒 False、eq 恒 False、ne 恒 True;
// datetime × 非 datetime → numpy 静默 False,本库 fail-loud → ErrDtype(受控偏差);
// - 单位换算(≡ astype(datetime64[to])):固定比例单位(W/D/h/m/s/ms/us/ns)
// 乘除、细→粗向负无穷截断;Y/M 与更细单位走**日历换算**(numpy 同:
// 2020-01 → 18262 天);Y↔M 为 12 倍关系。

import (
	"fmt"
	"math"
)

// —— 单位表 ——

// dtUnitRank 单位秩(粗 → 细)。未知单位 → false。
func dtUnitRank(u string) (int, bool) {
	switch u {
	case "Y":
		return 0, true
	case "M":
		return 1, true
	case "W":
		return 2, true
	case "D":
		return 3, true
	case "h":
		return 4, true
	case "m":
		return 5, true
	case "s":
		return 6, true
	case "ms":
		return 7, true
	case "us":
		return 8, true
	case "ns":
		return 9, true
	}
	return 0, false
}

const dtNsPerDay = int64(86400) * 1e9

// dtNsFactor 单位相对 ns 的固定因子;Y/M 无固定长度 → false。
func dtNsFactor(u string) (int64, bool) {
	switch u {
	case "W":
		return 7 * dtNsPerDay, true
	case "D":
		return dtNsPerDay, true
	case "h":
		return 3600 * 1e9, true
	case "m":
		return 60 * 1e9, true
	case "s":
		return 1e9, true
	case "ms":
		return 1e6, true
	case "us":
		return 1e3, true
	case "ns":
		return 1, true
	}
	return 0, false
}

// —— 日历(Hinnant 算法,1970-01-01 = 0)——

func floorDiv64(a, b int64) int64 {
	q := a / b
	if a%b != 0 && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

func floorMod64(a, b int64) int64 { return a - floorDiv64(a, b)*b }

// daysFromCivil 公历年月日 → 天数(1970-01-01 为 0)。
func daysFromCivil(y, m, d int64) int64 {
	if m <= 2 {
		y--
	}
	era := floorDiv64(y, 400)
	yoe := y - era*400
	mp := m - 3
	if m <= 2 {
		mp = m + 9
	}
	doy := (153*mp+2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return era*146097 + doe - 719468
}

// civilFromDays 天数 → 公历年月日。
func civilFromDays(z int64) (y, m, d int64) {
	z += 719468
	era := floorDiv64(z, 146097)
	doe := z - era*146097
	yoe := (doe - doe/1460 + doe/36524 - doe/146096) / 365
	y = yoe + era*400
	doy := doe - (365*yoe + yoe/4 - yoe/100)
	mp := (5*doy + 2) / 153
	d = doy - (153*mp+2)/5 + 1
	if mp < 10 {
		m = mp + 3
	} else {
		m = mp - 9
	}
	if m <= 2 {
		y++
	}
	return
}

// dtToDays 任意单位计数 → 天数(Y/M 按日历月初,细单位向负无穷截断)。
func dtToDays(v int64, u string) (int64, error) {
	if u == "Y" || u == "M" {
		months := v
		if u == "Y" {
			months = v * 12
		}
		y := 1970 + floorDiv64(months, 12)
		mo := floorMod64(months, 12) + 1
		return daysFromCivil(y, mo, 1), nil
	}
	if u == "W" {
		return v * 7, nil
	}
	if u == "D" {
		return v, nil
	}
	f, ok := dtNsFactor(u)
	if !ok {
		return 0, fmt.Errorf("num: datetime64 单位 %q 不支持: %w", u, ErrDtype)
	}
	return floorDiv64(v, dtNsPerDay/f), nil
}

// dtFromDays 天数 → 目标单位计数。
func dtFromDays(days int64, to string) (int64, error) {
	if to == "Y" || to == "M" {
		y, m, _ := civilFromDays(days)
		months := (y-1970)*12 + (m - 1)
		if to == "Y" {
			return floorDiv64(months, 12), nil
		}
		return months, nil
	}
	if to == "W" {
		return floorDiv64(days, 7), nil
	}
	if to == "D" {
		return days, nil
	}
	f, ok := dtNsFactor(to)
	if !ok {
		return 0, fmt.Errorf("num: datetime64 单位 %q 不支持: %w", to, ErrDtype)
	}
	return days * (dtNsPerDay / f), nil
}

// dtConvertValue 单值单位换算(≡ numpy astype(datetime64[to]))。
func dtConvertValue(v int64, from, to string) (int64, error) {
	if from == to {
		return v, nil
	}
	fr, ok1 := dtUnitRank(from)
	tr, ok2 := dtUnitRank(to)
	if !ok1 || !ok2 {
		return 0, fmt.Errorf("num: datetime64 单位 %q/%q 不支持: %w", from, to, ErrDtype)
	}
	isYM := func(u string) bool { return u == "Y" || u == "M" }
	if isYM(from) || isYM(to) {
		days, err := dtToDays(v, from)
		if err != nil {
			return 0, err
		}
		return dtFromDays(days, to)
	}
	ff, _ := dtNsFactor(from)
	tf, _ := dtNsFactor(to)
	if fr < tr { // 粗 → 细:乘 (ns/from) / (ns/to) = ff/tf
		return v * (ff / tf), nil
	}
	return floorDiv64(v, tf/ff), nil // 细 → 粗:除 (ns/to)/(ns/from),向负无穷截断
}

// CastDateTime 把 datetime64 数组换算到目标单位(≡ astype(datetime64[unit]))。
func CastDateTime(a *Array, unit string) (*Array, error) {
	if a.Dtype().Kind != Datetime64 {
		return nil, fmt.Errorf("num: CastDateTime 需 datetime64,得 %s: %w", a.Dtype(), ErrDtype)
	}
	if _, ok := dtUnitRank(unit); !ok {
		return nil, fmt.Errorf("num: datetime64 单位 %q 非法: %w", unit, ErrDtype)
	}
	if a.Dtype().Unit == unit {
		return a, nil
	}
	out, err := New(a.Shape(), DTime(unit))
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	src := a.AsContiguous()
	s, d := win[int64](engineWindow(src)), win[int64](engineWindow(out))
	for i, v := range s {
		if v == math.MinInt64 { // NaT 保持
			d[i] = math.MinInt64
			continue
		}
		c, err := dtConvertValue(v, a.Dtype().Unit, unit)
		if err != nil {
			return nil, err
		}
		d[i] = c
	}
	return out, nil
}

// —— 算术 ——

// dtAdd M8 ± int(单位不变;NaT 传播)。
func dtAdd(x, y *Array, sub bool) (*Array, error) {
	xdt := x.Dtype().Kind == Datetime64
	ydt := y.Dtype().Kind == Datetime64
	if xdt && ydt {
		return nil, fmt.Errorf("num: datetime64 之间不支持加法(numpy TypeError): %w", ErrDtype)
	}
	if !xdt && !ydt {
		return nil, fmt.Errorf("num: 无 datetime64 操作数: %w", ErrDtype)
	}
	m8, iv := x, y
	if ydt {
		if sub {
			return nil, fmt.Errorf("num: int - datetime64 不支持(numpy TypeError): %w", ErrDtype)
		}
		m8, iv = y, x
	}
	if !isIntegerKind(iv.Dtype().Kind) {
		return nil, fmt.Errorf("num: datetime64 ± %s 不支持(须整型,numpy TypeError): %w", iv.Dtype(), ErrDtype)
	}
	outSh, err := broadcastShapes(x.Shape(), y.Shape())
	if err != nil {
		return nil, fmt.Errorf("num: datetime64 算术: %w", err)
	}
	dt := m8.Dtype()
	mc, err := transformCopy(m8, outSh, dt)
	if err != nil {
		return nil, err
	}
	ic, err := transformCopy(iv, outSh, DInt64)
	if err != nil {
		return nil, err
	}
	out, err := New(outSh, dt)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	mw, iw, ow := win[int64](engineWindow(mc)), win[int64](engineWindow(ic)), win[int64](engineWindow(out))
	for i := range ow {
		if mw[i] == math.MinInt64 { // NaT 传播
			ow[i] = math.MinInt64
			continue
		}
		if sub {
			ow[i] = mw[i] - iw[i]
		} else {
			ow[i] = mw[i] + iw[i]
		}
	}
	return out, nil
}

// —— 比较 ——

// dtCompare datetime × datetime(单位归一 + NaT 口径)。
func dtCompare(op string, x, y *Array) (*Array, error) {
	if x.Dtype().Kind != Datetime64 || y.Dtype().Kind != Datetime64 {
		return nil, fmt.Errorf("num: datetime64 与 %s 不可比较(numpy 静默 False,本库 fail-loud): %w",
			func() Dtype {
				if x.Dtype().Kind != Datetime64 {
					return x.Dtype()
				}
				return y.Dtype()
			}(), ErrDtype)
	}
	// 归一到较细单位(numpy 比较同:粗 → 细)
	rx, _ := dtUnitRank(x.Dtype().Unit)
	ry, _ := dtUnitRank(y.Dtype().Unit)
	unit := x.Dtype().Unit
	if ry > rx {
		unit = y.Dtype().Unit
	}
	cx, err := CastDateTime(x, unit)
	if err != nil {
		return nil, err
	}
	cy, err := CastDateTime(y, unit)
	if err != nil {
		return nil, err
	}
	outSh, err := broadcastShapes(cx.Shape(), cy.Shape())
	if err != nil {
		return nil, fmt.Errorf("num: datetime64 比较: %w", err)
	}
	a, err := transformCopy(cx, outSh, DTime(unit))
	if err != nil {
		return nil, err
	}
	b, err := transformCopy(cy, outSh, DTime(unit))
	if err != nil {
		return nil, err
	}
	out, err := New(outSh, DBool)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	aw, bw, ow := win[int64](engineWindow(a)), win[int64](engineWindow(b)), win[bool](engineWindow(out))
	nat := int64(math.MinInt64)
	for i := range ow {
		if aw[i] == nat || bw[i] == nat {
			// NaT:有序比较与 eq 恒 False,ne 恒 True
			ow[i] = op == "not_equal"
			continue
		}
		switch op {
		case "less":
			ow[i] = aw[i] < bw[i]
		case "less_equal":
			ow[i] = aw[i] <= bw[i]
		case "greater":
			ow[i] = aw[i] > bw[i]
		case "greater_equal":
			ow[i] = aw[i] >= bw[i]
		case "equal":
			ow[i] = aw[i] == bw[i]
		case "not_equal":
			ow[i] = aw[i] != bw[i]
		}
	}
	return out, nil
}

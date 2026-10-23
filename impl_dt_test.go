package num

// 单测(B):datetime64 算术 / 单位换算 / 比较(NaT 口径)。
// 期望值 = numpy 2.5.4 实测探针字面量。

import (
	"math"
	"testing"
)

func dtArr(t *testing.T, unit string, vals ...int64) *Array {
	t.Helper()
	a, err := New([]int{len(vals)}, DTime(unit))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vals {
		if err := a.SetItem([]int{i}, v); err != nil {
			t.Fatal(err)
		}
	}
	return a
}

func bWantDt(t *testing.T, a *Array, unit string, want ...int64) {
	t.Helper()
	if a.Dtype().Kind != Datetime64 || a.Dtype().Unit != unit {
		t.Fatalf("dtype = %s, want datetime64[%s]", a.Dtype(), unit)
	}
	if a.Size() != len(want) {
		t.Fatalf("size = %d, want %d", a.Size(), len(want))
	}
	for i, w := range want {
		if g := itemAt(t, a, i).(int64); g != w {
			t.Fatalf("[%d] = %d, want %d", i, g, w)
		}
	}
}

const (
	nat = int64(math.MinInt64)
	d1  = int64(18262) // 2020-01-01
	d2  = int64(18263) // 2020-01-02
)

// M8 ± int(单位不变;NaT 传播)。
func TestDtArith(t *testing.T) {
	d := dtArr(t, "D", d1, d2)
	out, err := Add(d, bArr(t, []int64{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", d1+1, d2+2)

	// int + M8(反序同值)
	out, err = Add(bArr(t, []int64{1, 2}), d)
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", d1+1, d2+2)

	// M8 - int
	out, err = Subtract(d, bArr(t, []int64{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", d1-1, d2-2)

	// NaT 传播
	n := dtArr(t, "D", nat, d1)
	out, err = Add(n, bArr(t, []int64{5, 5}))
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", nat, d1+5)
	out, err = Subtract(n, bArr(t, []int64{5, 5}))
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", nat, d1-5)

	// 单位保持(M8[s] + int → M8[s])
	sec := dtArr(t, "s", 1577836801)
	out, err = Add(sec, bArr(t, []int64{1}))
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "s", 1577836802)
}

// 非法组合(numpy TypeError 同构)。
func TestDtArithErrors(t *testing.T) {
	d := dtArr(t, "D", d1)
	if _, err := Add(d, d); rootOf(err) != ErrDtype {
		t.Fatalf("M8+M8 err = %v, want ErrDtype", err)
	}
	if _, err := Subtract(d, d); rootOf(err) != ErrDtype {
		t.Fatalf("M8-M8(timedelta 不做)err = %v, want ErrDtype", err)
	}
	if _, err := Subtract(bArr(t, []int64{1}), d); rootOf(err) != ErrDtype {
		t.Fatalf("int-M8 err = %v, want ErrDtype", err)
	}
	if _, err := Add(d, bArr(t, []float64{1})); rootOf(err) != ErrDtype {
		t.Fatalf("M8+float err = %v, want ErrDtype", err)
	}
}

// 单位换算(≡ astype)。
func TestDtCast(t *testing.T) {
	// s → ms(×1000)
	s := dtArr(t, "s", 1577836801)
	out, err := CastDateTime(s, "ms")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "ms", 1577836801000)

	// ms → s(截断)
	ms := dtArr(t, "ms", 1577836801500)
	out, err = CastDateTime(ms, "s")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "s", 1577836801)

	// D → s(×86400)
	dd := dtArr(t, "D", d1)
	out, err = CastDateTime(dd, "s")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "s", 1577836800)

	// M → D(日历:2020-01 → 18262)
	mo := dtArr(t, "M", 600)
	out, err = CastDateTime(mo, "D")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", 18262)

	// Y → M(×12)
	yy := dtArr(t, "Y", 50)
	out, err = CastDateTime(yy, "M")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "M", 600)

	// W → D(×7)
	w := dtArr(t, "W", 2608)
	out, err = CastDateTime(w, "D")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "D", 18256)

	// us → ns / ns → us
	us := dtArr(t, "us", 1577836800000001)
	out, err = CastDateTime(us, "ns")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "ns", 1577836800000001000)
	out, err = CastDateTime(out, "us")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "us", 1577836800000001)

	// NaT 保持
	n := dtArr(t, "D", nat)
	out, err = CastDateTime(n, "s")
	if err != nil {
		t.Fatal(err)
	}
	bWantDt(t, out, "s", nat)

	// 非 datetime / 非法单位
	if _, err := CastDateTime(bArr(t, []int64{1}), "s"); rootOf(err) != ErrDtype {
		t.Fatalf("非 datetime err = %v", err)
	}
	if _, err := CastDateTime(dd, "bad"); rootOf(err) != ErrDtype {
		t.Fatalf("非法单位 err = %v", err)
	}
}

// 比较:NaT 口径 + 跨单位归一。
func TestDtCmp(t *testing.T) {
	d := dtArr(t, "D", d1, d2)
	n := dtArr(t, "D", nat, d2)

	// NaT:有序恒 False、eq 恒 False、ne 恒 True
	out, err := Equal(d, n)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, true)
	out, _ = NotEqual(d, n)
	bWantBool(t, out, true, false)
	out, _ = Less(n, d)
	bWantBool(t, out, false, false)
	out, _ = GreaterEqual(n, d)
	bWantBool(t, out, false, true)

	// 跨单位:M8[M] 600(2020-01)== M8[D] 18262(2020-01-01)
	mo := dtArr(t, "M", 600)
	out, err = Equal(mo, dtArr(t, "D", d1))
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true)

	// 与整数比较:fail-loud(numpy 静默 False,受控偏差)
	if _, err := Equal(d, bArr(t, []int64{18262, 18263})); rootOf(err) != ErrDtype {
		t.Fatalf("M8 vs int err = %v, want ErrDtype", err)
	}
}

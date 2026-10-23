package num

import (
	"testing"
)

// bench 基线。1e6 元素量级,标定引擎分发开销与工厂/IO 吞吐。
// - Add 走 _test_add 内核(ufunc_test.go 注册);引擎路径与 B 的正式 impl_*.go 内核
// 完全同构(同签名同分发),故数字可直接当正式算子的基线。
// - Sum/Take 已接线:Sum 走 B 的 pairwise.go 正式实现,Take 走 impl_gather.go 正式实现。
// - 校准命令:go test -run=^$ -bench=. -benchmem.

const benchN = 1 << 20 // 1e6

func benchPair(b *testing.B) (*Array, *Array) {
	b.Helper()
	x, err := Arange(0, benchN, 1, DFloat64)
	if err != nil {
		b.Fatalf("Arange: %v", err)
	}
	y, err := Arange(0, benchN, 1, DFloat64)
	if err != nil {
		b.Fatalf("Arange: %v", err)
	}
	return x, y
}

// 引擎全路径:广播判定 + promote + 归一直通 + 内核 + 输出分配。
func BenchmarkAddF64Contig1e6(b *testing.B) {
	x, y := benchPair(b)
	b.SetBytes(int64(benchN) * 8 * 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dispatchBinary("_test_add", x, y); err != nil {
			b.Fatal(err)
		}
	}
}

// 纯 Go 逐元素基线(同一 f64 zip-add,无引擎开销)。
func BenchmarkAddF64RawLoop1e6(b *testing.B) {
	dst := make([]float64, benchN)
	a := make([]float64, benchN)
	c := make([]float64, benchN)
	for i := range a {
		a[i] = float64(i)
		c[i] = 1
	}
	b.SetBytes(int64(benchN) * 8 * 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range dst {
			dst[j] = a[j] + c[j]
		}
	}
}

// 广播路径:标量(0 维)加到 1e6 向量 → 一侧走 transformCopy 整备。
func BenchmarkAddF64BroadcastScalar1e6(b *testing.B) {
	x, _ := benchPair(b)
	sc, err := Full([]int{}, 1.0, DFloat64)
	if err != nil {
		b.Fatalf("Full: %v", err)
	}
	b.SetBytes(int64(benchN) * 8 * 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dispatchBinary("_test_add", x, sc); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkArangeF64_1e6(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Arange(0, benchN, 1, DFloat64); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLinSpaceF64_1e6(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LinSpace(0, 1, benchN, true); err != nil {
			b.Fatal(err)
		}
	}
}

// npy 装载(v1 头 + 1e6 f64 = 8MB 体)。
func BenchmarkLoadNPYF64_1e6(b *testing.B) {
	a, err := Arange(0, benchN, 1, DFloat64)
	if err != nil {
		b.Fatalf("Arange: %v", err)
	}
	raw, err := SaveNPYBytes(a)
	if err != nil {
		b.Fatalf("SaveNPYBytes: %v", err)
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadNPYBytes(raw); err != nil {
			b.Fatal(err)
		}
	}
}

// 非连续 → 连续整备(1000×1000 转置,copyStrided 路径)。
func BenchmarkAsContiguousTransposed1e6(b *testing.B) {
	a, err := New([]int{1000, 1000}, DFloat64)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	tr, err := a.Transpose()
	if err != nil {
		b.Fatalf("Transpose: %v", err)
	}
	b.SetBytes(int64(benchN) * 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if tr.AsContiguous() == nil {
			b.Fatal("AsContiguous nil")
		}
	}
}

// Sum 全归约:axis=nil → 标量(0 维)结果,走 B 的 pairwise.go 正式实现。
func BenchmarkSumF64_1e6(b *testing.B) {
	x, _ := benchPair(b)
	b.SetBytes(int64(benchN) * 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Sum(x, nil, false); err != nil {
			b.Fatal(err)
		}
	}
}

// Take:1e6 f64 源 + int64 索引全量 gather,走 B 的 impl_gather.go 正式实现。
func BenchmarkTakeF64_1e6(b *testing.B) {
	src, _ := benchPair(b)
	idx, err := Arange(0, benchN, 1, DInt64)
	if err != nil {
		b.Fatalf("Arange: %v", err)
	}
	b.SetBytes(int64(benchN) * 8 * 2)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Take(src, idx); err != nil {
			b.Fatal(err)
		}
	}
}

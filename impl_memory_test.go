package num

import (
	"encoding/binary"
	"math"
	"runtime"
	"testing"
)

// 内存所有权语义:这些测试把 README「并发、内存与资源」一节的承诺钉住。

// FromBuffer 是零拷贝:调用方字节切片里改一个值,数组立刻看到。
func TestFromBufferIsZeroCopy(t *testing.T) {
	buf := make([]byte, 8*3)
	for i, v := range []float64{1, 2, 3} {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	a, err := FromBuffer(buf, DFloat64, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	// 改字节 → 数组应看到(零拷贝)
	binary.LittleEndian.PutUint64(buf[0:], math.Float64bits(99))
	if v, _ := a.Item(0); v != 99.0 {
		t.Fatalf("FromBuffer 未共享内存: %v, want 99", v)
	}
	// 反向:写数组 → 字节应变
	if err := a.SetItem([]int{1}, 42.0); err != nil {
		t.Fatal(err)
	}
	if got := math.Float64frombits(binary.LittleEndian.Uint64(buf[8:])); got != 42.0 {
		t.Fatalf("数组写入未反映到字节: %v", got)
	}
}

// Copy / AsContiguous 断开别名:改副本不影响源。
func TestCopyBreaksAliasing(t *testing.T) {
	src := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	tr, err := src.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	cont := tr.AsContiguous() // 落连续 = 新内存
	if err := cont.SetItem([]int{0, 0}, 77.0); err != nil {
		t.Fatal(err)
	}
	if v, _ := src.Item(0, 0); v != 1.0 {
		t.Fatalf("AsContiguous 未断开别名: src[0,0] = %v", v)
	}
	cp := Copy(src)
	if err := cp.SetItem([]int{0, 0}, 55.0); err != nil {
		t.Fatal(err)
	}
	if v, _ := src.Item(0, 0); v != 1.0 {
		t.Fatalf("Copy 未断开别名: src[0,0] = %v", v)
	}
}

// 结果数组与输入不共享内存(除显式视图外):改结果不影响输入。
func TestResultsOwnTheirMemory(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3})
	b := bArr(t, []float64{10, 20, 30})
	s, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetItem([]int{0}, -1.0); err != nil {
		t.Fatal(err)
	}
	if v, _ := a.Item(0); v != 1.0 {
		t.Fatalf("结果与输入共享内存: a[0] = %v", v)
	}
	// 归约/统计结果同样独立
	sum, err := Sum(a, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := sum.SetItem([]int{}, -1.0); err != nil {
		t.Fatal(err)
	}
	if v, _ := a.Item(0); v != 1.0 {
		t.Fatalf("归约结果与输入共享内存: a[0] = %v", v)
	}
}

// 反复分配/释放后堆不应持续增长(没有隐藏的全局缓存)。
func TestNoUnboundedGlobalGrowth(t *testing.T) {
	if testing.Short() {
		t.Skip("堆增长检查;`go test -short` 跳过")
	}
	// 预热,让运行时把初始堆/缓存稳定下来
	for i := 0; i < 50; i++ {
		a := mustMatShape(t, 32, 32)
		if _, err := MatMul(a, a); err != nil {
			t.Fatal(err)
		}
	}
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	for i := 0; i < 300; i++ {
		a := mustMatShape(t, 64, 64)
		s, err := MatMul(a, a)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Sum(s, nil, false); err != nil {
			t.Fatal(err)
		}
		if _, err := Sort(s, []int{0}); err != nil {
			t.Fatal(err)
		}
		if _, err := Svdvals(a); err != nil {
			t.Fatal(err)
		}
	}

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	// HeapInuse 在 GC 后应回到同一量级;放宽到 32 MiB 容差,只抓"泄漏式增长"。
	growth := int64(after.HeapInuse) - int64(before.HeapInuse)
	if growth > 32<<20 {
		t.Fatalf("GC 后堆增长 %d MiB,疑似全局状态持有内存", growth>>20)
	}
	// 包级注册表规模必须恒定(没有按调用累积)
	size := 0
	for k := range promoteIdx {
		_ = k
		size++
	}
	if size != len(promoteOrder) {
		t.Fatalf("promoteIdx 条目数 %d ≠ %d(全局表被改写)", size, len(promoteOrder))
	}
}

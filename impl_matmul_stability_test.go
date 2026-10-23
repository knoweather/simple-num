package num

import (
	"runtime"
	"testing"
)

// TestMatMulParallelStability 并行路径的稳定性:
// 同一对输入重复计算多次,结果必须**逐位相同**;若并行写存在重叠/竞争,
// 通常会出现结果随运行变化(本机无 gcc,-race 不可用,用这个做等效兜底)。
//
// 阈值 128³ = 2^21 正好落在并行分支上;同时切换 GOMAXPROCS 改变 worker 数量,
// 覆盖不同分块边界。
func TestMatMulParallelStability(t *testing.T) {
	const n = 128
	a := mustMat(t, []int{n, n}, make([]float64, n*n))
	b := mustMat(t, []int{n, n}, make([]float64, n*n))
	// 填充可直接判定的数据:结果第 (i,j) 项 = Σ_p (i*n+p) * (p*n+j)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if err := a.SetItem([]int{i, j}, float64(i*n+j)); err != nil {
				t.Fatal(err)
			}
			if err := b.SetItem([]int{i, j}, float64(pIdx(i, j))); err != nil {
				t.Fatal(err)
			}
		}
	}
	ref, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	refF := ref.AsContiguous().Flatten()
	refBytes := make([]byte, len(engineWindow(refF)))
	copy(refBytes, engineWindow(refF))

	prev := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(prev)
	for _, procs := range []int{1, 2, 4, runtime.NumCPU()} {
		if procs < 1 {
			continue
		}
		runtime.GOMAXPROCS(procs)
		for trial := 0; trial < 3; trial++ {
			got, err := MatMul(a, b)
			if err != nil {
				t.Fatalf("GOMAXPROCS=%d trial=%d: %v", procs, trial, err)
			}
			g := got.AsContiguous().Flatten()
			gb := engineWindow(g)
			if len(gb) != len(refBytes) {
				t.Fatalf("结果长度不一致")
			}
			for i := range refBytes {
				if gb[i] != refBytes[i] {
					t.Fatalf("GOMAXPROCS=%d trial=%d: 第 %d 字节不同(并行结果不稳定)", procs, trial, i)
				}
			}
		}
	}
}

// pIdx 构造第二个矩阵元素(让结果有区分度,但不至于溢出)。
func pIdx(i, j int) int {
	return (i*31 + j*17) % 1009
}

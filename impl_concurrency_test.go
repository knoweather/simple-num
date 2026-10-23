package num

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// TestConcurrentIndependentOps 并发调用互不相干的算子:
// 包级注册表在 init 后只读,每个 goroutine 用自己的数组,结果必须与串行一致。
func TestConcurrentIndependentOps(t *testing.T) {
	const workers = 8
	const iters = 200
	var wg sync.WaitGroup
	errs := make([]error, workers)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(id int) {
			defer wg.Done()
			for k := 0; k < iters; k++ {
				a, err := FromSlice([]float64{float64(id), float64(k), 3, 4})
				if err != nil {
					errs[id] = err
					return
				}
				b, err := FromSlice([]float64{2, 3, 4, 5})
				if err != nil {
					errs[id] = err
					return
				}
				s, err := Add(a, b)
				if err != nil {
					errs[id] = err
					return
				}
				// 结果自检:第一项 = id+2
				v, err := s.Item(0)
				if err != nil {
					errs[id] = err
					return
				}
				if v != float64(id)+2 {
					errs[id] = fmt.Errorf("worker %d iter %d: got %v, want %v", id, k, v, float64(id)+2)
					return
				}
				if _, err := Sum(s, nil, false); err != nil {
					errs[id] = err
					return
				}
				if _, err := MatMul(
					mustMatShape(t, 2, 2), mustMatShape(t, 2, 2)); err != nil {
					errs[id] = err
					return
				}
				if _, err := Einsum("ij,jk->ik", mustMatShape(t, 2, 2), mustMatShape(t, 2, 2)); err != nil {
					errs[id] = err
					return
				}
			}
		}(w)
	}
	wg.Wait()
	for id, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", id, err)
		}
	}
}

// mustMatShape 造一个 n×n 的常量数组(并发测试内部用,失败即 panic 以便快速定位)。
func mustMatShape(t *testing.T, n, m int) *Array {
	t.Helper()
	vals := make([]float64, n*m)
	for i := range vals {
		vals[i] = float64(i%7) + 1
	}
	a, err := FromSlice(vals)
	if err != nil {
		t.Fatal(err)
	}
	r, err := a.Reshape([]int{n, m})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestConcurrentReadSameArray 多个 goroutine 同时**只读**同一个数组:
// 结果必须一致,且数组内容不得被改变。
func TestConcurrentReadSameArray(t *testing.T) {
	const workers = 8
	a := mustMat(t, []int{4, 5}, []float64{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
		16, 17, 18, 19, 20,
	})
	before := make([]float64, a.Size())
	copy(before, win[float64](engineWindow(a.AsContiguous())))

	wantSum := 0.0
	for _, v := range before {
		wantSum += v
	}

	var wg sync.WaitGroup
	errs := make([]error, workers)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(id int) {
			defer wg.Done()
			for k := 0; k < 100; k++ {
				s, err := Sum(a, nil, false)
				if err != nil {
					errs[id] = err
					return
				}
				if v, _ := s.Item(); v != wantSum {
					errs[id] = fmt.Errorf("并发读 sum = %v, want %v", v, wantSum)
					return
				}
				// 排序 / 转置 / 切片都应只读
				if _, err := Sort(a, []int{1}); err != nil {
					errs[id] = err
					return
				}
				tr, err := a.Transpose()
				if err != nil {
					errs[id] = err
					return
				}
				if _, err := tr.Item(0, 0); err != nil {
					errs[id] = err
					return
				}
				if _, err := a.Sub(Slice(1, 3), All()); err != nil {
					errs[id] = err
					return
				}
			}
		}(w)
	}
	wg.Wait()
	for id, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", id, err)
		}
	}
	after := win[float64](engineWindow(a.AsContiguous()))
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("并发只读把数组改了:[%d] %v -> %v", i, before[i], after[i])
		}
	}
}

// TestNoFileHandleLeak 反复读写各格式文件:
// 若某个路径漏关文件,Windows 会在句柄上限(约 8192)处报 "too many open files"。
// 这里刻意跑 20000 轮,远超该上限。
func TestNoFileHandleLeak(t *testing.T) {
	if testing.Short() {
		t.Skip("20000 轮 I/O 往返;`go test -short` 跳过")
	}
	dir := t.TempDir()
	npy := filepath.Join(dir, "a.npy")
	npz := filepath.Join(dir, "a.npz")
	txt := filepath.Join(dir, "a.txt")

	small := bArr(t, []float64{1, 2, 3})
	if err := SaveNPY(npy, small); err != nil {
		t.Fatal(err)
	}
	if err := SaveZ(npz, []string{"x"}, []*Array{small}); err != nil {
		t.Fatal(err)
	}
	if err := SaveTxt(txt, small, "%.3f", " "); err != nil {
		t.Fatal(err)
	}

	const rounds = 20000
	for i := 0; i < rounds; i++ {
		if _, err := LoadNPY(npy); err != nil {
			t.Fatalf("第 %d 轮 LoadNPY: %v", i, err)
		}
		if _, err := LoadZ(npz); err != nil {
			t.Fatalf("第 %d 轮 LoadZ: %v", i, err)
		}
		if _, err := LoadTxt(txt, DFloat64, " ", 0); err != nil {
			t.Fatalf("第 %d 轮 LoadTxt: %v", i, err)
		}
		// 写路径同样要关句柄
		if i%(rounds/20) == 0 {
			if err := SaveNPY(npy, small); err != nil {
				t.Fatalf("第 %d 轮 SaveNPY: %v", i, err)
			}
			if err := SaveZ(npz, []string{"x"}, []*Array{small}); err != nil {
				t.Fatalf("第 %d 轮 SaveZ: %v", i, err)
			}
			if err := SaveTxt(txt, small, "%.3f", " "); err != nil {
				t.Fatalf("第 %d 轮 SaveTxt: %v", i, err)
			}
		}
	}
	// 收尾:临时目录可删说明没有残留句柄占用(Windows 上被占用的文件删不掉)
	if err := os.Remove(npy); err != nil {
		t.Fatalf("删除 npy 失败(可能仍有句柄未关): %v", err)
	}
}

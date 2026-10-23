package num

import (
	"math"
	"testing"
)

// 本文件的所有期望值都是 **numpy 2.5.4 现场运行的快照**
// (np.random.RandomState(seed) 的 legacy MT19937 路径),用于逐位验证兼容性。

// RandomSample:三个种子的 rand 序列。
func TestRandomSampleNumpyVectors(t *testing.T) {
	cases := []struct {
		seed int64
		n    int
		want []float64
	}{
		{0, 5, []float64{0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047}},
		{42, 5, []float64{0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652}},
		{12345, 5, []float64{0.9296160928171479, 0.3163755545817859, 0.18391881167709445, 0.2045602785530397, 0.5677250290816866}},
	}
	for _, c := range cases {
		r := NewRandomState(c.seed)
		got, err := r.RandomSample(c.n)
		if err != nil {
			t.Fatal(err)
		}
		if got.Dtype() != DFloat64 {
			t.Fatalf("dtype %s", got.Dtype())
		}
		bWantF64(t, got, c.want...)
	}
}

// StandardNormal:randn 必须逐位一致(含极坐标法的 spare 缓存)。
func TestStandardNormalNumpyVectors(t *testing.T) {
	r := NewRandomState(42)
	got, err := r.StandardNormal(5)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, got,
		0.4967141530112327, -0.13826430117118466, 0.6476885381006925,
		1.5230298564080254, -0.23415337472333597)

	r = NewRandomState(7)
	got, err = r.StandardNormal(4)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, got,
		1.690525703800356, -0.4659373705408328, 0.0328201636785844, 0.40751628299650783)
}

// RandInt:掩码拒绝采样 + 负区间 + 大范围(int64 路径)。
func TestRandIntNumpyVectors(t *testing.T) {
	r := NewRandomState(42)
	got, err := r.RandInt(0, 100, 5)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dtype() != DInt64 {
		t.Fatalf("dtype %s, want int64", got.Dtype())
	}
	bWantI64(t, got, 51, 92, 14, 71, 60)

	r = NewRandomState(7)
	got, err = r.RandInt(0, 10, 8)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, got, 4, 9, 6, 3, 3, 7, 7, 9)

	// 负区间
	r = NewRandomState(42)
	got, err = r.RandInt(-5, 5, 4)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, got, 1, -2, 2, -1)

	// 大范围(int64 路径)
	r = NewRandomState(42)
	big, err := r.RandInt(0, 1<<40, 3)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, big, 441507790259, 395924837646, 458615280711)

	// high <= low → ErrValue
	if _, err := r.RandInt(5, 5, 1); err == nil {
		t.Fatal("high == low 应报错")
	}
}

// Permutation / Shuffle:Fisher-Yates 与 numpy 同序。
func TestPermutationShuffle(t *testing.T) {
	r := NewRandomState(42)
	p, err := r.Permutation(10)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, p, 8, 1, 5, 0, 7, 2, 9, 4, 3, 6)

	// shuffle 与 permutation 在 numpy 里同一条 Fisher-Yates 路径
	r = NewRandomState(42)
	a, err := New([]int{10}, DInt64)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		if err := a.SetItem([]int{i}, int64(i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.Shuffle(a); err != nil {
		t.Fatal(err)
	}
	bWantI64(t, a, 8, 1, 5, 0, 7, 2, 9, 4, 3, 6)

	// 二维:沿第 0 轴整体搬移(行内容保持)
	r = NewRandomState(42)
	m, err := New([]int{3, 2}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		if err := m.SetItem([]int{i / 2, i % 2}, float64(i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.Shuffle(m); err != nil {
		t.Fatal(err)
	}
	// 每行必须是原本的 [0,1]/[2,3]/[4,5] 之一
	seen := map[float64]bool{}
	for i := 0; i < 3; i++ {
		v0, _ := m.Item(i, 0)
		v1, _ := m.Item(i, 1)
		f0, f1 := v0.(float64), v1.(float64)
		if f1 != f0+1 || math.Mod(f0, 2) != 0 {
			t.Fatalf("第 %d 行破坏了行内结构: %v %v", i, f0, f1)
		}
		seen[f0] = true
	}
	if len(seen) != 3 {
		t.Fatalf("打乱后行内容应互不相同,得 %v", seen)
	}
}

// Uniform / Normal:线性变换口径。
func TestUniformNormal(t *testing.T) {
	r := NewRandomState(42)
	u, err := r.Uniform(-1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, u, -0.250919762305275, 0.9014286128198323, 0.4639878836228102)

	r = NewRandomState(42)
	n, err := r.Normal(10, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, n, 10.993428306022466, 9.723471397657631, 11.295377076201385)
}

// Seed 重置后序列可重放。
func TestRandomSeedReplay(t *testing.T) {
	r := NewRandomState(42)
	first, err := r.RandomSample(3)
	if err != nil {
		t.Fatal(err)
	}
	_ = r
	r.Seed(42)
	again, err := r.RandomSample(3)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		a, _ := first.Item(i)
		b, _ := again.Item(i)
		if a != b {
			t.Fatalf("[%d] 重放不一致: %v vs %v", i, a, b)
		}
	}
	// 空 size
	if o, err := r.RandomSample(0); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("size 0 应为空,得 %d", o.Size())
	}
}

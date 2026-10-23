package num

// 单测(B):pairwise sum 与 numpy 逐位对拍。
// 证据来源:xcheck/reference/cases/sum_*.npy(gen_reference.py 用 numpy 2.5.4 生成,
// expect 为 0-d npy);本测试用库自带 LoadNPY 装载后直调内核比位型。
// 覆盖:n<8 朴素(初值 -0.0)、n=8/17/128 单块、n=129 折半、n=1000/100003
// 深递归、全 -0 输入、f32 累加器。

import (
	"math"
	"testing"
)

// flatF64 把 1 维 f64 数组摊平为 []float64(测试辅助;经公共 Item 接口)。
func flatF64(t *testing.T, a *Array) []float64 {
	t.Helper()
	if a.NDim() != 1 {
		t.Fatalf("flatF64 需 1 维,得 %v", a.Shape())
	}
	out := make([]float64, a.Size())
	for i := range out {
		out[i] = getI(t, a, i).(float64)
	}
	return out
}

func flatF32(t *testing.T, a *Array) []float32 {
	t.Helper()
	if a.NDim() != 1 {
		t.Fatalf("flatF32 需 1 维,得 %v", a.Shape())
	}
	out := make([]float32, a.Size())
	for i := range out {
		out[i] = getI(t, a, i).(float32)
	}
	return out
}

// scalarBits 取 0-d expect 的 f64 位型。
func scalarF64(t *testing.T, a *Array) float64 {
	t.Helper()
	if a.NDim() != 0 {
		t.Fatalf("expect 应为 0-d,得 %v", a.Shape())
	}
	v, err := a.Item()
	if err != nil {
		t.Fatalf("Item(): %v", err)
	}
	return v.(float64)
}

func scalarF32(t *testing.T, a *Array) float32 {
	t.Helper()
	if a.NDim() != 0 {
		t.Fatalf("expect 应为 0-d,得 %v", a.Shape())
	}
	v, err := a.Item()
	if err != nil {
		t.Fatalf("Item(): %v", err)
	}
	return v.(float32)
}

// TestPairwiseRef 逐案对拍(bitwise)。
func TestPairwiseRef(t *testing.T) {
	cases := []struct {
		name string
		f32  bool
	}{
		{"sum_f8_naive3", false},
		{"sum_f8_seven", false},
		{"sum_f8_eight", false},
		{"sum_f8_17", false},
		{"sum_f8_128", false},
		{"sum_f8_129", false},
		{"sum_f8_1000", false},
		{"sum_f8_big", false},
		{"sum_f8_negzero", false},
		{"sum_f4_128", true},
		{"sum_f4_1000", true},
		{"sum_f4_negzero", true},
	}
	for _, c := range cases {
		in, err := LoadNPY("xcheck/reference/cases/" + c.name + "_a.npy")
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		ex, err := LoadNPY("xcheck/reference/cases/" + c.name + "_out.npy")
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if c.f32 {
			got := sumPairwiseF32(flatF32(t, in))
			want := scalarF32(t, ex)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("%s: got %v (%#x), want %v (%#x)",
					c.name, got, math.Float32bits(got), want, math.Float32bits(want))
			}
			continue
		}
		got := sumPairwiseF64(flatF64(t, in))
		want := scalarF64(t, ex)
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("%s: got %v (%#x), want %v (%#x)",
				c.name, got, math.Float64bits(got), want, math.Float64bits(want))
		}
	}
}

// TestPairwiseKernelEdges 内核级边角(np.sum 口径,含 +0.0 恒等元)。
func TestPairwiseKernelEdges(t *testing.T) {
	// 空输入:np.sum(空) = +0.0(恒等元;C 函数本身给 -0.0,)
	if g := sumPairwiseF64(nil); math.Signbit(g) || g != 0 {
		t.Fatalf("sumPairwiseF64(nil) = %v, want +0.0", g)
	}
	// 全 -0:np.sum 实测 n=1..129 全为 +0.0(恒等元 0.0 + -0.0 = +0.0)
	if g := sumPairwiseF64([]float64{math.Copysign(0, -1), math.Copysign(0, -1)}); math.Signbit(g) {
		t.Fatalf("全 -0 求和 = %v, want +0.0", g)
	}
	// +0 与 -0 混合
	if g := sumPairwiseF64([]float64{0, math.Copysign(0, -1)}); math.Signbit(g) {
		t.Fatalf("0 + -0 = %v, want +0.0", g)
	}
	// 基本值
	if g := sumPairwiseF64([]float64{1, 2, 3, 4}); g != 10 {
		t.Fatalf("sum = %v, want 10", g)
	}
	if g := sumPairwiseF32([]float32{0.5, 0.25}); g != 0.75 {
		t.Fatalf("f32 sum = %v, want 0.75", g)
	}
	// NaN 传播
	if g := sumPairwiseF64([]float64{1, math.NaN()}); !math.IsNaN(g) {
		t.Fatalf("含 NaN 求和 = %v, want NaN", g)
	}
}

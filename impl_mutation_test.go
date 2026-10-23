package num

import (
	"math"
	"testing"
)

// TestNoInputMutation 跨模块回归:这些函数都不应该修改调用方的数组。
//
// 历史上踩过多次(method 在末轴时改到调用方、setFlatFloats 写进 Flatten 副本等),
// 这里一次性把常用"只读"函数跑一遍,前后逐字节比对。
func TestNoInputMutation(t *testing.T) {
	// 用 f64 且刻意含正负与零,便于发现任何位置被写坏
	mk := func() *Array {
		return mustMat(t, []int{3, 3}, []float64{4, -1, 2, 0, 5, 7, 2, 8, -6})
	}
	snapshot := func(a *Array) []float64 {
		flat := a.AsContiguous().Flatten()
		out := make([]float64, flat.Size())
		w := win[float64](engineWindow(flat))
		copy(out, w)
		return out
	}
	changed := func(name string, before, after []float64) {
		t.Helper()
		if len(before) != len(after) {
			t.Fatalf("%s: 长度变了 %d -> %d", name, len(before), len(after))
		}
		for i := range before {
			if before[i] != after[i] {
				t.Fatalf("%s: 入参被篡改 [%d] %v -> %v", name, i, before[i], after[i])
			}
		}
	}

	sq := mk()
	vec := bArr(t, []float64{3, -1, 4, 1, 5, -9, 2, 6})

	ops := []struct {
		name string
		run  func() error
	}{
		{"sort", func() error { _, err := Sort(sq, []int{1}); return err }},
		{"argsort", func() error { _, err := Argsort(sq, []int{1}); return err }},
		{"median", func() error { _, err := Median(sq, []int{0}, false); return err }},
		{"percentile", func() error { _, err := Percentile(sq, 40, []int{1}, false); return err }},
		{"percentile_many", func() error { _, err := PercentileMany(sq, []float64{10, 90}, []int{0}, false); return err }},
		{"quantile", func() error { _, err := Quantile(sq, 0.3, []int{1}, false); return err }},
		{"unique", func() error { _, err := Unique(vec); return err }},
		{"unique_axis", func() error { _, err := UniqueAxis(sq, 0); return err }},
		{"diff", func() error { _, err := Diff(vec, 1, []int{0}); return err }},
		{"cumsum", func() error { _, err := Cumsum(sq, []int{0}); return err }},
		{"cumprod", func() error { _, err := Cumprod(sq, []int{1}); return err }},
		{"pad", func() error { _, err := Pad(sq, [][2]int{{1, 1}, {1, 1}}, "edge", 0); return err }},
		{"pad_stat", func() error {
			_, err := PadWith(sq, [][2]int{{1, 1}, {0, 0}}, PadOpts{Mode: "maximum"})
			return err
		}},
		{"flip", func() error { _, err := Flip(sq, []int{1}); return err }},
		{"roll", func() error { _, err := Roll(sq, 2, []int{1}); return err }},
		{"repeat", func() error { _, err := Repeat(vec, 2, nil); return err }},
		{"tile", func() error { _, err := Tile(sq, []int{2, 1}); return err }},
		{"triu", func() error { _, err := Triu(sq, 0); return err }},
		{"tril", func() error { _, err := Tril(sq, 0); return err }},
		{"diag", func() error { _, err := Diag(sq, 0); return err }},
		{"trace", func() error { _, err := Trace(sq, 0); return err }},
		{"cov", func() error { _, err := Cov(sq, true, -1); return err }},
		{"corrcoef", func() error { _, err := CorrCoef(sq, true); return err }},
		{"det", func() error { _, err := Det(sq); return err }},
		{"inv", func() error { _, err := Inv(sq); return err }},
		{"norm", func() error { _, err := Norm(sq, "fro"); return err }},
		{"eigvalsh", func() error { _, err := Eigvalsh(sq); return err }},
		{"svdvals", func() error { _, err := Svdvals(sq); return err }},
		{"svd", func() error { _, err := Svd(sq); return err }},
		{"solve", func() error { _, err := Solve(sq, bArr(t, []float64{1, 2, 3})); return err }},
		{"convolve", func() error { _, err := Convolve(vec, bArr(t, []float64{1, 2}), "full"); return err }},
		{"correlate", func() error { _, err := Correlate(vec, bArr(t, []float64{1, 2}), "same"); return err }},
		{"unwrap", func() error { _, err := Unwrap(vec, 0, 0); return err }},
		{"polyfit", func() error {
			_, err := Polyfit(bArr(t, []float64{0, 1, 2, 3}), bArr(t, []float64{1, 3, 7, 13}), 2)
			return err
		}},
		{"isin", func() error { _, err := IsIn(vec, bArr(t, []float64{1, 5})); return err }},
		{"intersect1d", func() error { _, err := Intersect1D(vec, bArr(t, []float64{1, 5})); return err }},
		{"union1d", func() error { _, err := Union1D(vec, bArr(t, []float64{1, 5})); return err }},
		{"setdiff1d", func() error { _, err := Setdiff1D(vec, bArr(t, []float64{1, 5})); return err }},
		{"setxor1d", func() error { _, err := Setxor1D(vec, bArr(t, []float64{1, 5})); return err }},
		{"flatnonzero", func() error { _, err := Flatnonzero(sq); return err }},
		{"argwhere", func() error { _, err := Argwhere(sq); return err }},
	}
	beforeSQ := snapshot(sq)
	beforeVec := snapshot(vec)
	for _, op := range ops {
		if err := op.run(); err != nil {
			t.Fatalf("%s: %v", op.name, err)
		}
		changed(op.name+"(sq)", beforeSQ, snapshot(sq))
		changed(op.name+"(vec)", beforeVec, snapshot(vec))
	}
	// 浮点特化算子也不该改入参
	f8 := bArr(t, []float64{1.5, -2.25, 3, 4.5})
	bf := snapshot(f8)
	for _, f := range []func() error{
		func() error { _, err := Cast(f8, DFloat32); return err },
		func() error { _, err := CastUnsafe(f8, DInt32); return err },
		func() error { _, err := Sinc(f8); return err },
		func() error { _, err := Tanh(f8); return err },
		func() error { _, err := Around(f8, 1); return err },
		func() error { _, err := IsReal(f8); return err },
	} {
		if err := f(); err != nil {
			t.Fatal(err)
		}
		changed("float-specialized", bf, snapshot(f8))
	}
	_ = math.Abs
}

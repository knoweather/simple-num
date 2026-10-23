package num

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// TestEigSoakRandom 用确定性随机源压测特征值求解:
// 谱必须满足 Σλ = tr(A)、Πλ = det(A),且特征向量残差小 —— 不依赖 numpy 自校验。
func TestEigSoakRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(20261023))
	for trial := 0; trial < 120; trial++ {
		n := 2 + trial%7 // 2..8
		vals := make([]float64, n*n)
		for i := range vals {
			vals[i] = math.Round((rng.Float64()*20-10)*100) / 100
		}
		m := mustMat(t, []int{n, n}, vals)
		out, err := Eig(m)
		if err != nil {
			t.Fatalf("trial %d (n=%d): %v", trial, n, err)
		}
		w := complexValues(t, out[0])
		v := complexValues(t, out[1])

		// 1) 迹:Σλ = Σ a[i][i]
		var trace, sum float64
		for i := 0; i < n; i++ {
			trace += vals[i*n+i]
			sum += real(w[i])
		}
		if math.Abs(trace-sum) > 1e-6*math.Max(1, math.Abs(trace)) {
			t.Fatalf("trial %d (n=%d): Σλ = %v, trace = %v", trial, n, sum, trace)
		}
		// 2) 行列式:Πλ = det(A)(复数比较)
		det, err := Det(m)
		if err != nil {
			t.Fatal(err)
		}
		prod := complex(1, 0)
		for _, z := range w {
			prod *= z
		}
		if cmplx.Abs(prod-complex(det, 0)) > 1e-6*math.Max(1, math.Abs(det)) {
			t.Fatalf("trial %d (n=%d): Πλ = %v, det = %v", trial, n, prod, det)
		}
		// 3) 特征向量残差 ||A v - λ v||
		for j := 0; j < n; j++ {
			res := 0.0
			for i := 0; i < n; i++ {
				var s complex128
				for k := 0; k < n; k++ {
					s += complex(vals[i*n+k], 0) * v[k*n+j]
				}
				s -= w[j] * v[i*n+j]
				res += real(s)*real(s) + imag(s)*imag(s)
			}
			nrm := 0.0
			for i := 0; i < n; i++ {
				nrm += real(v[i*n+j])*real(v[i*n+j]) + imag(v[i*n+j])*imag(v[i*n+j])
			}
			if math.Sqrt(res) > 1e-6*math.Max(1, math.Sqrt(nrm)) {
				t.Fatalf("trial %d (n=%d) 列 %d: 残差 %.3e(λ=%v)", trial, n, j, math.Sqrt(res), w[j])
			}
		}
	}
}

// TestEigScaled 极端量级下仍应收敛:大/小尺度矩阵的谱按比例缩放。
func TestEigScaled(t *testing.T) {
	base := mustMat(t, []int{3, 3}, []float64{-7, 8, 9, -8, 9, 10, -9, 10, 11})
	ref, err := Eigvals(base)
	if err != nil {
		t.Fatal(err)
	}
	rw := complexValues(t, ref)

	for _, scale := range []float64{1e-6, 1e6} {
		vals := win[float64](engineWindow(base.AsContiguous()))
		sc := make([]float64, len(vals))
		for i, v := range vals {
			sc[i] = v * scale
		}
		sm := mustMat(t, []int{3, 3}, sc)
		got, err := Eigvals(sm)
		if err != nil {
			t.Fatalf("scale %g: %v", scale, err)
		}
		gw := complexValues(t, got)
		want := make([]complex128, len(rw))
		for i, z := range rw {
			want[i] = z * complex(scale, 0)
		}
		tol := 1e-6 * math.Max(1, math.Abs(scale)*10)
		matchSpectrum(t, gw, want, tol)
	}
}

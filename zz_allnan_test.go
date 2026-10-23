package num

import (
	"fmt"
	"math"
	"testing"
)

func TestZZAllNanProbe(t *testing.T) {
	p := func(name string, fn func() (*Array, error)) {
		a, err := fn()
		if err != nil {
			fmt.Println(name, "-> RAISE:", err)
			return
		}
		out := make([]any, a.Size())
		f := a.AsContiguous().Flatten()
		for i := range out {
			out[i], _ = f.Item(i)
		}
		fmt.Println(name, "->", out)
	}
	a := bArr(t, []float64{math.NaN(), math.NaN()})
	fmt.Println("期望: sum 0 / prod 1 / mean,var,std,min,max,median,percentile,quantile = NaN / argmax,argmin = 报错")
	p("sum", func() (*Array, error) { return NanSum(a, nil, false) })
	p("prod", func() (*Array, error) { return NanProd(a, nil, false) })
	p("mean", func() (*Array, error) { return NanMean(a, nil, false) })
	p("var", func() (*Array, error) { return NanVar(a, nil, false) })
	p("std", func() (*Array, error) { return NanStd(a, nil, false) })
	p("min", func() (*Array, error) { return NanMin(a, nil, false) })
	p("max", func() (*Array, error) { return NanMax(a, nil, false) })
	p("median", func() (*Array, error) { return NanMedian(a, nil, false) })
	p("percentile", func() (*Array, error) { return NanPercentile(a, 50, nil, false) })
	p("quantile", func() (*Array, error) { return NanQuantile(a, 0.5, nil, false) })
	p("argmax", func() (*Array, error) { return NanArgMax(a, nil, false) })
	p("argmin", func() (*Array, error) { return NanArgMin(a, nil, false) })
}

package num

import (
	"fmt"
	"math"
	"testing"
)

func TestZZNanProbe2(t *testing.T) {
	p := func(name string, fn func() (*Array, error)) {
		a, err := fn()
		if err != nil {
			fmt.Println(name, "ERR:", err)
			return
		}
		f := a.AsContiguous().Flatten()
		out := make([]any, f.Size())
		for i := range out {
			out[i], _ = f.Item(i)
		}
		fmt.Println(name, out)
	}
	a := bArr(t, []float64{1, math.NaN(), 3, 4, 5, math.NaN()})
	m, _ := a.Reshape([]int{2, 3})
	p("sum axis0      want [5 5 3]", func() (*Array, error) { return NanSum(m, []int{0}, false) })
	p("prod axis1     want [3 20]", func() (*Array, error) { return NanProd(m, []int{1}, false) })
	p("cumsum axis0   want [1 0 3 5 5 3]", func() (*Array, error) { return NanCumsum(m, []int{0}) })
	p("cumprod axis1  want [1 1 3 4 20 20]", func() (*Array, error) { return NanCumprod(m, []int{1}) })
	p("median axis1   want [2 4.5]", func() (*Array, error) { return NanMedian(m, []int{1}, false) })
	p("argmax axis1   want [2 1]", func() (*Array, error) { return NanArgMax(m, []int{1}, false) })
	p("var axis0       want [2.25 0 0]", func() (*Array, error) { return NanVar(m, []int{0}, false) })
	b, _ := FromSlice([]float64{math.NaN(), math.NaN(), 1, 2})
	b2, _ := b.Reshape([]int{2, 2})
	p("allnan sum ax1  want [0 3]", func() (*Array, error) { return NanSum(b2, []int{1}, false) })
	p("allnan prod ax1 want [1 2]", func() (*Array, error) { return NanProd(b2, []int{1}, false) })
	p("allnan median   want nan", func() (*Array, error) { return NanMedian(b, nil, false) })
	c, _ := FromSlice([]int64{1, 2, 3, 4})
	c2, _ := c.Reshape([]int{2, 2})
	p("int nansum ax0  want [4 6]", func() (*Array, error) { return NanSum(c2, []int{0}, false) })
	p("int nanmean ax0 want [2 3]", func() (*Array, error) { return NanMean(c2, []int{0}, false) })
	p("int nanprod ax0 want [3 8]", func() (*Array, error) { return NanProd(c2, []int{0}, false) })
}

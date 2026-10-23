package num

import (
	"math"
	"testing"
)

// BincountWeights:加权和(float64)与无权重(int64)两条路径。
func TestBincountWeights(t *testing.T) {
	x := bArr(t, []int64{1, 1, 2, 3})
	// 无权重 → 等价 Bincount
	c, err := BincountWeights(x, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if c.Dtype() != DInt64 {
		t.Fatalf("无权重 dtype %s, want int64", c.Dtype())
	}
	bWantI64(t, c, 0, 2, 1, 1)

	// 加权 → float64
	w := bArr(t, []float64{0.5, 0.25, 2.0, 1.0})
	wc, err := BincountWeights(x, w, 5)
	if err != nil {
		t.Fatal(err)
	}
	if wc.Dtype() != DFloat64 {
		t.Fatalf("加权 dtype %s, want float64", wc.Dtype())
	}
	// numpy:[0.0, 0.75, 2.0, 1.0, 0.0]
	bWantF64(t, wc, 0, 0.75, 2, 1, 0)

	// 长度不符 / 负值 / 非整型 x
	if _, err := BincountWeights(x, bArr(t, []float64{1}), 0); err == nil {
		t.Fatal("weights 长度不符应报错")
	}
	if _, err := BincountWeights(bArr(t, []int64{-1}), nil, 0); err == nil {
		t.Fatal("负值应报错")
	}
	if _, err := BincountWeights(bArr(t, []float64{1}), nil, 0); err == nil {
		t.Fatal("非整型应报错")
	}
	if _, err := BincountWeights(x, nil, -1); err == nil {
		t.Fatal("minlength 为负应报错")
	}
}

// Histogram2D:分箱、边缘、越界与自动 range。
func TestHistogram2D(t *testing.T) {
	x := bArr(t, []float64{0.1, 0.5, 0.9, 0.5})
	y := bArr(t, []float64{0.5, 1.5, 2.5, 0.5})
	out, err := Histogram2D(x, y, 2, 3, 0, 1, 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("应返回 (H, xedges, yedges),得 %d", len(out))
	}
	H := out[0]
	if !shapeEqual(H.Shape(), []int{2, 3}) {
		t.Fatalf("H 形状 %v, want [2 3]", H.Shape())
	}
	if H.Dtype() != DFloat64 {
		t.Fatalf("H dtype %s", H.Dtype())
	}
	// numpy:[[1,0,0],[1,1,1]]
	bWantF64(t, H.Flatten(), 1, 0, 0, 1, 1, 1)
	// edges
	bWantF64(t, out[1], 0, 0.5, 1)
	bWantF64(t, out[2], 0, 1, 2, 3)

	// 自动 range
	xa := bArr(t, []float64{0, 5})
	ya := bArr(t, []float64{0, 10})
	auto, err := Histogram2D(xa, ya, 1, 1, math.NaN(), math.NaN(), math.NaN(), math.NaN())
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := auto[0].Item(0, 0); v != 2.0 {
		t.Fatalf("自动 range 计数 = %v, want 2", v)
	}
	if v, _ := auto[1].Item(0); v != 0.0 {
		t.Fatalf("xedges[0] = %v", v)
	}
	if v, _ := auto[1].Item(1); v != 5.0 {
		t.Fatalf("xedges[1] = %v, want 5", v)
	}

	// 越界点被丢弃:y 超出 [0,1)
	o2, err := Histogram2D(bArr(t, []float64{0.5}), bArr(t, []float64{5}), 2, 2, 0, 1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := o2[0].Item(0, 0); v != 0.0 {
		t.Fatalf("越界点不应计入,得 %v", v)
	}

	// 参数校验
	if _, err := Histogram2D(x, y, 0, 3, 0, 1, 0, 3); err == nil {
		t.Fatal("bins=0 应报错")
	}
	if _, err := Histogram2D(x, bArr(t, []float64{1}), 2, 2, 0, 1, 0, 1); err == nil {
		t.Fatal("长度不等应报错")
	}
	if _, err := Histogram2D(mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4}), y, 2, 2, 0, 1, 0, 1); err == nil {
		t.Fatal("非 1-D 应报错")
	}
	if _, err := Histogram2D(x, y, 2, 2, 1, 0, 0, 1); err == nil {
		t.Fatal("hi<lo 应报错")
	}
}

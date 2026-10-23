package num

import "testing"

// pad 的 stat_length:前侧用前 N 个、后侧用后 N 个(期望值取自 numpy 2.5.4)。
func TestPadStatLength(t *testing.T) {
	a := bArr(t, []float64{1, 5, 3, 9, 2})
	// maximum,sl=1:前侧 max(1)=1,后侧 max(2)=2
	o1, err := PadWith(a, [][2]int{{2, 2}}, PadOpts{Mode: "maximum", StatLength: 1})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o1, 1, 1, 1, 5, 3, 9, 2, 2, 2)
	// maximum,sl=2:前侧 max(1,5)=5,后侧 max(9,2)=9
	o2, err := PadWith(a, [][2]int{{2, 2}}, PadOpts{Mode: "maximum", StatLength: 2})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o2, 5, 5, 1, 5, 3, 9, 2, 9, 9)
	// mean,sl=2:前侧 mean(1,5)=3,后侧 mean(9,2)=5.5
	o3, err := PadWith(a, [][2]int{{2, 2}}, PadOpts{Mode: "mean", StatLength: 2})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o3, 3, 3, 1, 5, 3, 9, 2, 5.5, 5.5)
	// 2-D:按列取
	m := mustMat(t, []int{2, 2}, []float64{1, 5, 3, 9})
	o4, err := PadWith(m, [][2]int{{1, 1}, {0, 0}}, PadOpts{Mode: "maximum", StatLength: 1})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o4.Flatten(), 1, 5, 1, 5, 3, 9, 3, 9)
	// sl 大于轴长 → 用整条
	o5, err := PadWith(a, [][2]int{{1, 1}}, PadOpts{Mode: "maximum", StatLength: 99})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o5, 9, 1, 5, 3, 9, 2, 9)
}

// Cumsum / Cumprod:默认展平,axis 指定时按轴;dtype 按 sum 规则提升。
func TestCumsumCumprod(t *testing.T) {
	m := mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})
	// 默认(axis 省略)→ 展平
	cs, err := Cumsum(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if cs.Dtype() != DInt64 {
		t.Fatalf("int32 cumsum dtype %s, want int64", cs.Dtype())
	}
	if !shapeEqual(cs.Shape(), []int{4}) {
		t.Fatalf("默认应展平为 (4,),得 %v", cs.Shape())
	}
	bWantI64(t, cs, 1, 3, 6, 10)
	// 按轴 0
	c0, err := Cumsum(m, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(c0.Shape(), []int{2, 2}) {
		t.Fatalf("axis=0 形状 %v", c0.Shape())
	}
	bWantI64(t, c0.Flatten(), 1, 2, 4, 6)
	// 按轴 1
	c1, err := Cumsum(m, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, c1.Flatten(), 1, 3, 3, 7)
	// cumprod
	cp, err := Cumprod(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	bWantI64(t, cp, 1, 2, 6, 24)

	// dtype 提升规则
	u := bArr(t, []uint8{1, 2, 3})
	if o, err := Cumsum(u, nil); err != nil {
		t.Fatal(err)
	} else if o.Dtype() != DUint64 {
		t.Fatalf("uint8 cumsum dtype %s, want uint64", o.Dtype())
	}
	f := bArr(t, []float32{1, 2, 3})
	if o, err := Cumsum(f, nil); err != nil {
		t.Fatal(err)
	} else if o.Dtype() != DFloat32 {
		t.Fatalf("f32 cumsum dtype %s, want float32", o.Dtype())
	}
	b := bArr(t, []bool{true, true, false})
	if o, err := Cumsum(b, nil); err != nil {
		t.Fatal(err)
	} else if o.Dtype() != DInt64 {
		t.Fatalf("bool cumsum dtype %s, want int64", o.Dtype())
	}
	// nil 数组
	if _, err := Cumsum(nil, nil); err == nil {
		t.Fatal("nil 应报错")
	}
}

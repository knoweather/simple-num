package num

import (
	"testing"
)

// Einsum:各种标签组合(期望值取自 numpy 2.5.4 现场)。
func TestEinsum(t *testing.T) {
	a := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	b := mustMat(t, []int{2, 2}, []float64{5, 6, 7, 8})

	// ij,jk->ik(矩阵乘)
	m, err := Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, m.Flatten(), 19, 22, 43, 50)

	// ij->ji(转置)
	tr, err := Einsum("ij->ji", a)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(tr.Shape(), []int{2, 2}) {
		t.Fatalf("转置形状 %v", tr.Shape())
	}
	bWantF64(t, tr.Flatten(), 1, 3, 2, 4)

	// ii->i(取对角)
	dg, err := Einsum("ii->i", a)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, dg, 1, 4)

	// ij->(全和)→ 0-d
	s0, err := Einsum("ij->", a)
	if err != nil {
		t.Fatal(err)
	}
	if s0.NDim() != 0 {
		t.Fatalf("全归约应为 0-d,得 %v", s0.Shape())
	}
	if v, _ := s0.Item(); v != 10.0 {
		t.Fatalf("全和 = %v, want 10", v)
	}

	// 隐式输出(只出现一次的标签按字母序)
	im, err := Einsum("ij,jk", a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, im.Flatten(), 19, 22, 43, 50)

	// i,i->(内积)
	v := bArr(t, []float64{1, 2, 3})
	w := bArr(t, []float64{4, 5, 6})
	dot, err := Einsum("i,i->", v, w)
	if err != nil {
		t.Fatal(err)
	}
	if x, _ := dot.Item(); x != 32.0 {
		t.Fatalf("内积 = %v, want 32", x)
	}

	// 整型保持整型
	ia := mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4})
	ib := mustMat(t, []int{2, 2}, []int32{5, 6, 7, 8})
	im2, err := Einsum("ij,jk->ik", ia, ib)
	if err != nil {
		t.Fatal(err)
	}
	if im2.Dtype() != DInt32 {
		t.Fatalf("dtype %s, want int32", im2.Dtype())
	}
	wantI32(t, im2.Flatten(), 19, 22, 43, 50)

	// 错误:标签数与 ndim 不符
	if _, err := Einsum("ijk->i", a); err == nil {
		t.Fatal("标签数不符应报错")
	}
	// 错误:标签尺寸冲突
	if _, err := Einsum("ij,jk->ik", a, mustMat(t, []int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})); err == nil {
		t.Fatal("尺寸冲突应报错")
	}
}

// Tensordot:收缩指定轴。
func TestTensordot(t *testing.T) {
	a := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	b := mustMat(t, []int{2, 2}, []float64{5, 6, 7, 8})
	out, err := Tensordot(a, b, []int{1}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, out.Flatten(), 19, 22, 43, 50)

	// 1-D 收缩 → 0-d
	v := bArr(t, []float64{1, 2, 3})
	w := bArr(t, []float64{4, 5, 6})
	s, err := Tensordot(v, w, []int{0}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if x, _ := s.Item(); x != 32.0 {
		t.Fatalf("tensordot 1-D = %v, want 32", x)
	}

	// 3-D × 2-D 收缩一条轴
	c := mustMat(t, []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	d := mustMat(t, []int{2, 2}, []float64{1, 0, 0, 1})
	o3, err := Tensordot(c, d, []int{2}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(o3.Shape(), []int{2, 2, 2}) {
		t.Fatalf("3-D tensordot 形状 %v, want [2 2 2]", o3.Shape())
	}
	bWantF64(t, o3.Flatten(), 1, 2, 3, 4, 5, 6, 7, 8)

	// 轴数不一致 / 收缩维不匹配
	if _, err := Tensordot(a, b, []int{1}, nil); err == nil {
		t.Fatal("轴数不一致应报错")
	}
	if _, err := Tensordot(a, mustMat(t, []int{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}), []int{1}, []int{0}); err == nil {
		t.Fatal("收缩维不匹配应报错")
	}
}

// UniqueIdx:values + 首次出现下标 + 计数(numpy 口径)。
func TestUniqueIdx(t *testing.T) {
	x := bArr(t, []int32{3, 1, 3, 2, 1})
	out, err := UniqueIdx(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("应返回 3 个数组,得 %d", len(out))
	}
	if out[0].Dtype() != DInt32 {
		t.Fatalf("values dtype %s, want int32", out[0].Dtype())
	}
	wantI32(t, out[0], 1, 2, 3)
	bWantI64(t, out[1], 1, 3, 0) // 首次出现的原始下标
	bWantI64(t, out[2], 2, 1, 2) // 出现次数

	// 二维输入先拉平(numpy 同)
	m := mustMat(t, []int{2, 2}, []float64{2, 1, 2, 1})
	o2, err := UniqueIdx(m)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, o2[0], 1, 2)
	bWantI64(t, o2[1], 1, 0)
	bWantI64(t, o2[2], 2, 2)
}

// 边界:空输入、0-d 操作数、UniqueIdx 空数组。
func TestEinsumEdges(t *testing.T) {
	// 空数组参与
	empty := mustNew(t, []int{0}, DFloat64)
	if o, err := Einsum("i->i", empty); err != nil {
		t.Fatal(err)
	} else if o.Size() != 0 {
		t.Fatalf("空输入结果应为空,得 %d", o.Size())
	}
	// 0-d 操作数(标量乘)
	sc, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := sc.SetItem([]int{}, 3.0); err != nil {
		t.Fatal(err)
	}
	v := bArr(t, []float64{1, 2})
	if o, err := Einsum(",i->i", sc, v); err != nil {
		t.Fatal(err)
	} else {
		bWantF64(t, o, 3, 6)
	}
	// 输出标签未出现 → 报错
	if _, err := Einsum("i->j", v); err == nil {
		t.Fatal("输出标签未出现应报错")
	}
	// 非法标签 / 省略号
	if _, err := Einsum("i j->ij", v, v); err == nil {
		t.Fatal("空格应被忽略但标签数不符应报错")
	}
	if _, err := Einsum("...->...", v); err == nil {
		t.Fatal("省略号应报错")
	}
	// UniqueIdx 空数组
	if outs, err := UniqueIdx(mustNew(t, []int{0}, DFloat64)); err != nil {
		t.Fatal(err)
	} else if outs[0].Size() != 0 || outs[1].Size() != 0 || outs[2].Size() != 0 {
		t.Fatal("空输入的 unique 三输出都应为空")
	}
	// Tensordot 全收缩(axes 覆盖所有轴)
	a := mustMat(t, []int{2, 2}, []float64{1, 2, 3, 4})
	if o, err := Tensordot(a, a, []int{0, 1}, []int{0, 1}); err != nil {
		t.Fatal(err)
	} else if x, _ := o.Item(); x != 30.0 {
		t.Fatalf("全收缩应得 1+4+9+16=30,得 %v", x)
	}
}

// einsum 各 dtype 分支(i8/i16/u8/f32)与 FFT 的非 2 的幂复数路径。
func TestEinsumDtypeSweep(t *testing.T) {
	cases := []struct {
		name string
		a, b *Array
		want func(t *testing.T, out *Array)
	}{
		{"i8", mustMat(t, []int{2, 2}, []int8{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []int8{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {
				if o.Dtype() != DInt8 {
					t.Fatalf("dtype %s", o.Dtype())
				}
			}},
		{"i16", mustMat(t, []int{2, 2}, []int16{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []int16{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {
				if o.Dtype() != DInt16 {
					t.Fatalf("dtype %s", o.Dtype())
				}
			}},
		{"u8", mustMat(t, []int{2, 2}, []uint8{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []uint8{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {
				if o.Dtype() != DUint8 {
					t.Fatalf("dtype %s", o.Dtype())
				}
			}},
		{"f32", mustMat(t, []int{2, 2}, []float32{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []float32{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {
				if o.Dtype() != DFloat32 {
					t.Fatalf("dtype %s", o.Dtype())
				}
			}},
		{"u16", mustMat(t, []int{2, 2}, []uint16{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []uint16{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {}},
		{"u32", mustMat(t, []int{2, 2}, []uint32{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []uint32{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {}},
		{"u64", mustMat(t, []int{2, 2}, []uint64{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []uint64{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {}},
		{"i64", mustMat(t, []int{2, 2}, []int64{1, 2, 3, 4}),
			mustMat(t, []int{2, 2}, []int64{5, 6, 7, 8}),
			func(t *testing.T, o *Array) {}},
	}
	for _, c := range cases {
		out, err := Einsum("ij,jk->ik", c.a, c.b)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !shapeEqual(out.Shape(), []int{2, 2}) {
			t.Fatalf("%s 形状 %v", c.name, out.Shape())
		}
		c.want(t, out)
	}
	// 不支持 dtype(复数)→ 报错
	cz := bArr(t, []complex128{1 + 1i})
	if _, err := Einsum("i->i", cz); err == nil {
		t.Fatal("复数 einsum 应报错")
	}
}

// FFT 非 2 的幂复数路径 + f32 输入。
func TestFFTExtraPaths(t *testing.T) {
	// 复数非幂次
	c := bArr(t, []complex128{1 + 1i, 2 - 1i, 3 + 0i})
	if o, err := FFT(c); err != nil {
		t.Fatal(err)
	} else if o.Size() != 3 {
		t.Fatalf("size %d", o.Size())
	}
	// f32 输入 → complex128 输出
	f32 := bArr(t, []float32{1, 2, 3, 4})
	if o, err := FFT(f32); err != nil {
		t.Fatal(err)
	} else if o.Dtype() != DComplex128 {
		t.Fatalf("dtype %s", o.Dtype())
	}
	// IFFT 非幂次
	if o, err := IFFT(c); err != nil {
		t.Fatal(err)
	} else if o.Size() != 3 {
		t.Fatalf("size %d", o.Size())
	}
	// RFFT 非幂次 + IRFFT 非幂次
	r, err := RFFT(bArr(t, []float64{1, 2, 3, 4, 5}))
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(r.Shape(), []int{3}) {
		t.Fatalf("rfft 非幂次形状 %v, want [3]", r.Shape())
	}
	back, err := IRFFT(r, 5)
	if err != nil {
		t.Fatal(err)
	}
	if back.Size() != 5 {
		t.Fatalf("irfft 非幂次 size %d", back.Size())
	}
	// 0-d 拒绝
	z, err := New([]int{}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if err := z.SetItem([]int{}, 1.0); err != nil {
		t.Fatal(err)
	}
	if _, err := FFT(z); err == nil {
		t.Fatal("0-d FFT 应报错")
	}
	if _, err := RFFT(z); err == nil {
		t.Fatal("0-d RFFT 应报错")
	}
	if _, err := IRFFT(z, 2); err == nil {
		t.Fatal("0-d IRFFT 应报错")
	}
}

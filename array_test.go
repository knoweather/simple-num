package num

import (
	"errors"
	"fmt"
	"testing"
)

func mustNew(t *testing.T, shape []int, dt Dtype) *Array {
	t.Helper()
	a, err := New(shape, dt)
	if err != nil {
		t.Fatalf("New %v %s: %v", shape, dt, err)
	}
	return a
}

func mustSlice[T any](t *testing.T, x []T) *Array {
	t.Helper()
	a, err := FromSlice(x)
	if err != nil {
		t.Fatalf("FromSlice %T: %v", x, err)
	}
	return a
}

func TestNewAndAccessors(t *testing.T) {
	a := mustNew(t, []int{2, 3}, DFloat64)
	if got := a.Shape(); !shapeEqual(got, []int{2, 3}) {
		t.Fatalf("Shape = %v", got)
	}
	if a.NDim() != 2 || a.Size() != 6 {
		t.Fatalf("NDim/Size = %d/%d", a.NDim(), a.Size())
	}
	st := a.Strides()
	if st[0] != 24 || st[1] != 8 {
		t.Fatalf("C 序步长错误 %v", st)
	}
	if a.Dtype() != DFloat64 {
		t.Fatalf("Dtype = %s", a.Dtype())
	}
	// 0-d:size 1、无步长
	z := mustNew(t, []int{}, DInt64)
	if z.Size() != 1 || z.NDim() != 0 || len(z.Strides()) != 0 {
		t.Fatalf("0-d 形状错:size=%d ndim=%d", z.Size(), z.NDim())
	}
	// 形状带 0:合法,size 0
	e := mustNew(t, []int{0, 3}, DFloat64)
	if e.Size() != 0 {
		t.Fatalf("0 元素数组 size = %d", e.Size())
	}
}

func TestDtypeValidation(t *testing.T) {
	bad := []Dtype{
		{Kind: Float32, Itemsize: 8},               // 宽度不符
		{Kind: Datetime64, Itemsize: 8, Unit: "x"}, // 单位非法
		{Kind: Bytes, Itemsize: 0},                 // 零宽
		{Kind: Kind(99)},                           // 未知
	}
	for _, dt := range bad {
		if _, err := New([]int{2}, dt); err == nil {
			t.Errorf("dtype %s 应拒收", dt)
		}
	}
}

func TestItemSetItemRoundtrip(t *testing.T) {
	cases := []struct {
		dt    Dtype
		write any
		want  any
	}{
		{DBool, true, true},
		{DBool, 7, true},
		{DInt8, -3, int8(-3)},
		{DInt16, 1200, int16(1200)},
		{DInt32, -70000, int32(-70000)},
		{DInt64, 1 << 40, int64(1 << 40)},
		{DUint8, 255, byte(255)},
		{DUint16, 65535, uint16(65535)},
		{DUint32, 4000000000, uint32(4000000000)},
		{DUint64, 1 << 62, uint64(1 << 62)},
		{DFloat32, float64(3.5), float32(3.5)},
		{DFloat64, 2.718, 2.718},
	}
	for _, c := range cases {
		dt := c.dt
		a := mustNew(t, []int{2, 2}, dt)
		// 多坐标写→读
		if err := a.SetItem([]int{1, 0}, c.write); err != nil {
			t.Fatalf("%s: SetItem(%v): %v", dt, c.write, err)
		}
		got, err := a.Item(1, 0)
		if err != nil {
			t.Fatalf("%s: Item:%v", dt, err)
		}
		if got != c.want {
			t.Errorf("%s: 写 %v 读 %v, want %v", dt, c.write, got, c.want)
		}
		// 其余位置保持零值
		zv, _ := a.Item(0, 1)
		var zOK bool
		switch z := zv.(type) {
		case bool:
			zOK = z == false
		case int8:
			zOK = z == 0
		case int16:
			zOK = z == 0
		case int32:
			zOK = z == 0
		case int64:
			zOK = z == 0
		case uint8:
			zOK = z == 0
		case uint16:
			zOK = z == 0
		case uint32:
			zOK = z == 0
		case uint64:
			zOK = z == 0
		case float32:
			zOK = z == 0
		case float64:
			zOK = z == 0
		}
		if !zOK {
			t.Errorf("%s 未写位置应零,得 %v", dt, zv)
		}
	}
}

// 赋值回绕(numpy/C 铸造语义:负→低位截断)。
func TestSetItemWrap(t *testing.T) {
	a := mustNew(t, []int{4}, DUint8)
	if err := a.SetItem([]int{0}, -1); err != nil {
		t.Fatal(err)
	}
	if v, _ := a.Item(0); v != byte(255) {
		t.Errorf("uint8 ← -1: %v", v)
	}
	b := mustNew(t, []int{2}, DInt16)
	if err := b.SetItem([]int{0}, 70000); err != nil {
		t.Fatal(err)
	}
	if v, _ := b.Item(0); v != int16(70000&0xFFFF) {
		t.Errorf("int16 ← 70000:%v", v)
	}
	// 浮点向整型截断(numpy 赋值同:向零)
	c := mustNew(t, []int{2}, DInt64)
	if err := c.SetItem([]int{0}, 3.99); err != nil {
		t.Fatal(err)
	}
	if v, _ := c.Item(0); v != int64(3) {
		t.Errorf("int64 ← 3.99:%v", v)
	}
}

func TestItemErrors(t *testing.T) {
	a := mustNew(t, []int{2, 3}, DFloat64)
	if _, err := a.Item(2, 0); err == nil { // 越界
		t.Error("Item(2,0) 应 ErrIndex")
	} else if !isErrIndex(err) {
		t.Errorf("错误包口径:%v", err)
	}
	if _, err := a.Item(5); err == nil { // 维度数不符
		t.Error("Item(5) 维度数应错")
	}
	if err := a.SetItem([]int{-1, 0}, 1.0); err == nil {
		t.Error("负索引应 ErrIndex")
	}
}

func isErrIndex(err error) bool { return rootOf(err) == ErrIndex }
func isErrShape(err error) bool { return rootOf(err) == ErrShape }

func rootOf(err error) error {
	for err != nil {
		switch err {
		case ErrShape, ErrDtype, ErrIndex, ErrZeroDiv:
			return err
		}
		if m, ok := err.(interface{ Unwrap() []error }); ok {
			// 多 %w( 起):按臂序取第一个非 nil 根;ErrDtype 恒列首臂。
			for _, b := range m.Unwrap() {
				if r := rootOf(b); r != nil {
					return r
				}
			}
			return nil
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return nil
		}
		err = u.Unwrap()
	}
	return nil
}

func TestReshapeViewAndCopy(t *testing.T) {
	a := mustSlice(t, []float64{1, 2, 3, 4, 5, 6})
	r, err := a.Reshape([]int{3, 2})
	if err != nil {
		t.Fatal(err)
	}
	if &r.buf[0] != &a.buf[0] {
		t.Fatalf("连续源 Reshape 应零拷贝共享缓冲")
	}
	if x, err := r.Item(1, 1); err != nil || x != 4.0 {
		t.Fatalf("Reshape (3,2)[1,1] = %v err=%v, want 4", x, err)
	}
	// -1 推断
	w, err := a.Reshape([]int{2, -1})
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(w.Shape(), []int{2, 3}) {
		t.Fatalf("-1 推断形状 %v", w.Shape())
	}
	// 错误路径
	if _, err := a.Reshape([]int{4, 2}); !isErrShape(err) {
		t.Errorf("Reshape 尺寸错应 ErrShape:%v", err)
	}
	if _, err := a.Reshape([]int{-1, -1}); !isErrShape(err) {
		t.Errorf("双 -1 应 ErrShape:%v", err)
	}
	// 转置源 reshape → 拷贝语义(numpy 同款)
	tr, err := r.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	c, err := tr.Reshape([]int{6})
	if err != nil {
		t.Fatal(err)
	}
	if &c.buf[0] == &a.buf[0] {
		t.Errorf("非连续源 Reshape 返回应拷贝新缓冲")
	}
	if x, _ := c.Item(0); x != 1.0 {
		t.Errorf("转置→flatten [0] = %v, want 1", x)
	}
}

func TestTranspose(t *testing.T) {
	a := mustSlice(t, []int64{1, 2, 3, 4, 5, 6})
	v, err := a.Reshape([]int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	tv, err := v.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(tv.Shape(), []int{3, 2}) {
		t.Fatalf("Transpose 形状 %v", tv.Shape())
	}
	if tv.Strides()[0] != 8 || tv.Strides()[1] != 24 {
		t.Fatalf("Transpose 步长 %v", tv.Strides())
	}
	if x, _ := tv.Item(2, 0); x != int64(3) {
		t.Errorf("T[2,0] = %v, want 3", x)
	}
	// 显式轴 (1,0) 同逆序
	u, err := v.Transpose(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(u.Shape(), tv.Shape()) {
		t.Fatal("显式 (1,0) ≠ 逆序")
	}
	// 错误轴
	if _, err := v.Transpose(0); !isErrShape(err) {
		t.Errorf("轴数不符应 ErrShape")
	}
	if _, err := v.Transpose(0, 2); !isErrIndex(err) {
		t.Errorf("越界轴应 ErrIndex")
	}
	if _, err := v.Transpose(0, 0); !isErrShape(err) {
		t.Errorf("重复轴应 ErrShape")
	}
}

func TestAsContiguous(t *testing.T) {
	a := mustSlice(t, []float64{1, 2, 3, 4, 5, 6})
	v, _ := a.Reshape([]int{2, 3})
	tr, _ := v.Transpose()
	c := tr.AsContiguous()
	if &c.buf[0] == &a.buf[0] {
		t.Fatal("转置源应拷贝新缓冲")
	}
	if c.Strides()[0] != 16 || c.Strides()[1] != 8 {
		t.Fatalf("AsContiguous 步长 %v", c.Strides())
	}
	// 内容 = 转置的 C 序展开:1,4,2,5,3,6
	want := []float64{1, 4, 2, 5, 3, 6}
	for i, w := range want {
		if x, _ := c.Item(i/2, i%2); x != w {
			t.Fatalf("c[%d]=%v, want %v", i, x, w)
		}
	}
}

func mustTypedSlice[T any](t *testing.T, x []T, want Dtype) {
	t.Helper()
	a, err := FromSlice(x)
	if err != nil {
		t.Fatalf("FromSlice %T: %v", x, err)
	}
	if a.Dtype() != want {
		t.Fatalf("FromSlice %T → %s, want %s", x, a.Dtype(), want)
	}
	got, _ := a.Item(0)
	if fmt.Sprint(got) != fmt.Sprint(x[0]) {
		t.Fatalf("element 0:%v vs %v", got, x[0])
	}
}

func TestFromSliceMatrix(t *testing.T) {
	mustTypedSlice(t, []bool{true}, DBool)
	mustTypedSlice(t, []int8{5}, DInt8)
	mustTypedSlice(t, []int16{-7}, DInt16)
	mustTypedSlice(t, []int32{9}, DInt32)
	mustTypedSlice(t, []int64{11}, DInt64)
	mustTypedSlice(t, []int{11}, DInt64) // int 别名 → i64
	mustTypedSlice(t, []uint8{5}, DUint8)
	mustTypedSlice(t, []uint16{7}, DUint16)
	mustTypedSlice(t, []uint32{9}, DUint32)
	mustTypedSlice(t, []uint64{11}, DUint64)
	mustTypedSlice(t, []float32{0.5}, DFloat32)
	mustTypedSlice(t, []float64{0.5}, DFloat64)
	// string → ErrDtype
	if a, err := FromSlice([]string{"x"}); a != nil || err == nil {
		t.Errorf("string FromSlice 应报暂未支持")
	}
}

func TestFromBytes(t *testing.T) {
	dt := DInt32
	buf := make([]byte, 4*3)
	if a, err := FromBytes(buf, dt, []int{3}); err != nil || a.Size() != 3 {
		t.Fatalf("FromBytes:%v", err)
	}
	if a, err := FromBytes(make([]byte, 5), dt, []int{3}); err == nil || a != nil {
		t.Fatalf("字节不足应 ErrShape")
	}
	if a, err := FromBytes(make([]byte, 16), dt, []int{2, 2}); err != nil {
		t.Fatal(err)
	} else if !a.isCContig() {
		t.Fatal("FromBytes 应连续")
	}
}

// —— Day5 补测:Ones/Offset/赋值转换矩阵/对齐整备/未到位 dtype 的编解码错误 ——

func TestOnesMatrix(t *testing.T) {
	numeric := []Dtype{DBool, DInt8, DInt16, DInt32, DInt64,
		DUint8, DUint16, DUint32, DUint64, DFloat32, DFloat64}
	for _, dt := range numeric {
		a, err := Ones([]int{2}, dt)
		if err != nil {
			t.Fatalf("Ones(%s): %v", dt, err)
		}
		v, err := a.Item(0)
		if err != nil {
			t.Fatalf("Ones(%s).Item: %v", dt, err)
		}
		switch got := v.(type) {
		case bool:
			if !got {
				t.Fatalf("Ones(%s) = false, want true", dt)
			}
		case int8:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case int16:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case int32:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case int64:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case uint8:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case uint16:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case uint32:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case uint64:
			if got != 1 {
				t.Fatalf("Ones(%s) = %d", dt, got)
			}
		case float32:
			if got != 1 {
				t.Fatalf("Ones(%s) = %v", dt, got)
			}
		case float64:
			if got != 1 {
				t.Fatalf("Ones(%s) = %v", dt, got)
			}
		default:
			t.Fatalf("Ones(%s) 值类型 %T", dt, v)
		}
	}

	// f16:1.0 位型(0x3C00 小端 00 3C);complex/datetime/S/U 走通用编码器
	af, err := Ones([]int{2}, DFloat16)
	if err != nil {
		t.Fatalf("Ones(f16): %v", err)
	}
	if af.buf[0] != 0x00 || af.buf[1] != 0x3C {
		t.Fatalf("Ones(f16) 位型 = % x", af.buf[:2])
	}
	ac, err := Ones([]int{1}, DComplex128)
	if err != nil {
		t.Fatalf("Ones(c128): %v", err)
	}
	if v, _ := ac.Item(0); v.(complex128) != complex(1, 0) {
		t.Fatalf("Ones(c128) = %v, want 1+0i", v)
	}
	am, err := Ones([]int{1}, DTime("us"))
	if err != nil {
		t.Fatalf("Ones(M8): %v", err)
	}
	if v, _ := am.Item(0); v.(int64) != 1 {
		t.Fatalf("Ones(M8) = %v, want 1(单位内计数)", v)
	}
}

func TestOffsetAndAlignmentFallback(t *testing.T) {
	a := filled(t, []int{4}, 0, DFloat64)
	v, err := a.Sub(Slice(1, 3))
	if err != nil {
		t.Fatalf("Sub: %v", err)
	}
	if v.Offset() != 8 {
		t.Fatalf("视图 Offset = %d, want 8", v.Offset())
	}

	// 外来错位缓冲:不可直通 → AsContiguous 整备成对齐独立缓冲
	raw := make([]byte, 40)
	buf := raw[1:17]
	src, err := FromBytes(buf, DFloat64, []int{2})
	if err != nil {
		t.Fatalf("FromBytes: %v", err)
	}
	for i := 0; i < 2; i++ {
		if err := src.SetItem([]int{i}, float64(i)+0.5); err != nil {
			t.Fatalf("SetItem: %v", err)
		}
	}
	if src.isAligned8() {
		t.Skip("测试缓冲恰好对齐(理论可达,概率极低)")
	}
	if src.fastWindow() != nil {
		t.Fatalf("错位缓冲不应有直通窗")
	}
	c := src.AsContiguous()
	if c == nil {
		t.Fatalf("AsContiguous nil")
	}
	if !c.isAligned8() || c.fastWindow() == nil {
		t.Fatalf("整备后应对齐且可直通")
	}
	for i := 0; i < 2; i++ {
		got, _ := c.Item(i)
		if got.(float64) != float64(i)+0.5 {
			t.Fatalf("整备后 [%d] = %v", i, got)
		}
	}
}

// 赋值转换矩阵:Go 各类数值 → f64/i64/bool 数组(coerce 全分支)。
func TestCoerceMatrix(t *testing.T) {
	f, _ := Zeros([]int{1}, DFloat64)
	vals := []struct {
		v    any
		want float64
	}{
		{true, 1}, {int(2), 2}, {int8(3), 3}, {int16(4), 4}, {int32(5), 5},
		{int64(6), 6}, {uint8(7), 7}, {uint16(8), 8}, {uint32(9), 9},
		{uint64(10), 10}, {uint(11), 11}, {float32(12.5), 12.5}, {float64(13.5), 13.5},
	}
	for _, c := range vals {
		if err := f.SetItem([]int{0}, c.v); err != nil {
			t.Fatalf("SetItem f64 ← %T: %v", c.v, err)
		}
		got, _ := f.Item(0)
		if got.(float64) != c.want {
			t.Fatalf("f64 ← %T = %v, want %v", c.v, got, c.want)
		}
	}

	i, _ := Zeros([]int{1}, DInt64)
	for _, c := range vals {
		if err := i.SetItem([]int{0}, c.v); err != nil {
			t.Fatalf("SetItem i64 ← %T: %v", c.v, err)
		}
		got, _ := i.Item(0)
		if got.(int64) != int64(c.want) {
			t.Fatalf("i64 ← %T = %v, want %v", c.v, got, int64(c.want))
		}
	}

	b, _ := Zeros([]int{1}, DBool)
	for _, c := range []any{true, 0, 1, int64(0), 0.0, 0.5, uint8(0)} {
		if err := b.SetItem([]int{0}, c); err != nil {
			t.Fatalf("SetItem bool ← %T: %v", c, err)
		}
	}
	for _, c := range []any{"x", struct{}{}} {
		if err := b.SetItem([]int{0}, c); !errors.Is(err, ErrDtype) {
			t.Fatalf("bool ← %T err = %v, want ErrDtype", c, err)
		}
	}
	if err := f.SetItem([]int{0}, "x"); !errors.Is(err, ErrDtype) {
		t.Fatalf("f64 ← string err = %v, want ErrDtype", err)
	}
	if err := i.SetItem([]int{0}, "x"); !errors.Is(err, ErrDtype) {
		t.Fatalf("i64 ← string err = %v, want ErrDtype", err)
	}
}

// – 到位后:全部 dtype 的 Item/SetItem 必须可往返,不得再报 ErrDtype。
func TestCodecAllDtypesRoundTrip(t *testing.T) {
	for _, dt := range []Dtype{DFloat16, DComplex64, DComplex128, DTime("us"), DBytes(4), DUnicode(2)} {
		a, err := New([]int{1}, dt)
		if err != nil {
			t.Fatalf("New(%s): %v", dt, err)
		}
		if err := a.SetItem([]int{0}, 1); err != nil {
			t.Fatalf("%s SetItem: %v", dt, err)
		}
		if v, err := a.Item(0); err != nil || v == nil {
			t.Fatalf("%s Item = %#v err=%v", dt, v, err)
		}
		// 读出的值可原样写回(编解码自洽)
		v, _ := a.Item(0)
		if err := a.SetItem([]int{0}, v); err != nil {
			t.Fatalf("%s SetItem(读回值): %v", dt, err)
		}
	}
}

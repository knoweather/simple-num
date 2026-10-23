package num

import (
	"errors"
	"testing"
)

// 形状操纵/拼接单测。语义期望来自 numpy 2.5.4 实测(gen_index.py,)。

func filled(t *testing.T, shape []int, start int64, dt Dtype) *Array {
	t.Helper()
	n := prodShape(shape)
	a, err := Arange(float64(start), float64(start+int64(n)), 1, dt)
	if err != nil {
		t.Fatalf("Arange: %v", err)
	}
	r, err := a.Reshape(shape)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	return r
}

func TestFlatten(t *testing.T) {
	// C 序源
	a := filled(t, []int{2, 3}, 0, DInt64)
	f := a.Flatten()
	wantShape(t, f, 6)
	for i := int64(0); i < 6; i++ {
		if got := int64At(t, f, int(i)); got != i {
			t.Fatalf("C flatten[%d] = %d", i, got)
		}
	}
	// F 序源(numpy asfortranarray):flatten 仍按 C 序(实测),故值保持逻辑序
	fo := mustLoadBytes(t, npyFF64)
	if fo.isCContig() {
		t.Fatalf("fixture 应为 F 序")
	}
	ff := fo.Flatten()
	wantShape(t, ff, 6)
	for i := int64(0); i < 6; i++ {
		v, _ := ff.Item(int(i))
		if v.(float64) != float64(i) {
			t.Fatalf("F flatten[%d] = %v, want %d", i, v, i)
		}
	}
	if !ff.isCContig() {
		t.Fatalf("Flatten 结果应 C 连续")
	}
	// 转置视图展平
	tr, err := a.Transpose()
	if err != nil {
		t.Fatalf("Transpose: %v", err)
	}
	tf := tr.Flatten()
	for i, w := range []int64{0, 3, 1, 4, 2, 5} {
		if got := int64At(t, tf, i); got != w {
			t.Fatalf("转置 flatten[%d] = %d, want %d", i, got, w)
		}
	}
}

func TestSqueezeExpandDims(t *testing.T) {
	a := filled(t, []int{1, 3, 1}, 0, DInt64)
	v, err := a.Squeeze()
	if err != nil {
		t.Fatalf("Squeeze: %v", err)
	}
	wantShape(t, v, 3)

	v, err = a.Squeeze(0)
	if err != nil {
		t.Fatalf("Squeeze(0): %v", err)
	}
	wantShape(t, v, 3, 1)

	v, err = a.Squeeze(-1)
	if err != nil {
		t.Fatalf("Squeeze(-1): %v", err)
	}
	wantShape(t, v, 1, 3)

	if _, err := filled(t, []int{2, 3}, 0, DInt64).Squeeze(0); !errors.Is(err, ErrShape) {
		t.Fatalf("Squeeze 轴长≠1 err = %v, want ErrShape", err)
	}
	if _, err := a.Squeeze(0, 0); !errors.Is(err, ErrShape) {
		t.Fatalf("Squeeze 重复轴 err = %v, want ErrShape", err)
	}
	if _, err := a.Squeeze(5); !errors.Is(err, ErrIndex) {
		t.Fatalf("Squeeze 越界轴 err = %v, want ErrIndex", err)
	}

	b := filled(t, []int{2, 3}, 0, DInt64)
	for _, c := range []struct {
		axis int
		want []int
	}{
		{1, []int{2, 1, 3}},
		{-1, []int{2, 3, 1}},
		{-3, []int{1, 2, 3}},
		{0, []int{1, 2, 3}},
		{2, []int{2, 3, 1}},
	} {
		v, err := b.ExpandDims(c.axis)
		if err != nil {
			t.Fatalf("ExpandDims(%d): %v", c.axis, err)
		}
		wantShape(t, v, c.want...)
	}
	if _, err := b.ExpandDims(3); !errors.Is(err, ErrIndex) {
		t.Fatalf("ExpandDims(3) err = %v, want ErrIndex", err)
	}
	if _, err := b.ExpandDims(-4); !errors.Is(err, ErrIndex) {
		t.Fatalf("ExpandDims(-4) err = %v, want ErrIndex", err)
	}
}

func TestSwapAxes(t *testing.T) {
	a := filled(t, []int{2, 3, 4}, 0, DInt64)
	v, err := a.SwapAxes(0, 2)
	if err != nil {
		t.Fatalf("SwapAxes: %v", err)
	}
	wantShape(t, v, 4, 3, 2)
	// v[i,j,k] == a[k,j,i]
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 2; k++ {
				if got, w := int64At(t, v, i, j, k), int64At(t, a, k, j, i); got != w {
					t.Fatalf("v[%d,%d,%d]=%d ≠ a[%d,%d,%d]=%d", i, j, k, got, k, j, i, w)
				}
			}
		}
	}
	v2, err := a.SwapAxes(0, -1)
	if err != nil {
		t.Fatalf("SwapAxes neg: %v", err)
	}
	wantShape(t, v2, 4, 3, 2)
	if _, err := a.SwapAxes(0, 3); !errors.Is(err, ErrIndex) {
		t.Fatalf("SwapAxes 越界 err = %v, want ErrIndex", err)
	}
}

func TestConcatenate(t *testing.T) {
	z := filled(t, []int{2, 3}, 0, DInt64)
	o := filled(t, []int{2, 1}, 100, DInt64)

	v, err := Concatenate(1, z, o)
	if err != nil {
		t.Fatalf("Concatenate: %v", err)
	}
	wantShape(t, v, 2, 4)
	if got := int64At(t, v, 0, 3); got != 100 {
		t.Fatalf("v[0,3] = %d, want 100", got)
	}
	if got := int64At(t, v, 1, 2); got != 5 {
		t.Fatalf("v[1,2] = %d, want 5", got)
	}

	vm, err := Concatenate(-1, z, o)
	if err != nil {
		t.Fatalf("Concatenate(-1): %v", err)
	}
	wantShape(t, vm, 2, 4)

	// 1 维拼接 + dtype 提升(int32 + f64 → f64,实测)
	p1, _ := Zeros([]int{2}, DInt32)
	p2, _ := Full([]int{2}, 1.5, DFloat64)
	pc, err := Concatenate(0, p1, p2)
	if err != nil {
		t.Fatalf("Concatenate promote: %v", err)
	}
	if pc.Dtype() != DFloat64 {
		t.Fatalf("promote dtype = %s, want float64", pc.Dtype())
	}
	wantShape(t, pc, 4)
	if v, _ := pc.Item(2); v.(float64) != 1.5 {
		t.Fatalf("pc[2] = %v, want 1.5", v)
	}

	if _, err := Concatenate(0, z, o); !errors.Is(err, ErrShape) {
		t.Fatalf("非拼接轴形状不等 err = %v, want ErrShape", err)
	}
	if _, err := Concatenate(0); !errors.Is(err, ErrShape) {
		t.Fatalf("空入参 err = %v, want ErrShape", err)
	}
	if _, err := Concatenate(0, z, filled(t, []int{6}, 0, DInt64)); !errors.Is(err, ErrShape) {
		t.Fatalf("维度不等 err = %v, want ErrShape", err)
	}
	if _, err := Concatenate(2, z, o); !errors.Is(err, ErrIndex) {
		t.Fatalf("轴越界 err = %v, want ErrIndex", err)
	}
}

func TestStackVStackHStack(t *testing.T) {
	a := filled(t, []int{3}, 0, DInt64)
	b := filled(t, []int{3}, 10, DInt64)

	s, err := Stack(0, a, b)
	if err != nil {
		t.Fatalf("Stack: %v", err)
	}
	wantShape(t, s, 2, 3)
	if got := int64At(t, s, 1, 0); got != 10 {
		t.Fatalf("s[1,0] = %d, want 10", got)
	}
	s2, err := Stack(-1, a, b)
	if err != nil {
		t.Fatalf("Stack(-1): %v", err)
	}
	wantShape(t, s2, 3, 2)
	if got := int64At(t, s2, 0, 1); got != 10 {
		t.Fatalf("s2[0,1] = %d, want 10", got)
	}
	if _, err := Stack(0, a, filled(t, []int{4}, 0, DInt64)); !errors.Is(err, ErrShape) {
		t.Fatalf("Stack 形状不等 err = %v, want ErrShape", err)
	}

	v, err := VStack(a, b)
	if err != nil {
		t.Fatalf("VStack: %v", err)
	}
	wantShape(t, v, 2, 3)

	// vstack 混合维:(2,3) + (3,) → (3,3)(实测)
	mix, err := VStack(filled(t, []int{2, 3}, 0, DInt64), filled(t, []int{3}, 100, DInt64))
	if err != nil {
		t.Fatalf("VStack mix: %v", err)
	}
	wantShape(t, mix, 3, 3)
	if got := int64At(t, mix, 2, 0); got != 100 {
		t.Fatalf("mix[2,0] = %d, want 100", got)
	}

	h, err := HStack(a, b)
	if err != nil {
		t.Fatalf("HStack: %v", err)
	}
	wantShape(t, h, 6)

	h2, err := HStack(filled(t, []int{2, 1}, 0, DInt64), filled(t, []int{2, 2}, 0, DInt64))
	if err != nil {
		t.Fatalf("HStack 2d: %v", err)
	}
	wantShape(t, h2, 2, 3)
}

func TestSplit(t *testing.T) {
	a := filled(t, []int{6}, 0, DInt64)
	parts, err := Split(a, 3, 0)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	if len(parts) != 3 {
		t.Fatalf("份数 = %d, want 3", len(parts))
	}
	for k, p := range parts {
		wantShape(t, p, 2)
		for j := 0; j < 2; j++ {
			if got, w := int64At(t, p, j), int64(k*2+j); got != w {
				t.Fatalf("parts[%d][%d] = %d, want %d", k, j, got, w)
			}
		}
	}

	// 视图零拷贝:写回原数组
	if err := parts[1].SetSubScalar(int64(77)); err != nil {
		t.Fatalf("SetSubScalar: %v", err)
	}
	if got := int64At(t, a, 3); got != 77 {
		t.Fatalf("a[3] = %d, want 77(视图)", got)
	}

	m := filled(t, []int{2, 4}, 0, DInt64)
	mp, err := Split(m, 2, 1)
	if err != nil {
		t.Fatalf("Split axis1: %v", err)
	}
	wantShape(t, mp[0], 2, 2)
	if got := int64At(t, mp[1], 0, 0); got != 2 {
		t.Fatalf("mp[1][0,0] = %d, want 2", got)
	}

	if _, err := Split(filled(t, []int{5}, 0, DInt64), 2, 0); !errors.Is(err, ErrShape) {
		t.Fatalf("不能等分 err = %v, want ErrShape", err)
	}
	if _, err := Split(a, 0, 0); !errors.Is(err, ErrShape) {
		t.Fatalf("份数 0 err = %v, want ErrShape", err)
	}
	if _, err := Split(m, 2, 2); !errors.Is(err, ErrIndex) {
		t.Fatalf("轴越界 err = %v, want ErrIndex", err)
	}
}

// HStack 的 0 维入参:atleast_1d → 拼接成 1 维(numpy:hstack([0-d,0-d]) → (2,))。
func TestHStackZeroDim(t *testing.T) {
	z0, err := New([]int{}, DInt64)
	if err != nil {
		t.Fatalf("New 0-d: %v", err)
	}
	if err := z0.SetItem([]int{}, int64(1)); err != nil {
		t.Fatalf("SetItem: %v", err)
	}
	z1, _ := New([]int{}, DInt64)
	if err := z1.SetItem([]int{}, int64(2)); err != nil {
		t.Fatalf("SetItem: %v", err)
	}
	h, err := HStack(z0, z1)
	if err != nil {
		t.Fatalf("HStack 0-d: %v", err)
	}
	wantShape(t, h, 2)
	if got := int64At(t, h, 1); got != 2 {
		t.Fatalf("h[1] = %d, want 2", got)
	}
}

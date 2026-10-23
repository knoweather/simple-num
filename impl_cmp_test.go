package num

// T9 单测(B)。期望值 = numpy 2.5.4 实测探针字面量(与 xcheck/reference fixtures
// 同源, §6)。bArr/bWantBool 借用 impl_arith_test.go(同包)。

import (
	"fmt"
	"math"
	"testing"
)

// bool 有序比较(numpy:False < True)。
func TestCmpBoolOrdered(t *testing.T) {
	x := bArr(t, []bool{true, false})
	y := bArr(t, []bool{false, false})
	out, err := Less(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, false, false)
	out, _ = LessEqual(x, y)
	bWantBool(t, out, false, true)
	out, _ = Greater(x, y)
	bWantBool(t, out, true, false)
	out, _ = GreaterEqual(x, y)
	bWantBool(t, out, true, true)
	out, _ = Equal(x, y)
	bWantBool(t, out, false, true)
	out, _ = NotEqual(x, y)
	bWantBool(t, out, true, false)
}

// f8 NaN 向量:有序比较恒 False;equal 恒 False;not_equal 恒 True;±0 相等。
func TestCmpF64NaN(t *testing.T) {
	a := bArr(t, []float64{1, math.NaN(), -1, math.NaN(), 0})
	b := bArr(t, []float64{1, 1, math.NaN(), math.NaN(), math.Copysign(0, -1)})
	out, err := Less(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, false, false, false, false, false)
	out, _ = LessEqual(a, b)
	bWantBool(t, out, true, false, false, false, true)
	out, _ = Greater(a, b)
	bWantBool(t, out, false, false, false, false, false)
	out, _ = GreaterEqual(a, b)
	bWantBool(t, out, true, false, false, false, true)
	out, _ = Equal(a, b)
	bWantBool(t, out, true, false, false, false, true) // 末行:0.0 == -0.0 → true
	out, _ = NotEqual(a, b)
	bWantBool(t, out, false, true, true, true, false)
}

// f4 NaN 与 inf 边角。
func TestCmpF32Edges(t *testing.T) {
	a := bArr(t, []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))})
	b := bArr(t, []float32{1, float32(math.Inf(1)), 0})
	out, err := Less(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, false, false, true) // nan<1=false; inf<inf=false; -inf<0=true
	out, _ = NotEqual(a, b)
	bWantBool(t, out, true, false, true)
}

// 晋级:Less(i8, i64) → 操作数 i64(引擎 promote),输出 bool。
func TestCmpPromote(t *testing.T) {
	x := bArr(t, []int8{-1, 2})
	y := bArr(t, []int64{0, 0})
	out, err := Less(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != DBool {
		t.Fatalf("dtype = %s, want bool", out.Dtype())
	}
	bWantBool(t, out, true, false)
}

// 广播:(2,3) >= (3,) → (2,3)。
func TestCmpBroadcast(t *testing.T) {
	xa := bArr(t, []int64{1, 2, 3, 4, 5, 6})
	x, _ := xa.Reshape([]int{2, 3})
	y := bArr(t, []int64{3, 4, 5})
	out, err := GreaterEqual(x, y)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(out.Shape(), []int{2, 3}) || out.Dtype() != DBool {
		t.Fatalf("out %v %s", out.Shape(), out.Dtype())
	}
	want := []bool{false, false, false, true, true, true}
	for i, w := range want {
		if g := getI(t, out, i/3, i%3).(bool); g != w {
			t.Fatalf("[%d] = %v, want %v", i, g, w)
		}
	}
}

// u64 边角:MaxUint64 > 0。
func TestCmpU64Edge(t *testing.T) {
	x := bArr(t, []uint64{math.MaxUint64, 0})
	y := bArr(t, []uint64{0, 0})
	out, err := Greater(x, y)
	if err != nil {
		t.Fatal(err)
	}
	bWantBool(t, out, true, false)
	out, _ = Equal(x, x)
	bWantBool(t, out, true, true)
}

// 引擎错误路径:形状不相容 → ErrShape。
func TestCmpShapeErr(t *testing.T) {
	x := bArr(t, []int64{1, 2, 3, 4})
	y := bArr(t, []int64{1, 2, 3})
	if _, err := Less(x, y); err == nil {
		t.Error("形状不相容应报错")
	} else if rootOf(err) != ErrShape {
		t.Errorf("错误根应为 ErrShape:%v", err)
	}
}

// —— 追加( 平移到注册式):引擎挂载路径与残留分支逐位对拍 ——

// parityOrd 引擎挂载路径(binaryReg per-idt 查找)与 cmpRun 直调参考同值。
func parityOrd[T OrdKinds](t *testing.T, op string, av, bv []T) {
	t.Helper()
	out, err := cmpDispatch(op, bArr(t, av), bArr(t, bv))
	if err != nil {
		t.Fatalf("%s %T×%T: %v", op, av, bv, err)
	}
	want := make([]bool, len(av))
	cmpRun(op, want, av, bv)
	bWantBool(t, out, want...)
}

// parityBool bool 窗:挂载路径与 cmpRunBool 直调参考同值。
func parityBool(t *testing.T, op string, av, bv []bool) {
	t.Helper()
	out, err := cmpDispatch(op, bArr(t, av), bArr(t, bv))
	if err != nil {
		t.Fatalf("%s bool: %v", op, err)
	}
	want := make([]bool, len(av))
	cmpRunBool(op, want, av, bv)
	bWantBool(t, out, want...)
}

// parityF16 f16 窗:挂载路径(闭包内加宽)与"加宽后 cmpRun"参考同值(取 f16 可精确表示的值)。
func parityF16(t *testing.T, op string, av, bv []float64) {
	t.Helper()
	out, err := cmpDispatch(op, f16Arr(t, av...), f16Arr(t, bv...))
	if err != nil {
		t.Fatalf("%s f16: %v", op, err)
	}
	a32, b32 := make([]float32, len(av)), make([]float32, len(bv))
	for i := range av {
		a32[i] = float32(av[i])
		b32[i] = float32(bv[i])
	}
	want := make([]bool, len(av))
	cmpRun(op, want, a32, b32)
	bWantBool(t, out, want...)
}

// parityComplex 复数窗:eq/ne 与 cmpEqNe 直调参考同值;有序比较须逐字复现原错误文案。
func parityComplex[T comparable](t *testing.T, op string, av, bv []T) {
	t.Helper()
	out, err := cmpDispatch(op, bArr(t, av), bArr(t, bv))
	if op != "equal" && op != "not_equal" {
		wantErr := fmt.Errorf("num: 复数不支持有序比较(numpy 同拒): %w", ErrDtype)
		if err == nil {
			t.Fatalf("%s %T: 应报错(复数不支持有序比较)", op, av)
		}
		if err.Error() != wantErr.Error() {
			t.Fatalf("%s %T: 错误文案 = %v, want %v", op, av, err, wantErr)
		}
		if rootOf(err) != ErrDtype {
			t.Fatalf("%s %T: 错误根应为 ErrDtype:%v", op, av, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("%s %T: %v", op, av, err)
	}
	want := make([]bool, len(av))
	cmpEqNe[T](op, want, av, bv)
	bWantBool(t, out, want...)
}

// parityStr S×S:cmpDispatch 残留分支(strCompare)与直调同值同错。
func parityStr(t *testing.T, op string, x, y *Array) {
	t.Helper()
	out, err := cmpDispatch(op, x, y)
	if err != nil {
		t.Fatalf("%s S: %v", op, err)
	}
	direct, err := strCompare(op, x, y)
	if err != nil {
		t.Fatalf("%s S(直调 strCompare): %v", op, err)
	}
	if out.Dtype() != DBool || direct.Dtype() != DBool || out.Size() != direct.Size() {
		t.Fatalf("%s S: out %s size %d vs direct %s size %d", op, out.Dtype(), out.Size(), direct.Dtype(), direct.Size())
	}
	for i := 0; i < out.Size(); i++ {
		if getI(t, out, i).(bool) != getI(t, direct, i).(bool) {
			t.Fatalf("%s S [%d]: 挂载路径 %v, strCompare 分支 %v", op, i, getI(t, out, i), getI(t, direct, i))
		}
	}
}

// TestCmpEngineParity:每 op × 代表性 kind 对,引擎挂载路径(per-idt 查找)与内核函数
// 直调参考逐位一致;S×S 走 strCompare 残留分支;复数有序错误逐字保留。
func TestCmpEngineParity(t *testing.T) {
	ops := []string{"less", "less_equal", "greater", "greater_equal", "equal", "not_equal"}
	for _, op := range ops {
		parityOrd[int8](t, op, []int8{-3, 0, 7}, []int8{-1, 0, 7})
		parityOrd[int16](t, op, []int16{-300, 0, 300}, []int16{-1, 0, 301})
		parityOrd[int32](t, op, []int32{-3, 0, 7}, []int32{-1, 0, 8})
		parityOrd[int64](t, op, []int64{math.MinInt64, 0, math.MaxInt64}, []int64{-1, 0, 1})
		parityOrd[uint8](t, op, []uint8{0, 255, 1}, []uint8{1, 255, 254})
		parityOrd[uint16](t, op, []uint16{0, 65535, 1}, []uint16{1, 65535, 65534})
		parityOrd[uint32](t, op, []uint32{0, 4294967295, 1}, []uint32{1, 4294967295, 4294967294})
		parityOrd[uint64](t, op, []uint64{0, math.MaxUint64, 1}, []uint64{1, math.MaxUint64, math.MaxUint64 - 1})
		parityOrd[float32](t, op, []float32{float32(math.NaN()), 1, -1}, []float32{1, float32(math.Inf(1)), 0})
		parityOrd[float64](t, op, []float64{math.NaN(), 1, -1}, []float64{1, math.Inf(-1), math.Copysign(0, -1)})
		parityBool(t, op, []bool{true, false, false}, []bool{false, true, false})
		parityF16(t, op, []float64{1.5, math.NaN(), -0.5}, []float64{2.5, 1, -0.5})
		parityComplex(t, op, []complex64{complex(1, 2), complex(0, 0)}, []complex64{complex(1, 2), complex(1, 0)})
		parityComplex(t, op, []complex128{complex(1, 2), complex(0, 0)}, []complex128{complex(1, 3), complex(0, 0)})
		// 混 dtype:i4×u2 → promote int32(表);参考 = 先 cast u2→i4 再比较
		parityMixed(t, op, []int32{-1, 2, 65535}, []uint16{0, 3, 1})
		// f8×int 混合 promote(表:int64×float64 → float64)
		parityMixedF64(t, op, []int64{-1, 2}, []float64{0, 2.5})
		// S×S:残留 strCompare 分支
		parityStr(t, op, strArr(t, 3, false, "ab", "abc", "zz"), strArr(t, 3, false, "ab", "abd", "zz"))
	}
}

// parityMixed i4×u2:引擎按 promote(int32) 归一后比较,与"先 cast 再比较"参考一致。
func parityMixed(t *testing.T, op string, av []int32, bv []uint16) {
	t.Helper()
	out, err := cmpDispatch(op, bArr(t, av), bArr(t, bv))
	if err != nil {
		t.Fatalf("%s i4×u2: %v", op, err)
	}
	cast := make([]int32, len(bv))
	for i, v := range bv {
		cast[i] = int32(v)
	}
	want := make([]bool, len(av))
	cmpRun(op, want, av, cast)
	bWantBool(t, out, want...)
}

// parityMixedF64 i8×f8:引擎按 promote(float64) 归一后比较,与"先 cast 再比较"参考一致。
func parityMixedF64(t *testing.T, op string, av []int64, bv []float64) {
	t.Helper()
	out, err := cmpDispatch(op, bArr(t, av), bArr(t, bv))
	if err != nil {
		t.Fatalf("%s i8×f8: %v", op, err)
	}
	a64 := make([]float64, len(av))
	for i, v := range av {
		a64[i] = float64(v)
	}
	want := make([]bool, len(av))
	cmpRun(op, want, a64, bv)
	bWantBool(t, out, want...)
}

// size-0 与 0-d:挂载路径下口径不变(size-0 空窗;0-d 单元素比较)。
func TestCmpEngineParityZeroSize(t *testing.T) {
	empty, err := New([]int{0}, DFloat64)
	if err != nil {
		t.Fatal(err)
	}
	out, err := Equal(empty, empty)
	if err != nil {
		t.Fatalf("Equal(size-0): %v", err)
	}
	if out.Dtype() != DBool || out.Size() != 0 {
		t.Fatalf("size-0 out = %s size %d, want bool size 0", out.Dtype(), out.Size())
	}
	d0, err := New([]int{}, DInt16)
	if err != nil {
		t.Fatal(err)
	}
	if err := d0.SetItem([]int{}, int16(5)); err != nil {
		t.Fatal(err)
	}
	d1, err := New([]int{}, DInt32)
	if err != nil {
		t.Fatal(err)
	}
	if err := d1.SetItem([]int{}, int32(6)); err != nil {
		t.Fatal(err)
	}
	out, err = Less(d0, d1) // promote(i2,i4)=i4
	if err != nil {
		t.Fatalf("Less(0-d): %v", err)
	}
	if out.Dtype() != DBool || len(out.Shape()) != 0 {
		t.Fatalf("0-d out = %s shape %v", out.Dtype(), out.Shape())
	}
	if got := getI(t, out).(bool); got != true {
		t.Fatalf("0-d Less = %v, want true", got)
	}
}

// TestCmpRegistryCoverage:6 个比较算子的 binaryReg 规格 = 14 kind(实际挂载集合),
// ForceOut==Bool;并锁定 divide 不受查找键改造影响(ForceOut==0 → 维持 outKind 键)。
func TestCmpRegistryCoverage(t *testing.T) {
	wantKinds := []Kind{
		Bool, Int8, Int16, Int32, Int64,
		Uint8, Uint16, Uint32, Uint64,
		Float16, Float32, Float64,
		Complex64, Complex128,
	}
	for _, op := range cmpOps {
		spec := binaryReg[op]
		if spec == nil {
			t.Fatalf("binaryReg[%q] 未挂", op)
		}
		if spec.ForceOut != Bool {
			t.Fatalf("%q ForceOut = %s, want Bool", op, spec.ForceOut)
		}
		if len(spec.Kernels) != len(wantKinds) {
			t.Fatalf("%q Kernels = %d kind, want %d", op, len(spec.Kernels), len(wantKinds))
		}
		for _, k := range wantKinds {
			if spec.Kernels[k] == nil {
				t.Fatalf("%q 缺 %s 内核(per-idt 键)", op, k)
			}
		}
	}

	// divide 实况(查证):无 ForceOut,Kernels 按 kind(= outKind)键 → 本次改造不触碰。
	dv := binaryReg["divide"]
	if dv == nil {
		t.Fatal("binaryReg[\"divide\"] 未注册")
	}
	if dv.ForceOut != 0 {
		t.Fatalf("divide ForceOut = %s, want 0(维持 outKind 查找)", dv.ForceOut)
	}
	for _, k := range []Kind{Float16, Float32, Float64, Complex64, Complex128} {
		if dv.Kernels[k] == nil {
			t.Fatalf("divide 缺 %s 内核(outKind 键)", k)
		}
	}

	// ForceOut==Bool 的算术特种算子:只在 bool×bool 路由,inKind==outKind==Bool,键不变。
	for _, op := range []string{"add_bool", "multiply_bool"} {
		s := binaryReg[op]
		if s == nil || s.ForceOut != Bool || s.Kernels[Bool] == nil {
			t.Fatalf("%s 规格异常(Bool 键)", op)
		}
	}
}

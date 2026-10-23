package xcheck

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	num "github.com/knoweather/simple-num"
)

// —— 参考数据对拍消费端 ——
//
// 生成端:xcheck/gen_reference.py + reference/manifest*.json + reference/cases/*.npy;
// 本文件是消费端,只读这些产物。
//
// 三层:
//  1. fixture 自洽:manifest → 本库 io_npy 装载(不埋私有解析)→
//     条目字段/重名校验 + 形状不变量(每个输入都须能广播到期望形状,走本库广播面);
//  2. 算子真值对拍:注册表把 op 名映射到算子入口,逐元素按 region 对拍;
//     多输出表(xcheckMultiOps)先行分流,gather/掩码表(xcheckGatherOps)先行短路,
//     其后按元数分一元(xcheckUnaryOps)/ 二元含比较(xcheckOps)/ 归约(xcheckReduceOps);
//  3. 比较器自检:diffElems 的语义本身用 FromSlice 造样本锁住。
//
// 第 1 层的广播形状断言只对非归约案成立(归约案的期望形状是轴被消掉**之后**的形状);
// 形状会变的算子(如 rfft / polyfit)由 manifest 的 out_shape 显式声明期望形状。
//
// 整数零除:**不设特例**,本库与 numpy 同为「元素 0、不报错」
// (ErrZeroDiv 留给真正会报错的路径,如 Arange step=0),故 reference 里那些 0 就是比对目标。

// xcheckOps op 名(manifest 的 "op" 字段)→ 二元算子入口。由 A 的注册文件
// 填充;未注册的 op 在第 1 层只做 fixture 校验并记 Log。
var xcheckOps = map[string]func(a, b *num.Array) (*num.Array, error){}

// xcheckUnaryOps 一元 op 名 → 入口(签名 `func(x *Array) (*Array, error)`)。
// 与 xcheckOps 同为 A 的注册文件填充。
var xcheckUnaryOps = map[string]func(x *num.Array) (*num.Array, error){}

// xcheckReduceOps 归约 op 名 → 入口(轴的表达走 manifest 的 `args{axis,keepdim}`)。
var xcheckReduceOps = map[string]func(a *num.Array, axis []int, keepdims bool) (*num.Array, error){}

// xcheckGatherOps gather/掩码 op 名(第三清单 manifest_gather.json)→ runner。
// ins=全部输入(count_nonzero/bincount 1 个,take/take_along_axis/compress 2 个,put 3 个),
// args 带 axis/minlength;返回 *Array(put 原地改写 src 后回传被改写的 src)。
var xcheckGatherOps = map[string]func(ins []*num.Array, args caseArgs) (*num.Array, error){}

// caseArgs manifest 的 args(轴/keepdim/endpoint/n/decimals;归约、gather 与 funcs 共用)。
type caseArgs struct {
	Axis       []int     `json:"axis,omitempty"`
	Keepdim    bool      `json:"keepdim,omitempty"`
	Endpoint   bool      `json:"endpoint,omitempty"`
	N          int       `json:"n,omitempty"`
	Decimals   int       `json:"decimals,omitempty"`
	Rtol       float64   `json:"rtol,omitempty"`
	Atol       float64   `json:"atol,omitempty"`
	Shift      int       `json:"shift,omitempty"`
	Repeats    int       `json:"repeats,omitempty"`
	Reps       []int     `json:"reps,omitempty"`
	Q          float64   `json:"q,omitempty"`
	Right      bool      `json:"right,omitempty"`
	XY         bool      `json:"xy,omitempty"`
	Pads       [][]int   `json:"pads,omitempty"`
	Mode       string    `json:"mode,omitempty"`
	CVal       float64   `json:"cval,omitempty"`
	Bins       int       `json:"bins,omitempty"`
	Lo         float64   `json:"lo,omitempty"`
	Hi         float64   `json:"hi,omitempty"`
	NCond      int       `json:"ncond,omitempty"`
	D          float64   `json:"d,omitempty"`
	Sub        string    `json:"sub,omitempty"`
	AxesA      []int     `json:"axesA,omitempty"`
	AxesB      []int     `json:"axesB,omitempty"`
	K          int       `json:"k,omitempty"`
	Offset     int       `json:"offset,omitempty"`
	Rowvar     bool      `json:"rowvar,omitempty"`
	Ddof       int       `json:"ddof,omitempty"`
	EndBefore  float64   `json:"endBefore,omitempty"`
	EndAfter   float64   `json:"endAfter,omitempty"`
	BinsX      int       `json:"binsX,omitempty"`
	BinsY      int       `json:"binsY,omitempty"`
	XLo        float64   `json:"xlo,omitempty"`
	XHi        float64   `json:"xhi,omitempty"`
	YLo        float64   `json:"ylo,omitempty"`
	YHi        float64   `json:"yhi,omitempty"`
	Qs         []float64 `json:"qs,omitempty"`
	Method     string    `json:"method,omitempty"`
	StatLength int       `json:"statLength,omitempty"`
	V1         float64   `json:"v1,omitempty"`
	V2         float64   `json:"v2,omitempty"`
	Sections   int       `json:"sections,omitempty"`
	Index      int       `json:"index,omitempty"`
	SplitIdx   []int     `json:"splitIdx,omitempty"`
	Deg        int       `json:"deg,omitempty"`
	Discont    float64   `json:"discont,omitempty"`
}

// manifestCase reference manifest 条目(B 的 gen_reference.py 落盘口径,字段以
// reference/manifest.json 为准;未知字段忽略,新字段按需消费)。
type manifestCase struct {
	Name   string   `json:"name"`
	Op     string   `json:"op"`
	Inputs []string `json:"inputs"` // 输入.npy(相对 reference/)
	Expect string   `json:"expect"` // 期望.npy(相对 reference/)
	Region string   `json:"region"` // bitwise | rel
	Tol    float64  `json:"tol,omitempty"`
	Note   string   `json:"note,omitempty"`
	Args   caseArgs `json:"args,omitempty"`
	// out_shape 仅出现在第三清单(gather):期望输出形状。Go encoding/json
	// {"out_shape":[]} → 非 nil(len 0);缺字段 → nil,故用 !=nil 区分"给了(含空)"与"未给"。
	OutShape []int `json:"out_shape"`
	// outputs / out_shapes 仅出现在第四清单(nonzero 这类多输出案):没有单一 expect,
	// outputs 与 out_shapes 等长,顺序 = np.nonzero 元组顺序(第 0 维在前),每个输出是 i64 一维数组。
	Outputs   []string `json:"outputs"`
	OutShapes [][]int  `json:"out_shapes"`
}

const refRoot = "../xcheck/reference"

func TestReferenceManifest(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(refRoot, "manifest.json"))
	if err != nil {
		t.Skipf("参考数据未生成(B 的 gen_reference.py),消费端待生成后生效:%v", err)
	}
	var cases []manifestCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("manifest 解码失败:%v", err)
	}
	// 归约案在 B 的第二份 manifest(manifest_agg.json):文件不存在只记一行继续;
	// 存在则并入同一 cases 切片(case 名跨文件查重,见下面的 seen)。
	// 注意:json.Unmarshal 会复用非空切片的已有元素,故解进临时切片再 append。
	if rawAgg, err := os.ReadFile(filepath.Join(refRoot, "manifest_agg.json")); err != nil {
		t.Logf("manifest_agg.json 未生成:%v(本轮只跑 manifest.json)", err)
	} else {
		var agg []manifestCase
		if err := json.Unmarshal(rawAgg, &agg); err != nil {
			t.Fatalf("manifest_agg.json 解码失败:%v", err)
		}
		cases = append(cases, agg...)
		t.Logf("manifest_agg.json 已并入,合计 %d 案", len(cases))
	}
	// gather/掩码案在 B 的第三份 manifest(manifest_gather.json):同口径——文件不存在只记一行继续;
	// 存在则并入同一 cases 切片(case 名跨清单查重,见下面的 seen)。
	if rawG, err := os.ReadFile(filepath.Join(refRoot, "manifest_gather.json")); err != nil {
		t.Logf("manifest_gather.json 未生成:%v(本轮不含 gather/掩码案)", err)
	} else {
		var g []manifestCase
		if err := json.Unmarshal(rawG, &g); err != nil {
			t.Fatalf("manifest_gather.json 解码失败:%v", err)
		}
		cases = append(cases, g...)
		t.Logf("manifest_gather.json 已并入,合计 %d 案", len(cases))
	}
	// 多输出案(nonzero)在 B 的第四份 manifest(manifest_nonzero.json):同口径——文件不存在
	// 只记一行继续;存在则并入同一 cases 切片(case 名跨清单查重,见下面的 seen)。
	if rawN, err := os.ReadFile(filepath.Join(refRoot, "manifest_nonzero.json")); err != nil {
		t.Logf("manifest_nonzero.json 未生成:%v(本轮不含多输出案)", err)
	} else {
		var n []manifestCase
		if err := json.Unmarshal(rawN, &n); err != nil {
			t.Fatalf("manifest_nonzero.json 解码失败:%v", err)
		}
		cases = append(cases, n...)
		t.Logf("manifest_nonzero.json 已并入,合计 %d 案", len(cases))
	}
	// numpy 高频函数面(round/clip/where 等)在第五份 manifest(manifest_funcs.json):
	// 同口径——文件不存在只记一行继续;存在则并入同一 cases 切片(跨清单查重不变)。
	if rawF, err := os.ReadFile(filepath.Join(refRoot, "manifest_funcs.json")); err != nil {
		t.Logf("manifest_funcs.json 未生成:%v(本轮不含 funcs 案)", err)
	} else {
		var f []manifestCase
		if err := json.Unmarshal(rawF, &f); err != nil {
			t.Fatalf("manifest_funcs.json 解码失败:%v", err)
		}
		cases = append(cases, f...)
		t.Logf("manifest_funcs.json 已并入,合计 %d 案", len(cases))
	}
	if len(cases) == 0 {
		t.Skip("manifest 空列表")
	}
	seen := make(map[string]bool, len(cases))
	nRan := 0
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			if c.Name == "" || c.Op == "" || len(c.Inputs) == 0 || (c.Expect == "" && len(c.Outputs) == 0) {
				t.Fatalf("manifest 条目字段不全:%+v", c)
			}
			if len(c.Outputs) > 0 && len(c.OutShapes) != len(c.Outputs) {
				t.Fatalf("多输出案 out_shapes 与 outputs 不等长:%+v", c)
			}
			if seen[c.Name] {
				t.Fatalf("case 名重复:%s", c.Name)
			}
			seen[c.Name] = true

			ins := make([]*num.Array, len(c.Inputs))
			for i, p := range c.Inputs {
				a, err := num.LoadNPY(filepath.Join(refRoot, p))
				if err != nil {
					t.Fatalf("装载输入 %s:%v", p, err)
				}
				ins[i] = a
			}
			// 多输出案(第四清单):没有单一 expect,逐输出对拍后返回,不走单输出的
			// 广播/out_shape 断言(输出形状由非零位置决定,与输入不可广播)。
			if len(c.Outputs) > 0 {
				got, ran := runMultiOp(t, c, ins)
				if !ran {
					return
				}
				nRan++
				if len(got) != len(c.Outputs) {
					t.Fatalf("op=%s 输出个数 %d ≠ manifest %d", c.Op, len(got), len(c.Outputs))
				}
				for i := range got {
					want, err := num.LoadNPY(filepath.Join(refRoot, c.Outputs[i]))
					if err != nil {
						t.Fatalf("装载期望 %s:%v", c.Outputs[i], err)
					}
					if !slices.Equal(got[i].Shape(), c.OutShapes[i]) {
						t.Errorf("op=%s 输出[%d] shape %v ≠ out_shapes[%d] %v",
							c.Op, i, got[i].Shape(), i, c.OutShapes[i])
					}
					t.Logf("op=%s 输出[%d] %s%v,对拍 %d 元素(region=%s, tol=%g)",
						c.Op, i, got[i].Dtype(), got[i].Shape(), want.Size(), c.Region, c.Tol)
					for _, d := range diffElems(got[i], want, c.Region, c.Tol) {
						t.Errorf("op=%s %s:%s", c.Op, c.Name, d)
					}
				}
				return
			}
			want, err := num.LoadNPY(filepath.Join(refRoot, c.Expect))
			if err != nil {
				t.Fatalf("装载期望 %s:%v", c.Expect, err)
			}
			// 形状不变量:numpy 右对齐广播 —— 每个输入都须能广播到期望形状。
			// 归约案的期望形状是归约**之后**的形状(轴被消掉),输入不须
			// 广播到它,该断言对归约案无意义 → 跳过;非归约案断言不变。
			// out_shape 豁免(gather):take/put/compress 等的输出形状由索引决定、与输入无关
			// (甚至不可广播),故按 manifest 的 out_shape 直接断言结果形状,跳过广播断言;
			// 未带 out_shape 的旧路径逐字不变。
			if isReduceCase(c) {
				t.Logf("op=%s 归约案:跳过广播形状断言", c.Op)
			} else if hasOutShape(c) {
				t.Logf("op=%s 带 out_shape:跳过广播形状断言(输出形状由索引决定)", c.Op)
			} else if want.NDim() == 0 {
				// 0-d 期望 = 聚合为标量(allclose 这类):输入不可能广播到,跳过。
				t.Logf("op=%s 0-d 期望:跳过广播形状断言(标量结果)", c.Op)
			} else {
				for i, in := range ins {
					if _, err := in.BroadcastTo(want.Shape()); err != nil {
						t.Errorf("输入[%d] shape %v 广播不到期望 %v:%v", i, in.Shape(), want.Shape(), err)
					}
				}
			}
			got, ran := runCaseOp(t, c, ins)
			if !ran {
				// 第五清单(funcs_*)由本项目自建,每个案都必须在注册表里 ——
				// 未注册一律失败,避免"跳过即绿"的假阳性。
				if strings.HasPrefix(c.Name, "funcs_") {
					t.Fatalf("op=%s 未在任何注册表中(funcs_ 案不允许静默跳过)", c.Op)
				}
				return
			}
			nRan++
			// out_shape 豁免:断言结果形状与 manifest 的 out_shape 相等(0-d 即空切片)。
			if hasOutShape(c) {
				if !slices.Equal(got.Shape(), c.OutShape) {
					t.Errorf("op=%s 结果 shape %v ≠ out_shape %v", c.Op, got.Shape(), c.OutShape)
				}
			}
			t.Logf("op=%s 输出 %s%v,对拍 %d 元素(region=%s, tol=%g)",
				c.Op, got.Dtype(), got.Shape(), want.Size(), c.Region, c.Tol)
			for _, d := range diffElems(got, want, c.Region, c.Tol) {
				t.Errorf("op=%s %s:%s", c.Op, c.Name, d)
			}
		})
	}
	if len(xcheckOps)+len(xcheckUnaryOps)+len(xcheckReduceOps)+len(xcheckMultiOps) > 0 && nRan == 0 {
		t.Errorf("注册表已填算子却没有任何 case 跑到对拍,manifest/注册表口径不一致")
	}
}

// isReduceCase 判定归约案( 豁免的口径):op 在归约注册表里,或 manifest 的 args
// 带了 axis / keepdim(B 的生成端只在归约案上写这两个字段)。
func isReduceCase(c manifestCase) bool {
	_, ok := xcheckReduceOps[c.Op]
	return ok || c.Args.Axis != nil || c.Args.Keepdim
}

// hasOutShape 判第三清单(gather)是否显式给了 out_shape(区分"给了空 []"(0-d)与"未给")。
func hasOutShape(c manifestCase) bool {
	return c.OutShape != nil
}

// runMultiOp 按 case 取多输出入口(第五张表 xcheckMultiOps,第四清单的 nonzero 这类案)。
// 未登记则只记 Log,判定交给 B 的内核测试(与 runCaseOp 同口径,这一层不静默通过)。
func runMultiOp(t *testing.T, c manifestCase, ins []*num.Array) ([]*num.Array, bool) {
	t.Helper()
	fn, ok := xcheckMultiOps[c.Op]
	if !ok {
		t.Logf("op=%s(多输出)入口未注册(内核在 B 的后续提交里);本分支只做 fixture 校验", c.Op)
		return nil, false
	}
	got, err := fn(ins, c.Args)
	if err != nil {
		t.Fatalf("op %s 报错:%v", c.Op, err)
	}
	return got, true
}

// runCaseOp 按 case 取入口并执行(返回 ran=false 表示本分支未注册该 op:只做 fixture
// 校验并记 Log,判定交给 B 的内核测试;这一层不做"静默通过")。多输出案由 runMultiOp
// (第五张表)先行分流、不进本函数;本函数内 gather/掩码表先行短路,
// 其后按元数分一元/二元/归约三表(既有分发行为零变化)。
// - gather/掩码(第三清单):xcheckGatherOps(take/take_along_axis/put/compress/count_nonzero/bincount);
// - 一元(1 输入):xcheckUnaryOps;
// - 二元(2 输入,含比较 → bool 输出):xcheckOps;
// - 归约(1 输入 + manifest 的 args):xcheckReduceOps。
func runCaseOp(t *testing.T, c manifestCase, ins []*num.Array) (*num.Array, bool) {
	t.Helper()
	// gather/掩码(第三清单):按 op 名先行短路,先于一元/二元/归约三表查询。
	// 其 op 名与三表互斥,故既有 len==1/二元/归约分发行为零变化。
	if fn, ok := xcheckGatherOps[c.Op]; ok {
		got, err := fn(ins, c.Args)
		if err != nil {
			t.Fatalf("op %s 报错:%v", c.Op, err)
		}
		return got, true
	}
	switch len(ins) {
	case 1:
		// 1 输入既可能是一元算子,也可能是归约(轴与 keepdim 走 manifest 的 args):
		// 先查一元表,未命中再查归约表;两张表都没登记才算"未注册"。
		if fn, ok := xcheckUnaryOps[c.Op]; ok {
			got, err := fn(ins[0])
			if err != nil {
				// 整数零除不在此列:按 numpy 的数组口径(元素 0,不报错,
				// 见 / 补注一),故这里**不设**特例。
				t.Fatalf("op %s 报错:%v", c.Op, err)
			}
			return got, true
		}
		if fn, ok := xcheckReduceOps[c.Op]; ok {
			got, err := fn(ins[0], c.Args.Axis, c.Args.Keepdim)
			if err != nil {
				t.Fatalf("op %s 报错:%v", c.Op, err)
			}
			return got, true
		}
		t.Logf("op=%s(1 元)一元/归约入口均未注册(内核在 B 的后续提交里);本分支只做 fixture 校验", c.Op)
	case 2:
		if fn, ok := xcheckOps[c.Op]; ok {
			got, err := fn(ins[0], ins[1])
			if err != nil {
				t.Fatalf("op %s 报错:%v", c.Op, err)
			}
			return got, true
		}
		t.Logf("op=%s 二元入口未注册(内核在 B 的后续提交里);本分支只做 fixture 校验", c.Op)
	default:
		if fn, ok := xcheckReduceOps[c.Op]; ok {
			got, err := fn(ins[0], c.Args.Axis, c.Args.Keepdim)
			if err != nil {
				t.Fatalf("op %s 报错:%v", c.Op, err)
			}
			return got, true
		}
		t.Logf("op=%s(%d 元)归约入口未注册;本分支只做 fixture 校验", c.Op, len(ins))
	}
	return nil, false
}

// diffElems 逐元素对拍,返回不符描述(空 = 全符)。
// - bitwise:逐位 —— 浮点比 IEEE 位型(含 NaN payload、±0)、复数拆两分量、
// 整数/布尔/文本比值语义;
// - rel:相对阈值 |got-want| ≤ tol·|want|(NaN/±Inf/0 按等值口径);tol=0 时取 4e-16。
func diffElems(got, want *num.Array, region string, tol float64) []string {
	if !slices.Equal(got.Shape(), want.Shape()) {
		return []string{fmt.Sprintf("形状 %v ≠ 期望 %v", got.Shape(), want.Shape())}
	}
	if got.Dtype() != want.Dtype() {
		return []string{fmt.Sprintf("dtype %s ≠ 期望 %s", got.Dtype(), want.Dtype())}
	}
	if region == "rel" && tol == 0 {
		tol = 4e-16
	}
	g, w := flat1D(got), flat1D(want)
	var bad []string
	for i := 0; i < w.Size(); i++ {
		gv, gerr := g.Item(i)
		wv, werr := w.Item(i)
		if gerr != nil || werr != nil {
			return append(bad, fmt.Sprintf("[%d] 取值失败:%v/%v", i, gerr, werr))
		}
		if region == "rel" {
			if !relClose(gv, wv, tol) {
				bad = append(bad, fmt.Sprintf("[%d] got %v want %v(rel tol %g)", i, gv, wv, tol))
			}
			continue
		}
		if gb, wb := bitsOf(gv), bitsOf(wv); gb != wb {
			bad = append(bad, fmt.Sprintf("[%d] got %v(%s) want %v(%s)", i, gv, gb, wv, wb))
		}
	}
	return bad
}

// flat1D 连续化后拉平成 1 维(便于按平坦下标取值;Array 无平坦取值口)。
func flat1D(a *num.Array) *num.Array {
	f, err := a.AsContiguous().Reshape([]int{-1})
	if err != nil { // 不可达:AsContiguous 后尺寸不变,单 -1 维恒可推
		return a
	}
	return f
}

// bitsOf 值的位语义表示(bitwise 对拍用)。
func bitsOf(v any) string {
	switch x := v.(type) {
	case float32:
		return fmt.Sprintf("f32:%08x", math.Float32bits(x))
	case float64:
		return fmt.Sprintf("f64:%016x", math.Float64bits(x))
	case complex64:
		return fmt.Sprintf("c64:%08x:%08x", math.Float32bits(real(x)), math.Float32bits(imag(x)))
	case complex128:
		return fmt.Sprintf("c128:%016x:%016x", math.Float64bits(real(x)), math.Float64bits(imag(x)))
	default: // bool/int64/uint64/string 的 == 即逐位
		return fmt.Sprintf("%T:%v", v, v)
	}
}

// relClose 相对阈值比较(非浮点退化为值相等)。
func relClose(g, w any, tol float64) bool {
	// 复数:按模的相对误差(|g-w| <= tol*|w|)比较(FFT 这类算法族需要);
	// NaN / Inf 按"模的等值"兜底,与实数分支口径一致。
	if gc, ok := asC128(g); ok {
		if wc, ok2 := asC128(w); ok2 {
			gm, wm := cmplxAbs(gc), cmplxAbs(wc)
			switch {
			case math.IsNaN(wm):
				return math.IsNaN(gm)
			case math.IsInf(wm, 0):
				return gm == wm
			case wm == 0:
				return gm == 0
			}
			return cmplxAbs(gc-wc) <= tol*wm
		}
	}
	gf, ok1 := asF64(g)
	wf, ok2 := asF64(w)
	if !ok1 || !ok2 {
		return bitsOf(g) == bitsOf(w)
	}
	switch {
	case math.IsNaN(wf):
		return math.IsNaN(gf)
	case math.IsInf(wf, 0):
		return gf == wf
	case wf == 0:
		return gf == 0
	}
	return math.Abs(gf-wf) <= tol*math.Abs(wf)
}

func asC128(v any) (complex128, bool) {
	switch x := v.(type) {
	case complex64:
		return complex128(x), true
	case complex128:
		return x, true
	}
	return 0, false
}

func cmplxAbs(c complex128) float64 { return math.Hypot(real(c), imag(c)) }

func asF64(v any) (float64, bool) {
	switch x := v.(type) {
	case float32:
		return float64(x), true
	case float64:
		return x, true
	}
	return 0, false
}

// 比较器自检:锁住 diffElems 的语义(否则第 2 层在合流前跑不到,语义没人守)。
func TestDiffElems(t *testing.T) {
	f := func(x ...float64) *num.Array {
		t.Helper()
		a, err := num.FromSlice(x)
		if err != nil {
			t.Fatalf("FromSlice(f8):%v", err)
		}
		return a
	}
	i := func(x ...int64) *num.Array {
		t.Helper()
		a, err := num.FromSlice(x)
		if err != nil {
			t.Fatalf("FromSlice(i8):%v", err)
		}
		return a
	}

	one, two := f(1, 2), f(1, 2)
	if d := diffElems(one, two, "bitwise", 0); len(d) != 0 {
		t.Errorf("相同数组应无差异:%v", d)
	}
	// ±0 位型不同,bitwise 必须报差异
	if d := diffElems(f(0), f(math.Copysign(0, -1)), "bitwise", 0); len(d) == 0 {
		t.Error("+0 与 -0 位型不同,bitwise 应报差异")
	}
	// NaN payload 相同 → 视为相同(逐位而非 ==)
	if d := diffElems(f(math.NaN()), f(math.NaN()), "bitwise", 0); len(d) != 0 {
		t.Errorf("NaN 同位型应判相同:%v", d)
	}
	// 1 与 1+1e-15:bitwise 报差异;rel 在 4e-14 内不报、在 4e-16 外报
	g, w := f(1), f(1+1e-15)
	if d := diffElems(g, w, "bitwise", 0); len(d) == 0 {
		t.Error("bitwise 应报 1 与 1+1e-15 的差异")
	}
	if d := diffElems(g, w, "rel", 4e-14); len(d) != 0 {
		t.Errorf("rel(4e-14)内不应报差异:%v", d)
	}
	if d := diffElems(g, w, "rel", 4e-16); len(d) == 0 {
		t.Error("rel(4e-16)外应报差异")
	}
	// 形状 / dtype 不符
	if d := diffElems(one, f(1), "bitwise", 0); len(d) == 0 {
		t.Error("形状不符应报差异")
	}
	if d := diffElems(one, i(1, 2), "bitwise", 0); len(d) == 0 {
		t.Error("dtype 不符应报差异")
	}
	// NaN vs ±Inf 的 rel 口径
	inf, nan := f(math.Inf(1)), f(math.NaN())
	if d := diffElems(inf, inf, "rel", 0); len(d) != 0 {
		t.Errorf("Inf 对 Inf 应相同:%v", d)
	}
	if d := diffElems(inf, nan, "rel", 0); len(d) == 0 {
		t.Error("Inf 对 NaN 应报差异")
	}
}

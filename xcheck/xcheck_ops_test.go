package xcheck

import (
	"fmt"
	"math"
	"slices"
	"testing"

	num "github.com/knoweather/simple-num"
)

// —— 算子注册表:fixture 校验 → 真值对拍的开关 ——
//
// 填了谁,消费端(`xcheck_test.go`)就真的跑谁并按 manifest 的 region 逐位/阈值对拍
// (bitwise 逐位含 NaN 位型与 ±0;rel 用该案的 `tol`)。没填的 op 只做 fixture 校验
// 并记 Log —— 那一层的判定由内核测试兜,不在这里静默通过。
//
// 为什么单独成一个文件:算子入口(`num.Add` 等)在 `impl_arith.go` 里 —— 只有
// 本文件登记算子入口;签名与实现逐项对齐,未注册的 `funcs_` 案会直接失败。
// `impl_arith.go`(四则 7 算子随 T7/T8 进公共前缀 `9986ae0`),故这里登记**四则 7 个**。
//
// **已追加(收尾)**:五张表已全量登记(清单取自 B 已提交的 manifest,签名用 `go doc.`
// 在合流树上实测核对):
//
//	xcheckOps(二元,签名 func(x, y *Array) (*Array, error)):四则 7 + 比较 6
// (比较输出 bool 数组,照 bitwise 比)
//
//	xcheckUnaryOps(一元,签名 func(x *Array) (*Array, error)):16 个
// (超越/取整/复数共轭)
//
//	xcheckReduceOps(归约,签名 func(a *Array, axis []int, keepdims bool) (*Array, error),
// 轴与 keepdim 由 manifest 的 `args` 带):10 个 —— 归约全真是 `num.AllTrue`,
// **不是** `num.All`(那里 `All` 是索引器,不是归约)。
//
//	xcheckGatherOps(gather/掩码,签名 func(ins []*Array, args caseArgs) (*Array, error)):
// 6 个(take/take_along_axis/put/compress/count_nonzero/bincount),取自第三清单
// manifest_gather.json;axis 取 args.Axis[0],minlength 取 args.N。
//
//	xcheckMultiOps(多输出,签名 func(ins []*Array, args caseArgs) ([]*Array, error)):
// 1 个(nonzero),取自第四清单 manifest_nonzero.json;返回按 np.nonzero 元组顺序
// (第 0 维在前)的 i64 一维数组,每维一个。多输出案没有单一 expect,消费端按
// outputs/out_shapes 逐输出对拍。
//
// 键一律取 manifest 的 `op` 名(numpy ufunc 名),不是 Go 的导出名 —— 对不上会被当成
// "未注册"记 Log(不会静默放过:消费端另有"表非空却一个 case 都没跑到"的兜底断言)。

// xcheckMultiOps 多输出 op 名(第四清单 manifest_nonzero.json)→ runner。
// 分发顺序:本表 → gather 表 → 元数三表(既有单输出行为逐字不变)。
var xcheckMultiOps = map[string]func(ins []*num.Array, args caseArgs) ([]*num.Array, error){}

func init() {
	// manifest 的 "op" 名 → 二元入口(四则 7 + 比较 6)。
	for name, fn := range map[string]func(x, y *num.Array) (*num.Array, error){
		"add":           num.Add,
		"subtract":      num.Subtract,
		"multiply":      num.Multiply,
		"divide":        num.Div,
		"floor_divide":  num.FloorDivide,
		"remainder":     num.Remainder,
		"power":         num.Power,
		"less":          num.Less,
		"less_equal":    num.LessEqual,
		"greater":       num.Greater,
		"greater_equal": num.GreaterEqual,
		"equal":         num.Equal,
		"not_equal":     num.NotEqual,
	} {
		xcheckOps[name] = fn
	}

	// manifest 的 "op" 名 → 一元入口。
	for name, fn := range map[string]func(x *num.Array) (*num.Array, error){
		"absolute": num.Abs,
		"exp":      num.Exp,
		"log":      num.Log,
		"log10":    num.Log10,
		"sqrt":     num.Sqrt,
		"sin":      num.Sin,
		"cos":      num.Cos,
		"tan":      num.Tan,
		"asin":     num.Asin,
		"acos":     num.Acos,
		"atan":     num.Atan,
		"floor":    num.Floor,
		"ceil":     num.Ceil,
		"trunc":    num.Trunc,
		"rint":     num.Rint,
		"conj":     num.Conj,
	} {
		xcheckUnaryOps[name] = fn
	}

	// manifest 的 "op" 名 → 归约入口(轴与 keepdim 由 manifest 的 `args` 带)。
	for name, fn := range map[string]func(a *num.Array, axis []int, keepdims bool) (*num.Array, error){
		"sum":    num.Sum,
		"mean":   num.Mean,
		"var":    num.Var,
		"std":    num.Std,
		"min":    num.Min,
		"max":    num.Max,
		"argmin": num.ArgMin,
		"argmax": num.ArgMax,
		"any":    num.AnyTrue,
		"all":    num.AllTrue,
		"nansum": func(a *num.Array, axis []int, keep bool) (*num.Array, error) { return num.NanSum(a, axis, keep) },
		"nanmax": func(a *num.Array, axis []int, keep bool) (*num.Array, error) { return num.NanMax(a, axis, keep) },
		"median": func(a *num.Array, axis []int, keep bool) (*num.Array, error) { return num.Median(a, axis, keep) },
		"nanmin": func(a *num.Array, axis []int, keep bool) (*num.Array, error) { return num.NanMin(a, axis, keep) },
		"nanmean": func(a *num.Array, axis []int, keep bool) (*num.Array, error) {
			return num.NanMean(a, axis, keep)
		},
		"nanvar": func(a *num.Array, axis []int, keep bool) (*num.Array, error) { return num.NanVar(a, axis, keep) },
		"nanstd": func(a *num.Array, axis []int, keep bool) (*num.Array, error) { return num.NanStd(a, axis, keep) },
	} {
		xcheckReduceOps[name] = fn
	}

	// manifest 的 "op" 名 → gather/掩码 runner(第三清单 manifest_gather.json)。
	// ins=全部输入;axis 取 args.Axis[0],minlength 取 args.N。
	for name, fn := range map[string]func(ins []*num.Array, args caseArgs) (*num.Array, error){
		"take": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Take(ins[0], ins[1])
		},
		"take_along_axis": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.TakeAlongAxis(ins[0], ins[1], args.Axis[0])
		},
		"put": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			// 原地 mutate:值写入 src;返回被原地改写的 src。
			if err := num.Put(ins[0], ins[1], ins[2]); err != nil {
				return nil, err
			}
			return ins[0], nil
		},
		"compress": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			if len(args.Axis) > 0 {
				return num.CompressAxis(ins[0], ins[1], args.Axis[0])
			}
			return num.Compress(ins[0], ins[1])
		},
		// —— 第五清单 manifest_funcs.json(numpy 高频函数面)——
		"where": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Where(ins[0], ins[1], ins[2])
		},
		"round": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Round(ins[0], args.Decimals)
		},
		"clip": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			// 边界以 0-d 数组作输入,便于 manifest 显式表达 dtype。
			lo, err := ins[1].Item()
			if err != nil {
				return nil, err
			}
			hi, err := ins[2].Item()
			if err != nil {
				return nil, err
			}
			return num.Clip(ins[0], lo, hi)
		},
		// —— 线代核心(dot/matmul/inner/outer)——
		"dot": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Dot(ins[0], ins[1]) },
		"matmul": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.MatMul(ins[0], ins[1])
		},
		"inner": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Inner(ins[0], ins[1]) },
		"outer": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Outer(ins[0], ins[1]) },
		// —— 杂项算法(searchsorted/digitize/interp)——
		"searchsorted": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.SearchSorted(ins[0], ins[1], args.Right)
		},
		"digitize": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Digitize(ins[0], ins[1], args.Right)
		},
		"interp": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Interp(ins[0], ins[1], ins[2])
		},
		// —— 杂项算法第二批(pad/select/putmask;histogram 在多输出表)——
		"pad": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			pw := make([][2]int, len(args.Pads))
			for i, p := range args.Pads {
				if len(p) != 2 {
					return nil, fmt.Errorf("pad: pads[%d] 需为 [before,after],得 %v", i, p)
				}
				pw[i] = [2]int{p[0], p[1]}
			}
			return num.PadWith(ins[0], pw, num.PadOpts{
				Mode:       args.Mode,
				CVal:       args.CVal,
				EndBefore:  args.EndBefore,
				EndAfter:   args.EndAfter,
				StatLength: args.StatLength,
			})
		},
		"cumsum": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Cumsum(ins[0], args.Axis)
		},
		"cumprod": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Cumprod(ins[0], args.Axis)
		},
		"bincount_w": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.BincountWeights(ins[0], ins[1], args.N)
		}, "unique_axis": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ax := 0
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			return num.UniqueAxis(ins[0], ax)
		},
		"select": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			n := args.NCond
			if n <= 0 || 2*n >= len(ins) {
				return nil, fmt.Errorf("select: args.ncond=%d 与输入数 %d 不匹配", n, len(ins))
			}
			return num.Select(ins[:n], ins[n:2*n], ins[2*n])
		},
		// —— 掩码数组(Step 5)——
		"ma_filled": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			// ins = [data, mask, fill(0-d)]
			m, err := num.NewMasked(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			v, err := ins[2].Item()
			if err != nil {
				return nil, err
			}
			return m.Filled(v)
		},
		"ma_compressed": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			m, err := num.NewMasked(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			return m.Compressed()
		},
		"ma_masked_invalid": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			m, err := num.MaskedInvalid(ins[0])
			if err != nil {
				return nil, err
			}
			return m.MaskArray(), nil
		},
		"ma_masked_where": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			m, err := num.MaskedWhere(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			return m.MaskArray(), nil
		},
		// —— 拼装/切分族(Step 24)——
		"append": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ax := -1
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			return num.Append(ins[0], ins[1], ax)
		},
		"insert": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ax := -1
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			return num.Insert(ins[0], args.Index, ins[1], ax)
		},
		"delete": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ax := -1
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			return num.Delete(ins[0], args.Index, ax)
		},
		"column_stack": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.ColumnStack(ins...)
		},
		"dstack": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.DStack(ins...)
		},
		"put_along_axis": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ax := 0
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			if err := num.PutAlongAxis(ins[0], ins[1], ins[2], ax); err != nil {
				return nil, err
			}
			return ins[0], nil
		},
		"copyto": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			if err := num.CopyTo(ins[0], ins[1]); err != nil {
				return nil, err
			}
			return ins[0], nil
		},
		// —— 收尾 4 个(Step 33)——
		"absolute": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Absolute(ins[0])
		},
		"any_red": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Any(ins[0], args.Axis, args.Keepdim)
		},
		"invert": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Invert(ins[0])
		},
		"cross": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Cross(ins[0], ins[1])
		},
		// —— 位运算 + 小工具(Step 34)——
		"bitwise_and": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.BitwiseAnd(ins[0], ins[1])
		},
		"bitwise_or": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.BitwiseOr(ins[0], ins[1])
		},
		"bitwise_xor": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.BitwiseXor(ins[0], ins[1])
		},
		"left_shift": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.LeftShift(ins[0], ins[1])
		},
		"right_shift": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.RightShift(ins[0], ins[1])
		},
		"rot90": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ax1, ax2 := 0, 1
			if len(args.Axis) >= 2 {
				ax1, ax2 = args.Axis[0], args.Axis[1]
			}
			return num.Rot90(ins[0], args.K, ax1, ax2)
		},
		"average": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			var w *num.Array
			if len(ins) > 1 {
				w = ins[1]
			}
			return num.Average(ins[0], args.Axis, w)
		},
		"ptp": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.PtP(ins[0], args.Axis, args.Keepdim)
		},
		"nanprod": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanProd(ins[0], args.Axis, args.Keepdim)
		},
		"nancumsum": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanCumsum(ins[0], args.Axis)
		},
		"kron": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Kron(ins[0], ins[1])
		}, // —— ufunc 补充批(Step 31)——
		"negative": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Negative(ins[0])
		},
		"square": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Square(ins[0])
		},
		"reciprocal": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Reciprocal(ins[0])
		},
		"log2": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Log2(ins[0])
		},
		"cbrt": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Cbrt(ins[0])
		},
		"arcsinh": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Arcsinh(ins[0])
		},
		"expm1": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Expm1(ins[0])
		},
		"real": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Real(ins[0])
		},
		"imag": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Imag(ins[0])
		}, // —— polyfit / unwrap(Step 28)——
		"polyfit": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Polyfit(ins[0], ins[1], args.Deg)
		},
		"unwrap": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Unwrap(ins[0], args.Discont, args.D)
		}, // —— 特征值/奇异值(Step 27)——
		"eigvalsh": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Eigvalsh(ins[0])
		},
		"svdvals": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Svdvals(ins[0])
		}, // —— 集合运算 + trapz/polyval(Step 26)——
		"isin": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.IsIn(ins[0], ins[1])
		},
		"intersect1d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Intersect1D(ins[0], ins[1])
		},
		"union1d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Union1D(ins[0], ins[1])
		},
		"setdiff1d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Setdiff1D(ins[0], ins[1])
		},
		"setxor1d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Setxor1D(ins[0], ins[1])
		},
		"trapz": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			var x *num.Array
			if len(ins) > 1 {
				x = ins[1]
			}
			dx := args.D
			if dx == 0 {
				dx = 1
			}
			v, err := num.Trapz(ins[0], x, dx)
			if err != nil {
				return nil, err
			}
			return scalarArr(v)
		},
		"polyval": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Polyval(ins[0], ins[1])
		},
		// —— 线代核心 + 常用别名(Step 25)——
		"det": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			v, err := num.Det(ins[0])
			if err != nil {
				return nil, err
			}
			return scalarArr(v)
		},
		"inv": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Inv(ins[0])
		},
		"solve": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Solve(ins[0], ins[1])
		},
		"norm": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			v, err := num.Norm(ins[0], args.Mode)
			if err != nil {
				return nil, err
			}
			return scalarArr(v)
		},
		"prod": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Prod(ins[0], args.Axis, args.Keepdim)
		},
		"around": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Around(ins[0], args.Decimals)
		},
		"flatnonzero": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Flatnonzero(ins[0])
		},
		"argwhere": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Argwhere(ins[0])
		},
		"isreal": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.IsReal(ins[0])
		},
		"identity": func(_ []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Identity(args.N)
		},
		"sinc": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Sinc(ins[0])
		},
		"tanh": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Tanh(ins[0])
		},
		"convolve": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Convolve(ins[0], ins[1], args.Mode)
		},
		"correlate": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Correlate(ins[0], ins[1], args.Mode)
		},
		"transpose_pkg": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Transpose(ins[0])
		},
		"ravel_pkg": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Ravel(ins[0])
		},
		"ma_pred": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			var (
				m   *num.MaskedArray
				err error
			)
			switch args.Mode {
			case "equal":
				m, err = num.MaskedEqual(ins[0], args.V1)
			case "not_equal":
				m, err = num.MaskedNotEqual(ins[0], args.V1)
			case "less":
				m, err = num.MaskedLess(ins[0], args.V1)
			case "less_equal":
				m, err = num.MaskedLessEqual(ins[0], args.V1)
			case "greater":
				m, err = num.MaskedGreater(ins[0], args.V1)
			case "greater_equal":
				m, err = num.MaskedGreaterEqual(ins[0], args.V1)
			case "inside":
				m, err = num.MaskedInside(ins[0], args.V1, args.V2)
			case "outside":
				m, err = num.MaskedOutside(ins[0], args.V1, args.V2)
			default:
				return nil, fmt.Errorf("ma_pred: 未知 mode %q", args.Mode)
			}
			if err != nil {
				return nil, err
			}
			return m.MaskArray(), nil
		}, "ma_median": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			m, err := num.NewMasked(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			r, err := m.Median(args.Axis, args.Keepdim)
			if err != nil {
				return nil, err
			}
			return r.Data(), nil
		},
		"ma_var": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			m, err := num.NewMasked(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			r, err := m.Var(args.Axis, args.Keepdim)
			if err != nil {
				return nil, err
			}
			return r.Data(), nil
		},
		"ma_std": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			m, err := num.NewMasked(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			r, err := m.Std(args.Axis, args.Keepdim)
			if err != nil {
				return nil, err
			}
			return r.Data(), nil
		}, "ma_sum": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			m, err := num.NewMasked(ins[0], ins[1])
			if err != nil {
				return nil, err
			}
			r, err := m.Sum(args.Axis, args.Keepdim)
			if err != nil {
				return nil, err
			}
			return r.Data(), nil
		}, // —— FFT 子集(沿最后一轴;reference 用 rel 区)——
		"fft":  func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.FFT(ins[0]) },
		"ifft": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.IFFT(ins[0]) },
		"rfft": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.RFFT(ins[0]) },
		"irfft": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.IRFFT(ins[0], args.N)
		},
		"fftfreq": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.FFTFreq(args.N, args.D)
		},
		// —— 对角线/三角/迹 + 协方差族(Step 12)——
		"diag":     func(ins []*num.Array, args caseArgs) (*num.Array, error) { return num.Diag(ins[0], args.K) },
		"diagflat": func(ins []*num.Array, args caseArgs) (*num.Array, error) { return num.Diagflat(ins[0], args.K) },
		"trace": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Trace(ins[0], args.Offset)
		},
		"triu": func(ins []*num.Array, args caseArgs) (*num.Array, error) { return num.Triu(ins[0], args.K) },
		"tril": func(ins []*num.Array, args caseArgs) (*num.Array, error) { return num.Tril(ins[0], args.K) },
		"fill_diagonal": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			v, err := ins[1].Item()
			if err != nil {
				return nil, err
			}
			if err := num.FillDiagonal(ins[0], v); err != nil {
				return nil, err
			}
			return ins[0], nil
		},
		"cov": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Cov(ins[0], args.Rowvar, args.Ddof)
		},
		"corrcoef": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.CorrCoef(ins[0], args.Rowvar)
		}, // —— 线代长尾(Step 9):einsum / tensordot ——
		"einsum": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Einsum(args.Sub, ins...)
		},
		"tensordot": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Tensordot(ins[0], ins[1], args.AxesA, args.AxesB)
		}, // —— ufunc 协议层(Step 10):out= / where= / reduce / accumulate / outer ——
		"add_out": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			if err := num.BinaryOut(num.Add, ins[2], ins[0], ins[1]); err != nil {
				return nil, err
			}
			return ins[2], nil
		},
		"add_where": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			if err := num.BinaryWhere(num.Add, ins[2], ins[0], ins[1], ins[3]); err != nil {
				return nil, err
			}
			return ins[2], nil
		},
		"add_reduce": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Reduce(num.Add, ins[0], args.Axis, args.Keepdim)
		},
		"mul_reduce": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Reduce(num.Multiply, ins[0], args.Axis, args.Keepdim)
		},
		"add_accumulate": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Accumulate(num.Add, ins[0], args.Axis)
		},
		"add_outer": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.UfuncOuter(num.Add, ins[0], ins[1])
		},
		"place": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			if err := num.Place(ins[0], ins[1], ins[2]); err != nil {
				return nil, err
			}
			return ins[0], nil // 原地改写后回传
		}, "putmask": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			if err := num.PutMask(ins[0], ins[1], ins[2]); err != nil {
				return nil, err
			}
			return ins[0], nil // 原地改写后回传
		},
		"isnan": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.IsNan(ins[0]) },
		"isinf": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.IsInf(ins[0]) }, "isfinite": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.IsFinite(ins[0]) },
		"logical_and": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.LogicalAnd(ins[0], ins[1])
		},
		"logical_or": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.LogicalOr(ins[0], ins[1])
		},
		"logical_xor": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.LogicalXor(ins[0], ins[1])
		},
		"logical_not": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.LogicalNot(ins[0])
		},
		"maximum": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Maximum(ins[0], ins[1])
		},
		"minimum": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Minimum(ins[0], ins[1])
		},
		"deg2rad": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Deg2Rad(ins[0]) },
		"rad2deg": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Rad2Deg(ins[0]) },
		"fabs":    func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Fabs(ins[0]) },
		"sign":    func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Sign(ins[0]) },
		"arctan2": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			return num.Arctan2(ins[0], ins[1])
		},
		"isclose": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.IsClose(ins[0], ins[1], args.Rtol, args.Atol)
		},
		"allclose": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			ok, err := num.AllClose(ins[0], ins[1], args.Rtol, args.Atol)
			if err != nil {
				return nil, err
			}
			out, e := num.New([]int{}, num.DBool) // 标量 → 0-d bool 承载
			if e != nil {
				return nil, e
			}
			if err := out.SetItem([]int{}, ok); err != nil {
				return nil, err
			}
			return out, nil
		},
		// —— 第六批:形状/构造族(flip/roll/repeat/tile/atleast_*/zeros_like)——
		"flip":   func(ins []*num.Array, args caseArgs) (*num.Array, error) { return num.Flip(ins[0], args.Axis) },
		"flipud": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.FlipUD(ins[0]) },
		"fliplr": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.FlipLR(ins[0]) },
		"roll": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Roll(ins[0], args.Shift, args.Axis)
		},
		"repeat": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Repeat(ins[0], args.Repeats, args.Axis)
		},
		"tile": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Tile(ins[0], args.Reps)
		},
		"atleast_1d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Atleast1D(ins[0]) },
		"atleast_2d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Atleast2D(ins[0]) },
		"atleast_3d": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Atleast3D(ins[0]) },
		// —— 第七批:排序与统计(sort/argsort/percentile/diff/unique;median 在归约表)——
		"sort": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Sort(ins[0], args.Axis)
		},
		"argsort": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Argsort(ins[0], args.Axis)
		},
		"nanmedian": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanMedian(ins[0], args.Axis, args.Keepdim)
		},
		"nanargmin": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanArgMin(ins[0], args.Axis, args.Keepdim)
		},
		"nanargmax": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanArgMax(ins[0], args.Axis, args.Keepdim)
		}, "quantile": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.QuantileMethod(ins[0], args.Q, args.Method, args.Axis, args.Keepdim)
		},
		"quantile_many": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.QuantileMany(ins[0], args.Qs, args.Axis, args.Keepdim)
		},
		"nanpercentile": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanPercentile(ins[0], args.Q, args.Axis, args.Keepdim)
		},
		"nanquantile": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.NanQuantile(ins[0], args.Q, args.Axis, args.Keepdim)
		}, "percentile_m": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.PercentileMethod(ins[0], args.Q, args.Method, args.Axis, args.Keepdim)
		},
		"percentile_many": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.PercentileMany(ins[0], args.Qs, args.Axis, args.Keepdim)
		}, "percentile": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Percentile(ins[0], args.Q, args.Axis, args.Keepdim)
		},
		"diff": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			n := args.N
			if n == 0 {
				n = 1 // manifest 未写 n 时按 numpy 默认
			}
			return num.Diff(ins[0], n, args.Axis)
		},
		"unique": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.Unique(ins[0]) }, "zeros_like": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.ZerosLike(ins[0]) },
		"ones_like": func(ins []*num.Array, _ caseArgs) (*num.Array, error) { return num.OnesLike(ins[0]) },
		"count_nonzero": func(ins []*num.Array, _ caseArgs) (*num.Array, error) {
			n, err := num.CountNonzero(ins[0])
			if err != nil {
				return nil, err
			}
			// 返回 int → 造 0-d i64 数组承载。
			out, e := num.New([]int{}, num.DInt64)
			if e != nil {
				return nil, e
			}
			if err := out.SetItem([]int{}, int64(n)); err != nil {
				return nil, err
			}
			return out, nil
		},
		"bincount": func(ins []*num.Array, args caseArgs) (*num.Array, error) {
			return num.Bincount(ins[0], args.N)
		},
	} {
		xcheckGatherOps[name] = fn
	}

	// manifest 的 "op" 名 → 多输出 runner(第四清单 manifest_nonzero.json)。
	// 返回按 np.nonzero 元组顺序(第 0 维在前)的输出数组,每维一个 i64 一维数组。
	for name, fn := range map[string]func(ins []*num.Array, args caseArgs) ([]*num.Array, error){
		"nonzero": func(ins []*num.Array, _ caseArgs) ([]*num.Array, error) {
			return num.Nonzero(ins[0])
		}, "meshgrid": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			return num.MeshGrid(ins, args.XY)
		},
		"gradient": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			return num.Gradient(ins[0], args.Axis)
		},
		"histogram": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			return num.Histogram(ins[0], args.Bins, args.Lo, args.Hi)
		},
		"unique_idx": func(ins []*num.Array, _ caseArgs) ([]*num.Array, error) {
			return num.UniqueIdx(ins[0])
		},
		"slogdet": func(ins []*num.Array, _ caseArgs) ([]*num.Array, error) {
			sign, logdet, err := num.Slogdet(ins[0])
			if err != nil {
				return nil, err
			}
			s, err := scalarArr(sign)
			if err != nil {
				return nil, err
			}
			l, err := scalarArr(logdet)
			if err != nil {
				return nil, err
			}
			return []*num.Array{s, l}, nil
		}, "array_split": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			ax := 0
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			return num.ArraySplit(ins[0], args.Sections, ax)
		},
		"split_at": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			ax := 0
			if len(args.Axis) > 0 {
				ax = args.Axis[0]
			}
			return num.SplitAt(ins[0], args.SplitIdx, ax)
		},
		"hsplit": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			return num.HSplit(ins[0], args.Sections)
		},
		"vsplit": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			return num.VSplit(ins[0], args.Sections)
		},
		"histogram2d": func(ins []*num.Array, args caseArgs) ([]*num.Array, error) {
			return num.Histogram2D(ins[0], ins[1], args.BinsX, args.BinsY,
				args.XLo, args.XHi, args.YLo, args.YHi)
		},
	} {
		xcheckMultiOps[name] = fn
	}
}

// TestMultiOutputRunner 自证多输出注册表与 nonzero 的元组顺序(不依赖 reference fixtures):
// 每维一个 i64 一维数组,第 0 维索引数组在前;全零案每维输出 shape [0]、Size==0。
func TestMultiOutputRunner(t *testing.T) {
	fn, ok := xcheckMultiOps["nonzero"]
	if !ok {
		t.Fatalf("多输出注册表未登记 nonzero")
	}
	mk := func(shape []int, x ...float64) *num.Array {
		t.Helper()
		a, err := num.FromSlice(x)
		if err != nil {
			t.Fatalf("FromSlice(f8):%v", err)
		}
		r, err := a.Reshape(shape)
		if err != nil {
			t.Fatalf("Reshape %v:%v", shape, err)
		}
		return r
	}
	vals := func(a *num.Array) []int64 {
		t.Helper()
		f := flat1D(a)
		out := make([]int64, 0, f.Size())
		for i := 0; i < f.Size(); i++ {
			v, err := f.Item(i)
			if err != nil {
				t.Fatalf("取值 [%d]:%v", i, err)
			}
			x, ok := v.(int64)
			if !ok {
				t.Fatalf("取值 [%d] 非 int64:%T", i, v)
			}
			out = append(out, x)
		}
		return out
	}
	type want struct {
		shape []int
		vals  []int64
	}
	check := func(t *testing.T, in *num.Array, outs []want) {
		t.Helper()
		got, err := fn([]*num.Array{in}, caseArgs{})
		if err != nil {
			t.Fatalf("nonzero:%v", err)
		}
		if len(got) != len(outs) {
			t.Fatalf("输出个数 %d ≠ 期望 %d", len(got), len(outs))
		}
		for i, w := range outs {
			if got[i].Dtype() != num.DInt64 {
				t.Errorf("输出[%d] dtype %s ≠ i64", i, got[i].Dtype())
			}
			if !slices.Equal(got[i].Shape(), w.shape) {
				t.Errorf("输出[%d] shape %v ≠ %v", i, got[i].Shape(), w.shape)
			}
			if got[i].Size() != len(w.vals) {
				t.Errorf("输出[%d] size %d ≠ %d", i, got[i].Size(), len(w.vals))
			}
			if g := vals(got[i]); !slices.Equal(g, w.vals) {
				t.Errorf("输出[%d] 值 %v ≠ %v", i, g, w.vals)
			}
		}
	}

	t.Run("2d", func(t *testing.T) {
		check(t, mk([]int{2, 3}, 0, 1, 0, 2, 0, 3), []want{
			{[]int{3}, []int64{0, 1, 1}}, // 第 0 维在前
			{[]int{3}, []int64{1, 0, 2}},
		})
	})
	t.Run("all_zero", func(t *testing.T) {
		check(t, mk([]int{3, 4}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), []want{
			{[]int{0}, []int64{}},
			{[]int{0}, []int64{}},
		})
	})
	t.Run("1d", func(t *testing.T) {
		check(t, mk([]int{6}, 0, 5, 0, -3, 0, 7), []want{
			{[]int{3}, []int64{1, 3, 5}},
		})
	})
	t.Run("1d_nan_and_negzero", func(t *testing.T) {
		// NaN 计入(真值口径 NaN != 0 为真),-0.0 不计入
		check(t, mk([]int{4}, 0, math.NaN(), math.Copysign(0, -1), 2), []want{
			{[]int{2}, []int64{1, 3}},
		})
	})
}

// scalarArr 用 float64 值造 0-d 数组(det/norm 这类标量结果的对拍需要)。
func scalarArr(v float64) (*num.Array, error) {
	a, err := num.New([]int{}, num.DFloat64)
	if err != nil {
		return nil, err
	}
	if err := a.SetItem([]int{}, v); err != nil {
		return nil, err
	}
	return a, nil
}

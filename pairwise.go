package num

// :pairwise 求和(numpy 逐位复刻,B 拥有)。
//
// 源码考据:numpy/_core/src/umath/loops_utils.h.src 的 @TYPE@_pairwise_sum
// (BSD-3;结构照抄,实现按 Go 习语重写,):
// - PW_BLOCKSIZE = 128;
// - n < 8:朴素累加,初值 **-0.0**(numpy 注释:全 -0 输入求和应得 -0,
// 因为 0 + -0 = +0 而 -0 + -0 = -0);
// - 8 <= n <= 128:8 路累加器展开(每轮 8 个),收尾树形合并
// ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)),再补 n%8 尾链;
// - n > 128:n2 = n/2 向下取整到 8 的倍数,折半递归后相加。
//
// 逐位可行性:纯加法链、无乘法面(无 FMA 融合),Go 规范禁止 FP 重结合
// (spec:"floating-point operations may not be reassociated")→ 与 numpy 逐位
// 相等(amd64 基线;证据 = reference sum 案 bitwise 全过)。
//
// 恒等元(,实测 n=1..129):C 函数对全 -0 输入返回 -0.0,但 np.sum 在所有
// n 下返回 **+0.0** —— 归约层把结果再加一次 +0.0 恒等元(0.0 + -0.0 = +0.0;
// 对非零值/NaN/Inf 该加法是精确恒等)。故公开包装层加恒等元,np.sum 逐位对齐。

import "math"

const pwBlockSize = 128

// sumPairwise 逐位复刻 numpy pairwise sum(f32 用 f32 累加器,numpy 同)。
func sumPairwise[T FloatKinds](x []T) T {
	n := len(x)
	if n < 8 {
		res := T(math.Copysign(0, -1)) // -0.0(见文件头)
		for _, v := range x {
			res += v
		}
		return res
	}
	if n <= pwBlockSize {
		r0, r1, r2, r3 := x[0], x[1], x[2], x[3]
		r4, r5, r6, r7 := x[4], x[5], x[6], x[7]
		i := 8
		for ; i < n-(n%8); i += 8 {
			r0 += x[i]
			r1 += x[i+1]
			r2 += x[i+2]
			r3 += x[i+3]
			r4 += x[i+4]
			r5 += x[i+5]
			r6 += x[i+6]
			r7 += x[i+7]
		}
		// 树形合并(顺序即 numpy 的顺序,不可改写)
		res := ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
		for ; i < n; i++ {
			res += x[i]
		}
		return res
	}
	// 折半递归:避免切在非 8 倍数上(numpy 同)
	n2 := n / 2
	n2 -= n2 % 8
	return sumPairwise(x[:n2]) + sumPairwise(x[n2:])
}

// sumPairwiseF64 契约名( §2.3)。含 +0.0 恒等元(np.sum 同)。
func sumPairwiseF64(x []float64) float64 { return 0 + sumPairwise(x) }

// sumPairwiseF32 契约名( §2.3)。含 +0.0 恒等元(np.sum 同)。
func sumPairwiseF32(x []float32) float32 { return 0 + sumPairwise(x) }

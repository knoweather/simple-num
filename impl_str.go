package num

// :定宽字符串 S/U 的比较(B 拥有)。
//
// 语义基准(numpy 2.5.4 实测,):
// - S(定宽字节)/U(定宽 4 字节码元)支持 6 种比较,输出 bool;
// - 比较口径:尾 NUL 裁剪后按**字节序**(S)/**码点序**(U)—— Go 的 string
// 比较为字节序,UTF-8 保序 → 与 U 的码点序一致;
// - 同 kind 不同宽度(S3 vs S4)可比较(按值,非按填充);S × U 混合 →
// numpy TypeError → ErrDtype;
// - numpy 的 np.add(S,S)/np.add(U,U) 支持**字符串拼接**(→ 更宽 dtype),
// 本库明确"字符运算不做" → 返回 ErrDtype;
// - 存取/编解码/npy 往返由 codec 承担(本文件只补比较面)。

import "fmt"

// isStrKind 定宽字符串 kind(S/U)。
func isStrKind(k Kind) bool { return k == Bytes || k == Unicode }

// strCompare S/U 比较(广播 + 逐元素 decode 比较;尾 NUL 由 codec 裁剪)。
func strCompare(op string, x, y *Array) (*Array, error) {
	kx, ky := x.Dtype().Kind, y.Dtype().Kind
	if kx != Bytes && kx != Unicode {
		return nil, fmt.Errorf("num: 比较不支持 %s: %w", x.Dtype(), ErrDtype)
	}
	if kx != ky {
		return nil, fmt.Errorf("num: S 与 U 不可比较(numpy TypeError 同拒): %w", ErrDtype)
	}
	outSh, err := broadcastShapes(x.Shape(), y.Shape())
	if err != nil {
		return nil, fmt.Errorf("num: 字符串比较: %w", err)
	}
	out, err := New(outSh, DBool)
	if err != nil {
		return nil, err
	}
	if out.Size() == 0 {
		return out, nil
	}
	xs := elemStepsBroadcast(x.Shape(), x.Strides(), outSh)
	ys := elemStepsBroadcast(y.Shape(), y.Strides(), outSh)
	ow := win[bool](engineWindow(out))
	idx := make([]int, len(outSh))
	for k := 0; k < out.Size(); k++ {
		xo, yo := x.Offset(), y.Offset()
		for d := range idx {
			xo += idx[d] * xs[d]
			yo += idx[d] * ys[d]
		}
		xv, err := decodeOne(x.Dtype(), xo, x.buf)
		if err != nil {
			return nil, err
		}
		yv, err := decodeOne(y.Dtype(), yo, y.buf)
		if err != nil {
			return nil, err
		}
		a, b := xv.(string), yv.(string)
		switch op {
		case "less":
			ow[k] = a < b
		case "less_equal":
			ow[k] = a <= b
		case "greater":
			ow[k] = a > b
		case "greater_equal":
			ow[k] = a >= b
		case "equal":
			ow[k] = a == b
		case "not_equal":
			ow[k] = a != b
		}
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < outSh[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out, nil
}

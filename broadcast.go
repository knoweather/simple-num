package num

import "fmt"

// —— T4 广播引擎(numpy 右对齐语义,参考数据对拍在 broadcast_test)——

// broadcastShapes 两形状的 numpy 同款结果(右对齐补 1;不兼容 → ErrShape)。
func broadcastShapes(x, y []int) ([]int, error) {
	nd := len(x)
	if len(y) > nd {
		nd = len(y)
	}
	out := make([]int, nd)
	for i := 1; i <= nd; i++ {
		dx, dy := 1, 1 // 缺失的头部维按 1(numpy 同款)
		if i <= len(x) {
			dx = x[len(x)-i]
		}
		if i <= len(y) {
			dy = y[len(y)-i]
		}
		switch {
		case dx == dy:
			out[nd-i] = dx
		case dx == 1:
			out[nd-i] = dy
		case dy == 1:
			out[nd-i] = dx
		default:
			return nil, fmt.Errorf("num: 广播 %v × %v 长度不相容: %w", x, y, ErrShape)
		}
	}
	return out, nil
}

// BroadcastTo 形状广播视图(零拷贝;广播维写 stride 0)。不兼容 → ErrShape。
// numpy broadcast_to 输出的数组为"只读共享"视图约定,本库由不可变视图承担。
func (a *Array) BroadcastTo(shape []int) (*Array, error) {
	sh, _, err := normShapeCopy(shape)
	if err != nil {
		return nil, err
	}
	if !broadcastOk(a.shape, sh) {
		return nil, fmt.Errorf("num: %v 无法广播到 %v: %w", a.shape, shape, ErrShape)
	}
	st := elemStepsBroadcast(a.shape, a.st, sh)
	return newViewInternal(a, sh, st, a.off), nil
}

// BroadcastTo 已覆盖"取广播视图"的唯一需求;早先的 broadcastOf 原型(BroadcastTo
// 两连)在引擎改走 normalizeOperand 一体化归一路径后已无调用者,Day5 删除。

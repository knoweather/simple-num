package num

import (
	"fmt"
	"math"
)

// —— 掩码数组(Step 5):MaskedArray ——
//
// 设计:独立类型(不改造 Array),持有一份数据与一份 bool 掩码(同形状),
// 语义对齐 np.ma.MaskedArray:
// - 掩码为 true 的位置"无效";归约时跳过,取值时由 Filled 指定替身;
// - 归约结果仍是 MaskedArray:整条切片全被掩码时,该位置结果被掩码
// (与 numpy 的 masked constant 一致);
// - 二元运算的掩码取并集(任一侧无效则结果无效)。
//
// 实现策略:复用既有引擎 —— 求和把掩码位填 0、求极值填 dtype 极值,
// 这样结果与 numpy 的"跳过无效值"口径逐位一致,无需另写归约内核。

// MaskedArray 掩码数组:数据 + bool 掩码。
type MaskedArray struct {
	data *Array
	mask *Array
}

// NewMasked 由数据与掩码构造;两者形状必须一致。
func NewMasked(data, mask *Array) (*MaskedArray, error) {
	if data == nil || mask == nil {
		return nil, fmt.Errorf("num: ma: 数据或掩码为 nil: %w", ErrDtype)
	}
	if !shapeEqual(data.Shape(), mask.Shape()) {
		return nil, fmt.Errorf("num: ma: 数据形状 %v ≠ 掩码形状 %v: %w", data.Shape(), mask.Shape(), ErrShape)
	}
	if mask.Dtype() != DBool {
		c, err := Cast(mask, DBool)
		if err != nil {
			return nil, fmt.Errorf("num: ma: 掩码须可转 bool: %w", err)
		}
		mask = c
	}
	return &MaskedArray{data: data, mask: mask}, nil
}

// AsMasked 把普通数组包成全 False 掩码的 MaskedArray(numpy 的 np.ma.array 同)。
func AsMasked(a *Array) *MaskedArray {
	m, err := New(a.Shape(), DBool)
	if err != nil { // 形状来自既有数组,构造不会失败
		panic("num: ma: AsMasked 内部构造失败")
	}
	return &MaskedArray{data: a, mask: m}
}

// Data 返回底层数据(只读约定:修改会影响本对象)。
func (m *MaskedArray) Data() *Array { return m.data }

// MaskArray 返回底层掩码。
func (m *MaskedArray) MaskArray() *Array { return m.mask }

// Shape / Size / NDim / Dtype 与 Array 同名同义。
func (m *MaskedArray) Shape() []int { return m.data.Shape() }
func (m *MaskedArray) Size() int    { return m.data.Size() }
func (m *MaskedArray) NDim() int    { return m.data.NDim() }
func (m *MaskedArray) Dtype() Dtype { return m.data.Dtype() }

// IsMasked 是否有任何被掩码的元素(numpy 的 ma.is_masked)。
func (m *MaskedArray) IsMasked() bool {
	w := win[uint8](engineWindow(m.mask.AsContiguous()))
	for _, v := range w {
		if v != 0 {
			return true
		}
	}
	return false
}

// Count 未被掩码的元素个数(numpy 的 ma.count)。
func (m *MaskedArray) Count() int {
	return m.Size() - m.maskedCount()
}

func (m *MaskedArray) maskedCount() int {
	w := win[uint8](engineWindow(m.mask.AsContiguous()))
	n := 0
	for _, v := range w {
		if v != 0 {
			n++
		}
	}
	return n
}

// Filled ≡ ma.filled(v):把被掩码的位置替换为 v,返回普通数组。
func (m *MaskedArray) Filled(v any) (*Array, error) {
	out, err := transformCopy(m.data, m.data.Shape(), m.data.Dtype())
	if err != nil {
		return nil, err
	}
	if err := PutMask(out, m.mask, scalarArrOf(m.data.Dtype(), v)); err != nil {
		return nil, err
	}
	return out, nil
}

// scalarArrOf 把标量按目标 dtype 转成 1 元素数组(供 PutMask 填值)。
func scalarArrOf(dt Dtype, v any) *Array {
	a, err := New([]int{1}, dt)
	if err != nil {
		return nil
	}
	if err := a.SetItem([]int{0}, v); err != nil {
		return nil
	}
	return a
}

// Compressed ≡ ma.compressed:取出所有未掩码元素(1-D)。
func (m *MaskedArray) Compressed() (*Array, error) {
	notMasked, err := LogicalNot(m.mask)
	if err != nil {
		return nil, err
	}
	return Compress(m.data, notMasked)
}

// reduction 掩码归约的公共骨架:按"填值 + 复用普通归约 + 计数掩码"实现。
func (m *MaskedArray) reduction(axis []int, keepdims bool,
	fill any, rn func(*Array, []int, bool) (*Array, error),
) (*MaskedArray, error) {
	// 计数:未被掩码的数量(按同样轴归约)
	notM, err := LogicalNot(m.mask)
	if err != nil {
		return nil, err
	}
	cntF, err := Cast(notM, DFloat64)
	if err != nil {
		return nil, err
	}
	counts, err := Sum(cntF, axis, keepdims)
	if err != nil {
		return nil, err
	}
	// 填值后走普通归约
	filled, err := m.Filled(fill)
	if err != nil {
		return nil, err
	}
	res, err := rn(filled, axis, keepdims)
	if err != nil {
		return nil, err
	}
	// 计数为 0 的位置 → 结果被掩码
	cw := win[float64](engineWindow(counts.AsContiguous()))
	mask, err := New(counts.Shape(), DBool)
	if err != nil {
		return nil, err
	}
	mw := win[uint8](engineWindow(mask))
	for i, c := range cw {
		if c == 0 {
			mw[i] = 1
		}
	}
	return &MaskedArray{data: res, mask: mask}, nil
}

// Sum ≡ ma.sum:跳过被掩码元素求和(全掩码的切片结果被掩码)。
func (m *MaskedArray) Sum(axis []int, keepdims bool) (*MaskedArray, error) {
	return m.reduction(axis, keepdims, 0, Sum)
}

// Mean ≡ ma.mean:跳过被掩码元素求均值(按未掩码个数归一)。
func (m *MaskedArray) Mean(axis []int, keepdims bool) (*MaskedArray, error) {
	sums, err := m.reduction(axis, keepdims, 0, Sum)
	if err != nil {
		return nil, err
	}
	notM, err := LogicalNot(m.mask)
	if err != nil {
		return nil, err
	}
	cntF, err := Cast(notM, DFloat64)
	if err != nil {
		return nil, err
	}
	counts, err := Sum(cntF, axis, keepdims)
	if err != nil {
		return nil, err
	}
	// 0 计数位置用 1 兜底避免 0/0,结果仍会被掩码
	cw := win[float64](engineWindow(counts.AsContiguous()))
	for i, c := range cw {
		if c == 0 {
			cw[i] = 1
		}
	}
	sd, err := Cast(sums.data, DFloat64)
	if err != nil {
		return nil, err
	}
	mean, err := Div(sd, counts)
	if err != nil {
		return nil, err
	}
	return &MaskedArray{data: mean, mask: sums.mask}, nil
}

// Min ≡ ma.min:跳过被掩码元素取最小。
func (m *MaskedArray) Min(axis []int, keepdims bool) (*MaskedArray, error) {
	return m.reduction(axis, keepdims, maxOfDtype(m.data.Dtype()), Min)
}

// Max ≡ ma.max:跳过被掩码元素取最大。
func (m *MaskedArray) Max(axis []int, keepdims bool) (*MaskedArray, error) {
	return m.reduction(axis, keepdims, minOfDtype(m.data.Dtype()), Max)
}

// maxOfDtype / minOfDtype 取 dtype 的极值(用作极值归约的填充值)。
func maxOfDtype(dt Dtype) any {
	switch dt.Kind {
	case Float64:
		return math.Inf(1)
	case Float32:
		return float32(math.Inf(1))
	case Float16: // 由 f16 路径处理,这里给 f32 无穷
		return float32(math.Inf(1))
	case Int8:
		return int8(math.MaxInt8)
	case Int16:
		return int16(math.MaxInt16)
	case Int32:
		return int32(math.MaxInt32)
	case Int64:
		return int64(math.MaxInt64)
	case Uint8:
		return uint8(math.MaxUint8)
	case Uint16:
		return uint16(math.MaxUint16)
	case Uint32:
		return uint32(math.MaxUint32)
	case Uint64:
		return uint64(math.MaxUint64)
	}
	return 0
}

func minOfDtype(dt Dtype) any {
	switch dt.Kind {
	case Float64:
		return math.Inf(-1)
	case Float32:
		return float32(math.Inf(-1))
	case Float16:
		return float32(math.Inf(-1))
	case Int8:
		return int8(math.MinInt8)
	case Int16:
		return int16(math.MinInt16)
	case Int32:
		return int32(math.MinInt32)
	case Int64:
		return int64(math.MinInt64)
	case Uint8:
		return uint8(0)
	case Uint16:
		return uint16(0)
	case Uint32:
		return uint32(0)
	case Uint64:
		return uint64(0)
	}
	return 0
}

// binaryMasked 二元运算骨架:数据走普通算子,掩码取并集。
func binaryMasked(a, b *MaskedArray, op func(x, y *Array) (*Array, error)) (*MaskedArray, error) {
	if !shapeEqual(a.Shape(), b.Shape()) {
		// 交给普通算子做广播;掩码也必须广播到同一形状
		sh, err := broadcastShapes(a.Shape(), b.Shape())
		if err != nil {
			return nil, fmt.Errorf("num: ma: 形状不可广播: %w", err)
		}
		da, err := normalizeOperand(a.data, sh, a.data.Dtype())
		if err != nil {
			return nil, err
		}
		db, err := normalizeOperand(b.data, sh, b.data.Dtype())
		if err != nil {
			return nil, err
		}
		ma, err := normalizeOperand(a.mask, sh, DBool)
		if err != nil {
			return nil, err
		}
		mb, err := normalizeOperand(b.mask, sh, DBool)
		if err != nil {
			return nil, err
		}
		res, err := op(da, db)
		if err != nil {
			return nil, err
		}
		mm, err := LogicalOr(ma, mb)
		if err != nil {
			return nil, err
		}
		return &MaskedArray{data: res, mask: mm}, nil
	}
	res, err := op(a.data, b.data)
	if err != nil {
		return nil, err
	}
	mm, err := LogicalOr(a.mask, b.mask)
	if err != nil {
		return nil, err
	}
	return &MaskedArray{data: res, mask: mm}, nil
}

// MaskedAdd ≡ ma.add:数据相加,掩码取并集。
func MaskedAdd(a, b *MaskedArray) (*MaskedArray, error) { return binaryMasked(a, b, Add) }

// MaskedSubtract ≡ ma.subtract。
func MaskedSubtract(a, b *MaskedArray) (*MaskedArray, error) { return binaryMasked(a, b, Subtract) }

// MaskedMultiply ≡ ma.multiply。
func MaskedMultiply(a, b *MaskedArray) (*MaskedArray, error) { return binaryMasked(a, b, Multiply) }

// MaskedDivide ≡ ma.divide。
func MaskedDivide(a, b *MaskedArray) (*MaskedArray, error) { return binaryMasked(a, b, Div) }

package num

import (
	"fmt"
	"math"
)

// —— 工厂:Arange / LinSpace / Eye ——
//
// 语义一律"numpy 2.x 实测同构"( /):
// - Arange:长度 = ceil((stop-start)/step)(step=0 → ErrZeroDiv,numpy ZeroDivisionError 对应),
// 元素 = start + i*step(分离乘加,不融合 —— Go 不做 FMA 收缩,与 numpy 逐位一致);
// 整型 dtype 且三参均为整值 → int64 直算(大整数不被 f64 截断;溢出静默回绕,numpy 同)。
// - LinSpace:div = (endpoint && n>1) ? n-1 : n;step = (stop-start)/div;
// y[i] = i*step + start;末点(endpoint && n>1)精确置 stop —— 与 numpy 逐位。
// - Eye:n×n 单位阵(方阵口径;numpy eye 的 M/k 变体出界,记)。
// 上界熔断:长度/n 超过 maxFactoryLen → ErrShape(防 int 溢出与荒谬分配;numpy 同 raise)。

const maxFactoryLen = 1 << 40

// isIntKind 整型族判定(bool 不计入)。
func isIntKind(k Kind) bool {
	switch k {
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
		return true
	}
	return false
}

// Arange [start, stop) 步长 step 的等差数列(dtype 决定元素类型)。
func Arange(start, stop, step float64, dt Dtype) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	if step == 0 {
		return nil, fmt.Errorf("num: Arange step=0: %w", ErrZeroDiv)
	}
	q := (stop - start) / step
	if math.IsNaN(q) || q > maxFactoryLen {
		return nil, fmt.Errorf("num: Arange(start=%g,stop=%g,step=%g) 长度 %g 非法: %w",
			start, stop, step, q, ErrShape)
	}
	length := 0
	if q > 0 {
		length = int(math.Ceil(q))
	}
	out, err := New([]int{length}, dt)
	if err != nil {
		return nil, err
	}
	it := dt.Itemsize
	// 整型快径:三参整值 → int64 精确路径(Go 有符号溢出按二补回绕,numpy 同)。
	if isIntKind(dt.Kind) &&
		start == math.Trunc(start) && stop == math.Trunc(stop) && step == math.Trunc(step) {
		si := int64(start)
		st := int64(step)
		for i := 0; i < length; i++ {
			storeI64(dt.Kind, out.buf[i*it:(i+1)*it], si+int64(i)*st)
		}
		return out, nil
	}
	for i := 0; i < length; i++ {
		if err := storeElemF64(dt, i*it, out.buf, start+float64(i)*step); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// LinSpace [start, stop] 上 n 点等距采样(endpoint=false 时为左闭右开)。返回 f64。
func LinSpace(start, stop float64, n int, endpoint bool) (*Array, error) {
	if n < 0 {
		return nil, fmt.Errorf("num: LinSpace n=%d <0: %w", n, ErrShape)
	}
	if n > maxFactoryLen {
		return nil, fmt.Errorf("num: LinSpace n=%d 超界: %w", n, ErrShape)
	}
	out, err := New([]int{n}, DFloat64)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return out, nil
	}
	div := n
	if endpoint && n > 1 {
		div = n - 1
	}
	step := (stop - start) / float64(div)
	for i := 0; i < n; i++ {
		storeF64(Float64, out.buf[i*8:(i+1)*8], start+float64(i)*step)
	}
	if endpoint && n > 1 { // 末点精确置 stop(消累积误差,numpy 同)
		storeF64(Float64, out.buf[(n-1)*8:n*8], stop)
	}
	return out, nil
}

// Eye n×n 单位阵(dtype 决定 1 的编码:bool→true、整型→1、浮点→1.0)。
func Eye(n int, dt Dtype) (*Array, error) {
	if n < 0 {
		return nil, fmt.Errorf("num: Eye n=%d <0: %w", n, ErrShape)
	}
	if n > maxFactoryLen {
		return nil, fmt.Errorf("num: Eye n=%d 超界: %w", n, ErrShape)
	}
	out, err := Zeros([]int{n, n}, dt)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return out, nil
	}
	it := dt.Itemsize
	row := n * it
	for i := 0; i < n; i++ {
		if err := storeElemF64(dt, i*row+i*it, out.buf, 1); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// storeElemF64 数值元素直存(免 any 装箱);非数值 kind 交通用编码器
// (complex/datetime/S/U 在 – 前返回对应错误)。
func storeElemF64(dt Dtype, off int, buf []byte, v float64) error {
	b := buf[off : off+dt.Itemsize]
	switch dt.Kind {
	case Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64:
		storeI64(dt.Kind, b, int64(v)) // 向零截断 + 回绕(numpy 赋值 cast 同)
		return nil
	case Float32, Float64:
		storeF64(dt.Kind, b, v)
		return nil
	}
	return encodeOne(dt, off, buf, v)
}

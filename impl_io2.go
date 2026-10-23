package num

import (
	"fmt"
	"io"
	"os"
)

// FromBuffer ≡ np.frombuffer(buf, dtype, count=-1, offset=0):把字节缓冲按 dtype
// 解释成 **1-D** 数组。
//
// - **零拷贝视图**:与 numpy 一致,返回的数组直接引用 b[offset:] 的内存,
// 调用方改动 b 会反映到数组上(要独立副本请自行拷贝);
// - count < 0 表示"取到缓冲末尾";长度不足时按实际可容纳的元素数截断
// (numpy 会报错,本库选择截断以便流式读取,属受控差异,文档注明);
// - offset 越界 → ErrIndex;dtype 非法 → ErrDtype。
func FromBuffer(b []byte, dt Dtype, count, offset int) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	es := dt.Itemsize
	if offset < 0 || offset > len(b) {
		return nil, fmt.Errorf("num: frombuffer: offset %d 越界(缓冲 %dB): %w", offset, len(b), ErrIndex)
	}
	avail := (len(b) - offset) / es
	if count < 0 || count > avail {
		count = avail
	}
	return FromBytes(b[offset:offset+count*es], dt, []int{count})
}

// FromFile ≡ np.fromfile(path, dtype, count=-1, offset=0):从**裸二进制**文件按
// dtype 读 1-D 数组(仅二进制形态;numpy 的 sep= 文本模式不在本接口内,
// 文本请用 LoadTxt 一类的后续接口)。
//
// - count < 0 表示读到文件末尾;
// - offset 为字节偏移,越界 → ErrIndex;
// - 与 numpy 同:不做字节序交换(文件按本机小端解释;需要大端请先自行翻转)。
func FromFile(path string, dt Dtype, count, offset int) (*Array, error) {
	if err := checkDt(dt); err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("num: fromfile: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("num: fromfile: %w", err)
	}
	size := info.Size()
	if offset < 0 || int64(offset) > size {
		return nil, fmt.Errorf("num: fromfile: offset %d 越界(文件 %dB): %w", offset, size, ErrIndex)
	}
	es := int64(dt.Itemsize)
	avail := (size - int64(offset)) / es
	n := int64(count)
	if n < 0 || n > avail {
		n = avail
	}
	buf := make([]byte, n*es)
	if _, err := f.ReadAt(buf, int64(offset)); err != nil && err != io.EOF {
		return nil, fmt.Errorf("num: fromfile: %w", err)
	}
	return FromBytes(buf, dt, []int{int(n)})
}

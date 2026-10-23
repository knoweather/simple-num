package num

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// FromBuffer:零拷贝视图 + count/offset。
func TestFromBuffer(t *testing.T) {
	a := bArr(t, []float64{1, 2, 3, 4})
	arr, err := FromBuffer(a.buf, DFloat64, -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(arr.Shape(), []int{4}) || arr.Dtype() != DFloat64 {
		t.Fatalf("形状/dtype 不符: %v %s", arr.Shape(), arr.Dtype())
	}
	bWantF64(t, arr, 1, 2, 3, 4)

	// 零拷贝:整字改成 9.0,数组应跟着变
	bits := math.Float64bits(9.0)
	binary.LittleEndian.PutUint64(a.buf[0:8], bits)
	if v, _ := arr.Item(0); v != 9.0 {
		t.Fatalf("FromBuffer 应为零拷贝视图,得 %v", v)
	}

	// count + offset(offset=8 字节 = 跳过第 1 个 f64)
	part, err := FromBuffer(a.buf, DFloat64, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, part, 2, 3)

	// count 超出可容纳数 → 截断(numpy 会报错,本库受控差异)
	all, err := FromBuffer(a.buf, DFloat64, 100, 0)
	if err != nil {
		t.Fatal(err)
	}
	if all.Size() != 4 {
		t.Fatalf("count 超界应截断到 4,得 %d", all.Size())
	}

	if _, err := FromBuffer(a.buf, DFloat64, 1, 999); !errors.Is(err, ErrIndex) {
		t.Fatalf("offset 越界应 ErrIndex,得 %v", err)
	}
}

// FromFile:与 numpy tofile 的裸二进制互读。
func TestFromFile(t *testing.T) {
	a := bArr(t, []float64{1.5, -2.25, 3, 4.75})
	dir := t.TempDir()
	p := filepath.Join(dir, "raw.bin")
	if err := os.WriteFile(p, a.buf, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := FromFile(p, DFloat64, -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !shapeEqual(got.Shape(), []int{4}) {
		t.Fatalf("形状 %v, want [4]", got.Shape())
	}
	bWantF64(t, got, 1.5, -2.25, 3, 4.75)

	// count=2, offset=8
	part, err := FromFile(p, DFloat64, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	bWantF64(t, part, -2.25, 3)

	// 整型 dtype:i4 读同一文件(4 个 f64 = 32 字节 → 8 个 i4)
	i4, err := FromFile(p, DInt32, -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if i4.Size() != 8 {
		t.Fatalf("按 i4 读应得 8 个元素,得 %d", i4.Size())
	}
	// 1.5 的 IEEE 位型低 32 位 = 0x00000000(小端前 4 字节 = 0)
	if v, _ := i4.Item(0); v != int32(0) {
		t.Fatalf("[0] 按 i4 解释 = %v, want 0", v)
	}

	if _, err := FromFile(p, DFloat64, -1, 999); !errors.Is(err, ErrIndex) {
		t.Fatalf("offset 越界应 ErrIndex,得 %v", err)
	}
	if _, err := FromFile(filepath.Join(dir, "nope.bin"), DFloat64, -1, 0); err == nil {
		t.Fatal("文件不存在应报错")
	}
}

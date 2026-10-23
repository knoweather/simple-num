package num

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

// TestNPYHeaderNumpyLayout 锁住.npy 头部布局(numpy 的硬性规范):
// - magic "\x93NUMPY" + ver(1) + 0 + 2 字节长度(LE)
// - 头部 = dict 字面量 + **空格填充** + **末尾单个 '\n'**
// - (10 + 头部长度) 必须是 64 的倍数
//
// 曾经的 bug:把 '\n' 放在填充空格之前,我们自己的 reader 能读、但 numpy 的
// ast.literal_eval 会因"第二行空格"报 IndentationError —— 靠与 numpy 的双向
// 互操作实测才暴露出来。
func TestNPYHeaderNumpyLayout(t *testing.T) {
	cases := []*Array{
		bArr(t, []float64{1.5, -2.25}),
		mustMat(t, []int{2, 2}, []int32{1, 2, 3, 4}),
		func() *Array {
			// S/U 用构造器建(FromSlice 不支持 string)
			u, err := New([]int{2}, DUnicode(2))
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range []string{"ab", "cd"} {
				if err := u.SetItem([]int{i}, v); err != nil {
					t.Fatal(err)
				}
			}
			return u
		}(),
	}
	for _, a := range cases {
		raw, err := SaveNPYBytes(a)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.HasPrefix(raw, []byte("\x93NUMPY")) {
			t.Fatalf("%s: magic 不对", a.Dtype())
		}
		ver := raw[6]
		if ver != 1 && ver != 2 {
			t.Fatalf("%s: 版本 %d 不支持", a.Dtype(), ver)
		}
		var hlen int
		var off int
		if ver == 1 {
			hlen = int(binary.LittleEndian.Uint16(raw[8:10]))
			off = 10
		} else {
			hlen = int(binary.LittleEndian.Uint32(raw[8:12]))
			off = 12
		}
		hdr := raw[off : off+hlen]
		if off+hlen > len(raw) {
			t.Fatalf("%s: 头部长度 %d 越界", a.Dtype(), hlen)
		}
		// 末尾必须是单个 '\n'
		if hdr[len(hdr)-1] != '\n' {
			t.Fatalf("%s: 头部未以 '\\n' 结尾: %q", a.Dtype(), hdr)
		}
		// 除末尾外不允许出现 '\n'(否则 numpy 解析器会看到多行)
		if strings.Count(string(hdr), "\n") != 1 {
			t.Fatalf("%s: 头部含多个换行: %q", a.Dtype(), hdr)
		}
		// 64 字节对齐
		if (off+hlen)%64 != 0 {
			t.Fatalf("%s: 头部总长 %d 不是 64 的倍数", a.Dtype(), off+hlen)
		}
		// 我们自己的 reader 仍能读回
		back, err := LoadNPYBytes(raw)
		if err != nil {
			t.Fatalf("%s: 自读失败: %v", a.Dtype(), err)
		}
		if !shapeEqual(back.Shape(), a.Shape()) || back.Dtype() != a.Dtype() {
			t.Fatalf("%s: 自读形状/dtype 不符", a.Dtype())
		}
	}
}

package num

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
)

// —— npz 多数组归档(Step 13):SaveZ / SaveZCompressed / LoadZ ——
//
// 与 numpy 的格式一致:一个 ZIP 容器,每个数组一条 `<name>.npy` 条目
// (内容就是标准.npy 字节,复用既有的 SaveNPYBytes/LoadNPYBytes)。
// np.savez 用**不压缩**(Store),np.savez_compressed 用 Deflate。

// NpzFile 已加载的 npz 归档:保持条目顺序 + 名称→数组映射。
type NpzFile struct {
	names []string
	data  map[string]*Array
}

// Names 返回归档内的数组名(去掉.npy 后缀,保持写入顺序/numpy 的 zip 顺序)。
func (z *NpzFile) Names() []string { return append([]string(nil), z.names...) }

// Get 按名取数组。
func (z *NpzFile) Get(name string) (*Array, bool) {
	a, ok := z.data[name]
	return a, ok
}

// Len 归档内数组个数。
func (z *NpzFile) Len() int { return len(z.names) }

// SaveZ ≡ np.savez(path, **kwds):把多个数组写成不压缩的.npz。
// 路径缺少.npz 后缀时自动补(numpy 同)。
func SaveZ(path string, names []string, arrays []*Array) error {
	return saveZ(path, names, arrays, zip.Store)
}

// SaveZCompressed ≡ np.savez_compressed:同上但用 Deflate 压缩。
func SaveZCompressed(path string, names []string, arrays []*Array) error {
	return saveZ(path, names, arrays, zip.Deflate)
}

func saveZ(path string, names []string, arrays []*Array, method uint16) error {
	if len(names) != len(arrays) {
		return fmt.Errorf("num: savez: 名字 %d 个 ≠ 数组 %d 个: %w", len(names), len(arrays), ErrShape)
	}
	if !strings.HasSuffix(strings.ToLower(path), ".npz") {
		path += ".npz"
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("num: savez: %w", err)
	}
	defer f.Close()
	zw := zip.NewWriter(f)
	seen := map[string]bool{}
	for i, name := range names {
		if name == "" {
			name = fmt.Sprintf("arr_%d", i) // numpy 的匿名数组命名规则
		}
		if seen[name] {
			return fmt.Errorf("num: savez: 重复的数组名 %q: %w", name, ErrValue)
		}
		seen[name] = true
		raw, err := SaveNPYBytes(arrays[i])
		if err != nil {
			return fmt.Errorf("num: savez: 序列化 %q 失败: %w", name, err)
		}
		w, err := zw.CreateHeader(&zip.FileHeader{
			Name:   name + ".npy",
			Method: method,
		})
		if err != nil {
			return fmt.Errorf("num: savez: %w", err)
		}
		if _, err := w.Write(raw); err != nil {
			return fmt.Errorf("num: savez: 写入 %q 失败: %w", name, err)
		}
	}
	if err := zw.Close(); err != nil {
		return fmt.Errorf("num: savez: 收尾失败: %w", err)
	}
	return nil
}

// LoadZ ≡ np.load(path)(针对.npz):读入全部数组。
func LoadZ(path string) (*NpzFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("num: loadz: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("num: loadz: %w", err)
	}
	zr, err := zip.NewReader(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("num: loadz: 不是有效的 npz(zip)文件: %w", ErrValue)
	}
	out := &NpzFile{data: map[string]*Array{}}
	for _, zf := range zr.File {
		if !strings.HasSuffix(strings.ToLower(zf.Name), ".npy") {
			continue // 忽略非 npy 条目(如 numpy 可能写入的元数据)
		}
		rc, err := zf.Open()
		if err != nil {
			return nil, fmt.Errorf("num: loadz: 打开 %q 失败: %w", zf.Name, err)
		}
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, rc); err != nil {
			rc.Close()
			return nil, fmt.Errorf("num: loadz: 读取 %q 失败: %w", zf.Name, err)
		}
		rc.Close()
		arr, err := LoadNPYBytes(buf.Bytes())
		if err != nil {
			return nil, fmt.Errorf("num: loadz: 解析 %q 失败: %w", zf.Name, err)
		}
		name := strings.TrimSuffix(zf.Name, ".npy")
		if _, dup := out.data[name]; dup {
			return nil, fmt.Errorf("num: loadz: 重复条目 %q: %w", name, ErrValue)
		}
		out.names = append(out.names, name)
		out.data[name] = arr
	}
	if len(out.names) == 0 {
		return nil, fmt.Errorf("num: loadz: 归档里没有 .npy 条目: %w", ErrValue)
	}
	return out, nil
}

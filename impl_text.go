package num

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// —— 文本 IO 与小函数收尾(Step 11):savetxt / loadtxt / place / piecewise ——

// SaveTxt ≡ np.savetxt(path, a, fmt, delimiter):
// 2-D 每行一条记录、1-D 每元素一行;fmt 为 Go 的格式化串(空则浮点用 %.18e、整数用 %d)。
func SaveTxt(path string, a *Array, fmtStr, delim string) error {
	if a.NDim() > 2 {
		return fmt.Errorf("num: savetxt: 只支持 1-D/2-D(得 %d 维): %w", a.NDim(), ErrShape)
	}
	if fmtStr == "" {
		switch a.Dtype().Kind {
		case Float16, Float32, Float64:
			fmtStr = "%.18e"
		case Complex64, Complex128:
			fmtStr = "(%.18e%+.18ej)"
		default:
			fmtStr = "%d"
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("num: savetxt: %w", err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	// numpy 口径:2-D 每行一条记录;1-D 每元素一行;0-d 单行单值。
	rows, cols := 1, a.Size()
	switch a.NDim() {
	case 2:
		rows, cols = a.Shape()[0], a.Shape()[1]
	case 1:
		rows, cols = a.Size(), 1
	case 0:
		rows, cols = 1, 1
	}
	flat := a.AsContiguous().Flatten()
	for r := 0; r < rows; r++ {
		for cc := 0; cc < cols; cc++ {
			v, err := flat.Item(r*cols + cc)
			if err != nil {
				return err
			}
			if cc > 0 && delim != "" {
				if _, err := w.WriteString(delim); err != nil {
					return err
				}
			}
			if _, err := fmt.Fprintf(w, fmtStr, v); err != nil {
				return fmt.Errorf("num: savetxt: 格式化失败(%v): %w", v, ErrValue)
			}
		}
		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}
	return nil
}

// LoadTxt ≡ np.loadtxt(path, dtype, delimiter, skiprows):
// 单列返回 1-D,多列返回 2-D(rows × cols);空行与 '#' 注释行跳过(numpy 默认 comments='#')。
func LoadTxt(path string, dt Dtype, delim string, skipRows int) (*Array, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("num: loadtxt: %w", err)
	}
	defer f.Close()
	var rows [][]float64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	lineNo := 0
	for sc.Scan() {
		line := sc.Text()
		if lineNo < skipRows {
			lineNo++
			continue
		}
		lineNo++
		t := strings.TrimSpace(line)
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		var fields []string
		if delim == "" {
			fields = strings.Fields(t)
		} else {
			fields = strings.Split(t, delim)
		}
		vals := make([]float64, 0, len(fields))
		for _, fd := range fields {
			fd = strings.TrimSpace(fd)
			if fd == "" {
				continue
			}
			v, err := strconv.ParseFloat(fd, 64)
			if err != nil {
				return nil, fmt.Errorf("num: loadtxt: 第 %d 行无法解析 %q: %w", lineNo, fd, ErrValue)
			}
			vals = append(vals, v)
		}
		if len(vals) == 0 {
			continue
		}
		if len(rows) > 0 && len(vals) != len(rows[0]) {
			return nil, fmt.Errorf("num: loadtxt: 第 %d 行列数 %d ≠ 首行 %d: %w",
				lineNo, len(vals), len(rows[0]), ErrShape)
		}
		rows = append(rows, vals)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("num: loadtxt: %w", err)
	}
	if len(rows) == 0 {
		out, err := New([]int{0}, dt)
		if err != nil {
			return nil, err
		}
		return out, nil
	}
	cols := len(rows[0])
	flatVals := make([]float64, 0, len(rows)*cols)
	for _, r := range rows {
		flatVals = append(flatVals, r...)
	}
	ff, err := FromSlice(flatVals)
	if err != nil {
		return nil, err
	}
	var sh []int
	if cols == 1 {
		sh = []int{len(rows)}
	} else {
		sh = []int{len(rows), cols}
	}
	reshaped, err := ff.Reshape(sh)
	if err != nil {
		return nil, err
	}
	if dt == DFloat64 {
		return reshaped, nil
	}
	return CastUnsafe(reshaped, dt) // 文本按 dtype 落值(numpy 的 dtype= 口径)
}

// Place ≡ np.place(a, mask, vals):把 mask 为真的位置**按顺序**填入 vals(不足则循环)。
//
// 与 PutMask 的区别(与 numpy 一致):putmask 按**下标** `vals[i%len]` 取,place 按
// **命中顺序**依次取。
func Place(a, mask, vals *Array) error {
	sh := a.Shape()
	nm, err := normalizeOperand(mask, sh, mask.Dtype())
	if err != nil {
		return fmt.Errorf("num: place: mask 归一失败: %w", err)
	}
	m, err := truthyMask(nm)
	if err != nil {
		return err
	}
	if vals.Size() == 0 {
		return nil
	}
	nv, err := normalizeOperand(vals.AsContiguous().Flatten(), []int{vals.Size()}, a.Dtype())
	if err != nil {
		return fmt.Errorf("num: place: vals 归一失败: %w", err)
	}
	es := a.Dtype().Itemsize
	if a.fastWindow() == nil {
		flat := nv.Flatten()
		k := 0
		for i, t := range m {
			if !t {
				continue
			}
			v, err := flat.Item(k % vals.Size())
			if err != nil {
				return err
			}
			if err := a.SetItem(flatToIdx(sh, i), v); err != nil {
				return err
			}
			k++
		}
		return nil
	}
	db := engineWindow(a)
	vb := engineWindow(nv)
	k := 0
	for i, t := range m {
		if !t {
			continue
		}
		j := k % vals.Size()
		copy(db[i*es:(i+1)*es], vb[j*es:(j+1)*es])
		k++
	}
	return nil
}

// Piecewise ≡ np.piecewise(x, condlist, funclist):按条件依次套用函数。
//
// len(fns) == len(conds) 时,未被任何条件命中的位置填 0;len(fns) == len(conds)+1 时,
// 最后一个函数作为默认分支(numpy 同)。
func Piecewise(x *Array, conds []*Array, fns []UnaryFunc) (*Array, error) {
	if len(conds) == 0 || len(fns) < len(conds) {
		return nil, fmt.Errorf("num: piecewise: conds/fns 数量不合法: %w", ErrShape)
	}
	if len(fns) > len(conds)+1 {
		return nil, fmt.Errorf("num: piecewise: fns 最多比 conds 多一个: %w", ErrShape)
	}
	out, err := New(x.Shape(), x.Dtype())
	if err != nil {
		return nil, err
	}
	// 未被命中处:默认分支(若有)或 0
	chosen := make([]bool, out.Size())
	if len(fns) == len(conds)+1 {
		def, err := fns[len(fns)-1](x)
		if err != nil {
			return nil, err
		}
		if err := copyResultInto(out, def); err != nil {
			return nil, err
		}
	}
	for k := range conds {
		mask, err := truthyMask2(conds[k], x.Shape())
		if err != nil {
			return nil, err
		}
		// 只对尚未命中的位置计算(避免不必要的工作)
		any := false
		for i, m := range mask {
			if m && !chosen[i] {
				any = true
				break
			}
		}
		if !any {
			continue
		}
		res, err := fns[k](x)
		if err != nil {
			return nil, err
		}
		if err := copyWhereInto(out, res, maskToArrayUnchecked(mask, x.Shape())); err != nil {
			return nil, err
		}
		for i, m := range mask {
			if m {
				chosen[i] = true
			}
		}
	}
	return out, nil
}

// maskToArrayUnchecked 把 bool 切片包成 bool 数组(内部已知长度匹配)。
func maskToArrayUnchecked(m []bool, sh []int) *Array {
	out, err := New(sh, DBool)
	if err != nil {
		return nil
	}
	w := win[uint8](engineWindow(out))
	for i, v := range m {
		if v {
			w[i] = 1
		}
	}
	return out
}

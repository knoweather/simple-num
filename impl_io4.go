package num

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// GenfromtxtOptions 是 Genfromtxt 的读取配置。
//
// 零值即默认口径:逗号以外的分隔由 Delimiter 指定(空则按任意空白切分);
// 以 "#" 开头的行与空行忽略;缺失字段填 NaN。
type GenfromtxtOptions struct {
	Delimiter    string  // 字段分隔符;空表示按连续空白切分
	SkipHeader   int     // 跳过文件开头多少行(物理行,含注释行)
	SkipFooter   int     // 跳过文件末尾多少行
	Usecols      []int   // 只取这些列(空表示全部)
	Comments     string  // 注释行前缀,默认 "#";设为 "\x00" 表示不跳过任何行
	FillingValue float64 // 缺失字段的填充值
	HasFilling   bool    // 为 true 时缺失字段用 FillingValue 而不是 NaN
}

// Genfromtxt ≡ np.genfromtxt:按文本读入二维表,缺失字段填 NaN。
//
// 与 numpy 一致的要点:参差行(列数不一致)直接报错并给出行号;单列结果返回 1-D;
// 空字段、字面 "nan"/"NaN" 都按 NaN 处理,"inf"/"-inf" 按无穷处理。
func Genfromtxt(path string, opts GenfromtxtOptions) (*Array, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("num: genfromtxt 读 %q: %w", path, err)
	}
	return genfromtxtParse(raw, opts)
}

// GenfromtxtBytes 与 Genfromtxt 同一口径,但直接从内存字节解析(免落盘)。
func GenfromtxtBytes(raw []byte, opts GenfromtxtOptions) (*Array, error) {
	return genfromtxtParse(raw, opts)
}

func genfromtxtParse(raw []byte, opts GenfromtxtOptions) (*Array, error) {
	lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	// 末尾换行会多出一个空元素,不计入行数(否则 skip_footer 会吃掉它)
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	if opts.SkipHeader > 0 {
		if opts.SkipHeader >= len(lines) {
			lines = nil
		} else {
			lines = lines[opts.SkipHeader:]
		}
	}
	if opts.SkipFooter > 0 {
		if opts.SkipFooter >= len(lines) {
			lines = nil
		} else {
			lines = lines[:len(lines)-opts.SkipFooter]
		}
	}
	comment := "#"
	if opts.Comments != "" {
		comment = opts.Comments
	}
	skipComments := comment != "\x00"

	var rows [][]float64
	maxCols := 0
	lineNo := opts.SkipHeader
	for _, ln := range lines {
		lineNo++
		trimmed := strings.TrimSpace(ln)
		if trimmed == "" {
			continue
		}
		if skipComments && strings.HasPrefix(trimmed, comment) {
			continue
		}
		var fields []string
		if opts.Delimiter == "" {
			fields = strings.Fields(trimmed)
		} else {
			fields = strings.Split(trimmed, opts.Delimiter)
			for i := range fields {
				fields[i] = strings.TrimSpace(fields[i])
			}
		}
		vals := make([]float64, len(fields))
		for i, f := range fields {
			v, err := parseTextField(f, opts)
			if err != nil {
				return nil, fmt.Errorf("num: genfromtxt 第 %d 行第 %d 列(%q): %w", lineNo, i+1, f, err)
			}
			vals[i] = v
		}
		if len(rows) > 0 && len(vals) != maxCols {
			return nil, fmt.Errorf("num: genfromtxt 第 %d 行有 %d 列,前面是 %d 列: %w",
				lineNo, len(vals), maxCols, ErrShape)
		}
		maxCols = len(vals)
		rows = append(rows, vals)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("num: genfromtxt 没有可解析的数据行: %w", ErrShape)
	}
	// 选列
	cols := opts.Usecols
	if len(cols) == 0 {
		cols = make([]int, maxCols)
		for i := range cols {
			cols[i] = i
		}
	}
	for _, c := range cols {
		if c < 0 || c >= maxCols {
			return nil, fmt.Errorf("num: genfromtxt 列号 %d 越界(共 %d 列): %w", c, maxCols, ErrIndex)
		}
	}
	nr := len(rows)
	nc := len(cols)
	buf := make([]float64, nr*nc)
	for i, r := range rows {
		for j, c := range cols {
			buf[i*nc+j] = r[c]
		}
	}
	out, err := FromSlice(buf)
	if err != nil {
		return nil, err
	}
	if nc == 1 {
		return out, nil
	}
	return out.Reshape([]int{nr, nc})
}

// parseTextField 解析单个字段;空字段按填充值(默认 NaN)。
func parseTextField(f string, opts GenfromtxtOptions) (float64, error) {
	if f == "" {
		if opts.HasFilling {
			return opts.FillingValue, nil
		}
		return math.NaN(), nil
	}
	switch strings.ToLower(f) {
	case "nan", "+nan", "-nan":
		return math.NaN(), nil
	case "inf", "+inf", "infinity", "+infinity":
		return math.Inf(1), nil
	case "-inf", "-infinity":
		return math.Inf(-1), nil
	}
	v, err := strconv.ParseFloat(f, 64)
	if err != nil {
		return 0, fmt.Errorf("num: 无法解析为数值: %w", ErrValue)
	}
	return v, nil
}

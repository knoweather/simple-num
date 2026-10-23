//go:build js && wasm

// Package wasm 是 simple-num 的 JS/WASM 胶水层。
//
// 把数组 API 挂到 globalThis.simpleNum;错误约定 = 数组类方法失败返回 null
// 且错误记入 lastError(成功清空);数组对象的 JS 形态为
// {dtype:"f8", shape:[2,3], data:Uint8Array}(小端 C 序字节)。
//
// 核心逻辑在 api.go(纯 Go、可宿主测试),本文件只做 syscall/js 桥接。
package wasm

import (
	"fmt"
	"syscall/js"

	num "github.com/knoweather/simple-num"
)

// keepFuncs 防 GC:JS 侧持有的函数包装,Go 侧须保持引用直至进程退出。
var keepFuncs []js.Func

// funcOf 用 FuncOf 包装 fn 并登记防 GC。
func funcOf(f func(this js.Value, args []js.Value) any) js.Func {
	fn := js.FuncOf(f)
	keepFuncs = append(keepFuncs, fn)
	return fn
}

// argAt 取第 i 个参数;越界返回 undefined。
func argAt(args []js.Value, i int) js.Value {
	if i < 0 || i >= len(args) {
		return js.Undefined()
	}
	return args[i]
}

// strAt 取第 i 个字符串参数;非字符串(或缺失)返回空串,交由下游校验报错。
func strAt(args []js.Value, i int) string {
	v := argAt(args, i)
	switch v.Type() {
	case js.TypeString:
		return v.String()
	default:
		return ""
	}
}

// isTypedArray 判断 v 是否为 JS TypedArray(有.buffer 属性)。
func isTypedArray(v js.Value) bool {
	if v.Type() != js.TypeObject {
		return false
	}
	return v.Get("buffer").Type() == js.TypeObject
}

// bytesArg 从 JS 值提取原始字节。接受 Uint8Array/Uint8ClampedArray、ArrayBuffer、
// 任意 TypedArray(经 new Uint8Array(buffer, byteOffset, byteLength) 取视图)、
// number[](每元素须为 [0,256) 的整数,按 1 字节写)。
func bytesArg(v js.Value) ([]byte, error) {
	if v.IsUndefined() || v.IsNull() {
		return nil, fmt.Errorf("wasm: data 不能为空")
	}
	// TypedArray(含 Uint8/Int/Float 等):经统一视图读底层字节。
	if isTypedArray(v) {
		buf := v.Get("buffer")
		offset := 0
		if bo := v.Get("byteOffset"); bo.Type() == js.TypeNumber {
			offset = bo.Int()
		}
		bl := v.Get("byteLength").Int()
		view := js.Global().Get("Uint8Array").New(buf, offset, bl)
		out := make([]byte, bl)
		js.CopyBytesToGo(out, view)
		return out, nil
	}
	// ArrayBuffer:有 byteLength 但无 buffer。
	if byl := v.Get("byteLength"); byl.Type() == js.TypeNumber {
		bl := byl.Int()
		view := js.Global().Get("Uint8Array").New(v, 0, bl)
		out := make([]byte, bl)
		js.CopyBytesToGo(out, view)
		return out, nil
	}
	// number[]:每元素一个字节。
	if v.Type() == js.TypeObject {
		n := v.Length()
		out := make([]byte, n)
		for i := range out {
			e := v.Index(i)
			if e.Type() != js.TypeNumber {
				return nil, fmt.Errorf("wasm: data[%d] 须为数字", i)
			}
			x := e.Int()
			if x < 0 || x > 255 {
				return nil, fmt.Errorf("wasm: data[%d]=%d 超出字节范围 [0,256)", i, x)
			}
			out[i] = byte(x)
		}
		return out, nil
	}
	return nil, fmt.Errorf("wasm: data 须为 TypedArray/ArrayBuffer/number[]")
}

// toJSBytes Go 字节 → JS Uint8Array(新建后 CopyBytesToJS)。
func toJSBytes(b []byte) js.Value {
	u8 := js.Global().Get("Uint8Array").New(len(b))
	if len(b) > 0 {
		js.CopyBytesToJS(u8, b)
	}
	return u8
}

// fail 记录错误并返回 null(数组类方法失败路径)。
func fail(err error) any {
	NoteErr(err)
	return js.Null()
}

// arrayOrNull 把 Array 转成 JS 对象;转换失败走 fail。成功则 ClearErr 后返回对象。
func arrayOrNull(a *num.Array) any {
	o, err := arrayToJS(a)
	if err != nil {
		return fail(err)
	}
	ClearErr()
	return o
}

// jsToArray JS 数组对象 → Array:须有 dtype/shape/data;否则报 "wasm: 非法数组对象"。
func jsToArray(v js.Value) (*num.Array, error) {
	if v.Type() != js.TypeObject {
		return nil, fmt.Errorf("wasm: 非法数组对象")
	}
	dtype := propString(v, "dtype")
	shape, err := propShape(v, "shape")
	if err != nil {
		return nil, fmt.Errorf("wasm: 非法数组对象:%s", err.Error())
	}
	data, err := bytesArg(v.Get("data"))
	if err != nil {
		return nil, fmt.Errorf("wasm: 非法数组对象:%s", err.Error())
	}
	return FromBytes(dtype, shape, data)
}

// propString 取对象 o 的字符串属性 p;缺失/非字符串返回空串。
func propString(o js.Value, p string) string {
	v := o.Get(p)
	switch v.Type() {
	case js.TypeString:
		return v.String()
	default:
		return ""
	}
}

// propShape 取对象 o 的 number[] 属性 p。
func propShape(o js.Value, p string) ([]int, error) {
	v := o.Get(p)
	if v.Type() != js.TypeObject {
		return nil, fmt.Errorf("缺 %q 或不是数组", p)
	}
	n := v.Length()
	out := make([]int, n)
	for i := range out {
		e := v.Index(i)
		if e.Type() != js.TypeNumber {
			return nil, fmt.Errorf("%q[%d] 须为数字", p, i)
		}
		out[i] = e.Int()
	}
	return out, nil
}

// arrayToJS Array → JS 对象 {dtype, shape, data:Uint8Array}(data 为小端 C 序连续字节)。
func arrayToJS(a *num.Array) (js.Value, error) {
	if a == nil {
		return js.Null(), fmt.Errorf("wasm: 数组为 nil")
	}
	data, err := Bytes(a)
	if err != nil {
		return js.Null(), err
	}
	obj := js.Global().Get("Object").New()
	obj.Set("dtype", DtypeString(a.Dtype()))
	sh := a.Shape()
	shArr := make([]any, len(sh))
	for i, d := range sh {
		shArr[i] = d
	}
	obj.Set("shape", shArr)
	obj.Set("data", toJSBytes(data))
	return obj, nil
}

// valueToJS 单个元素(any)→ JS 值:bool/string 原样;float32/64、int/uint 变体 → number;
// complex → [re, im];其余 → String。
func valueToJS(v any) js.Value {
	switch x := v.(type) {
	case bool:
		return js.ValueOf(x)
	case float32:
		return js.ValueOf(float64(x))
	case complex64:
		return js.ValueOf([]any{float64(real(x)), float64(imag(x))})
	case complex128:
		return js.ValueOf([]any{float64(real(x)), float64(imag(x))})
	case string:
		return js.ValueOf(x)
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float64:
		return js.ValueOf(x)
	default:
		return js.ValueOf(fmt.Sprintf("%v", x))
	}
}

// parseShape 取第 i 个 number[] 参数。
func parseShape(args []js.Value, i int) ([]int, error) {
	if i >= len(args) {
		return nil, fmt.Errorf("缺少 shape 参数")
	}
	v := args[i]
	if v.Type() != js.TypeObject {
		return nil, fmt.Errorf("shape 须为 number[]")
	}
	n := v.Length()
	out := make([]int, n)
	for j := range out {
		e := v.Index(j)
		if e.Type() != js.TypeNumber {
			return nil, fmt.Errorf("shape[%d] 须为数字", j)
		}
		out[j] = e.Int()
	}
	return out, nil
}

// parseAxis reduce 的 axis(第 2 位):null/undefined/缺省 → nil(全归约);number → 单维;number[] → 数组。
func parseAxis(args []js.Value) ([]int, error) {
	if len(args) < 3 {
		return nil, nil
	}
	v := args[2]
	switch {
	case v.IsUndefined() || v.IsNull():
		return nil, nil
	case v.Type() == js.TypeNumber:
		return []int{v.Int()}, nil
	case v.Type() == js.TypeObject:
		n := v.Length()
		out := make([]int, n)
		for i := range out {
			e := v.Index(i)
			if e.Type() != js.TypeNumber {
				return nil, fmt.Errorf("axis[%d] 须为数字", i)
			}
			out[i] = e.Int()
		}
		return out, nil
	default:
		return nil, fmt.Errorf("axis 须为 number/number[]/null")
	}
}

// parseKeepdims reduce 的 keepdims(第 3 位);缺省 false。
func parseKeepdims(args []js.Value) bool {
	if len(args) < 4 {
		return false
	}
	v := args[3]
	switch v.Type() {
	case js.TypeBoolean:
		return v.Bool()
	default:
		return false
	}
}

// —— 各方法 ——

func zerosFunc(this js.Value, args []js.Value) any {
	shape, err := parseShape(args, 0)
	if err != nil {
		return fail(err)
	}
	dt, err := ParseDtype(strAt(args, 1))
	if err != nil {
		return fail(err)
	}
	a, err := num.Zeros(shape, dt)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(a)
}

func arangeFunc(this js.Value, args []js.Value) any {
	if len(args) < 4 {
		return fail(fmt.Errorf("wasm: arange 需 start/stop/step/dtype"))
	}
	for i, name := range []string{"start", "stop", "step"} {
		if args[i].Type() != js.TypeNumber {
			return fail(fmt.Errorf("wasm: arange %s 须为数字", name))
		}
	}
	dt, err := ParseDtype(strAt(args, 3))
	if err != nil {
		return fail(err)
	}
	a, err := num.Arange(args[0].Float(), args[1].Float(), args[2].Float(), dt)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(a)
}

func fromBytesFunc(this js.Value, args []js.Value) any {
	if len(args) < 3 {
		return fail(fmt.Errorf("wasm: fromBytes 需 dtype/shape/data"))
	}
	shape, err := parseShape(args, 1)
	if err != nil {
		return fail(err)
	}
	data, err := bytesArg(args[2])
	if err != nil {
		return fail(err)
	}
	a, err := FromBytes(strAt(args, 0), shape, data)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(a)
}

// toBytes arr → 数组对象;data 为往返后的连续字节(小端 C 序)。
func toBytesFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(a)
}

func valuesFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	vals, err := Values(a)
	if err != nil {
		return fail(err)
	}
	out := make([]any, len(vals))
	for i, v := range vals {
		out[i] = valueToJS(v)
	}
	ClearErr()
	return out
}

func opFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 1))
	if err != nil {
		return fail(err)
	}
	b, err := jsToArray(argAt(args, 2))
	if err != nil {
		return fail(err)
	}
	r, err := Op(strAt(args, 0), a, b)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}

func unaryMethod(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 1))
	if err != nil {
		return fail(err)
	}
	r, err := Unary(strAt(args, 0), a)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}

func reduceMethod(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 1))
	if err != nil {
		return fail(err)
	}
	axis, err := parseAxis(args)
	if err != nil {
		return fail(err)
	}
	r, err := Reduce(strAt(args, 0), a, axis, parseKeepdims(args))
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}

func linalgFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 1))
	if err != nil {
		return fail(err)
	}
	var b *num.Array
	if len(args) > 2 && args[2].Type() != js.TypeUndefined && args[2].Type() != js.TypeNull {
		if b, err = jsToArray(args[2]); err != nil {
			return fail(err)
		}
	}
	r, err := Linalg(strAt(args, 0), a, b)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}

func cumulativeFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 1))
	if err != nil {
		return fail(err)
	}
	axis, err := parseAxis(args)
	if err != nil {
		return fail(err)
	}
	r, err := Cumulative(strAt(args, 0), a, axis)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}
func castFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	dt, err := ParseDtype(strAt(args, 1))
	if err != nil {
		return fail(err)
	}
	r, err := num.Cast(a, dt)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}

func reshapeFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	sh, err := parseShape(args, 1)
	if err != nil {
		return fail(err)
	}
	r, err := a.Reshape(sh)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(r)
}

func itemFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	idx := make([]int, len(args)-1)
	for i := range idx {
		e := args[1+i]
		if e.Type() != js.TypeNumber {
			return fail(fmt.Errorf("wasm: item 索引 %d 须为数字", i))
		}
		idx[i] = e.Int()
	}
	v, err := a.Item(idx...)
	if err != nil {
		return fail(err)
	}
	ClearErr()
	return valueToJS(v)
}

func loadNPYFunc(this js.Value, args []js.Value) any {
	b, err := bytesArg(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	a, err := num.LoadNPYBytes(b)
	if err != nil {
		return fail(err)
	}
	return arrayOrNull(a)
}

// saveNPY arr → Uint8Array(完整.npy 字节)。
func saveNPYFunc(this js.Value, args []js.Value) any {
	a, err := jsToArray(argAt(args, 0))
	if err != nil {
		return fail(err)
	}
	b, err := num.SaveNPYBytes(a)
	if err != nil {
		return fail(err)
	}
	ClearErr()
	return toJSBytes(b)
}

// lastErrorFunc 直接返回 LastErr;不做 NoteErr/ClearErr。
func lastErrorFunc(this js.Value, args []js.Value) any {
	return LastErr()
}

// versionFunc 返回 Version(必须是函数:JS 侧按 simpleNum.version 调用)。
func versionFunc(this js.Value, args []js.Value) any {
	return Version
}

// Register 把 API 挂到 g.simpleNum。
func Register(g js.Value) {
	ns := js.Global().Get("Object").New()
	ns.Set("version", funcOf(versionFunc))
	ns.Set("zeros", funcOf(zerosFunc))
	ns.Set("arange", funcOf(arangeFunc))
	ns.Set("fromBytes", funcOf(fromBytesFunc))
	ns.Set("toBytes", funcOf(toBytesFunc))
	ns.Set("values", funcOf(valuesFunc))
	ns.Set("op", funcOf(opFunc))
	ns.Set("unary", funcOf(unaryMethod))
	ns.Set("reduce", funcOf(reduceMethod))
	ns.Set("linalg", funcOf(linalgFunc))
	ns.Set("cumulative", funcOf(cumulativeFunc))
	ns.Set("cast", funcOf(castFunc))
	ns.Set("reshape", funcOf(reshapeFunc))
	ns.Set("item", funcOf(itemFunc))
	ns.Set("loadNPY", funcOf(loadNPYFunc))
	ns.Set("saveNPY", funcOf(saveNPYFunc))
	ns.Set("lastError", funcOf(lastErrorFunc))
	g.Set("simpleNum", ns)
}

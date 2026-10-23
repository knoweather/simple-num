//go:build js && wasm

// Package main 是 simple-num WASM 模块的入口(go build GOOS=js GOARCH=wasm./wasm/main)。
package main

import (
	"syscall/js"

	"github.com/knoweather/simple-num/wasm"
)

func main() {
	wasm.Register(js.Global())
	if c := js.Global().Get("console"); c.Type() == js.TypeObject {
		c.Call("log", "simple-num wasm v"+wasm.Version)
	}
	select {} // 常驻,保持事件循环存活
}

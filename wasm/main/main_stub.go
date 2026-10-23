//go:build !js || !wasm

// Package main(stub):非 js/wasm 构建时兜底,提示正确用法并退出,
// 保证宿主机 go build./... 不产生可执行产物、且保持干净。
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "simple-num wasm 只能 js/wasm 构建:GOOS=js GOARCH=wasm go build -o ... ./wasm/main")
	os.Exit(1)
}

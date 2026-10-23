#!/bin/sh
# Build wasm/simple-num.wasm plus the matching wasm_exec.js from the local Go
# toolchain. Works on Linux, macOS and Git Bash on Windows.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

# Resolve go: GO_BIN_DIR wins if it holds a go binary, otherwise PATH.
if [ -n "${GO_BIN_DIR:-}" ] && [ -x "$GO_BIN_DIR/go" ]; then
	GO="$GO_BIN_DIR/go"
else
	if ! command -v go >/dev/null 2>&1; then
		echo "go not found: put it on PATH, or set GO_BIN_DIR to the directory containing go" >&2
		exit 1
	fi
	GO=go
fi

goroot=$("$GO" env GOROOT)
exec_js="$goroot/lib/wasm/wasm_exec.js"
if [ ! -f "$exec_js" ]; then
	exec_js="$goroot/misc/wasm/wasm_exec.js"
fi
if [ ! -f "$exec_js" ]; then
	echo "cannot find wasm_exec.js under $goroot" >&2
	exit 1
fi

version=$(git describe --tags --always --dirty 2>/dev/null || echo "0.0.0-dev")
version=${version#v}
GOOS=js GOARCH=wasm CGO_ENABLED=0 "$GO" build -trimpath \
	-ldflags "-X github.com/knoweather/simple-num/wasm.Version=$version" \
	-o wasm/simple-num.wasm ./wasm/main
cp "$exec_js" wasm/wasm_exec.js

echo "built  wasm/simple-num.wasm  ($(wc -c < wasm/simple-num.wasm) bytes, version $version)"
echo "copied wasm/wasm_exec.js"
echo "usage: load wasm/wasm_exec.js and instantiate the module; then call globalThis.simpleNum.* (see README)."

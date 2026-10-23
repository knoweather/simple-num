#Requires -Version 5.1
<#
.SYNOPSIS
    Builds simple-num's WASM module and stages its runtime glue.

.DESCRIPTION
    Compiles the js/wasm entry (wasm/main) into a .wasm file and copies the Go
    wasm_exec.js host into the same directory. Both outputs are gitignored
    (see .gitignore: "wasm/*.wasm" and "wasm/wasm_exec.js") because they bind
    tightly to the Go toolchain version:
      * wasm/simple-num.wasm  compiled module (GOOS=js GOARCH=wasm, CGO_ENABLED=0)
      * wasm/wasm_exec.js     runtime glue copied from $GOROOT/lib/wasm/wasm_exec.js
    The script fails fast on any error and restores the environment variables it
    sets (GOOS / GOARCH / CGO_ENABLED) in a finally block.

.PARAMETER None

.EXAMPLE
    .\wasm\build.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-Go {
    # GO_BIN_DIR can point at the directory holding go.exe; otherwise use PATH.
    if ($env:GO_BIN_DIR) {
        $cand = Join-Path $env:GO_BIN_DIR 'go.exe'
        if (Test-Path -LiteralPath $cand) { return $cand }
    }
    if (-not (Get-Command 'go' -ErrorAction SilentlyContinue)) {
        throw 'go not found: put it on PATH, or set GO_BIN_DIR to the directory containing go.exe'
    }
    return 'go'
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$goBin = Resolve-Go
if ($goBin -like '*.exe') {
    # full path to go.exe -> GOROOT is two levels up (...\bin\go.exe)
    $goroot = Split-Path (Split-Path -Parent $goBin) -Parent
} else {
    $goroot = & $goBin env GOROOT
}

# stage wasm_exec.js source location: prefer lib/wasm, fall back to misc/wasm.
$execJsSrc = Join-Path $goroot 'lib\wasm\wasm_exec.js'
if (-not (Test-Path -LiteralPath $execJsSrc)) {
    $execJsSrc = Join-Path $goroot 'misc\wasm\wasm_exec.js'
}
if (-not (Test-Path -LiteralPath $execJsSrc)) {
    throw "cannot find wasm_exec.js; tried $execJsSrc"
}

$touchedEnv = $false
$savedGoos   = $null
$savedGoarch = $null
$savedCgo    = $null
try {
    $savedGoos   = $env:GOOS
    $savedGoarch = $env:GOARCH
    $savedCgo    = $env:CGO_ENABLED
    $touchedEnv  = $true

    $env:GOOS        = 'js'
    $env:GOARCH      = 'wasm'
    $env:CGO_ENABLED = '0'

    & $goBin build -trimpath -o wasm/simple-num.wasm ./wasm/main
    if ($LASTEXITCODE -ne 0) { throw "go build failed with exit code $LASTEXITCODE" }

    Copy-Item -LiteralPath $execJsSrc -Destination 'wasm/wasm_exec.js'

    $wasmOut = Join-Path $repoRoot 'wasm\simple-num.wasm'
    $size    = (Get-Item -LiteralPath $wasmOut).Length
    Write-Host ("built  {0}  ({1:N0} bytes)" -f $wasmOut, $size)
    Write-Host "usage: load wasm/wasm_exec.js and instantiate the module; then call globalThis.simpleNum.* (see README)."
}
finally {
    if ($touchedEnv) {
        if ($null -ne $savedGoos)   { $env:GOOS = $savedGoos } else { Remove-Item Env:GOOS -ErrorAction SilentlyContinue }
        if ($null -ne $savedGoarch) { $env:GOARCH = $savedGoarch } else { Remove-Item Env:GOARCH -ErrorAction SilentlyContinue }
        if ($null -ne $savedCgo)    { $env:CGO_ENABLED = $savedCgo } else { Remove-Item Env:CGO_ENABLED -ErrorAction SilentlyContinue }
    }
}

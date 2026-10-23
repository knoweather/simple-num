// wasm/smoke/smoke.mjs —— simple-num WASM 接口冒烟测试(Node,无需浏览器)。
//
// 运行(仓库根目录):
// node wasm/smoke/smoke.mjs
//
// 流程:GOOS=js GOARCH=wasm CGO_ENABLED=0 go build -o <tmp>/simple-num.wasm./wasm/main
// → 载入 $GOROOT/lib/wasm/wasm_exec.js → Node 里实例化 →
// 调 globalThis.simpleNum 构数组/算张量/npy 往返,断言数值与错误路径。
// 任一步失败即非零退出。

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { TextEncoder, TextDecoder } from "node:util";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");

// go 可执行文件:默认走 PATH;若设了 GO_BIN_DIR 则优先用其中的 go。
let goExe = "go";
if (process.env.GO_BIN_DIR) {
	const cand = path.join(process.env.GO_BIN_DIR, process.platform === "win32" ? "go.exe" : "go");
	if (fs.existsSync(cand)) goExe = cand;
}

const env = {
	...process.env,
	CGO_ENABLED: "0",
	GOOS: "js",
	GOARCH: "wasm",
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "simplenum-wasm-smoke-"));
const wasmPath = path.join(tmp, "simple-num.wasm");
console.log("[smoke] go build ->", wasmPath);
try {
	execFileSync(goExe, ["build", "-trimpath", "-o", wasmPath, "./wasm/main"], {
		cwd: root,
		env,
		stdio: "inherit",
	});
} catch (e) {
	console.error(
		`\n[smoke] \`go build\` 失败:找不到可用的 go(${goExe})。\n` +
			`        请把 go 放进 PATH,或设置环境变量 GO_BIN_DIR 指向其所在目录。`,
	);
	process.exit(1);
}

const goroot = execFileSync(goExe, ["env", "GOROOT"], { cwd: root, env })
	.toString()
	.trim();
let wasmExec = path.join(goroot, "lib", "wasm", "wasm_exec.js");
if (!fs.existsSync(wasmExec)) wasmExec = path.join(goroot, "misc", "wasm", "wasm_exec.js");
if (!fs.existsSync(wasmExec)) throw new Error(`找不到 wasm_exec.js(GOROOT=${goroot})`);
console.log("[smoke] wasm_exec.js ->", wasmExec);

globalThis.TextEncoder = TextEncoder;
globalThis.TextDecoder = TextDecoder;
globalThis.performance ??= performance;
globalThis.crypto ??= crypto;

vm.runInThisContext(fs.readFileSync(wasmExec, "utf8"), { filename: wasmExec });

const go = new Go();
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath), go.importObject);
go.run(instance); // 不 await:Go main 会 select{} 常驻

async function waitFor(fn, what, timeoutMs = 15000) {
	const deadline = Date.now() + timeoutMs;
	for (;;) {
		const v = fn();
		if (v) return v;
		if (Date.now() > deadline) throw new Error(`等待超时: ${what}`);
		await new Promise((r) => setTimeout(r, 20));
	}
}

const api = await waitFor(() => globalThis.simpleNum, "globalThis.simpleNum");
await waitFor(() => typeof api.op === "function", "simpleNum.op");

let failed = 0;
function check(cond, msg) {
	if (cond) console.log("[ok]", msg);
	else { failed++; console.error("[FAIL]", msg); }
}
function close(a, b, eps = 1e-12) {
	return Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b));
}
function f8bytes(vals) {
	return new Uint8Array(new Float64Array(vals).buffer);
}
function mustArr(res, what) {
	if (res === null || res === undefined) throw new Error(`${what} 返回 null:${api.lastError()}`);
	return res;
}

// 1) 版本
check(api.version() === "0.1.0", `version() = ${api.version()}`);

// 2) 构造 + Values + reshape
const a = mustArr(api.fromBytes("f8", [2, 3], f8bytes([0, 1, 2, 3, 4, 5])), "fromBytes");
check(JSON.stringify(a.shape) === "[2,3]", `fromBytes 形状 ${JSON.stringify(a.shape)}`);
check(a.dtype === "f8", `fromBytes dtype = ${a.dtype}`);
check(api.values(a).every((v, i) => v === i), `Values 行主序 0..5`);
const flat = mustArr(api.reshape(a, [6]), "reshape");
check(JSON.stringify(flat.shape) === "[6]", `reshape [6] -> ${JSON.stringify(flat.shape)}`);

// 3) arange + 广播加
const b = mustArr(api.fromBytes("f8", [3], f8bytes([10, 20, 30])), "fromBytes b");
const sum = mustArr(api.op("add", a, b), "op add"); // (2,3)+(3,) 右对齐广播
check(JSON.stringify(sum.shape) === "[2,3]", `add 广播形状 ${JSON.stringify(sum.shape)}`);
const got = api.values(sum);
const want = [10, 21, 32, 13, 24, 35];
check(got.every((v, i) => close(v, want[i])), `add 广播值 ${got.join(",")} (期望 ${want.join(",")})`);

// 4) 归约 sum(全归约 -> 0 维)
const all = mustArr(api.reduce("sum", a, null, false), "reduce sum");
check(JSON.stringify(all.shape) === "[]", `全归约 0 维 -> ${JSON.stringify(all.shape)}`);
check(close(api.values(all)[0], 15), `sum(0..5) = ${api.values(all)[0]}(期望 15)`);

// 5) 归约 keepdims + axis
const ax0 = mustArr(api.reduce("sum", a, 0, true), "reduce sum axis=0 keepdims");
check(JSON.stringify(ax0.shape) === "[1,3]", `keepdims 形状 ${JSON.stringify(ax0.shape)}`);
check(api.values(ax0).every((v, i) => close(v, [3, 5, 7][i])), `axis=0 列和 3,5,7`);

// 6) 一元 sqrt
const s = mustArr(api.unary("sqrt", b), "unary sqrt");
check(api.values(s).every((v, i) => close(v, [Math.sqrt(10), Math.sqrt(20), Math.sqrt(30)][i])), `sqrt 逐元素`);

// 7) 比较输出 bool dtype
const gt = mustArr(api.op("greater", a, b), "op greater");
check(gt.dtype === "b", `greater dtype = ${gt.dtype}(期望 b)`);
check(api.values(gt).every((v) => v === false), "greater 全 false(0..5 vs 10..30)");

// 8) cast(same_kind 口径:f8→i4 属跨 kind,被拒;同族 f8→f4 合法)
const cf = mustArr(api.cast(b, "f4"), "cast f4");
check(cf.dtype === "f4", `cast dtype = ${cf.dtype}`);
check(api.values(cf).every((v, i) => close(v, [10, 20, 30][i])), "cast f8->f4 值不变");
check(api.cast(b, "i4") === null && String(api.lastError()).length > 0,
	`cast f8→i4 被 same_kind 拒绝:${api.lastError()}`);

// 9) zeros / arange / item(胶水直通)
const z = mustArr(api.zeros([2, 2], "f4"), "zeros");
check(api.values(z).every((v) => v === 0), "zeros 全 0");
const ar = mustArr(api.arange(0, 5, 1, "i8"), "arange i8");
check(ar.dtype === "i8" && api.values(ar).every((v, i) => v === i), "arange 0..4 i8");
check(api.item(ar, 3) === 3, `item(ar,3) = ${api.item(ar, 3)}`);
const z2 = mustArr(api.reshape(z, [4]), "reshape zeros");
check(api.item(z2, 0) === 0, "item 单下标");

// 10) npy 往返
const npy = api.saveNPY(a);
check(npy instanceof Uint8Array && npy.length > 64, `saveNPY ${npy.length} bytes`);
const back = mustArr(api.loadNPY(npy), "loadNPY");
check(back.dtype === "f8" && JSON.stringify(back.shape) === "[2,3]", `loadNPY dtype/shape = ${back.dtype} ${JSON.stringify(back.shape)}`);
check(api.values(back).every((v, i) => v === i), "npy 往返值一致");

// 11) toBytes 往返
const tb = mustArr(api.toBytes(a), "toBytes");
const tbBack = mustArr(api.fromBytes(tb.dtype, tb.shape, tb.data), "fromBytes(toBytes)");
check(api.values(tbBack).every((v, i) => v === i), "toBytes/fromBytes 往返值一致");

// 12) 错误路径:未知算子 -> null + lastError
const bad = api.op("nope", a, b);
check(bad === null, `未知算子返回 null(实得 ${bad})`);
check(String(api.lastError()).includes("未知算子"), `lastError = ${api.lastError()}`);

// 13) 错误路径:字节长度不符
const badLen = api.fromBytes("f8", [2, 3], f8bytes([1]));
check(badLen === null, `字节长度不符返回 null(实得 ${badLen})`);
check(String(api.lastError()).length > 0, `lastError 有记录:${api.lastError()}`);

// 14) 线代分发:det / inv / norm_fro
const m22 = mustArr(api.fromBytes("f8", [2, 2], f8bytes([1, 2, 3, 4])), "mat2x2");
const detArr = mustArr(api.linalg("det", m22), "linalg det");
const detVal = api.values(detArr)[0];
check(Math.abs(detVal - (-2)) < 1e-12, `det [[1,2],[3,4]] = ${detVal}(应 -2)`);
const invArr = mustArr(api.linalg("inv", m22), "linalg inv");
check(invArr.shape.join(",") === "2,2", `inv 形状 ${invArr.shape}`);
const froArr = mustArr(api.linalg("norm_fro", m22), "linalg norm_fro");
check(Math.abs(api.values(froArr)[0] - Math.sqrt(30)) < 1e-12, "norm_fro = sqrt(30)");
const mm = mustArr(api.linalg("matmul", m22, m22), "linalg matmul");
check(Math.abs(api.item(mm, 0, 0) - 7) < 1e-12, `matmul[0,0] = ${api.item(mm, 0, 0)}(应 7)`);

// 15) 前缀累积:cumsum(默认展平)/ cumsum(axis=0)
const cs = mustArr(api.cumulative("cumsum", a), "cumsum");
check(api.values(cs).join(",") === "0,1,3,6,10,15", `cumsum = ${api.values(cs)}`);
const cs0 = mustArr(api.cumulative("cumsum", m22, 0), "cumsum axis=0");
check(api.values(cs0).join(",") === "1,2,4,6", `cumsum axis=0 = ${api.values(cs0)}`);

// 16) 扩充的一元/归约算子
const tn = mustArr(api.unary("tanh", mustArr(api.fromBytes("f8", [1], f8bytes([0])), "zero")), "tanh");
check(Math.abs(api.values(tn)[0]) < 1e-15, "tanh(0) = 0");
const ns = mustArr(api.reduce("nansum", a, null, false), "nansum");
check(Math.abs(api.values(ns)[0] - 15) < 1e-12, `nansum = ${api.values(ns)[0]}(应 15)`);
if (failed) {
	console.error(`\n[smoke] ${failed} 项失败`);
	process.exit(1);
}
console.log("\n[smoke] all passed");
process.exit(0);

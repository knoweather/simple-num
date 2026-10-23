#!/usr/bin/env python3
"""simple-num reference fixture 生成器。

产出 npy fixtures + manifest.json。
消费端口径以 A 落地的 xcheck/xcheck_test.go 代码为准:
    manifest 路径 xcheck/reference/manifest.json(JSON 数组)
    条目字段  name/op/inputs[]/expect/region/args{axis,keepdim,endpoint,n}
             (+ §2.4 的 generator/generated/tol 附加字段,json 多字段无害)


运行(需 numpy>=2 的 Python 环境;建议独立 venv):
    python xcheck/gen_reference.py
输出:
    xcheck/reference/manifest.json
    xcheck/reference/cases/<name>_{a,b,out}.npy
边角覆盖:±0、±Inf、NaN、MinInt64、MaxUint64/MaxInt64+1 回绕、|x|>1 幂、
负底数分数指数、0^0、0**neg、bool 特判矩阵。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "reference")
CASES = os.path.join(REFDIR, "cases")

# numpy dtype 简写 → 真身(v0 数值全集;f16 待,c/datetime/S/U 待 -)
DT = {
    "b": np.bool_,
    "i1": np.int8, "i2": np.int16, "i4": np.int32, "i8": np.int64,
    "u1": np.uint8, "u2": np.uint16, "u4": np.uint32, "u8": np.uint64,
    "f2": np.float16, "f4": np.float32, "f8": np.float64,
    "c8": np.complex64, "c16": np.complex128,
}

# 生成器标识只写 numpy 版本;需要附加信息(如环境名)用环境变量 REFERENCE_GENERATOR_NOTE,
# 避免把本机路径写进 fixtures。
_GENERATOR_NOTE = os.environ.get("REFERENCE_GENERATOR_NOTE", "").strip()
GENERATOR = f"numpy {np.__version__}" + (f" ({_GENERATOR_NOTE})" if _GENERATOR_NOTE else "")


def arr(dtype: str, vals, shape=None):
    a = np.array(vals, dtype=DT[dtype])
    if shape is not None:
        a = a.reshape(shape)
    return a


def nan_with_bits(bits: int, dtype: str):
    """构造指定 NaN 位型的标量(payload 传播对拍用;NB §2.8)。"""
    import struct
    if dtype == "f8":
        return np.float64(struct.unpack("<d", struct.pack("<Q", bits))[0])
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


class Manifest:
    def __init__(self):
        self.entries: list[dict] = []

    def emit(self, name, op, a, b, expect, region="bitwise", tol=None, note=""):
        fa, fb, fo = f"cases/{name}_a.npy", f"cases/{name}_b.npy", f"cases/{name}_out.npy"
        np.save(os.path.join(REFDIR, fa), a, allow_pickle=False)
        np.save(os.path.join(REFDIR, fb), b, allow_pickle=False)
        np.save(os.path.join(REFDIR, fo), expect, allow_pickle=False)
        entry = {
            "name": name,
            "op": op,
            "inputs": [fa, fb],
            "expect": fo,
            "region": region,
            "generator": GENERATOR,
            "generated": _now(),
            "note": note,
        }
        if tol is not None:
            entry["tol"] = tol
        self.entries.append(entry)

    def emit1(self, name, op, a, expect, region="bitwise", tol=None, note="", args=None):
        """一元算子:inputs 只有一项。args 进 manifest(axis/keepdim 等)。"""
        fa, fo = f"cases/{name}_a.npy", f"cases/{name}_out.npy"
        np.save(os.path.join(REFDIR, fa), a, allow_pickle=False)
        np.save(os.path.join(REFDIR, fo), np.asarray(expect), allow_pickle=False)
        entry = {
            "name": name,
            "op": op,
            "inputs": [fa],
            "expect": fo,
            "region": region,
            "generator": GENERATOR,
            "generated": _now(),
            "note": note,
        }
        if tol is not None:
            entry["tol"] = tol
        if args:
            entry["args"] = args
        self.entries.append(entry)


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_prev_generated(path: str) -> dict:
    """读旧 manifest,取 name→generated 映射,用于保留既有时间戳。

    避免每次全量重生成把 100+ 条时间戳全改掉造成 git churn。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {e["name"]: e.get("generated") for e in data if isinstance(e, dict)}


def opfn(op):
    return getattr(np, op)


def binop(m: Manifest, name, op, adt, avals, bdt, bvals, ashape=None, bshape=None,
          region="bitwise", tol=None, note=""):
    a = arr(adt, avals, ashape)
    b = arr(bdt, bvals, bshape)
    with np.errstate(all="ignore"):  # 边角故意触发 warning;值本身照记
        out = opfn(op)(a, b)
    m.emit(name, op, a, b, out, region=region, tol=tol, note=note)


def unop(m: Manifest, name, op, dt, vals, region="rel", tol=4e-16, note=""):
    a = arr(dt, vals)
    with np.errstate(all="ignore"):
        out = opfn(op)(a)
    m.emit1(name, op, a, out, region=region, tol=tol, note=note)


def agg(m: Manifest, name, op, dt, vals, shape=None, axis=None, keepdims=False,
        region="bitwise", tol=None, note=""):
    """聚合算子(axis/keepdims 进 manifest args)。"""
    a = arr(dt, vals, shape)
    kw = {}
    args = {}
    if axis is not None:
        kw["axis"] = tuple(axis) if isinstance(axis, (list, tuple)) else axis
        args["axis"] = list(axis) if isinstance(axis, (list, tuple)) else [axis]
    if keepdims:
        kw["keepdims"] = True
        args["keepdim"] = True
    with np.errstate(all="ignore"):
        out = getattr(np, op)(a, **kw)
    m.emit1(name, op, a, out, region=region, tol=tol, note=note, args=args or None)


def main(argv=None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if np.__version__ < "2":
        print(f"numpy>=2 required, got {np.__version__}", file=sys.stderr)
        return 1
    os.makedirs(CASES, exist_ok=True)
    if "--gather" in argv:
        return _main_gather()
    if "--nonzero" in argv:
        return _main_nonzero()
    if "--funcs" in argv:
        return _main_funcs()
    m = Manifest()   # 逐元素算子(消费端 layer1/2 直接消费)
    ma = Manifest()  # 归约算子(输出形状与输入不可广播;
 # impl_agg_test/pairwise_test 按名直读 cases/*.npy)

 # —— add:同形 × dtype 面 ——
    binop(m, "add_f8_same", "add", "f8", [1.5, -2.25, 3.125, 0.0], "f8", [0.5, 0.25, -6.0, 0.0],
          note="同形 f8")
    binop(m, "add_i8_same", "add", "i8", [-128, 100, 127, 0], "i8", [1, 3, -1, 0],
          note="i8 回绕边角(-128-1=127, 100+3=103)")
    binop(m, "add_u4_same", "add", "u4", [1, 2**31, 2**32 - 1], "u4", [2, 1, 1],
          note="u4 回绕边角")
    binop(m, "add_f4_same", "add", "f4", [1.5, 2.5], "f4", [0.25, -0.5],
          note="同形 f4")
    binop(m, "add_bool_same", "add", "b", [True, True, False], "b", [True, False, False],
          note="bool×bool → bool(OR;DB1)")

 # —— add:广播/晋级 ——
    binop(m, "add_f8_bcast_r", "add", "f8", [1, 2, 3, 4, 5, 6], "f8", [10, 20, 30],
          ashape=(2, 3), note="右广播 (2,3)+(3,)")
    binop(m, "add_i8_i64_prom", "add", "i1", [-1, 2], "i8", [10, 20],
          note="晋级 i8+i64 → i64")
    binop(m, "add_f4_i8_prom", "add", "f4", [7.5, -7.5], "i1", [2, 2],
          note="晋级 f4+i8 → f4(promote 表)")

 # —— subtract ——
    binop(m, "subtract_i8_wrap", "subtract", "i8", [-128, 0], "i8", [1, 1],
          note="i8 回绕(-128-1=127)")
    binop(m, "subtract_f8_same", "subtract", "f8", [7.5, -7.5, 0.0], "f8", [2.5, 2.5, 0.0],
          note="同形 f8")

 # —— multiply ——
    binop(m, "multiply_i8_overflow", "multiply", "i8", [100, -100], "i8", [3, 3],
          note="i8 溢出回绕 44/-44")
    binop(m, "multiply_f8_edge", "multiply", "f8", [1e150, -2.5, 0.0], "f8", [1e150, 4.0, 1e300],
          note="大数相乘 → Inf 边角")
    binop(m, "multiply_bool_same", "multiply", "b", [True, False, True], "b", [True, True, False],
          note="bool×bool → bool(AND;DB1)")

 # —— divide(true_divide)——
    binop(m, "divide_i8_zerodiv", "divide", "i8", [7, -7, 0], "i8", [2, -2, 0],
          note="整型对 → f8;0 除 IEEE(inf/-inf/nan)")
    binop(m, "divide_i8_minint", "divide", "i8", [-2**63, 2**63 - 1], "i8", [-1, -1],
          note="MinInt64/-1 → 9.223372036854776e+18")
    binop(m, "divide_bool", "divide", "b", [True, False, False], "b", [False, False, False],
          note="bool 对 → f8;[inf,nan,nan]")
    binop(m, "divide_f4_i8_prom", "divide", "f4", [7.5, -7.5, 1.0], "i1", [2, 4, 0],
          note="f4/i8 → f4(promote=f4;IEEE 零除)")
    binop(m, "divide_i4_f8_prom", "divide", "i4", [7, -7], "f8", [2.0, 0.0],
          note="i4/f8 → f8(promote(i4,f8)=f8)")

 # —— floor_divide ——
    binop(m, "floor_divide_i8_edges", "floor_divide", "i8",
          [-2**63, -7, -1, 0, 7], "i8", [-1, 0, 2, 3, -2],
          note="i64 边角:MinInt64/-1 回绕、零除→0、负号 floor")
    binop(m, "floor_divide_u8_zero", "floor_divide", "u8", [2**64 - 1, 7], "u8", [0, 3],
          note="u64 零除 → 0")
    binop(m, "floor_divide_f8_edges", "floor_divide", "f8",
          [7.5, -7.5, np.inf, -np.inf, np.nan, 5.0, -5.0, 0.0],
          "f8", [np.inf, -np.inf, 0.0, -0.0, 2.0, np.inf, -np.inf, 0.0],
          note="npy_divmod 逐位复刻案:±inf/±0/NaN/snap")
    binop(m, "floor_divide_f8_infdiv", "floor_divide", "f8",
          [np.inf, -np.inf, -5.0, 5.0, -7.5, 7.5],
          "f8", [2.0, 2.0, np.inf, -np.inf, -np.inf, np.inf],
          note="inf 被除数与 snap 特例(-5//inf=-1, -7.5//-inf=+0)")
    binop(m, "floor_divide_f4_same", "floor_divide", "f4", [7.5, -7.5, -5.0], "f4",
          [2.0, 2.0, np.inf], note="f32 逐位:3/-4/snap=-1")

 # —— remainder ——
    binop(m, "remainder_i8_edges", "remainder", "i8",
          [-2**63, -7, -1, 0, 7], "i8", [-1, 0, 2, 3, -2],
          note="i64 边角:符号随除数、零除→0")
    binop(m, "remainder_f8_edges", "remainder", "f8",
          [7.5, -7.5, np.inf, -np.inf, np.nan, 5.0, -5.0, 0.0],
          "f8", [np.inf, -np.inf, 0.0, -0.0, 2.0, np.inf, -np.inf, 0.0],
          note="npy_divmod 余数:±inf 符号修正、±0 → NaN")
    binop(m, "remainder_f8_sign", "remainder", "f8", [-5.0, 5.0, 7.5, -7.5],
          "f8", [np.inf, -np.inf, 2.0, 2.0],
          note="符号修正:rem(-5,inf)=+inf、rem(5,-inf)=-inf")

 # —— NaN 位型传播(B §2.8:npFmod 复刻 UCRT)fmod ——
    p123 = nan_with_bits(0x7FF8000000000123, "f8")
    binop(m, "floor_divide_f8_nanprop", "floor_divide", "f8", [p123, p123], "f8", [2.0, 0.0],
          note="NaN payload 原样传播(f8 floor_divide;C fmod 语义)")
    binop(m, "remainder_f8_nanprop", "remainder", "f8", [p123, p123], "f8", [2.0, 0.0],
          note="NaN payload 原样传播(f8 remainder)")
    binop(m, "floor_divide_f4_inf", "floor_divide", "f4",
          [np.inf, -np.inf, np.nan, 0.0], "f4", [2.0, 2.0, 2.0, 0.0],
          note="f32 inf/±0 NaN 位型(UCRT fmodf=0xFFC00000;nan 传播 0x7FC00000)")
    binop(m, "remainder_f4_inf", "remainder", "f4",
          [np.inf, -np.inf, np.nan, 0.0], "f4", [2.0, 2.0, 2.0, 0.0],
          note="同上(f32 remainder)")

 # —— 比较 6(:输出 bool;NaN 有序恒 False、eq 恒 False/ne 恒 True)——
    binop(m, "less_i8_same", "less", "i8", [-3, -1, 0, 2], "i8", [-1, -1, 0, -2],
          note="同形 i8")
    binop(m, "less_equal_i8_same", "less_equal", "i8", [-3, -1, 0, 2], "i8", [-1, -1, 0, -2],
          note="同形 i8")
    binop(m, "greater_i8_same", "greater", "i8", [-3, -1, 0, 2], "i8", [-1, -1, 0, -2],
          note="同形 i8")
    binop(m, "greater_equal_i8_same", "greater_equal", "i8", [-3, -1, 0, 2],
          "i8", [-1, -1, 0, -2], note="同形 i8")
    binop(m, "equal_i8_same", "equal", "i8", [-3, -1, 0, 2], "i8", [-1, -1, 0, -2],
          note="同形 i8")
    binop(m, "not_equal_i8_same", "not_equal", "i8", [-3, -1, 0, 2],
          "i8", [-1, -1, 0, -2], note="同形 i8")
 # NaN 口径
    nanv = [1.0, np.nan, -1.0, np.nan, 0.0]
    nanb = [1.0, 1.0, np.nan, np.nan, -0.0]
    for op in ["less", "less_equal", "greater", "greater_equal", "equal", "not_equal"]:
        binop(m, f"{op}_f8_nan", op, "f8", nanv, "f8", nanb,
              note="f8 NaN 向量:有序恒 False、eq 恒 False、ne 恒 True、±0 相等")
 # bool 有序比较(False < True)
    binop(m, "less_bool_same", "less", "b", [True, False], "b", [False, False],
          note="bool 有序:numpy False<True")
    binop(m, "greater_equal_bool_same", "greater_equal", "b", [True, False],
          "b", [False, False], note="bool 有序")
 # 混合 dtype 晋级 + 广播
    binop(m, "less_i8_i64_prom", "less", "i1", [-1, 2], "i8", [0, 0],
          note="晋级 i8×i64 → 操作数 i64,输出 bool")
    binop(m, "greater_equal_f8_bcast_r", "greater_equal", "f8", [1, 2, 3, 4, 5, 6],
          "f8", [3, 4, 5], ashape=(2, 3), note="右广播 (2,3) >= (3,)")

 # —— 超越 11(;rel 门槛 f8 4e-16 / f4 1e-6 / f16 1e-3)——
    unop(m, "exp_f8", "exp", "f8", [1.0, 10.0, 710.0, -745.0], tol=4e-16,
         note="f8 exp(含溢出 +Inf / denormal)")
    unop(m, "log_f8_edges", "log", "f8", [np.e, 0.0, -1.0], tol=4e-16,
         note="log(0)=-Inf、log(负)=NaN")
    unop(m, "log10_f8_edges", "log10", "f8", [100.0, 0.0, -1.0], tol=4e-16,
         note="log10(0)=-Inf、log10(负)=NaN")
    unop(m, "sqrt_f8_edges", "sqrt", "f8", [2.0, -1.0, -0.0, np.inf], tol=4e-16,
         note="sqrt(负)=NaN、sqrt(-0)=-0")
    unop(m, "sin_f8", "sin", "f8", [1.0, np.inf], tol=4e-16, note="sin(inf)=NaN")
    unop(m, "cos_f8", "cos", "f8", [1.0, np.inf], tol=4e-16, note="cos(inf)=NaN")
    unop(m, "tan_f8", "tan", "f8", [1.0, np.inf], tol=4e-16, note="tan(inf)=NaN")
    unop(m, "asin_f8", "asin", "f8", [0.5, 2.0], tol=4e-16, note="asin(|x|>1)=NaN")
    unop(m, "acos_f8", "acos", "f8", [0.5, 2.0], tol=4e-16, note="acos(|x|>1)=NaN")
    unop(m, "atan_f8", "atan", "f8", [1.0, np.inf, -np.inf], tol=4e-16,
         note="atan(±inf)=±π/2")
    unop(m, "exp_f4", "exp", "f4", [1.0], tol=1e-6, note="f32 exp")
    unop(m, "sqrt_f4", "sqrt", "f4", [2.0], tol=1e-6, note="f32 sqrt")
    unop(m, "sin_f4", "sin", "f4", [1.0], tol=1e-6, note="f32 sin")
 # 整型/bool → 浮点类型解析:i8/u8/b → f16,i16/u16 → f32,其余 → f64
    unop(m, "exp_i8_f16", "exp", "i1", [1, 2], tol=1e-3,
         note="i8 exp → f16(numpy 最小可无损容纳)")
    unop(m, "exp_i16_f32", "exp", "i2", [1], tol=1e-6, note="i16 exp → f32")
    unop(m, "exp_i32_f64", "exp", "i4", [1], tol=4e-16, note="i32 exp → f64")
    unop(m, "sin_bool_f16", "sin", "b", [True, False], tol=1e-3,
         note="bool sin → f16")
 # abs(保 dtype;bitwise)
    unop(m, "abs_i8", "absolute", "i1", [-128, -1, 0, 5], region="bitwise",
         note="MinInt 回绕(-128 → -128)")
    unop(m, "abs_f8", "absolute", "f8", [-0.0, -1.5, 2.0, np.inf], region="bitwise",
         note="abs(-0)=+0、abs(-inf)=inf")
    unop(m, "abs_bool", "absolute", "b", [True, False], region="bitwise",
         note="bool abs 恒等")

 # —— 取整 4(;结果精确整值 → bitwise)——
    edge7 = [np.inf, -np.inf, np.nan, -0.0, 0.0, -1.7, 1.7]
    for op in ["floor", "ceil", "trunc", "rint"]:
        unop(m, f"{op}_f8_edges", op, "f8", edge7, region="bitwise",
             note="±Inf/NaN/±0 保值(含 -0 符号)")
    v5 = [2.5, 3.5, -2.5, 1.1, -1.1]
    for op in ["floor", "ceil", "trunc", "rint"]:
        unop(m, f"{op}_f4", op, "f4", v5, region="bitwise",
             note="f32 取整;rint 银行家舍入")
    for op in ["floor", "rint"]:
        unop(m, f"{op}_f16", op, "f2", [2.5, 3.5, -2.5], region="bitwise",
             note="f16 取整(HALF 策略)")
    unop(m, "floor_i8_identity", "floor", "i1", [-3, 0, 5], region="bitwise",
         note="整型恒等(numpy 整型循环)")
    unop(m, "ceil_bool_identity", "ceil", "b", [True, False], region="bitwise",
         note="bool 恒等")
    unop(m, "rint_i8_f16", "rint", "i1", [1, 2], region="bitwise",
         note="rint 整型 → f16(类型解析同超越)")
    unop(m, "rint_u4_f64", "rint", "u4", [3], region="bitwise",
         note="rint u32 → f64")
    unop(m, "rint_f8_big", "rint", "f8", [2.0**52 + 0.5, 1e16 + 1.0],
         region="bitwise", note="大值银行家舍入(2^52+0.5 → 2^52)")

 # —— pairwise sum(:numpy PW_BLOCKSIZE=128/8 路展开/折半递归)——
 # 各分支:n<8 朴素、n=8 单块、n=17 块+尾、n=128 单块满、n=129 折半、大 n 递归
    unop(ma, "sum_f8_naive3", "sum", "f8", [1e16, 1.0, -1e16], region="bitwise",
         note="n=3 朴素链:1e16+1=1e16,+(-1e16)=0")
    unop(ma, "sum_f8_seven", "sum", "f8", [1e16, 1, 1, 1, 1, 1, -1e16],
         region="bitwise", note="n=7 朴素链")
    unop(ma, "sum_f8_eight", "sum", "f8", [1e16, 1, 1, 1, 1, 1, 1, 1],
         region="bitwise", note="n=8 单块 8 路累加")
    unop(ma, "sum_f8_17", "sum", "f8", [1e16] + [1.0] * 15 + [-1e16],
         region="bitwise", note="n=17 块 + 9 尾")
    unop(ma, "sum_f8_128", "sum", "f8", [1e16] + [1.0] * 127,
         region="bitwise", note="n=128 单块满")
    unop(ma, "sum_f8_129", "sum", "f8", [1e16] + [1.0] * 128,
         region="bitwise", note="n=129 折半 64+65")
    unop(ma, "sum_f8_1000", "sum", "f8", np.linspace(-1e10, 1e10, 1000),
         region="bitwise", note="n=1000 折半递归")
    unop(ma, "sum_f8_big", "sum", "f8", np.linspace(-1.0, 1.0, 100003),
         region="bitwise", note="n=100003 深递归")
    unop(ma, "sum_f8_negzero", "sum", "f8", [-0.0] * 4, region="bitwise",
         note="全 -0 输入 → -0.0(numpy 初值 -0.0 的理由)")
    unop(ma, "sum_f4_128", "sum", "f4", [1e8] + [1.0] * 127, region="bitwise",
         note="f32 单块(1e8+1 被吸收)")
    unop(ma, "sum_f4_1000", "sum", "f4", np.linspace(-1e4, 1e4, 1000),
         region="bitwise", note="f32 折半递归")
    unop(ma, "sum_f4_negzero", "sum", "f4", [-0.0] * 4, region="bitwise",
         note="f32 全 -0 → -0.0")

 # —— 聚合 9(:axis/keepdims/dtype 规则/NaN/多轴展平)——
    a23 = [1, 2, 3, 4, 5, 6]
    agg(ma, "agg_sum_i8_axis0", "sum", "i1", a23, shape=(2, 3), axis=0,
        note="i8 轴 0 → i64")
    agg(ma, "agg_sum_i8_axis1", "sum", "i1", a23, shape=(2, 3), axis=1,
        note="i8 轴 1 → i64")
    agg(ma, "agg_sum_bool", "sum", "b", [True, False, True, True],
        note="bool → i64 计数")
    agg(ma, "agg_sum_u4", "sum", "u4", [1, 2**32 - 1], note="u32 → u64")
    agg(ma, "agg_sum_f8_axis1", "sum", "f8", [1e16, 1.0, -1e16, 1.0, 2.0, 3.0],
        shape=(2, 3), axis=1, note="轴 pairwise 抵消")
    agg(ma, "agg_sum_f8_axes01", "sum", "f8", [1e16, -1e16, 1.0, 1.0],
        shape=(2, 2), axis=(0, 1), note="多轴展平 pairwise(→2.0,非逐轴 1.0)")
    agg(ma, "agg_sum_f8_keepdims", "sum", "f8", a23, shape=(2, 3), axis=1,
        keepdims=True, note="keepdims (2,1)")
    agg(ma, "agg_sum_f8_negaxis", "sum", "f8", a23, shape=(2, 3), axis=-1,
        note="负轴")
    agg(ma, "agg_sum_f4_axis0", "sum", "f4", [1e8, 1.0, 2.0, 3.0], shape=(2, 2),
        axis=0, note="f32 累加")
    agg(ma, "agg_sum_f8_negzero_axis", "sum", "f8", [-0.0] * 4, shape=(2, 2),
        axis=1, note="每输出元素 +0.0 恒等元")
    agg(ma, "agg_mean_f8_axis1", "mean", "f8", a23, shape=(2, 3), axis=1,
        note="mean f8")
    agg(ma, "agg_mean_f4_axis0", "mean", "f4", [1, 2, 3, 4], shape=(2, 2),
        axis=0, note="mean f32 保 dtype")
    agg(ma, "agg_mean_i8", "mean", "i1", [1, 2, 3, 4], note="mean 整型 → f64")
    agg(ma, "agg_var_f8_axis1", "var", "f8", [1, 2, 3, 4, 1e16], shape=(1, 5),
        axis=1, region="rel", tol=1e-15, note="var(ddof=0)")
    agg(ma, "agg_std_f8", "std", "f8", [1, 2, 3, 4, 1e16], region="rel",
        tol=1e-15, note="std")
    agg(ma, "agg_var_f4", "var", "f4", [1, 2, 3, 4], region="rel", tol=1e-6,
        note="var f32")
 # f16 聚合:HALF 策略,输出 dtype = f16(numpy 实测 mean/var/std(f16) → f16,
 # 内部按 f32 计算后 cast 回 f16)。
    agg(ma, "agg_mean_f16", "mean", "f2", [1, 2, 3, 4, 5], note="mean f16 → f16(HALF)")
    agg(ma, "agg_var_f16", "var", "f2", [1, 2, 3, 4, 5], region="rel", tol=1e-2,
        note="var f16 (HALF)")
    agg(ma, "agg_std_f16", "std", "f2", [1, 2, 3, 4, 5], region="rel", tol=1e-2,
        note="std f16 (HALF)")
    agg(ma, "agg_min_f16", "min", "f2", [3.0, 1.0, 5.0, 2.0, 4.0],
        note="min f16 → f16")
    agg(ma, "agg_max_f16", "max", "f2", [3.0, 1.0, 5.0, 2.0, 4.0],
        note="max f16 → f16")
    agg(ma, "agg_argmin_f16", "argmin", "f2", [3.0, 1.0, 5.0, 2.0, 4.0],
        note="argmin f16 → i64")
    agg(ma, "agg_argmax_f16", "argmax", "f2", [3.0, 1.0, 5.0, 2.0, 4.0],
        note="argmax f16 → i64")
    agg(ma, "agg_any_f16", "any", "f2", [0.0, 1.0, 0.0], note="any f16 → bool")
    agg(ma, "agg_all_f16", "all", "f2", [1.0, 1.0, 0.0], note="all f16 → bool")
    agg(ma, "agg_min_f16_nan", "min", "f2", [3.0, np.nan, 1.0],
        note="min f16 NaN 传播")
    agg(ma, "agg_argmin_f16_nan", "argmin", "f2", [3.0, np.nan, 1.0],
        note="argmin f16 首个 NaN 位置胜出")
    agg(ma, "agg_min_f8_nan", "min", "f8", [3.0, np.nan, 1.0],
        note="min NaN 传播")
    agg(ma, "agg_max_i8_axis1", "max", "i1", [1, -5, 7, 2, 0, -3], shape=(2, 3),
        axis=1, note="max i8 轴")
    agg(ma, "agg_min_bool", "min", "b", [True, False, True], note="bool min")
    agg(ma, "agg_argmin_f8_nan", "argmin", "f8", [3.0, np.nan, 1.0],
        note="首个 NaN 位置胜出")
    agg(ma, "agg_argmax_i8_axis1", "argmax", "i1", [1, 5, 3, 2, 0, 3],
        shape=(2, 3), axis=1, note="argmax 轴,首现")
    agg(ma, "agg_any_f8_nan", "any", "f8", [0.0, np.nan], note="NaN 为真")
    agg(ma, "agg_all_i8", "all", "i1", [1, 2, 0], note="all 有 0 → false")
    agg(ma, "agg_all_f8_axis1", "all", "f8", [1.0, 2.0, 0.0, 4.0], shape=(2, 2),
        axis=1, note="all 轴")

 # —— f16 算子面(:HALF 策略 f16→f32→运算→f16)——
    a16 = [1.5, 0.1, -2.5]
    b16 = [2.25, 0.2, 0.0]
    for op in ["add", "subtract", "multiply", "divide", "floor_divide", "remainder"]:
        binop(m, f"{op}_f16_same", op, "f2", a16, "f2", b16,
              note="f16 HALF 策略(f32 中间值→f16)")
    binop(m, "power_f16_same", "power", "f2", a16, "f2", b16, region="rel",
          tol=1e-3, note="f16 pow(f32 pow→f16)")
    binop(m, "less_f16_same", "less", "f2", [1.5, -0.5], "f2", [2.5, 0.5],
          note="f16 比较 → bool")
    binop(m, "add_f16_i8_prom", "add", "f2", [1.0], "i1", [1], note="f16+i8 → f16")
    binop(m, "add_f16_f4_prom", "add", "f2", [1.0], "f4", [1.0],
          note="f16+f32 → f32(promote 表)")
    unop(ma, "sum_f16", "sum", "f2", a16, region="bitwise",
         note="f16 求和:f32 pairwise 累加→f16")

 # —— 复数 c64/c128(:add/sub/mul 逐位;div 缩放算法;abs/conj)——
    ca = [1 + 2j, -3.5 + 0.5j, 0 + 0j]
    cb = [3 - 1j, 0.5 + 2j, 0 + 0j]
    binop(m, "add_c128_same", "add", "c16", ca, "c16", cb, note="c128 加法(逐位)")
    binop(m, "subtract_c128_same", "subtract", "c16", ca, "c16", cb, note="c128 减法")
    binop(m, "multiply_c128_same", "multiply", "c16", ca, "c16", cb,
          note="c128 乘法 naive 公式(逐位)")
    binop(m, "divide_c128_same", "divide", "c16", ca, "c16", cb, region="rel",
          tol=4e-16, note="c128 除法缩放算法(numpy 同式)")
    binop(m, "divide_c128_edges", "divide", "c16", [1 + 1j, 0 + 0j], "c16",
          [0 + 0j, 0 + 0j], region="rel", tol=4e-16,
          note="除零:(1+1j)/0 → inf+infj;0/0 → nan+nanj")
    unop(m, "abs_c128", "absolute", "c16", ca, region="bitwise",
         note="c128 abs = hypot → f64")
    unop(m, "abs_c64", "absolute", "c8", ca, region="rel", tol=1e-6,
         note="c64 abs → f32")
    unop(m, "conj_c128", "conj", "c16", ca, region="bitwise", note="共轭")
    binop(m, "eq_c128_same", "equal", "c16", ca, "c16", ca, note="c128 eq → bool")
    binop(m, "mul_c64_same", "multiply", "c8", ca, "c8", cb,
          note="c64 乘法(逐位)")
    binop(m, "add_c64_i8_prom", "add", "c8", [1 + 1j], "i1", [1],
          note="c64+i8 → c64")
    binop(m, "add_c64_f8_prom", "add", "c8", [1 + 1j], "f8", [1.0],
          note="c64+f8 → c128")

 # —— datetime64(;:算术/NaT/跨单位比较)——
    def dtarr(unit, counts):
        return np.array(counts, dtype=f"datetime64[{unit}]")

    dts = dtarr("D", [18262, 18263])  # 2020-01-01/02
    ones = np.array([1, 2], dtype=np.int64)
    m.emit("dt_add_i64", "add", dts, ones, np.add(dts, ones), region="bitwise",
           note="M8[D] + i64(单位不变)")
    m.emit("dt_sub_i64", "subtract", dts, ones, np.subtract(dts, ones),
           region="bitwise", note="M8[D] - i64")
    natd = dtarr("D", [np.iinfo(np.int64).min, 18263])
    fives = np.array([5, 5], dtype=np.int64)
    m.emit("dt_nat_add", "add", natd, fives, np.add(natd, fives),
           region="bitwise", note="NaT 参与算术 → NaT")
    m.emit("dt_eq_nat", "equal", dts, natd, np.equal(dts, natd),
           region="bitwise", note="NaT eq 恒 False")
    m.emit("dt_lt_nat", "less", natd, dts, np.less(natd, dts),
           region="bitwise", note="NaT 有序比较恒 False")
    m.emit("dt_eq_crossunit", "equal", dtarr("M", [600]), dtarr("D", [18262]),
           np.equal(dtarr("M", [600]), dtarr("D", [18262])), region="bitwise",
           note="跨单位比较:2020-01 == 2020-01-01")

 # —— 定宽字符串 S/U 比较——
    def sarr(w, vals):
        return np.array(vals, dtype=f"|S{w}")

    def uarr(w, vals):
        return np.array(vals, dtype=f"<U{w}")

    sa = sarr(3, [b"ab", b"abc", b"ab", b"zz"])
    sb = sarr(3, [b"ab", b"abd", b"ab", b"zz"])
    m.emit("str_eq_s", "equal", sa, sb, np.equal(sa, sb), region="bitwise",
           note="S 尾 NUL 裁剪 + 字节序")
    m.emit("str_lt_s", "less", sa, sb, np.less(sa, sb), region="bitwise",
           note="S 字节序")
    ua = uarr(3, ["ab", "abc", "中", "zz"])
    ub = uarr(3, ["ab", "abd", "中", "zz"])
    m.emit("str_eq_u", "equal", ua, ub, np.equal(ua, ub), region="bitwise",
           note="U 码点序")
    m.emit("str_lt_u", "less", ua, ub, np.less(ua, ub), region="bitwise",
           note="U 码点序")
    sw = sarr(4, [b"ab", b"abc", b"ab", b"zz"])
    m.emit("str_eq_width", "equal", sa, sw, np.equal(sa, sw), region="bitwise",
           note="S3 vs S4 按值比较(非按填充)")

 # —— power ——
    binop(m, "power_i8_wrap", "power", "i8", [2, 3, -2, 0], "i8", [63, 40, 10, 0],
          region="bitwise", note="i64 回绕:2^63=MinInt64、3^40 回绕、0^0=1")
    binop(m, "power_f8_edges", "power", "f8", [2.0, -8.0, 0.0, 1.0],
          "f8", [-1.0, 1.0 / 3.0, -1.0, np.inf],
          region="rel", tol=4e-16,
          note="2^-1=0.5、(-8)^(1/3)=NaN、0^-1=+Inf、1^inf=1")
    binop(m, "power_f4_rel", "power", "f4", [2.0, 9.0], "f4", [0.5, 0.5],
          region="rel", tol=1e-6,
          note="f32 pow 门槛 1ulp×裕量")

 # newline="\n":仓库.gitattributes 规定 *.json 落 LF,避免 Windows 文本模式
 # 把 manifest 写成 CRLF 造成 git 反复 churn。
    for path, entries in (
        (os.path.join(REFDIR, "manifest.json"), m.entries),
        (os.path.join(REFDIR, "manifest_agg.json"), ma.entries),
    ):
 # 保留既有 case 的时间戳,只对新增 case 打新戳(见 _load_prev_generated)。
        prev_gen = _load_prev_generated(path)
        stamp = _now()
        for e in entries:
            e["generated"] = prev_gen.get(e["name"], stamp)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(entries, f, ensure_ascii=False, indent=1)
            f.write("\n")
        n_bit = sum(1 for e in entries if e["region"] == "bitwise")
        print(f"manifest : {path}({len(entries)} cases: {n_bit} bitwise / {len(entries) - n_bit} rel)")
    print(f"generator: {GENERATOR}")
    print(f"fixtures : {CASES} (*.npy)")
    return 0


# —— --gather 模式(/ gather/掩码族,manifest_gather.json 第三份双载)——
#
# 只产出前缀 gather_/put_/mask_/bincount_ 的新案 + 对应 cases/*.npy;不碰既有两份
# manifest 及其 npy(默认无参路径字节不变,哈希可证)。幂等:重跑保留首跑时间戳
# (同 _load_prev_generated 惯例,见默认路径写盘块)。

def _emit_gather(name, op, inputs, out, note, args=None, out_shape=None):
    """--gather 单案发射器:变元数输入(suffix 来自 input 列表),可选 out_shape
    (仅当期望输出形状不等于输入广播结果时出现;无此字段语义与既有清单一致)。"""
    paths = []
    for s, a in inputs:
        p = f"cases/{name}_{s}.npy"
        np.save(os.path.join(REFDIR, p), a, allow_pickle=False)
        paths.append(p)
    fo = f"cases/{name}_out.npy"
    np.save(os.path.join(REFDIR, fo), out, allow_pickle=False)
    entry = {
        "name": name,
        "op": op,
        "inputs": paths,
        "expect": fo,
        "region": "bitwise",
        "generator": GENERATOR,
        "generated": _now(),
        "note": note,
    }
    if args is not None:
        entry["args"] = args
    if out_shape is not None:
        entry["out_shape"] = out_shape
    return entry


def _take_along_axis_go(a, idx, axis):
    """复刻 Go TakeAlongAxis 语义:非轴维**不**广播到源形状,输出形状=idx.shape;
    负索引按轴维长归一。numpy take_along_axis 会把非轴维 1 广播 → 本例结果不同,
    故此路用 Go 语义复刻生成期望值(见 manifest note)"""
    out = np.empty(idx.shape, dtype=a.dtype)
    nd = a.ndim
    for k in range(out.size):
        pos = [0] * nd
        kk = k
        for d in range(nd - 1, -1, -1):
            pos[d] = kk % idx.shape[d]
            kk //= idx.shape[d]
        v = int(idx.flat[k])
        if v < 0:
            v += a.shape[axis]
        tup = list(pos)
        tup[axis] = v
        out[tuple(pos)] = a[tuple(tup)]
    return out


def _main_gather() -> int:
    """--gather:只生成 gather/掩码族新案,写 manifest_gather.json + 新 cases/*.npy。"""
    entries = []

 # —— take 族(≡ Go Take / TakeAlongAxis)——
    a23 = arr("i1", [10, 20, 30, 40, 50, 60], shape=(2, 3))
    idx1d = arr("i1", [0, -1, 2, 5])
    entries.append(_emit_gather(
        "gather_take_i8_1d", "take", [("a", a23), ("b", idx1d)],
        np.take(a23, idx1d),
        note="flat take:负索引归一+重复;out shape=idx (4,),a(2,3)不可广播到它",
        out_shape=[4]))
    idx2d = arr("i1", [-2, 1, 3, 5], shape=(2, 2))
    entries.append(_emit_gather(
        "gather_take_i8_2d", "take", [("a", a23), ("b", idx2d)],
        np.take(a23, idx2d),
        note="非方 idx (2,2)→out (2,2);a(2,3)不可广播到它",
        out_shape=[2, 2]))
    af8 = np.empty(2, dtype=np.float64)
    af8[0] = nan_with_bits(0x7FF8000000000123, "f8")
    af8[1] = 1.5
    entries.append(_emit_gather(
        "gather_take_f8_nan", "take", [("a", af8), ("b", arr("i1", [0]))],
        np.take(af8, [0]),
        note="f8 字节级拷贝:NaN payload 保留(bitwise)",
        out_shape=[1]))

    a34 = arr("i1", list(range(12)), shape=(3, 4))
    idx_ta = arr("i1", [-1, 0, 1, -1, 0, 2], shape=(3, 2))
    entries.append(_emit_gather(
        "gather_take_along_axis_i8", "take_along_axis", [("a", a34), ("b", idx_ta)],
        np.take_along_axis(a34, idx_ta, axis=1),
        note="a(3,4)×idx(3,2) axis=1:负+重复;out shape=idx (3,2)",
        args={"axis": [1]}, out_shape=[3, 2]))
    a22 = arr("i1", list(range(4)), shape=(2, 2))
    idx_bcast = arr("i1", [0, 1], shape=(1, 2))
    entries.append(_emit_gather(
        "gather_take_along_axis_bcast", "take_along_axis", [("a", a22), ("b", idx_bcast)],
        _take_along_axis_go(a22, idx_bcast, 1),
        note="非轴维 1:Go out=idx (1,2)≠numpy 广播结果;期望值按 Go 语义复刻生成",
        args={"axis": [1]}, out_shape=[1, 2]))

 # —— put(≡ Go Put:3 输入,原地 mutate)——
    src6 = arr("i1", [10, 20, 30, 40, 50, 60])
    idx_put = arr("i1", [-1, 0, -1])
    vals3 = arr("i1", [7, 8, 9])
    src6_out = src6.copy()
    np.put(src6_out, idx_put, vals3)
    entries.append(_emit_gather(
        "put_i8_neg_dup", "put", [("src", src6), ("idx", idx_put), ("vals", vals3)],
        src6_out,
        note="原地 mutate:负归一+重复 last-write-wins;out=src 形状 (6,),3 输入不可广播→out_shape",
        out_shape=[6]))
    src6b = arr("i1", [10, 20, 30, 40, 50, 60])
    vals_f8 = arr("f8", [1.9, 2.9])
    src6b_out = src6b.copy()
    np.put(src6b_out, [0, 1], vals_f8)
    entries.append(_emit_gather(
        "put_i8_unsafe_f8", "put", [("src", src6b), ("idx", arr("i1", [0, 1])), ("vals", vals_f8)],
        src6b_out,
        note="不安全转换 f8→i8 向零截断(1.9→1);原地 mutate;out_shape=[6]",
        out_shape=[6]))

 # —— 掩码压缩(≡ Go Compress / CompressAxis)——
    a6 = arr("i1", list(range(6)), shape=(2, 3))
    cond_short = arr("b", [True, False])
    entries.append(_emit_gather(
        "mask_compress_short", "compress", [("a", a6), ("cond", cond_short)],
        np.compress(cond_short, a6),
        note="flat compress:短条件只作用于前 min(len,size) 个 flat 元素;out (1,)",
        out_shape=[1]))
    cond_full = arr("b", [False, True, False, True, False, True])
    entries.append(_emit_gather(
        "mask_compress_full", "compress", [("a", a6), ("cond", cond_full)],
        np.compress(cond_full, a6),
        note="flat compress:out (3,) 保留 flat 位 1/3/5",
        out_shape=[3]))
    a12 = arr("i1", list(range(12)), shape=(3, 4))
    cond_ax = arr("b", [True, False, True])
    entries.append(_emit_gather(
        "mask_compress_axis_i8", "compress", [("a", a12), ("cond", cond_ax)],
        np.compress(cond_ax, a12, axis=0),
        note="轴压缩:保留行 0/2→(2,4)",
        args={"axis": [0]}, out_shape=[2, 4]))

 # —— count_nonzero(≡ Go CountNonzero:返回标量 int)——
    a22c = arr("i1", [0, 1, 2, 0], shape=(2, 2))
    entries.append(_emit_gather(
        "mask_count_nonzero_i8", "count_nonzero", [("x", a22c)],
        np.array(np.count_nonzero(a22c), dtype=np.int64),
        note="0-d 标量输出(Go 返回 int);out_shape [] = 0-d",
        out_shape=[]))
    anan = arr("f8", [0.0, float("nan")])
    entries.append(_emit_gather(
        "mask_count_nonzero_nan", "count_nonzero", [("x", anan)],
        np.array(np.count_nonzero(anan), dtype=np.int64),
        note="NaN 计入非零;0-d 输出",
        out_shape=[]))

 # —— bincount(≡ Go Bincount:minlength 走 args.n)——
    x4 = arr("i1", [0, 1, 1, 3])
    entries.append(_emit_gather(
        "bincount_i8", "bincount", [("x", x4)],
        np.bincount(x4),
        note="输出 i64 长=max+1;形状与输入同可广播→无 out_shape",
        args={"n": 0}))
    x2 = arr("i1", [0, 1])
    entries.append(_emit_gather(
        "bincount_minlength", "bincount", [("x", x2)],
        np.bincount(x2, minlength=5),
        note="minlength=5 补零到 (5,)",
        args={"n": 5}, out_shape=[5]))
    xu = arr("u1", [0, 1, 2])
    entries.append(_emit_gather(
        "bincount_u8", "bincount", [("x", xu)],
        np.bincount(xu),
        note="u8 输入→i64 输出(dtype 变化)",
        args={"n": 0}))

 # 写 manifest_gather.json(第三份;不碰旧两份 manifest 及其 npy)
    mg_path = os.path.join(REFDIR, "manifest_gather.json")
    prev_gen = _load_prev_generated(mg_path)
    stamp = _now()
    for e in entries:
        e["generated"] = prev_gen.get(e["name"], stamp)
    with open(mg_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_bit = sum(1 for e in entries if e["region"] == "bitwise")
    print(f"manifest_gather : {mg_path}({len(entries)} cases: {n_bit} bitwise / {len(entries) - n_bit} rel)")
    print(f"generator: {GENERATOR}")
    print(f"fixtures : {CASES} (*.npy,--gather 前缀新增)")
    return 0


# —— --nonzero 模式(nonzero 多输出族,第四份清单 manifest_nonzero.json)——
#
# 只产出 nonzero 新案 + 对应 cases/*.npy;不碰旧三份 manifest 及其 npy(默认无参路径字节不变,
# 哈希可证)。幂等:重跑保留首跑时间戳(同 _load_prev_generated 惯例,见默认路径写盘块)。
# 多输出条目无 expect 字段;outputs/out_shapes 按元组序(第 0 维索引在前),每个输出
# intp→i64 落盘前 astype,保证 DInt64;空结果也落 npy(形状 [0])。

def _emit_nonzero(name, a, outs, note):
    """--nonzero 单案发射器:np.nonzero 多输出;单输出沿用 {name}_out.npy 惯例,
    多输出 {name}_out{i}.npy(元组序,第 0 维索引在前)。"""
    paths = [f"cases/{name}_a.npy"]
    np.save(os.path.join(REFDIR, paths[0]), a, allow_pickle=False)
    fo_paths = []
    for i, o in enumerate(outs):
        p = f"cases/{name}_out.npy" if len(outs) == 1 else f"cases/{name}_out{i}.npy"
        np.save(os.path.join(REFDIR, p), o.astype(np.int64), allow_pickle=False)
        fo_paths.append(p)
    entry = {
        "name": name,
        "op": "nonzero",
        "inputs": paths,
        "outputs": fo_paths,
        "out_shapes": [list(o.shape) for o in outs],
        "region": "bitwise",
        "generator": GENERATOR,
        "generated": _now(),
        "note": note,
    }
    return entry


def _main_nonzero() -> int:
    """--nonzero:只生成 nonzero 多输出族,写 manifest_nonzero.json + 新 cases/*.npy。"""
    entries = []

 # —— 1D i8 ——
    a = arr("i1", [0, 5, 0, -3, 0, 7])
    with np.errstate(all="ignore"):
        outs = np.nonzero(a)
    entries.append(_emit_nonzero("nonzero_i8_1d", a, outs,
                                 note="i8 1D:[0,5,0,-3,0,7] nonzero 位置 [1,3,5];负值亦为真"))

 # —— 2D i8 ——
    a = arr("i1", [[0, 1, 0], [2, 0, 3]])
    with np.errstate(all="ignore"):
        outs = np.nonzero(a)
    entries.append(_emit_nonzero("nonzero_i8_2d", a, outs,
                                 note="i8(2,3):非零元 (0,1)/(1,0)/(1,2),rows[0,1,1] cols[1,0,2]"))

 # —— 3D f8 特值(NaN/inf 为真、-0.0 为假)——
    af = arr("f8", [0.0, -0.0, float("nan"), float("inf"),
                   1.5, -2.5, 3.75, -4.25], shape=(2, 2, 2))
    with np.errstate(all="ignore"):
        outs = np.nonzero(af)
    entries.append(_emit_nonzero("nonzero_f8_3d", af, outs,
                                 note="f8(2,2,2):NaN/inf 为真、-0.0 为假(nonzero 数=6)"))

 # —— 全零 2D ——
    a = arr("i1", [0] * 12, shape=(3, 4))
    with np.errstate(all="ignore"):
        outs = np.nonzero(a)
    entries.append(_emit_nonzero("nonzero_all_zero", a, outs,
                                 note="i8(3,4) 全零:两输出均空(形状 [0]),验空 npy 落盘"))

 # —— bool 2D ——
    ab = arr("b", [[True, False, True], [False, True, False]])
    with np.errstate(all="ignore"):
        outs = np.nonzero(ab)
    entries.append(_emit_nonzero("nonzero_bool_2d", ab, outs,
                                 note="bool(2,3):True 为真、False 为假"))

 # —— 1D u8 ——
    a = arr("u1", [0, 1, 0, 255, 0])
    with np.errstate(all="ignore"):
        outs = np.nonzero(a)
    entries.append(_emit_nonzero("nonzero_u1_1d", a, outs,
                                 note="u8 1D:含 0 与非零(255),位置 [1,3]"))

 # 写 manifest_nonzero.json(第四份;不碰旧三份 manifest 及其 npy)
    nz_path = os.path.join(REFDIR, "manifest_nonzero.json")
    prev_gen = _load_prev_generated(nz_path)
    stamp = _now()
    for e in entries:
        e["generated"] = prev_gen.get(e["name"], stamp)
    with open(nz_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_bit = sum(1 for e in entries if e["region"] == "bitwise")
    print(f"manifest_nonzero : {nz_path}({len(entries)} cases: {n_bit} bitwise / {len(entries) - n_bit} rel)")
    print(f"generator: {GENERATOR}")
    print(f"fixtures : {CASES} (*.npy,--nonzero 前缀新增)")
    return 0


# —— --funcs 模式(round/clip/where 第五清单 manifest_funcs.json)——
#
# 只产出 funcs_ 前缀新案 + 对应 cases/*.npy;不碰既有四份 manifest 及其 npy(默认无参路径
# 字节不变,哈希可证)。幂等:重跑保留首跑时间戳(同 _load_prev_generated 惯例,见默认路径写盘块)。

def _emit_funcs(name, op, inputs, out, note, args=None, out_shape=None):
    """--funcs 单案发射器:变元数输入(suffix 来自 input 列表),输出固定 {name}_out.npy。
    round/clip/where 输出形状=输入广播结果,故不带 out_shape(与既有清单一致)。"""
    paths = []
    for s, a in inputs:
        p = f"cases/{name}_{s}.npy"
        np.save(os.path.join(REFDIR, p), a, allow_pickle=False)
        paths.append(p)
    fo = f"cases/{name}_out.npy"
    np.save(os.path.join(REFDIR, fo), out, allow_pickle=False)
    entry = {
        "name": name,
        "op": op,
        "inputs": paths,
        "expect": fo,
        "region": "bitwise",
        "generator": GENERATOR,
        "generated": _now(),
        "note": note,
    }
    if args is not None:
        entry["args"] = args
    if out_shape is not None:
        entry["out_shape"] = list(out_shape)
    return entry


def _emit_shape(name, op, inputs, out, note, args=None):
    """形状改变类发射器(repeat/tile/atleast_*):必须带 out_shape,消费端据此
    跳过"输入须广播到期望形状"的断言并直接断言结果形状。"""
    return _emit_funcs(name, op, inputs, out, note, args=args, out_shape=np.shape(out))

def _emit_multi_funcs(name, op, inputs, outs, note, args=None, tol=None):
    """多输出 funcs 发射器(meshgrid/gradient):outputs + out_shapes,与 nonzero 清单同口径。"""
    paths = []
    for s, a in inputs:
        p = f"cases/{name}_{s}.npy"
        np.save(os.path.join(REFDIR, p), a, allow_pickle=False)
        paths.append(p)
    opaths, oshapes = [], []
    for i, o in enumerate(outs):
        p = f"cases/{name}_out{i}.npy"
        np.save(os.path.join(REFDIR, p), o, allow_pickle=False)
        opaths.append(p)
        oshapes.append(list(np.shape(o)))
    entry = {
        "name": name,
        "op": op,
        "inputs": paths,
        "outputs": opaths,
        "out_shapes": oshapes,
        "region": "bitwise" if not tol else "rel",
        "generator": GENERATOR,
        "generated": _now(),
        "note": note,
    }
    if tol:
        entry["tol"] = tol
    if args is not None:
        entry["args"] = args
    return entry

def _emit_rel(name, op, inputs, out, note, tol, args=None, out_shape=None):
    """rel 区发射器(FFT 这类算法族:不同实现末位有差异,用相对容差)。"""
    paths = []
    for s, a in inputs:
        p = f"cases/{name}_{s}.npy"
        np.save(os.path.join(REFDIR, p), a, allow_pickle=False)
        paths.append(p)
    fo = f"cases/{name}_out.npy"
    np.save(os.path.join(REFDIR, fo), out, allow_pickle=False)
    entry = {
        "name": name,
        "op": op,
        "inputs": paths,
        "expect": fo,
        "region": "rel",
        "tol": tol,
        "generator": GENERATOR,
        "generated": _now(),
        "note": note,
    }
    if args is not None:
        entry["args"] = args
    if out_shape is not None:
        entry["out_shape"] = list(out_shape)
    return entry

def _main_funcs() -> int:
    """--funcs:只生成 round/clip/where 新案,写 manifest_funcs.json + 新 cases/*.npy。"""
    entries = []

 # —— round(一元,args.decimals)——
    a = arr("f8", [0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -0.0])
    with np.errstate(all="ignore"):
        out = np.round(a, 0)
    entries.append(_emit_funcs("funcs_round_f8_d0", "round", [("a", a)], out,
                               note="f8 decimals=0:half-to-even(0.5→0、2.5→2、3.5→4)+ -0.5/-0.0→-0.0",
                               args={"decimals": 0}))

    a = arr("f8", [2.675, 1.005, 3.14159])
    with np.errstate(all="ignore"):
        out = np.round(a, 2)
    entries.append(_emit_funcs("funcs_round_f8_d2", "round", [("a", a)], out,
                               note="f8 decimals=2:含二进制非精确十进制值(2.675/1.005)验 numpy 实际位型",
                               args={"decimals": 2}))

    a = arr("f8", [1250.0, -1250.0, 1499.0])
    with np.errstate(all="ignore"):
        out = np.round(a, -2)
    entries.append(_emit_funcs("funcs_round_f8_dn2", "round", [("a", a)], out,
                               note="f8 decimals=-2:百位取整(1250→1200、-1250→-1200、1499→1500)",
                               args={"decimals": -2}))

    a = arr("f4", [1.25, 2.35, -1.75, 0.05, 9.85, 3.14])
    with np.errstate(all="ignore"):
        out = np.round(a, 1)
    entries.append(_emit_funcs("funcs_round_f4_d1", "round", [("a", a)], out,
                               note="f4 decimals=1:6 值,half-to-even;out dtype f32",
                               args={"decimals": 1}))

    a = arr("i4", [5, -3, 0, 42, -17])
    with np.errstate(all="ignore"):
        out = np.round(a, 0)
    entries.append(_emit_funcs("funcs_round_i4_d0", "round", [("a", a)], out,
                               note="i4 decimals=0:整型原样,out dtype i32",
                               args={"decimals": 0}))

    a = arr("c16", [1.5 + 2.5j, -0.5 - 0.5j, 3.2 + 4.8j])
    with np.errstate(all="ignore"):
        out = np.round(a, 0)
    entries.append(_emit_funcs("funcs_round_c16_d0", "round", [("a", a)], out,
                               note="c16 decimals=0:实虚分量各自取整,out dtype c128",
                               args={"decimals": 0}))

    a = arr("f8", [np.nan, np.inf, -np.inf, 1.5, -2.5])
    with np.errstate(all="ignore"):
        out = np.round(a, 0)
    entries.append(_emit_funcs("funcs_round_f8_special", "round", [("a", a)], out,
                               note="f8 decimals=0:NaN/±Inf 不变、其余取整",
                               args={"decimals": 0}))

 # —— clip(三输入:a + 0-d lo + 0-d hi)——
    a = arr("f8", [2.0, -3.0, 0.5, 1.75, -1.25])
    lo = np.array(-0.5, dtype=np.float64)
    hi = np.array(1.5, dtype=np.float64)
    with np.errstate(all="ignore"):
        out = np.clip(a, lo, hi)
    entries.append(_emit_funcs("funcs_clip_f8", "clip", [("a", a), ("lo", lo), ("hi", hi)], out,
                               note="f8 常规:5 元素+0-d f64 界;out dtype f64"))

    a = arr("f8", [2.0, np.nan, -3.0, 0.5, 1.75])
    lo = np.array(-0.5, dtype=np.float64)
    hi = np.array(1.5, dtype=np.float64)
    with np.errstate(all="ignore"):
        out = np.clip(a, lo, hi)
    entries.append(_emit_funcs("funcs_clip_f8_nan", "clip", [("a", a), ("lo", lo), ("hi", hi)], out,
                               note="f8 含 NaN:NaN 传播(同界);out dtype f64"))

    a = arr("i4", [5, -1, 3, 7])
    lo = np.array(-2, dtype=np.int32)
    hi = np.array(4, dtype=np.int32)
    with np.errstate(all="ignore"):
        out = np.clip(a, lo, hi)
    entries.append(_emit_funcs("funcs_clip_i4", "clip", [("a", a), ("lo", lo), ("hi", hi)], out,
                               note="i4 数组+i32 0-d 界;out dtype i32"))

    a = arr("i4", [0, 2, 5])
    lo = np.array(3, dtype=np.int32)
    hi = np.array(1, dtype=np.int32)
    with np.errstate(all="ignore"):
        out = np.clip(a, lo, hi)
    entries.append(_emit_funcs("funcs_clip_i4_lohi", "clip", [("a", a), ("lo", lo), ("hi", hi)], out,
                               note="i4 lo>hi(3/1):numpy 实际结果恒为 hi=1"))

    a = arr("f8", [2.5, -1.75, 0.5])
    lo = np.array(0, dtype=np.int32)
    hi = np.array(2, dtype=np.int32)
    with np.errstate(all="ignore"):
        out = np.clip(a, lo, hi)
    entries.append(_emit_funcs("funcs_clip_f8_prom", "clip", [("a", a), ("lo", lo), ("hi", hi)], out,
                               note="f8 数组+i32 0-d 界:promote 到 f64"))

 # —— where(三输入:cond + x + y)——
    cond = arr("b", [[True, False, True], [False, True, False]], shape=(2, 3))
    x = arr("i4", [10, 20, 30])
    y = np.array(99, dtype=np.int32)
    with np.errstate(all="ignore"):
        out = np.where(cond, x, y)
    entries.append(_emit_funcs("funcs_where_b23_i4", "where", [("cond", cond), ("x", x), ("y", y)], out,
                               note="cond bool(2,3)+x i4(3,)+y i4 0-d→out (2,3) i32;广播+晋级"))

    cond = arr("b", [[True, False, True], [False, True, False]], shape=(2, 3))
    x = arr("i4", [[1, 2, 3], [4, 5, 6]], shape=(2, 3))
    y = arr("f8", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], shape=(2, 3))
    with np.errstate(all="ignore"):
        out = np.where(cond, x, y)
    entries.append(_emit_funcs("funcs_where_b23_i4f8", "where", [("cond", cond), ("x", x), ("y", y)], out,
                               note="cond bool(2,3)+x i4(2,3)+y f8(2,3)→out (2,3) f64;晋级"))

    cond = arr("i4", [0, 1, -1, 0, 2])
    x = arr("f8", [1.5, 2.5, 3.5, 4.5, 5.5])
    y = arr("f8", [10.0, 20.0, 30.0, 40.0, 50.0])
    with np.errstate(all="ignore"):
        out = np.where(cond, x, y)
    entries.append(_emit_funcs("funcs_where_i4_f8", "where", [("cond", cond), ("x", x), ("y", y)], out,
                               note="cond i4(5,)含 0/非 0(!=0 为真)+x/y f8(5,)→out (5,) f64"))

 # —— Step 1B: isnan/isinf/isfinite/logical_and/or/not/xor/maximum/minimum(9 算子,14 案)——
    a = arr("f8", [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0])
    with np.errstate(all="ignore"):
        out = np.isnan(a)
    entries.append(_emit_funcs("funcs_isnan_f8", "isnan", [("a", a)], out,
                               note="f8 [NaN,±Inf,±0,1]→[T,T,T,F,F,F];out dtype bool"))

    a = arr("f4", [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0])
    with np.errstate(all="ignore"):
        out = np.isnan(a)
    entries.append(_emit_funcs("funcs_isnan_f4", "isnan", [("a", a)], out,
                               note="f4 同 f8 形;out dtype bool"))

    a = arr("c16", [1 + 2j, np.nan + 0j, np.nan * 1j])
    with np.errstate(all="ignore"):
        out = np.isnan(a)
    entries.append(_emit_funcs("funcs_isnan_c16", "isnan", [("a", a)], out,
                               note="c128:实/虚部 NaN→T;[F,T,T];out dtype bool"))

    a = arr("f8", [np.nan, np.inf, -np.inf, 0.0, 5.0])
    with np.errstate(all="ignore"):
        out = np.isinf(a)
    entries.append(_emit_funcs("funcs_isinf_f8", "isinf", [("a", a)], out,
                               note="f8 [NaN,±Inf,0,5]→[F,T,T,F,F];out dtype bool"))

    a = arr("c16", [1 + 2j, np.inf + 0j])
    with np.errstate(all="ignore"):
        out = np.isinf(a)
    entries.append(_emit_funcs("funcs_isinf_c16", "isinf", [("a", a)], out,
                               note="c128 [有限,inf]→[F,T];out dtype bool"))

    a = arr("f8", [np.nan, np.inf, -np.inf, 0.0, 5.0])
    with np.errstate(all="ignore"):
        out = np.isfinite(a)
    entries.append(_emit_funcs("funcs_isfinite_f8", "isfinite", [("a", a)], out,
                               note="f8 [NaN,±Inf,0,5]→[F,F,F,T,T];out dtype bool"))

    x = arr("i4", [0, 1, 2])
    y = arr("i4", [1, 1, 0])
    with np.errstate(all="ignore"):
        out = np.logical_and(x, y)
    entries.append(_emit_funcs("funcs_logical_and_i4", "logical_and", [("x", x), ("y", y)], out,
                               note="i4×i4→bool [F,T,F];非零为真"))

    x = arr("b", [[True], [False]], shape=(2, 1))
    y = arr("b", [[True, False, True]], shape=(1, 3))
    with np.errstate(all="ignore"):
        out = np.logical_and(x, y)
    entries.append(_emit_funcs("funcs_logical_and_bcast", "logical_and", [("x", x), ("y", y)], out,
                               note="bool(2,1)×bool(1,3)→out (2,3) [[T,F,T],[F,F,F]];广播"))

    x = arr("b", [True, False, False])
    y = arr("b", [False, False, True])
    with np.errstate(all="ignore"):
        out = np.logical_or(x, y)
    entries.append(_emit_funcs("funcs_logical_or_b", "logical_or", [("x", x), ("y", y)], out,
                               note="bool×bool→[T,F,T];out dtype bool"))

    a = arr("f8", [0.0, 1.0, np.nan, -0.0])
    with np.errstate(all="ignore"):
        out = np.logical_not(a)
    entries.append(_emit_funcs("funcs_logical_not_f8", "logical_not", [("a", a)], out,
                               note="f8 [0,1,NaN,-0]→not [T,F,F,T];NaN 非零;out dtype bool"))

    x = arr("i4", [0, 1, 2])
    y = arr("f8", [1.0, 0.0, np.nan])
    with np.errstate(all="ignore"):
        out = np.logical_xor(x, y)
    entries.append(_emit_funcs("funcs_logical_xor_mix", "logical_xor", [("x", x), ("y", y)], out,
                               note="i4×f8 混型→bool [T,T,F];NaN 非零"))

    x = arr("f8", [1.0, np.nan, -0.0, 3.0])
    y = arr("f8", [2.0, 5.0, 0.0, -1.0])
    with np.errstate(all="ignore"):
        out = np.maximum(x, y)
    entries.append(_emit_funcs("funcs_maximum_f8", "maximum", [("x", x), ("y", y)], out,
                               note="f8:NaN 传播;max(-0.0,+0.0)=+0.0(相等取第二操作数);out [2,nan,+0,3] f64"))

    x = arr("f8", [1.0, np.nan, -0.0, 3.0])
    y = arr("f8", [2.0, 5.0, 0.0, -1.0])
    with np.errstate(all="ignore"):
        out = np.minimum(x, y)
    entries.append(_emit_funcs("funcs_minimum_f8", "minimum", [("x", x), ("y", y)], out,
                               note="f8:NaN 传播;min(-0.0,+0.0)=+0.0(相等取第二操作数);out [1,nan,+0,-1] f64"))

    x = arr("i4", [1, 5, -3])
    y = arr("f8", [2.0, 0.5, -9.0])
    with np.errstate(all="ignore"):
        out = np.maximum(x, y)
    entries.append(_emit_funcs("funcs_maximum_mix", "maximum", [("x", x), ("y", y)], out,
                               note="i4×f8→promote 到 f64 [2.0,5.0,-3.0]"))

 # —— Step 1C: deg2rad/rad2deg/fabs/sign/arctan2(5算子,12案;全部 region=bitwise)——
    a = arr("f8", [0, 90, 180, -45, 1e8, 0.1])
    with np.errstate(all="ignore"):
        out = np.deg2rad(a)
    entries.append(_emit_funcs("funcs_deg2rad_f8", "deg2rad", [("a", a)], out,
                               note="f8 6值;out dtype f64"))

    a = arr("f4", [0, 90, 180, -45, 1e8, 0.1])
    with np.errstate(all="ignore"):
        out = np.deg2rad(a)
    entries.append(_emit_funcs("funcs_deg2rad_f4", "deg2rad", [("a", a)], out,
                               note="f4 同值;out dtype f32"))

    a = arr("i4", [90, 180, -45])
    with np.errstate(all="ignore"):
        out = np.deg2rad(a)
    entries.append(_emit_funcs("funcs_deg2rad_i4", "deg2rad", [("a", a)], out,
                               note="i4→deg2rad:输出dtype由numpy决定=f64;3值"))

    a = arr("f8", [0, 1.5707963267948966, 3.141592653589793, -0.7853981633974483, 1e8])
    with np.errstate(all="ignore"):
        out = np.rad2deg(a)
    entries.append(_emit_funcs("funcs_rad2deg_f8", "rad2deg", [("a", a)], out,
                               note="f8 5值(含±π/4、π、1e8);out dtype f64"))

    a = arr("f8", [-3, 3, -0.0, np.nan, -np.inf])
    with np.errstate(all="ignore"):
        out = np.fabs(a)
    entries.append(_emit_funcs("funcs_fabs_f8", "fabs", [("a", a)], out,
                               note="f8 [±3,-0,nan,-inf];out dtype f64"))

    a = arr("i1", [-5, 0, 5])
    with np.errstate(all="ignore"):
        out = np.fabs(a)
    entries.append(_emit_funcs("funcs_fabs_i1", "fabs", [("a", a)], out,
                               note="i8→np.fabs:输出dtype由numpy决定=f16(最小可无损容纳整型浮点);bitwise"))

    a = arr("f8", [-2, 0, 3, -0.0, np.nan])
    with np.errstate(all="ignore"):
        out = np.sign(a)
    entries.append(_emit_funcs("funcs_sign_f8", "sign", [("a", a)], out,
                               note="f8 [-2,0,3,-0,nan]→[-1,0,1,0,nan];out dtype f64"))

    a = arr("i4", [-5, 0, 7])
    with np.errstate(all="ignore"):
        out = np.sign(a)
    entries.append(_emit_funcs("funcs_sign_i4", "sign", [("a", a)], out,
                               note="i4[-5,0,7]→[-1,0,1];out dtype i32(整型保dtype)"))

    a = arr("c16", [3 + 4j, 0 + 0j, -1 + 0j])
    with np.errstate(all="ignore"):
        out = np.sign(a)
    entries.append(_emit_funcs("funcs_sign_c16", "sign", [("a", a)], out,
                               note="c128:复数按x/|x|(3+4j→0.6+0.8j、0→0、-1→-1);out dtype c128"))

    y = arr("f8", [1, -1, 1, 0])
    x = arr("f8", [1, -1, -1, 0])
    with np.errstate(all="ignore"):
        out = np.arctan2(y, x)
    entries.append(_emit_funcs("funcs_arctan2_f8", "arctan2", [("y", y), ("x", x)], out,
                               note="f8 y×x 4组:π/4、-3π/4、3π/4、0;out dtype f64"))

    y = arr("i1", [1])
    x = arr("i1", [1])
    with np.errstate(all="ignore"):
        out = np.arctan2(y, x)
    entries.append(_emit_funcs("funcs_arctan2_i1", "arctan2", [("y", y), ("x", x)], out,
                               note="i8×i8→arctan2:输出dtype由numpy决定=f16;out [0.785] f16"))

    y = arr("f8", [[1], [-1]], shape=(2, 1))
    x = arr("f8", [[1, 0, -1]], shape=(1, 3))
    with np.errstate(all="ignore"):
        out = np.arctan2(y, x)
    entries.append(_emit_funcs("funcs_arctan2_bcast", "arctan2", [("y", y), ("x", x)], out,
                               note="f8 (2,1)×(1,3)→out (2,3);广播;out dtype f64"))

 # —— Step 2a: nansum/nanmax/nanmin/isclose/allclose(5算子,10案;全部 region=bitwise)——
    a = arr("f8", [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], shape=(2, 3))
    with np.errstate(all="ignore"):
        out = np.nansum(a)
    entries.append(_emit_funcs("funcs_nansum_f8", "nansum", [("a", a)], out,
                               note="f8(2,3)含NaN全归约:1+3+4+5=13;out 0-d f64"))

    with np.errstate(all="ignore"):
        out = np.nansum(a, axis=0)
    entries.append(_emit_funcs("funcs_nansum_axis0", "nansum", [("a", a)], out,
                               note="f8(2,3)含NaN轴0归约:[5,5,3];out (3,) f64", args={"axis": [0]}))

    b = arr("f8", [1.0, np.nan, 3.0, -2.0])
    with np.errstate(all="ignore"):
        out = np.nanmax(b)
    entries.append(_emit_funcs("funcs_nanmax_f8", "nanmax", [("a", b)], out,
                               note="f8 [1,nan,3,-2]跳过NaN取最大:3;out 0-d f64"))

    with np.errstate(all="ignore"):
        out = np.nanmax(a, axis=1)
    entries.append(_emit_funcs("funcs_nanmax_axis1", "nanmax", [("a", a)], out,
                               note="f8(2,3)含NaN轴1归约:[3,5];out (2,) f64", args={"axis": [1]}))

    with np.errstate(all="ignore"):
        out = np.nanmin(b)
    entries.append(_emit_funcs("funcs_nanmin_f8", "nanmin", [("a", b)], out,
                               note="f8 [1,nan,3,-2]跳过NaN取最小:-2;out 0-d f64"))

    d = arr("i4", [-5, 3, 9])
    with np.errstate(all="ignore"):
        out = np.nanmin(d)
    entries.append(_emit_funcs("funcs_nanmin_i4", "nanmin", [("a", d)], out,
                               note="i4整型委托:跳过NaN取最小:-5;out 0-d i32"))

    a = arr("f8", [1.0, 1.0, np.nan, np.inf, np.inf, 0.0])
    b = arr("f8", [1.0, 1.000009, np.nan, np.inf, 1.0, 1e-9])
    with np.errstate(all="ignore"):
        out = np.isclose(a, b, rtol=1e-5, atol=1e-8)
    entries.append(_emit_funcs("funcs_isclose_f8", "isclose", [("a", a), ("b", b)], out,
                               note="f8(6,)含NaN/±Inf:同号inf相等T、NaN≠NaN F;out (6,) bool",
                               args={"rtol": 1e-5, "atol": 1e-8}))

    a = arr("f8", [0.0, 0.0])
    b = arr("f8", [1e-9, 1e-6])
    with np.errstate(all="ignore"):
        out = np.isclose(a, b, rtol=0.0, atol=1e-8)
    entries.append(_emit_funcs("funcs_isclose_atol", "isclose", [("a", a), ("b", b)], out,
                               note="f8(2,)rtol=0仅atol:|0-1e-9|<=1e-8 T、|0-1e-6| F;out (2,) bool",
                               args={"rtol": 0.0, "atol": 1e-8}))

    a = arr("f8", [[1.0], [2.0]], shape=(2, 1))
    b = arr("f8", [[1.0, 2.0, 3.0]], shape=(1, 3))
    with np.errstate(all="ignore"):
        out = np.isclose(a, b, rtol=0.0, atol=0.5)
    entries.append(_emit_funcs("funcs_isclose_bcast", "isclose", [("a", a), ("b", b)], out,
                               note="f8(2,1)×(1,3)广播→out (2,3) bool [[T,F,F],[F,T,F]]",
                               args={"rtol": 0.0, "atol": 0.5}))

    a = arr("f8", [1.0, 2.0, 3.0])
    b = arr("f8", [1.0, 2.0000001, 3.0])
    with np.errstate(all="ignore"):
        out = np.allclose(a, b, rtol=1e-5, atol=1e-8)
    entries.append(_emit_funcs("funcs_allclose", "allclose", [("a", a), ("b", b)],
                               np.array(out),
                               note="f8(3,)全close→0-d bool True;np.allclose 返回标量,落盘 0-d bool",
                               args={"rtol": 1e-5, "atol": 1e-8}))

 # —— nanmean / nanvar / nanstd(归约,args.axis 同既有归约案)——
    a = arr("f8", [1.0, np.nan, 3.0, 5.0])
    with np.errstate(all="ignore"):
        out = np.nanmean(a)
    entries.append(_emit_funcs("funcs_nanmean_f8", "nanmean", [("a", a)], out,
                               note="f8 [1,nan,3,5] → 非 NaN 均值 3.0"))

    a2 = arr("f8", [[1.0, np.nan], [3.0, 5.0]])
    with np.errstate(all="ignore"):
        out = np.nanmean(a2, axis=1)
    entries.append(_emit_funcs("funcs_nanmean_axis1", "nanmean", [("a", a2)], out,
                               note="f8 (2,2) axis=1 → [1.0, 4.0]",
                               args={"axis": [1]}))
    with np.errstate(all="ignore"):
        out = np.nanvar(a2, axis=1)
    entries.append(_emit_funcs("funcs_nanvar_axis1", "nanvar", [("a", a2)], out,
                               note="f8 (2,2) axis=1 → [0.0, 1.0](ddof=0)",
                               args={"axis": [1]}))
    with np.errstate(all="ignore"):
        out = np.nanstd(a2, axis=1)
    entries.append(_emit_funcs("funcs_nanstd_axis1", "nanstd", [("a", a2)], out,
                               note="f8 (2,2) axis=1 → [0.0, 1.0](√var)",
                               args={"axis": [1]}))
    with np.errstate(all="ignore"):
        out = np.nanvar(a)
    entries.append(_emit_funcs("funcs_nanvar_f8", "nanvar", [("a", a)], out,
                               note="f8 [1,nan,3,5] → 8/3(ddof=0)"))

 # —— flip 族 / roll / repeat / tile / atleast_*(Step 3)——
    a1 = arr("f8", [1.0, 2.0, 3.0, 4.0])
    a2 = arr("f8", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    entries.append(_emit_funcs("funcs_flip_f8_1d", "flip", [("a", a1)], np.flip(a1),
                               note="1-D 全轴反转"))
    entries.append(_emit_funcs("funcs_flip_2d_ax0", "flip", [("a", a2)], np.flip(a2, axis=0),
                               note="2-D 轴 0 反转", args={"axis": [0]}))
    entries.append(_emit_funcs("funcs_flip_2d_all", "flip", [("a", a2)], np.flip(a2),
                               note="2-D 全轴反转"))
    entries.append(_emit_funcs("funcs_flipud", "flipud", [("a", a2)], np.flipud(a2),
                               note="flipud = flip axis 0"))
    entries.append(_emit_funcs("funcs_fliplr", "fliplr", [("a", a2)], np.fliplr(a2),
                               note="fliplr = flip axis 1"))
    entries.append(_emit_funcs("funcs_roll_flat", "roll", [("a", a1)], np.roll(a1, 1),
                               note="axis=None:先拉平再循环右移 1", args={"shift": 1}))
    entries.append(_emit_funcs("funcs_roll_ax0", "roll", [("a", a2)], np.roll(a2, 1, axis=0),
                               note="沿轴 0 循环右移 1", args={"shift": 1, "axis": [0]}))
    entries.append(_emit_funcs("funcs_roll_neg", "roll", [("a", a1)], np.roll(a1, -1),
                               note="负移位 = 左移", args={"shift": -1}))
    entries.append(_emit_shape("funcs_repeat_flat", "repeat", [("a", a1)], np.repeat(a1, 2),
                               note="axis=None:每元素重复 2 次", args={"repeats": 2}))
    entries.append(_emit_shape("funcs_repeat_ax0", "repeat", [("a", a2)], np.repeat(a2, 2, axis=0),
                               note="沿轴 0 每行重复 2 次", args={"repeats": 2, "axis": [0]}))
    entries.append(_emit_shape("funcs_tile_1d", "tile", [("a", a1)], np.tile(a1, 2),
                               note="1-D 平铺 ×2", args={"reps": [2]}))
    entries.append(_emit_shape("funcs_tile_2d", "tile", [("a", a2)], np.tile(a2, (2, 1)),
                               note="2-D 行方向 ×2、列不变", args={"reps": [2, 1]}))
    entries.append(_emit_shape("funcs_tile_pad", "tile", [("a", a2)], np.tile(a2, 1),
                               note="reps 短于 ndim 时前置补 1(等价不铺贴)", args={"reps": [1]}))
    entries.append(_emit_shape("funcs_atleast_1d_0d", "atleast_1d", [("a", arr("f8", 5.0))],
                               np.atleast_1d(arr("f8", 5.0)), note="0-d → (1,)"))
    entries.append(_emit_shape("funcs_atleast_2d_1d", "atleast_2d", [("a", a1)],
                               np.atleast_2d(a1), note="1-D → (1,n)"))
    entries.append(_emit_shape("funcs_atleast_3d_1d", "atleast_3d", [("a", a1)],
                               np.atleast_3d(a1), note="1-D → (1,n,1)"))
    entries.append(_emit_shape("funcs_atleast_3d_2d", "atleast_3d", [("a", a2)],
                               np.atleast_3d(a2), note="2-D → (m,n,1)"))
    entries.append(_emit_funcs("funcs_zeros_like", "zeros_like", [("a", a2)], np.zeros_like(a2),
                               note="同形状同 dtype 全零"))

 # —— 排序与统计(Step 6):sort/argsort/median/percentile/diff/unique ——
    a1 = arr("f8", [3.0, 1.0, 2.0])
    a2 = arr("f8", [[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]])
    entries.append(_emit_shape("funcs_sort_f8_1d", "sort", [("a", a1)], np.sort(a1),
                               note="1-D 升序;形状不变"))
    entries.append(_emit_shape("funcs_sort_axis1", "sort", [("a", a2)], np.sort(a2, axis=1),
                               note="2-D 沿轴 1 排序", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_sort_axis0", "sort", [("a", a2)], np.sort(a2, axis=0),
                               note="2-D 沿轴 0 排序", args={"axis": [0]}))
    nan1 = arr("f8", [2.0, np.nan, 1.0])
    entries.append(_emit_shape("funcs_sort_nan", "sort", [("a", nan1)], np.sort(nan1),
                               note="NaN 排最后(numpy 同)"))
    entries.append(_emit_shape("funcs_argsort_f8", "argsort", [("a", a1)], np.argsort(a1),
                               note="下标升序 [1,2,0]"))
    entries.append(_emit_shape("funcs_argsort_axis1", "argsort", [("a", a2)], np.argsort(a2, axis=1),
                               note="2-D 沿轴 1 的下标", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_median_f8", "median", [("a", a1)], np.median(a1),
                               note="奇数个 → 中间值 2.0"))
    a4 = arr("f8", [4.0, 1.0, 3.0, 2.0])
    entries.append(_emit_shape("funcs_median_even", "median", [("a", a4)], np.median(a4),
                               note="偶数个 → 中间两数均值 2.5"))
    entries.append(_emit_shape("funcs_median_axis1", "median", [("a", a2)], np.median(a2, axis=1),
                               note="2-D 沿轴 1 中位数", args={"axis": [1]}))
    p4 = arr("f8", [1.0, 2.0, 3.0, 4.0])
    entries.append(_emit_shape("funcs_percentile_q25", "percentile", [("a", p4)], np.percentile(p4, 25),
                               note="线性插值:1 + 0.75*(2-1) = 1.75", args={"q": 25.0}))
    entries.append(_emit_shape("funcs_percentile_axis1", "percentile", [("a", a2)],
                               np.percentile(a2, 75, axis=1),
                               note="2-D 沿轴 1 的 75 分位", args={"q": 75.0, "axis": [1]}))
    d4 = arr("f8", [1.0, 2.0, 4.0, 7.0])
    entries.append(_emit_shape("funcs_diff_1", "diff", [("a", d4)], np.diff(d4),
                               note="一阶差分 [1,2,3]", args={"n": 1}))
    entries.append(_emit_shape("funcs_diff_axis0", "diff", [("a", a2)], np.diff(a2, axis=0),
                               note="沿轴 0 一阶差分", args={"n": 1, "axis": [0]}))
    u1 = arr("f8", [3.0, 1.0, 3.0, 2.0, 1.0])
    entries.append(_emit_shape("funcs_unique_f8", "unique", [("a", u1)], np.unique(u1),
                               note="排序去重 → [1,2,3]"))
    ui = arr("i4", [5, 5, 2, 9, 2])
    entries.append(_emit_shape("funcs_unique_i4", "unique", [("a", ui)], np.unique(ui),
                               note="整型去重保持 dtype"))
 # —— 线代核心(Step 7):dot / matmul / inner / outer ——
    ma = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    mb = arr("f8", [[5.0, 6.0], [7.0, 8.0]])
    entries.append(_emit_shape("funcs_dot_2d", "dot", [("a", ma), ("b", mb)], np.dot(ma, mb),
                               note="2-D·2-D → (2,2)"))
    v1 = arr("f8", [1.0, 2.0, 3.0])
    v2 = arr("f8", [4.0, 5.0, 6.0])
    entries.append(_emit_shape("funcs_dot_1d", "dot", [("a", v1), ("b", v2)], np.dot(v1, v2),
                               note="1-D·1-D → 0-d 标量 32"))
    entries.append(_emit_shape("funcs_dot_2d1d", "dot", [("a", ma), ("b", arr("f8", [5.0, 6.0]))],
                               np.dot(ma, arr("f8", [5.0, 6.0])), note="2-D·1-D → (2,)"))
    entries.append(_emit_shape("funcs_dot_1d2d", "dot", [("a", arr("f8", [1.0, 2.0])), ("b", mb)],
                               np.dot(arr("f8", [1.0, 2.0]), mb), note="1-D·2-D → (2,)"))
    ia = arr("i4", [[1, 2], [3, 4]])
    ib = arr("i4", [[5, 6], [7, 8]])
    entries.append(_emit_shape("funcs_dot_int", "dot", [("a", ia), ("b", ib)], np.dot(ia, ib),
                               note="int32·int32 → int32(同宽累加)"))
    entries.append(_emit_shape("funcs_matmul_2d", "matmul", [("a", ma), ("b", mb)], np.matmul(ma, mb),
                               note="matmul 2-D·2-D"))
    entries.append(_emit_shape("funcs_matmul_1d1d", "matmul", [("a", v1), ("b", v2)], np.matmul(v1, v2),
                               note="matmul 1-D·1-D → 0-d"))
    entries.append(_emit_shape("funcs_matmul_2d1d", "matmul", [("a", ma), ("b", arr("f8", [5.0, 6.0]))],
                               np.matmul(ma, arr("f8", [5.0, 6.0])), note="matmul 2-D·1-D → (2,)"))
    st = arr("f8", [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    entries.append(_emit_shape("funcs_matmul_stack", "matmul", [("a", st), ("b", mb)], np.matmul(st, mb),
                               note="batch (2,2,2)·(2,2):b 广播到两个 batch"))
    ina = arr("f8", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    inb = arr("f8", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    entries.append(_emit_shape("funcs_inner_2d", "inner", [("a", ina), ("b", inb)], np.inner(ina, inb),
                               note="inner (2,3)·(2,3) → (2,2)"))
    entries.append(_emit_shape("funcs_outer", "outer", [("a", arr("f8", [1.0, 2.0])),
                                                        ("b", arr("f8", [3.0, 4.0, 5.0]))],
                               np.outer(arr("f8", [1.0, 2.0]), arr("f8", [3.0, 4.0, 5.0])),
                               note="outer:[1,2]×[3,4,5] → (2,3)"))
    entries.append(_emit_shape("funcs_outer_flat", "outer", [("a", ma), ("b", arr("f8", [1.0, 10.0]))],
                               np.outer(ma, arr("f8", [1.0, 10.0])),
                               note="二维输入先拉平 → (4,2)"))
 # —— 杂项算法(Step 8 第一批):searchsorted/digitize/interp/meshgrid/gradient ——
    sarr = arr("f8", [1.0, 3.0, 5.0, 7.0])
    sv = arr("f8", [0.0, 1.0, 3.0, 4.0, 7.0, 9.0])
    entries.append(_emit_shape("funcs_searchsorted_left", "searchsorted", [("a", sarr), ("b", sv)],
                               np.searchsorted(sarr, sv, side="left"),
                               note="lower bound:首个 >= v 的下标", args={"right": False}))
    entries.append(_emit_shape("funcs_searchsorted_right", "searchsorted", [("a", sarr), ("b", sv)],
                               np.searchsorted(sarr, sv, side="right"),
                               note="upper bound:首个 > v 的下标", args={"right": True}))
    bins = arr("f8", [0.0, 1.0, 2.0, 3.0])
    dx = arr("f8", [-1.0, 0.0, 0.5, 1.0, 2.5, 4.0])
    entries.append(_emit_shape("funcs_digitize", "digitize", [("a", dx), ("b", bins)],
                               np.digitize(dx, bins, right=False),
                               note="right=False:bins[i-1] <= x < bins[i]", args={"right": False}))
    entries.append(_emit_shape("funcs_digitize_right", "digitize", [("a", dx), ("b", bins)],
                               np.digitize(dx, bins, right=True),
                               note="right=True:边界归左箱", args={"right": True}))
    xp = arr("f8", [0.0, 1.0, 2.0])
    fp = arr("f8", [0.0, 10.0, 20.0])
    ix = arr("f8", [-1.0, 0.5, 1.5, 3.0])
    entries.append(_emit_shape("funcs_interp", "interp", [("a", ix), ("b", xp), ("c", fp)],
                               np.interp(ix, xp, fp),
                               note="线性插值 + 端点钳制:[-1→0, 0.5→5, 1.5→15, 3→20]"))
    xp2 = arr("f8", [0.0, 2.0, 3.0])
    fp2 = arr("f8", [0.0, 4.0, 9.0])
    entries.append(_emit_shape("funcs_interp_nonuniform", "interp",
                               [("a", arr("f8", [1.0, 2.5])), ("b", xp2), ("c", fp2)],
                               np.interp(arr("f8", [1.0, 2.5]), xp2, fp2),
                               note="非等距 xp:x=1→2, x=2.5→6.5"))
    g1 = arr("f8", [1.0, 2.0])
    g2 = arr("f8", [3.0, 4.0, 5.0])
    mg_ij = np.meshgrid(g1, g2, indexing="ij")
    entries.append(_emit_multi_funcs("funcs_meshgrid_ij", "meshgrid", [("a", g1), ("b", g2)],
                                     list(mg_ij), note="ij 索引序:输出形状 (2,3)"))
    mg_xy = np.meshgrid(g1, g2, indexing="xy")
    entries.append(_emit_multi_funcs("funcs_meshgrid_xy", "meshgrid", [("a", g1), ("b", g2)],
                                     list(mg_xy), note="xy 索引序:前两维交换 → (3,2)",
                                     args={"xy": True}))
    gf = arr("f8", [1.0, 2.0, 4.0, 7.0])
    entries.append(_emit_multi_funcs("funcs_gradient_1d", "gradient", [("a", gf)],
                                     [np.gradient(gf)], note="1-D:边界一阶、内部中心差分 [1,1.5,2.5,3]"))
    gm = arr("f8", [[1.0, 2.0], [4.0, 6.0]])
    entries.append(_emit_multi_funcs("funcs_gradient_2d", "gradient", [("a", gm)],
                                     list(np.gradient(gm)), note="2-D 无 axis:每轴一个数组,形状同输入"))
 # —— 杂项算法(Step 8b):pad / histogram / select / putmask ——
    p1 = arr("f8", [1.0, 2.0, 3.0])
    entries.append(_emit_shape("funcs_pad_constant", "pad", [("a", p1)],
                               np.pad(p1, (1, 1), mode="constant", constant_values=9.0),
                               note="constant:两端补 9", args={"pads": [[1, 1]], "mode": "constant", "cval": 9.0}))
    entries.append(_emit_shape("funcs_pad_edge", "pad", [("a", p1)],
                               np.pad(p1, (1, 2), mode="edge"),
                               note="edge:复制边缘", args={"pads": [[1, 2]], "mode": "edge"}))
    entries.append(_emit_shape("funcs_pad_reflect", "pad", [("a", p1)],
                               np.pad(p1, (2, 2), mode="reflect"),
                               note="reflect:不重复边缘", args={"pads": [[2, 2]], "mode": "reflect"}))
    entries.append(_emit_shape("funcs_pad_symmetric", "pad", [("a", p1)],
                               np.pad(p1, (2, 2), mode="symmetric"),
                               note="symmetric:重复边缘", args={"pads": [[2, 2]], "mode": "symmetric"}))
    entries.append(_emit_shape("funcs_pad_wrap", "pad", [("a", p1)],
                               np.pad(p1, (2, 2), mode="wrap"),
                               note="wrap:循环", args={"pads": [[2, 2]], "mode": "wrap"}))
    p2 = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    entries.append(_emit_shape("funcs_pad_2d", "pad", [("a", p2)],
                               np.pad(p2, ((1, 0), (0, 1)), mode="constant", constant_values=0.0),
                               note="2-D 每轴不同宽度", args={"pads": [[1, 0], [0, 1]], "mode": "constant"}))
    h1 = arr("f8", [0.0, 0.1, 0.2, 0.9, 1.0])
    hist, hedges = np.histogram(h1, bins=2, range=(0.0, 1.0))
    entries.append(_emit_multi_funcs("funcs_histogram_uniform", "histogram", [("a", h1)],
                                     [hist, hedges], note="等分 2 箱:边界归属(1.0 落末箱)",
                                     args={"bins": 2, "lo": 0.0, "hi": 1.0}))
    h2 = arr("f8", [0.0, 5.0, 10.0])
    hist2, hedges2 = np.histogram(h2, bins=2, range=(0.0, 10.0))
    entries.append(_emit_multi_funcs("funcs_histogram_range", "histogram", [("a", h2)],
                                     [hist2, hedges2], note="显式 range [0,10] 两箱",
                                     args={"bins": 2, "lo": 0.0, "hi": 10.0}))
    sc1 = arr("b", [True, False, False])
    sc2 = arr("b", [False, True, False])
    sch1 = arr("f8", [1.0, 1.0, 1.0])
    sch2 = arr("f8", [2.0, 2.0, 2.0])
    sdef = arr("i4", -1)
    entries.append(_emit_shape("funcs_select", "select",
                               [("c1", sc1), ("c2", sc2), ("h1", sch1), ("h2", sch2), ("d", sdef)],
                               np.select([sc1, sc2], [sch1, sch2], default=sdef),
                               note="首个为真胜出;default 参与 dtype 提升", args={"ncond": 2}))
    pm_a = arr("f8", [1.0, 2.0, 3.0, 4.0])
    pm_m = arr("b", [True, False, True, False])
    pm_v = arr("f8", [9.0, 8.0])
    expect_pm = pm_a.copy()
    np.putmask(expect_pm, pm_m, pm_v)
    entries.append(_emit_shape("funcs_putmask", "putmask", [("a", pm_a), ("m", pm_m), ("v", pm_v)],
                               expect_pm, note="原地覆盖,values 循环取用(0→9, 2→8)"))
    pm2_a = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    pm2_m = arr("b", [[False, True], [False, True]])
    pm2_v = arr("f8", 0.0)
    expect_pm2 = pm2_a.copy()
    np.putmask(expect_pm2, pm2_m, pm2_v)
    entries.append(_emit_shape("funcs_putmask_2d", "putmask",
                               [("a", pm2_a), ("m", pm2_m), ("v", pm2_v)],
                               expect_pm2, note="2-D 同形 mask"))
 # —— FFT 子集(Step 8d;region=rel,FFT 实现不同末位有差异)——
    fa = arr("f8", [1.0, 2.0, 3.0, 4.0])
    entries.append(_emit_rel("funcs_fft4", "fft", [("a", fa)], np.fft.fft(fa),
                             note="n=4 的 FFT(radix-2)", tol=1e-12))
    fn5 = arr("f8", [1.0, 2.0, 3.0, 4.0, 5.0])
    entries.append(_emit_rel("funcs_fft5", "fft", [("a", fn5)], np.fft.fft(fn5),
                             note="n=5 非 2 的幂(朴素 DFT)", tol=1e-12))
    fc = arr("c16", [1 + 1j, 2 - 1j])
    entries.append(_emit_rel("funcs_fft_complex", "fft", [("a", fc)], np.fft.fft(fc),
                             note="复数输入", tol=1e-12))
    entries.append(_emit_rel("funcs_ifft_rt", "ifft", [("a", np.fft.fft(fn5))], np.fft.ifft(np.fft.fft(fn5)),
                             note="ifft 往返还原(1/n 归一)", tol=1e-12))
    entries.append(_emit_rel("funcs_rfft4", "rfft", [("a", fa)], np.fft.rfft(fa),
                             note="半谱 n/2+1=3", tol=1e-12, out_shape=[3]))
    half = np.fft.rfft(fa)
    entries.append(_emit_rel("funcs_irfft4", "irfft", [("a", half)], np.fft.irfft(half, 4),
                             note="半谱还原实数序列", tol=1e-12, args={"n": 4}, out_shape=[4]))
    entries.append(_emit_rel("funcs_fftfreq8", "fftfreq", [("a", arr("f8", 0.0))],
                             np.fft.fftfreq(8, 1.0), note="频率分档(n=8, d=1)",
                             tol=1e-15, args={"n": 8, "d": 1.0}, out_shape=[8]))
    f2d = arr("f8", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    entries.append(_emit_rel("funcs_fft_2d", "fft", [("a", f2d)], np.fft.fft(f2d),
                             note="2-D 沿最后一轴", tol=1e-12))
 # —— 掩码数组(Step 5):filled / compressed ——
    mad = arr("f8", [1.0, 2.0, 3.0, 4.0])
    mam = arr("b", [False, True, False, True])
    _ma = np.ma.MaskedArray(mad, mask=mam)
    entries.append(_emit_shape("funcs_ma_filled", "ma_filled",
                               [("a", mad), ("m", mam), ("v", arr("f8", 0.0))],
                               _ma.filled(0.0), note="被掩码位置填 0"))
    entries.append(_emit_shape("funcs_ma_compressed", "ma_compressed",
                               [("a", mad), ("m", mam)],
                               _ma.compressed(), note="取出未掩码元素 (1-D)"))
    _ma2 = np.ma.MaskedArray(arr("f8", [[1.0, 2.0], [3.0, 4.0]]),
                             mask=arr("b", [[False, True], [False, False]]))
    entries.append(_emit_shape("funcs_ma_sum_axis0", "ma_sum",
                               [("a", _ma2.data), ("m", np.ma.getmaskarray(_ma2))],
                               np.ma.getdata(_ma2.sum(axis=0)),
                               note="按轴 0 求和,跳过掩码(第 1 列只有 4 有效)",
                               args={"axis": [0]}))

 # —— 线代长尾(Step 9):einsum / tensordot / unique 多输出 ——
    ea = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    eb = arr("f8", [[5.0, 6.0], [7.0, 8.0]])
    entries.append(_emit_shape("funcs_einsum_matmul", "einsum", [("a", ea), ("b", eb)],
                               np.einsum("ij,jk->ik", ea, eb),
                               note="ij,jk->ik 矩阵乘", args={"sub": "ij,jk->ik"}))
    entries.append(_emit_shape("funcs_einsum_transpose", "einsum", [("a", ea)],
                               np.einsum("ij->ji", ea), note="ij->ji 转置",
                               args={"sub": "ij->ji"}))
    entries.append(_emit_shape("funcs_einsum_diag", "einsum", [("a", ea)],
                               np.einsum("ii->i", ea), note="ii->i 取对角",
                               args={"sub": "ii->i"}))
    entries.append(_emit_shape("funcs_einsum_fullsum", "einsum", [("a", ea)],
                               np.einsum("ij->", ea), note="ij-> 全和 → 0-d",
                               args={"sub": "ij->"}))
    ev = arr("f8", [1.0, 2.0, 3.0])
    ew = arr("f8", [4.0, 5.0, 6.0])
    entries.append(_emit_shape("funcs_einsum_dot", "einsum", [("a", ev), ("b", ew)],
                               np.einsum("i,i->", ev, ew), note="i,i-> 内积 → 0-d",
                               args={"sub": "i,i->"}))
    entries.append(_emit_shape("funcs_tensordot_2d", "tensordot", [("a", ea), ("b", eb)],
                               np.tensordot(ea, eb, axes=([1], [0])),
                               note="2-D 收缩末轴与首轴", args={"axesA": [1], "axesB": [0]}))
    entries.append(_emit_shape("funcs_tensordot_1d", "tensordot", [("a", ev), ("b", ew)],
                               np.tensordot(ev, ew, axes=([0], [0])),
                               note="1-D 点积 → 0-d", args={"axesA": [0], "axesB": [0]}))
    e3 = arr("f8", [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    eid = arr("f8", [[1.0, 0.0], [0.0, 1.0]])
    entries.append(_emit_shape("funcs_tensordot_3d", "tensordot", [("a", e3), ("b", eid)],
                               np.tensordot(e3, eid, axes=([2], [0])),
                               note="3-D × 2-D 收缩一条轴", args={"axesA": [2], "axesB": [0]}))
    ux = arr("i4", [3, 1, 3, 2, 1])
    uv, ui, uc = np.unique(ux, return_index=True, return_counts=True)
    entries.append(_emit_multi_funcs("funcs_unique_idx", "unique_idx", [("a", ux)],
                                     [uv, ui, uc],
                                     note="values + 首次出现下标 + 计数(numpy 口径)"))

 # —— ufunc 协议层(Step 10):out= / where= / reduce / accumulate / outer ——
    pa = arr("f8", [1.0, 2.0, 3.0])
    pb = arr("f8", [10.0, 20.0, 30.0])
    pdst = arr("f8", [0.0, 0.0, 0.0])
    _o = pdst.copy(); np.add(pa, pb, out=_o)
    entries.append(_emit_shape("funcs_ufunc_out", "add_out", [("a", pa), ("b", pb), ("d", pdst)],
                               _o, note="out= 写回已有数组"))
    pcond = arr("b", [True, False, True])
    _w = arr("f8", [-1.0, -1.0, -1.0]); np.add(pa, pb, out=_w, where=pcond)
    entries.append(_emit_shape("funcs_ufunc_where", "add_where",
                               [("a", pa), ("b", pb), ("d", arr("f8", [-1.0, -1.0, -1.0])), ("c", pcond)],
                               _w, note="where= 只写条件为真的位置"))
    pr = arr("f8", [1.0, 2.0, 3.0, 4.0])
    entries.append(_emit_shape("funcs_ufunc_reduce", "add_reduce", [("a", pr)],
                               np.add.reduce(pr), note="add.reduce 全归约 → 0-d",
                               args={}))
    p2 = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    entries.append(_emit_shape("funcs_ufunc_reduce_axis", "add_reduce", [("a", p2)],
                               np.add.reduce(p2, axis=0), note="add.reduce axis=0",
                               args={"axis": [0]}))
    entries.append(_emit_shape("funcs_ufunc_reduce_mul", "mul_reduce", [("a", pr)],
                               np.multiply.reduce(pr), note="multiply.reduce 全归约"))
    entries.append(_emit_shape("funcs_ufunc_accumulate", "add_accumulate", [("a", pr)],
                               np.add.accumulate(pr), note="add.accumulate 前缀和"))
    entries.append(_emit_shape("funcs_ufunc_accumulate_axis", "add_accumulate", [("a", p2)],
                               np.add.accumulate(p2, axis=0), note="按列前缀和",
                               args={"axis": [0]}))
    po = arr("f8", [1.0, 2.0])
    entries.append(_emit_shape("funcs_ufunc_outer", "add_outer",
                               [("a", po), ("b", arr("f8", [3.0, 4.0]))],
                               np.add.outer(po, arr("f8", [3.0, 4.0])),
                               note="add.outer 逐对外积"))

 # —— 文本 IO / 小函数收尾(Step 11):place(与 putmask 的取值规则不同)——
    pla = arr("f8", [1.0, 2.0, 3.0, 4.0])
    plm = arr("b", [True, False, True, False])
    plv = arr("f8", [9.0, 8.0])
    expect_pl = pla.copy()
    np.place(expect_pl, plm, plv)
    entries.append(_emit_shape("funcs_place", "place", [("a", pla), ("m", plm), ("v", plv)],
                               expect_pl, note="place 按**命中顺序**取 vals(位置 0→9、位置 2→8)"))

 # —— 对角线/三角/迹 + 协方差族(Step 12)——
    d1 = arr("f8", [1.0, 2.0, 3.0])
    entries.append(_emit_shape("funcs_diag_1d", "diag", [("a", d1)], np.diag(d1),
                               note="1-D → 3x3 对角阵", args={"k": 0}))
    entries.append(_emit_shape("funcs_diag_1d_k1", "diag", [("a", arr("f8", [1.0, 2.0]))],
                               np.diag(arr("f8", [1.0, 2.0]), k=1),
                               note="k=1:上移一条对角线", args={"k": 1}))
    d3 = arr("f8", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    entries.append(_emit_shape("funcs_diag_2d", "diag", [("a", d3)], np.diag(d3),
                               note="2-D → 抽取主对角", args={"k": 0}))
    entries.append(_emit_shape("funcs_diag_2d_k-1", "diag", [("a", d3)], np.diag(d3, k=-1),
                               note="抽取 k=-1 对角", args={"k": -1}))
    entries.append(_emit_shape("funcs_diagflat", "diagflat", [("a", arr("f8", [[1.0, 2.0], [3.0, 4.0]]))],
                               np.diagflat(arr("f8", [[1.0, 2.0], [3.0, 4.0]])),
                               note="二维先拉平再造对角阵"))
    entries.append(_emit_shape("funcs_trace", "trace", [("a", d3)], np.trace(d3),
                               note="对角线之和 15", args={"offset": 0}))
    entries.append(_emit_shape("funcs_trace_offset", "trace", [("a", d3)], np.trace(d3, offset=1),
                               note="offset=1 → 8", args={"offset": 1}))
    entries.append(_emit_shape("funcs_triu", "triu", [("a", d3)], np.triu(d3),
                               note="上三角", args={"k": 0}))
    entries.append(_emit_shape("funcs_triu_k1", "triu", [("a", d3)], np.triu(d3, k=1),
                               note="上三角 k=1", args={"k": 1}))
    entries.append(_emit_shape("funcs_tril", "tril", [("a", d3)], np.tril(d3),
                               note="下三角", args={"k": 0}))
    entries.append(_emit_shape("funcs_tril_k-1", "tril", [("a", d3)], np.tril(d3, k=-1),
                               note="下三角 k=-1", args={"k": -1}))
    fd = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    expect_fd = fd.copy()
    np.fill_diagonal(expect_fd, 9.0)
    entries.append(_emit_shape("funcs_fill_diagonal", "fill_diagonal",
                               [("a", fd), ("v", arr("f8", 9.0))], expect_fd,
                               note="原地写对角线"))
    cx = arr("f8", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    entries.append(_emit_shape("funcs_cov", "cov", [("a", cx)], np.cov(cx),
                               note="2 变量 3 观测的协方差矩阵", args={"rowvar": True, "ddof": -1}))
    entries.append(_emit_shape("funcs_corrcoef", "corrcoef", [("a", cx)], np.corrcoef(cx),
                               note="相关系数矩阵", args={"rowvar": True}))
    cy = arr("f8", [1.0, 2.0, 3.0])
    entries.append(_emit_shape("funcs_cov_1d", "cov", [("a", cy)], np.cov(cy),
                               note="1-D → 0-d 方差", args={"rowvar": True, "ddof": -1}))

 # —— pad 模式扩展 + unique(axis)(Step 14)——
    pa1 = arr("f8", [1.0, 2.0, 3.0])
    entries.append(_emit_shape("funcs_pad_linear_ramp", "pad", [("a", pa1)],
                               np.pad(pa1, (2, 0), mode="linear_ramp"),
                               note="linear_ramp 前侧 2", args={"pads": [[2, 0]], "mode": "linear_ramp"}))
    entries.append(_emit_shape("funcs_pad_linear_ramp_end", "pad", [("a", pa1)],
                               np.pad(pa1, (0, 3), mode="linear_ramp", end_values=9.0),
                               note="linear_ramp 后侧 3,末端值 9",
                               args={"pads": [[0, 3]], "mode": "linear_ramp", "endAfter": 9.0}))
    entries.append(_emit_shape("funcs_pad_maximum", "pad", [("a", pa1)],
                               np.pad(pa1, (2, 1), mode="maximum"),
                               note="maximum 两侧都用该轴最大值 3",
                               args={"pads": [[2, 1]], "mode": "maximum"}))
    entries.append(_emit_shape("funcs_pad_minimum", "pad", [("a", pa1)],
                               np.pad(pa1, (2, 1), mode="minimum"),
                               note="minimum 两侧都用最小值 1",
                               args={"pads": [[2, 1]], "mode": "minimum"}))
    entries.append(_emit_shape("funcs_pad_mean", "pad", [("a", pa1)],
                               np.pad(pa1, (2, 1), mode="mean"),
                               note="mean 两侧都用均值 2",
                               args={"pads": [[2, 1]], "mode": "mean"}))
    entries.append(_emit_shape("funcs_pad_median", "pad", [("a", pa1)],
                               np.pad(pa1, (2, 1), mode="median"),
                               note="median 两侧都用中位数 2",
                               args={"pads": [[2, 1]], "mode": "median"}))
    pm2 = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    entries.append(_emit_shape("funcs_pad_max_2d", "pad", [("a", pm2)],
                               np.pad(pm2, ((1, 1), (0, 0)), mode="maximum"),
                               note="2-D 轴 0:每列取最大填充",
                               args={"pads": [[1, 1], [0, 0]], "mode": "maximum"}))
    entries.append(_emit_shape("funcs_pad_mean_2d", "pad", [("a", pm2)],
                               np.pad(pm2, ((0, 0), (1, 1)), mode="mean"),
                               note="2-D 轴 1:每行取均值填充",
                               args={"pads": [[0, 0], [1, 1]], "mode": "mean"}))
    upad = arr("i4", [1, 2, 3])
    entries.append(_emit_shape("funcs_pad_max_int", "pad", [("a", upad)],
                               np.pad(upad, (2, 1), mode="maximum"),
                               note="整型输入:输出仍为 int32",
                               args={"pads": [[2, 1]], "mode": "maximum"}))
    ua = arr("i4", [[1, 2], [1, 2], [3, 4], [1, 2]])
    entries.append(_emit_shape("funcs_unique_axis0", "unique_axis", [("a", ua)],
                               np.unique(ua, axis=0), note="按行去重(字典序)", args={"axis": [0]}))
    ub = arr("f8", [[1.0, 2.0], [1.0, 3.0]])
    entries.append(_emit_shape("funcs_unique_axis1", "unique_axis", [("a", ub)],
                               np.unique(ub, axis=1), note="按列去重(字典序)", args={"axis": [1]}))

 # —— 统计面补齐(Step 15):bincount(weights) / histogram2d ——
    bw_x = arr("i8", [1, 1, 2, 3])
    bw_w = arr("f8", [0.5, 0.25, 2.0, 1.0])
    entries.append(_emit_shape("funcs_bincount_weights", "bincount_w",
                               [("x", bw_x), ("w", bw_w)],
                               np.bincount(bw_x, weights=bw_w, minlength=5),
                               note="加权 bincount → float64 和", args={"n": 5}))
    h2x = arr("f8", [0.1, 0.5, 0.9, 0.5])
    h2y = arr("f8", [0.5, 1.5, 2.5, 0.5])
    H, xe, ye = np.histogram2d(h2x, h2y, bins=(2, 3), range=((0.0, 1.0), (0.0, 3.0)))
    entries.append(_emit_multi_funcs("funcs_histogram2d", "histogram2d",
                                     [("x", h2x), ("y", h2y)], [H, xe, ye],
                                     note="2x3 分箱 → (H, xedges, yedges)",
                                     args={"binsX": 2, "binsY": 3,
                                           "xlo": 0.0, "xhi": 1.0, "ylo": 0.0, "yhi": 3.0}))
    h2x2 = arr("f8", [0.5, 0.5, 0.9])
    h2y2 = arr("f8", [0.5, 0.9, 0.5])
    H2, xe2, ye2 = np.histogram2d(h2x2, h2y2, bins=(2, 2), range=((0.0, 1.0), (0.0, 1.0)))
    entries.append(_emit_multi_funcs("funcs_histogram2d_2", "histogram2d",
                                     [("x", h2x2), ("y", h2y2)], [H2, xe2, ye2],
                                     note="2x2 分箱,同箱多点",
                                     args={"binsX": 2, "binsY": 2,
                                           "xlo": 0.0, "xhi": 1.0, "ylo": 0.0, "yhi": 1.0}))

 # —— percentile 方法与 q 数组(Step 16)——
    pq5 = arr("f8", [1.0, 2.0, 3.0, 4.0, 5.0])
    entries.append(_emit_shape("funcs_pct_linear", "percentile_m", [("a", pq5)],
                               np.percentile(pq5, 25, method="linear"),
                               note="linear q=25", args={"q": 25.0, "method": "linear"}))
    entries.append(_emit_shape("funcs_pct_weibull", "percentile_m", [("a", pq5)],
                               np.percentile(pq5, 25, method="weibull"),
                               note="weibull q=25 → 1.5", args={"q": 25.0, "method": "weibull"}))
    entries.append(_emit_shape("funcs_pct_hazen", "percentile_m", [("a", pq5)],
                               np.percentile(pq5, 25, method="hazen"),
                               note="hazen q=25 → 1.75", args={"q": 25.0, "method": "hazen"}))
    pq6 = arr("f8", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    entries.append(_emit_shape("funcs_pct_nearest", "percentile_m", [("a", pq6)],
                               np.percentile(pq6, 50, method="nearest"),
                               note="nearest q=50(n=6)→ 3(银行家舍入)",
                               args={"q": 50.0, "method": "nearest"}))
    entries.append(_emit_shape("funcs_pct_closest", "percentile_m", [("a", pq5)],
                               np.percentile(pq5, 50, method="closest_observation"),
                               note="closest_observation q=50 → 2",
                               args={"q": 50.0, "method": "closest_observation"}))
    entries.append(_emit_shape("funcs_pct_many", "percentile_many", [("a", pq5)],
                               np.percentile(pq5, [25, 50, 75]),
                               note="q 数组 → (3,)", args={"qs": [25.0, 50.0, 75.0]}))
    pm23 = arr("f8", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    entries.append(_emit_shape("funcs_pct_many_axis", "percentile_many", [("a", pm23)],
                               np.percentile(pm23, [25, 75], axis=1),
                               note="q 数组 + 轴归约 → (2,2)",
                               args={"qs": [25.0, 75.0], "axis": [1]}))

 # —— quantile 族(Step 17)——
    qa = arr("f8", [1.0, 2.0, 3.0, 4.0, 5.0])
    entries.append(_emit_shape("funcs_quantile", "quantile", [("a", qa)],
                               np.quantile(qa, 0.25), note="quantile 0.25 → 2",
                               args={"q": 0.25}))
    entries.append(_emit_shape("funcs_quantile_method", "quantile", [("a", qa)],
                               np.quantile(qa, 0.25, method="weibull"),
                               note="quantile weibull 0.25 → 1.5",
                               args={"q": 0.25, "method": "weibull"}))
    entries.append(_emit_shape("funcs_quantile_many", "quantile_many", [("a", qa)],
                               np.quantile(qa, [0.25, 0.5]), note="q 数组 → [2,3]",
                               args={"qs": [0.25, 0.5]}))
    qn = arr("f8", [1.0, 2.0, np.nan, 4.0, 5.0])
    with np.errstate(all="ignore"):
        np_out = np.nanpercentile(qn, 50)
    entries.append(_emit_shape("funcs_nanpercentile", "nanpercentile", [("a", qn)],
                               np_out, note="剔除 NaN 后中位数 3", args={"q": 50.0}))
    qm = arr("f8", [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    with np.errstate(all="ignore"):
        np_out2 = np.nanpercentile(qm, 50, axis=1)
    entries.append(_emit_shape("funcs_nanpercentile_axis", "nanpercentile", [("a", qm)],
                               np_out2, note="按轴剔除 NaN → [2,5]",
                               args={"q": 50.0, "axis": [1]}))
    entries.append(_emit_shape("funcs_nanquantile", "nanquantile", [("a", qn)],
                               np.nanquantile(qn, 0.75), note="剔除 NaN 后 0.75 → 4.25",
                               args={"q": 0.75}))

 # —— NaN 族补齐(Step 18):nanmedian / nanargmin / nanargmax ——
    na = arr("f8", [3.0, np.nan, 1.0, 2.0, np.nan])
    with np.errstate(all="ignore"):
        nm = np.nanmedian(na)
    entries.append(_emit_shape("funcs_nanmedian", "nanmedian", [("a", na)], nm,
                               note="剔除 NaN 后中位数 2"))
    nm2 = arr("f8", [[3.0, np.nan, 1.0], [5.0, 2.0, 4.0]])
    with np.errstate(all="ignore"):
        nmo = np.nanmedian(nm2, axis=1)
    entries.append(_emit_shape("funcs_nanmedian_axis", "nanmedian", [("a", nm2)], nmo,
                               note="按轴剔除 NaN → [2,4]", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_nanargmin", "nanargmin", [("a", na)], np.nanargmin(na),
                               note="跳过 NaN 的最小下标 2"))
    entries.append(_emit_shape("funcs_nanargmax", "nanargmax", [("a", na)], np.nanargmax(na),
                               note="跳过 NaN 的最大下标 0"))
    entries.append(_emit_shape("funcs_nanargmin_axis", "nanargmin", [("a", nm2)],
                               np.nanargmin(nm2, axis=1),
                               note="按轴的最小下标 → [2,1]", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_nanargmax_axis0", "nanargmax", [("a", nm2)],
                               np.nanargmax(nm2, axis=0),
                               note="按轴 0 的最大下标 → [1,1,1]", args={"axis": [0]}))

 # —— pad stat_length + cumsum/cumprod(Step 19)——
    sa = arr("f8", [1.0, 5.0, 3.0, 9.0, 2.0])
    entries.append(_emit_shape("funcs_pad_stat_len1", "pad", [("a", sa)],
                               np.pad(sa, (2, 2), mode="maximum", stat_length=1),
                               note="stat_length=1:前侧 max(1)、后侧 max(2)",
                               args={"pads": [[2, 2]], "mode": "maximum", "statLength": 1}))
    entries.append(_emit_shape("funcs_pad_stat_len2_mean", "pad", [("a", sa)],
                               np.pad(sa, (2, 2), mode="mean", stat_length=2),
                               note="mean stat_length=2:前 3、后 5.5",
                               args={"pads": [[2, 2]], "mode": "mean", "statLength": 2}))
    sm = arr("f8", [[1.0, 5.0], [3.0, 9.0]])
    entries.append(_emit_shape("funcs_pad_stat_len_2d", "pad", [("a", sm)],
                               np.pad(sm, ((1, 1), (0, 0)), mode="maximum", stat_length=1),
                               note="2-D 按列取边缘 1 个",
                               args={"pads": [[1, 1], [0, 0]], "mode": "maximum", "statLength": 1}))
    cm = arr("i4", [[1, 2], [3, 4]])
    entries.append(_emit_shape("funcs_cumsum_default", "cumsum", [("a", cm)],
                               np.cumsum(cm), note="默认展平 + dtype 提升到 int64"))
    entries.append(_emit_shape("funcs_cumsum_axis0", "cumsum", [("a", cm)],
                               np.cumsum(cm, axis=0), note="按轴 0 前缀和", args={"axis": [0]}))
    entries.append(_emit_shape("funcs_cumsum_axis1", "cumsum", [("a", cm)],
                               np.cumsum(cm, axis=1), note="按轴 1 前缀和", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_cumprod_default", "cumprod", [("a", cm)],
                               np.cumprod(cm), note="默认展平前缀积"))
    cf = arr("f4", [1.5, 2.5, 3.5])
    entries.append(_emit_shape("funcs_cumsum_f32", "cumsum", [("a", cf)],
                               np.cumsum(cf), note="f32 保持 f32"))

 # —— 掩码数组长尾(Step 20):masked_invalid / masked_where / Median·Var·Std ——
    mi_a = arr("f8", [1.0, 2.0, np.nan, np.inf, 5.0])
    entries.append(_emit_shape("funcs_ma_masked_invalid", "ma_masked_invalid", [("a", mi_a)],
                               np.ma.getmaskarray(np.ma.masked_invalid(mi_a)),
                               note="NaN/Inf 被标为无效 → 掩码数组"))
    mw_c = arr("b", [False, True, False, True])
    mw_a = arr("f8", [1.0, 2.0, 3.0, 4.0])
    entries.append(_emit_shape("funcs_ma_masked_where", "ma_masked_where",
                               [("c", mw_c), ("a", mw_a)],
                               np.ma.getmaskarray(np.ma.masked_where(mw_c, mw_a)),
                               note="条件为真处标为无效"))
    md_a = arr("f8", [5.0, 1.0, 9.0, 3.0])
    md_m = arr("b", [False, False, True, False])
    _mma = np.ma.MaskedArray(md_a, mask=md_m)
    entries.append(_emit_shape("funcs_ma_median", "ma_median", [("a", md_a), ("m", md_m)],
                               np.ma.getdata(np.ma.median(_mma)), note="跳过掩码的中位数 3"))
    entries.append(_emit_shape("funcs_ma_var", "ma_var", [("a", md_a), ("m", md_m)],
                               np.ma.getdata(np.ma.var(_mma)), note="跳过掩码的方差 8/3"))
    entries.append(_emit_shape("funcs_ma_std", "ma_std", [("a", md_a), ("m", md_m)],
                               np.ma.getdata(np.ma.std(_mma)), note="跳过掩码的标准差 sqrt(8/3)"))
    m2_a = arr("f8", [[1.0, 5.0], [3.0, 9.0]])
    m2_m = arr("b", [[False, True], [False, False]])
    _mm2 = np.ma.MaskedArray(m2_a, mask=m2_m)
    entries.append(_emit_shape("funcs_ma_median_axis", "ma_median",
                               [("a", m2_a), ("m", m2_m)],
                               np.ma.getdata(np.ma.median(_mm2, axis=1)),
                               note="按轴中位数 → [1,6]", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_ma_std_axis0", "ma_std",
                               [("a", m2_a), ("m", m2_m)],
                               np.ma.getdata(np.ma.std(_mm2, axis=0)),
                               note="按轴 0 标准差 → [1,0]", args={"axis": [0]}))

 # —— 掩码谓词构造器(Step 21)——
    pv = arr("f8", [1.0, 2.0, 3.0, 4.0])
    _preds = [
        ("equal", lambda: np.ma.masked_equal(pv, 2.0), {"mode": "equal", "v1": 2.0}),
        ("not_equal", lambda: np.ma.masked_not_equal(pv, 2.0), {"mode": "not_equal", "v1": 2.0}),
        ("less", lambda: np.ma.masked_less(pv, 3.0), {"mode": "less", "v1": 3.0}),
        ("greater", lambda: np.ma.masked_greater(pv, 3.0), {"mode": "greater", "v1": 3.0}),
        ("less_equal", lambda: np.ma.masked_less_equal(pv, 3.0), {"mode": "less_equal", "v1": 3.0}),
        ("greater_equal", lambda: np.ma.masked_greater_equal(pv, 3.0),
         {"mode": "greater_equal", "v1": 3.0}),
        ("inside", lambda: np.ma.masked_inside(pv, 2.0, 3.0),
         {"mode": "inside", "v1": 2.0, "v2": 3.0}),
        ("outside", lambda: np.ma.masked_outside(pv, 2.0, 3.0),
         {"mode": "outside", "v1": 2.0, "v2": 3.0}),
    ]
    for _name, _fn, _args in _preds:
        entries.append(_emit_shape("funcs_ma_pred_" + _name, "ma_pred", [("a", pv)],
                                   np.ma.getmaskarray(_fn()),
                                   note="masked_" + _name + " 的掩码", args=_args))

 # —— FFT Bluestein 路径(Step 23):非 2 的幂大长度 ——
    fb1000 = arr("f8", [np.sin(i * 0.37) + (i % 11) * 0.25 for i in range(1000)])
    entries.append(_emit_rel("funcs_fft_bluestein_1000", "fft", [("a", fb1000)],
                             np.fft.fft(fb1000),
                             note="n=1000 非 2 的幂(Bluestein)", tol=1e-10))
    fb999 = arr("f8", [np.cos(i * 0.21) + (i % 7) * 0.5 for i in range(999)])
    entries.append(_emit_rel("funcs_fft_bluestein_999", "fft", [("a", fb999)],
                             np.fft.fft(fb999),
                             note="n=999(质因数分解含 3*3*3*37)", tol=1e-10))
    _rf = np.fft.rfft(fb1000)
    entries.append(_emit_rel("funcs_rfft_bluestein_1000", "rfft", [("a", fb1000)],
                             _rf, note="实数半谱 n=1000(Bluestein)", tol=1e-10,
                             out_shape=np.shape(_rf)))

 # —— 拼装/切分族(Step 24)——
    sp6 = arr("f8", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    entries.append(_emit_multi_funcs("funcs_array_split", "array_split", [("a", sp6)],
                                     list(np.array_split(sp6, 4)),
                                     note="不等分切 4 段(前 2 段多一个)",
                                     args={"sections": 4, "axis": [0]}))
    entries.append(_emit_multi_funcs("funcs_split_at", "split_at", [("a", sp6)],
                                     list(np.split(sp6, [2, 5])),
                                     note="按索引 [2,5] 切分",
                                     args={"splitIdx": [2, 5], "axis": [0]}))
    spm = arr("f8", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    entries.append(_emit_multi_funcs("funcs_hsplit", "hsplit", [("a", spm)],
                                     list(np.hsplit(spm, 3)), note="2-D 沿轴 1 切 3 段",
                                     args={"sections": 3}))
    entries.append(_emit_multi_funcs("funcs_vsplit", "vsplit", [("a", spm)],
                                     list(np.vsplit(spm, 2)), note="沿轴 0 切 2 段",
                                     args={"sections": 2}))
    apv = arr("f8", [[9.0, 9.0, 9.0]])
    entries.append(_emit_shape("funcs_append_axis0", "append", [("a", spm), ("v", apv)],
                               np.append(spm, apv, axis=0), note="沿轴 0 追加一行",
                               args={"axis": [0]}))
    entries.append(_emit_shape("funcs_append_flat", "append", [("a", spm),
                               ("v", arr("f8", [9.0, 9.0]))],
                               np.append(spm, arr("f8", [9.0, 9.0])),
                               note="axis=None 展平追加"))
    entries.append(_emit_shape("funcs_insert_flat", "insert", [("a", sp6),
                               ("v", arr("f8", 99.0))],
                               np.insert(sp6, 2, 99.0), note="展平插入(axis=None)",
                               args={"index": 2}))
    entries.append(_emit_shape("funcs_insert_axis0", "insert", [("a", spm),
                               ("v", arr("f8", [7.0, 8.0, 9.0]))],
                               np.insert(spm, 1, arr("f8", [7.0, 8.0, 9.0]), axis=0),
                               note="沿轴 0 插入一行", args={"index": 1, "axis": [0]}))
    entries.append(_emit_shape("funcs_delete_neg", "delete", [("a", sp6)],
                               np.delete(sp6, -1), note="展平删除最后一个",
                               args={"index": -1}))
    entries.append(_emit_shape("funcs_delete_axis0", "delete", [("a", spm)],
                               np.delete(spm, 0, axis=0), note="沿轴 0 删除第一行",
                               args={"index": 0, "axis": [0]}))
    cs1, cs2 = arr("f8", [1.0, 2.0]), arr("f8", [3.0, 4.0])
    entries.append(_emit_shape("funcs_column_stack", "column_stack", [("a", cs1), ("b", cs2)],
                               np.column_stack([cs1, cs2]), note="1-D 变列 → (2,2)"))
    entries.append(_emit_shape("funcs_dstack", "dstack", [("a", cs1), ("b", cs2)],
                               np.dstack([cs1, cs2]), note="dstack → (1,2,2)"))
    pda = arr("f8", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pdi = arr("i8", [[0], [2]])
    pdv = arr("f8", [[5.0], [6.0]])
    _expect_pda = pda.copy()
    np.put_along_axis(_expect_pda, pdi, pdv, axis=1)
    entries.append(_emit_shape("funcs_put_along_axis", "put_along_axis",
                               [("a", pda), ("i", pdi), ("v", pdv)], _expect_pda,
                               note="按 idx 沿轴写入", args={"axis": [1]}))
    ctsrc = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    ctdst = arr("f8", [[0.0, 0.0], [0.0, 0.0]])
    entries.append(_emit_shape("funcs_copyto", "copyto", [("d", ctdst), ("s", ctsrc)],
                               ctsrc.copy(), note="copyto 原地覆盖"))

 # —— 线代核心 + 常用别名(Step 25)——
    la = arr("f8", [[4.0, 3.0, 2.0], [1.0, 5.0, 7.0], [2.0, 8.0, 6.0]])
    entries.append(_emit_rel("funcs_det", "det", [("a", la)], np.asarray(np.linalg.det(la)),
                              note="3x3 行列式 -84(LU 实现末位可能有差异)", tol=1e-12))
    entries.append(_emit_rel("funcs_inv", "inv", [("a", la)], np.linalg.inv(la),
                              note="3x3 求逆(LU 实现末位可能有差异)", tol=1e-12))
    lb = arr("f8", [1.0, 2.0, 3.0])
    entries.append(_emit_rel("funcs_solve", "solve", [("a", la), ("b", lb)],
                              np.linalg.solve(la, lb), note="解 Ax=b(末位可能有差异)", tol=1e-12))
    entries.append(_emit_shape("funcs_norm_fro", "norm", [("a", la)],
                               np.linalg.norm(la, "fro"), note="矩阵 Frobenius 范数",
                               args={"mode": "fro"}))
    lv = arr("f8", [3.0, -4.0, 0.0])
    entries.append(_emit_shape("funcs_norm_vec2", "norm", [("a", lv)],
                               np.linalg.norm(lv, 2), note="向量 2-范数 5",
                               args={"mode": "2"}))
    pi32 = arr("i4", [1, 2, 3, 4])
    entries.append(_emit_shape("funcs_prod", "prod", [("a", pi32)], np.prod(pi32),
                               note="连乘 + 整型提升到 int64"))
    entries.append(_emit_shape("funcs_around", "around", [("a", arr("f8", [1.234, 2.789]))],
                               np.around(arr("f8", [1.234, 2.789]), 1), note="保留 1 位",
                               args={"decimals": 1}))
    fz = arr("f8", [[1.0, 0.0], [0.0, 2.0]])
    entries.append(_emit_shape("funcs_flatnonzero", "flatnonzero", [("a", fz)],
                               np.flatnonzero(fz), note="非零元素的平坦下标 [0,3]"))
    entries.append(_emit_shape("funcs_argwhere", "argwhere", [("a", fz)],
                               np.argwhere(fz), note="非零位置 (2,2)"))
    cz = arr("c16", [1 + 0j, 1 + 1j])
    entries.append(_emit_shape("funcs_isreal", "isreal", [("a", cz)], np.isreal(cz),
                               note="复数虚部为 0 才算实数"))
    sx = arr("f8", [0.0, 1.0, 0.5, -1.0])
    entries.append(_emit_rel("funcs_sinc", "sinc", [("a", sx)], np.sinc(sx),
                              note="np.sinc(π 的乘法顺序不同 → 末位差异)", tol=1e-12))
    tx = arr("f8", [0.0, 1.0, -2.0])
    entries.append(_emit_shape("funcs_tanh", "tanh", [("a", tx)], np.tanh(tx), note="双曲正切"))
    ca = arr("f8", [1.0, 2.0, 3.0, 4.0])
    cv = arr("f8", [1.0, 10.0])
    entries.append(_emit_shape("funcs_convolve_full", "convolve", [("a", ca), ("v", cv)],
                               np.convolve(ca, cv, "full"), note="卷积 full 模式",
                               args={"mode": "full"}))
    entries.append(_emit_shape("funcs_convolve_same", "convolve", [("a", ca), ("v", cv)],
                               np.convolve(ca, cv, "same"), note="卷积 same 模式",
                               args={"mode": "same"}))
    entries.append(_emit_shape("funcs_correlate_full", "correlate", [("a", ca), ("v", cv)],
                               np.correlate(ca, cv, "full"), note="互相关 full 模式",
                               args={"mode": "full"}))
    tm = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    entries.append(_emit_shape("funcs_transpose_pkg", "transpose_pkg", [("a", tm)],
                               np.transpose(tm), note="包级 transpose 便捷式"))
    entries.append(_emit_shape("funcs_ravel", "ravel_pkg", [("a", tm)], np.ravel(tm),
                               note="展平"))

 # —— 集合运算 + trapz/polyval(Step 26)——
    sa1 = arr("i4", [3, 1, 2, 5])
    sb1 = arr("i4", [2, 3, 7])
    entries.append(_emit_shape("funcs_isin", "isin", [("e", sa1), ("t", sb1)],
                               np.isin(sa1, sb1), note="逐元素成员判断"))
    sm2 = arr("i4", [[1, 2], [3, 4]])
    entries.append(_emit_shape("funcs_isin_2d", "isin", [("e", sm2), ("t", sb1)],
                               np.isin(sm2, sb1), note="2-D 逐元素判断(形状保持)"))
    entries.append(_emit_shape("funcs_intersect1d", "intersect1d", [("a", sa1), ("b", sb1)],
                               np.intersect1d(sa1, sb1), note="交集排序去重 [2,3]"))
    entries.append(_emit_shape("funcs_union1d", "union1d", [("a", sa1), ("b", sb1)],
                               np.union1d(sa1, sb1), note="并集排序去重"))
    entries.append(_emit_shape("funcs_setdiff1d", "setdiff1d", [("a", sa1), ("b", sb1)],
                               np.setdiff1d(sa1, sb1), note="差集 [1,5]"))
    entries.append(_emit_shape("funcs_setxor1d", "setxor1d", [("a", sa1), ("b", sb1)],
                               np.setxor1d(sa1, sb1), note="对称差 [1,5,7]"))
    ty = arr("f8", [1.0, 2.0, 3.0, 4.0])
    entries.append(_emit_rel("funcs_trapz", "trapz", [("y", ty)],
                             np.asarray(np.trapezoid(ty)), note="梯形积分 7.5", tol=1e-12))
    entries.append(_emit_rel("funcs_trapz_dx", "trapz", [("y", ty)],
                             np.asarray(np.trapezoid(ty, dx=2.0)), note="dx=2 → 15",
                             args={"d": 2.0}, tol=1e-12))
    tx = arr("f8", [0.0, 1.0, 3.0, 6.0])
    entries.append(_emit_rel("funcs_trapz_x", "trapz", [("y", ty), ("x", tx)],
                             np.asarray(np.trapezoid(ty, x=tx)), note="非等距 → 17",
                             tol=1e-12))
    tp = arr("f8", [2.0, -1.0, 3.0])
    txv = arr("f8", [0.0, 1.0, 2.0])
    entries.append(_emit_shape("funcs_polyval", "polyval", [("p", tp), ("x", txv)],
                               np.polyval(tp, txv), note="2x²-x+3 → [3,4,9]"))

 # —— 特征值/奇异值(Step 27)——
    ea = arr("f8", [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
    entries.append(_emit_rel("funcs_eigvalsh", "eigvalsh", [("a", ea)],
                             np.linalg.eigvalsh(ea),
                             note="对称矩阵特征值(升序)", tol=1e-10))
    entries.append(_emit_rel("funcs_svdvals", "svdvals",
                             [("a", arr("f8", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))],
                             np.linalg.svd(arr("f8", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                                           compute_uv=False),
                             note="奇异值(降序)", tol=1e-10))
    sa = arr("f8", [[4.0, 3.0, 2.0], [1.0, 5.0, 7.0], [2.0, 8.0, 6.0]])
    entries.append(_emit_rel("funcs_norm_mat2", "norm", [("a", sa)],
                             np.asarray(np.linalg.norm(sa, 2)),
                             note="矩阵 2-范数(最大奇异值)", args={"mode": "2"}, tol=1e-10))
    entries.append(_emit_rel("funcs_norm_mat_neg2", "norm", [("a", sa)],
                             np.asarray(np.linalg.norm(sa, -2)),
                             note="矩阵 -2 范数(最小奇异值)", args={"mode": "-2"}, tol=1e-10))

 # —— polyfit / unwrap(Step 28)——
    fx = arr("f8", [0.0, 1.0, 2.0, 3.0])
    fy = arr("f8", [1.0, 3.0, 7.0, 13.0])
    _pf2 = np.polyfit(fx, fy, 2)
    entries.append(_emit_rel("funcs_polyfit2", "polyfit", [("x", fx), ("y", fy)],
                             _pf2, note="二次拟合 y=x²+x+1", args={"deg": 2},
                             tol=1e-10, out_shape=np.shape(_pf2)))
 # 一次拟合用截距非零的数据:否则 numpy 的截距是 ~1e-15 的噪声,
 # 相对容差对近零值没有意义。
    fy1 = arr("f8", [2.0, 5.0, 8.0, 11.0])
    _pf1 = np.polyfit(fx, fy1, 1)
    entries.append(_emit_rel("funcs_polyfit1", "polyfit", [("x", fx), ("y", fy1)],
                             _pf1, note="一次拟合 y=3x+2", args={"deg": 1},
                             tol=1e-10, out_shape=np.shape(_pf1)))
    up = arr("f8", [0.0, 3.5, 7.0, 1.5, 5.0])
    entries.append(_emit_rel("funcs_unwrap", "unwrap", [("p", up)], np.unwrap(up),
                             note="默认 period=2π", tol=1e-12))
    entries.append(_emit_rel("funcs_unwrap_period4", "unwrap", [("p", up)],
                             np.unwrap(up, period=4.0), note="period=4", args={"d": 4.0},
                             tol=1e-12))
    up2 = arr("f8", [[0.0, 3.5, 7.0], [1.5, 5.0, 8.5]])
    entries.append(_emit_rel("funcs_unwrap_2d", "unwrap", [("p", up2)], np.unwrap(up2),
                             note="2-D 沿最后一轴,行间独立", tol=1e-12))

 # —— ufunc 补充批(Step 31)——
    ua = arr("f8", [1.0, -2.0, 0.0])
    entries.append(_emit_shape("funcs_negative", "negative", [("a", ua)],
                               np.negative(ua), note="取负(注意 -0.0 逐位保留)"))
    entries.append(_emit_shape("funcs_square", "square",
                               [("a", arr("i4", [2, 3, -4]))],
                               np.square(arr("i4", [2, 3, -4])),
                               note="平方,保持 int32"))
    entries.append(_emit_shape("funcs_reciprocal", "reciprocal",
                               [("a", arr("f8", [2.0, 4.0, 0.5]))],
                               np.reciprocal(arr("f8", [2.0, 4.0, 0.5])), note="1/x"))
    entries.append(_emit_shape("funcs_log2", "log2", [("a", arr("f8", [1.0, 2.0, 8.0]))],
                               np.log2(arr("f8", [1.0, 2.0, 8.0])), note="以 2 为底对数"))
    entries.append(_emit_shape("funcs_cbrt", "cbrt", [("a", arr("f8", [8.0, -27.0]))],
                               np.cbrt(arr("f8", [8.0, -27.0])),
                               note="立方根(保留负号,与 pow 不同)"))
    entries.append(_emit_shape("funcs_arcsinh", "arcsinh", [("a", arr("f8", [0.0, 1.0]))],
                               np.arcsinh(arr("f8", [0.0, 1.0])), note="反双曲正弦"))
    entries.append(_emit_shape("funcs_expm1_log1p", "expm1", [("a", arr("f8", [0.0, 1.0]))],
                               np.expm1(arr("f8", [0.0, 1.0])), note="exp(x)-1(小 x 更精确)"))
    cz1 = arr("c16", [1 + 2j, -3 + 0.5j])
    entries.append(_emit_shape("funcs_real", "real", [("a", cz1)], np.real(cz1),
                               note="复数取实部"))
    entries.append(_emit_shape("funcs_imag", "imag", [("a", cz1)], np.imag(cz1),
                               note="复数取虚部"))

 # —— 收尾 4 个:absolute / any / invert / cross(Step 33)——
    entries.append(_emit_shape("funcs_absolute", "absolute",
                               [("a", arr("f8", [-1.0, 2.0]))],
                               np.absolute(arr("f8", [-1.0, 2.0])), note="absolute = abs"))
    entries.append(_emit_shape("funcs_any", "any_red",
                               [("a", arr("i4", [0, 0, 1]))], np.any(arr("i4", [0, 0, 1])),
                               note="any 归约"))
    entries.append(_emit_shape("funcs_any_axis", "any_red",
                               [("a", arr("i4", [[0, 1], [0, 0]]))],
                               np.any(arr("i4", [[0, 1], [0, 0]]), axis=1),
                               note="any 按轴 → [True, False]", args={"axis": [1]}))
    entries.append(_emit_shape("funcs_invert_int", "invert",
                               [("a", arr("i4", [1, -2]))], np.invert(arr("i4", [1, -2])),
                               note="整型按位取反"))
    entries.append(_emit_shape("funcs_invert_bool", "invert",
                               [("a", arr("b", [True, False]))],
                               np.invert(arr("b", [True, False])), note="bool 逻辑取反"))
    cv1 = arr("f8", [1.0, 2.0, 3.0])
    cv2 = arr("f8", [4.0, 5.0, 6.0])
    entries.append(_emit_shape("funcs_cross", "cross", [("a", cv1), ("b", cv2)],
                               np.cross(cv1, cv2), note="3 维叉积"))
    cb = arr("f8", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    entries.append(_emit_shape("funcs_cross_batch", "cross",
                               [("a", cb), ("b", arr("f8", [0.0, 0.0, 1.0]))],
                               np.cross(cb, arr("f8", [0.0, 0.0, 1.0])),
                               note="批量叉积(前导维广播)"))

 # —— 位运算 + 小工具(Step 34)——
    ba1 = arr("i4", [6, 3, -4])
    bb1 = arr("i4", [3, 5, 1])
    entries.append(_emit_shape("funcs_bitwise_and", "bitwise_and", [("a", ba1), ("b", bb1)],
                           np.bitwise_and(ba1, bb1), note="按位与"))
    entries.append(_emit_shape("funcs_bitwise_or", "bitwise_or", [("a", ba1), ("b", bb1)],
                           np.bitwise_or(ba1, bb1), note="按位或"))
    entries.append(_emit_shape("funcs_bitwise_xor", "bitwise_xor", [("a", ba1), ("b", bb1)],
                           np.bitwise_xor(ba1, bb1), note="按位异或"))
    entries.append(_emit_shape("funcs_left_shift", "left_shift", [("a", ba1), ("b", bb1)],
                           np.left_shift(ba1, bb1), note="左移"))
    entries.append(_emit_shape("funcs_right_shift", "right_shift", [("a", ba1), ("b", bb1)],
                           np.right_shift(ba1, bb1), note="右移"))
    rm = arr("f8", [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    entries.append(_emit_shape("funcs_rot90", "rot90", [("a", rm)], np.rot90(rm, 1, axes=(0, 1)),
                           note="逆时针 90 度", args={"k": 1, "axis": [0, 1]}))
    entries.append(_emit_shape("funcs_rot90_k3", "rot90", [("a", rm)], np.rot90(rm, 3, axes=(0, 1)),
                           note="逆时针 270 度", args={"k": 3, "axis": [0, 1]}))
    av1 = arr("f8", [1.0, 2.0, 3.0])
    avw = arr("f8", [1.0, 2.0, 2.0])
    entries.append(_emit_rel("funcs_average", "average", [("a", av1), ("w", avw)],
                         np.asarray(np.average(av1, weights=avw)),
                         note="加权平均 2.2", tol=1e-12))
    av2 = arr("f8", [[1.0, 2.0], [3.0, 4.0]])
    avw2 = arr("f8", [1.0, 3.0])
    entries.append(_emit_rel("funcs_average_axis", "average", [("a", av2), ("w", avw2)],
                         np.average(av2, axis=1, weights=avw2),
                         note="按轴加权平均", args={"axis": [1]}, tol=1e-12))
    entries.append(_emit_shape("funcs_ptp", "ptp", [("a", arr("f8", [3.0, 1.0, 4.0, 1.0, 5.0]))],
                           np.ptp(arr("f8", [3.0, 1.0, 4.0, 1.0, 5.0])), note="峰谷差 4"))
    entries.append(_emit_shape("funcs_ptp_axis", "ptp", [("a", av2)],
                           np.ptp(av2, axis=0), note="按轴峰谷差", args={"axis": [0]}))
    npd = arr("f8", [1.0, np.nan, 3.0])
    entries.append(_emit_shape("funcs_nanprod", "nanprod", [("a", npd)],
                           np.nanprod(npd), note="忽略 NaN 的连乘 3"))
    entries.append(_emit_shape("funcs_nancumsum", "nancumsum", [("a", npd)],
                           np.nancumsum(npd), note="NaN 当 0 的前缀和 [1,1,4]"))
    ka = arr("i4", [[1, 2], [3, 4]])
    kb = arr("i4", [[0, 5], [6, 7]])
    entries.append(_emit_shape("funcs_kron", "kron", [("a", ka), ("b", kb)], np.kron(ka, kb),
                           note="Kronecker 积"))
    sa = arr("f8", [[4.0, 3.0], [2.0, 1.0]])
    _sign, _logdet = np.linalg.slogdet(sa)
    entries.append(_emit_multi_funcs("funcs_slogdet", "slogdet", [("a", sa)],
                                 [np.asarray(_sign), np.asarray(_logdet)],
                                 note="(符号, log|det|)"))

# —— 线代补全(Step 35):QR / Cholesky / Pinv / Lstsq ——
    qa = arr("f8", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _q, _r = np.linalg.qr(qa)
    entries.append(_emit_multi_funcs("funcs_qr", "qr", [("a", qa)], [_q, _r],
                                     note="Householder QR(reduced) → (Q, R)", tol=1e-10))
    sa2 = arr("f8", [[4.0, 2.0], [2.0, 3.0]])
    entries.append(_emit_rel("funcs_cholesky", "cholesky", [("a", sa2)],
                             np.linalg.cholesky(sa2), note="下三角 Cholesky 分解",
                             tol=1e-12))
    _pv = np.linalg.pinv(qa)
    entries.append(_emit_rel("funcs_pinv", "pinv", [("a", qa)], _pv,
                             note="Moore-Penrose 伪逆", tol=1e-10,
                             out_shape=np.shape(_pv)))
    lb = arr("f8", [1.0, 2.0, 4.0])
    _x, _res, _rank, _sv = np.linalg.lstsq(qa, lb, rcond=None)
    entries.append(_emit_multi_funcs("funcs_lstsq", "lstsq", [("a", qa), ("b", lb)],
                                     [_x, _res, np.asarray(_rank, dtype="<i8"), _sv],
                                     note="最小二乘 (x, residuals, rank, s)", tol=1e-10))

# 写 manifest_funcs.json(第五份;不碰旧四份 manifest 及其 npy)
    mf_path = os.path.join(REFDIR, "manifest_funcs.json")
    prev_gen = _load_prev_generated(mf_path)
    stamp = _now()
    for e in entries:
        e["generated"] = prev_gen.get(e["name"], stamp)
    with open(mf_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_bit = sum(1 for e in entries if e["region"] == "bitwise")
    print(f"manifest_funcs : {mf_path}({len(entries)} cases: {n_bit} bitwise / {len(entries) - n_bit} rel)")
    print(f"generator: {GENERATOR}")
    print(f"fixtures : {CASES} (*.npy,--funcs 前缀新增)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

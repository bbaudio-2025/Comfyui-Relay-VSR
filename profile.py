"""RelayVSR 各阶段耗时统计。

每个 stage 前后各取一次 perf_counter；GPU 阶段额外 synchronize，保证测到的是
设备上的实际执行时间，而不是 kernel 下发时间。统计只做累加，等节点执行结束时
由 report() 汇总打印一次，因此流式节点也不会逐帧刷屏。

编码阶段跑在后台线程里，与推理并行，所以它的占比不能直接和其他阶段相加。
"""

import logging
import time
from contextlib import contextmanager

import torch

_times = {}
_counts = {}
_started = None

# 嵌套在 wan/flash 内部的阶段，汇总时不计入顶层耗时。
_NESTED = ("attention_local", "attention_full")

_LABELS = {
    "load": "Model load",
    "io": "Image tensor conversion",
    "wan": "Wan reference frames",
    "flash": "Flash trunk",
    "decode": "Flash pixel decode",
    "attention_local": "Local attention",
    "attention_full": "Full attention",
    "preprocess": "Input preprocess",
    "postprocess": "Postprocess (interp/uint8/CPU copy)",
    "read": "Video decode",
    "encode": "Video encode (thread)",
    "stitch": "Spatial tile stitch",
}


def reset():
    """节点开始执行时调用，清空上一轮统计。"""
    global _started
    _times.clear()
    _counts.clear()
    _started = time.perf_counter()


def record(name, seconds, count=1):
    _times[name] = _times.get(name, 0.0) + seconds
    _counts[name] = _counts.get(name, 0) + count


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def stage(name, sync=True):
    """给一个阶段计时；sync=False 用于纯 CPU 或后台线程阶段。"""
    if sync:
        _sync()
    started = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            _sync()
        record(name, time.perf_counter() - started)


def report(title, frames=None):
    """把累计耗时汇总成一段日志；frames 非空时附带吞吐。"""
    if _started is None or not _times:
        return
    wall = time.perf_counter() - _started
    head = f"{title}: total {wall:.2f} s"
    if frames:
        head += f", {frames} frames, {frames / wall:.2f} fps"
    lines = [head]
    top = 0.0
    for name, seconds in sorted(_times.items(), key=lambda item: -item[1]):
        share = seconds / wall * 100 if wall else 0.0
        prefix = "  |_ " if name in _NESTED else "  "
        lines.append(
            f"{prefix}{_LABELS.get(name, name)}: {seconds:.2f} s ({share:.1f}%), {_counts[name]} calls"
        )
        if name not in _NESTED:
            top += seconds
    if top < wall:
        lines.append(f"  Other (setup/scheduling/wait): {wall - top:.2f} s")
    logging.info("\n".join(lines))

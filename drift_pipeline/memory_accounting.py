"""
E4 — memory accounting for continual-learning state.

The project's motivation for replay-free CL is edge memory limits. FYP1 asserted
that replay is expensive and never measured it. This module measures it.

## Why byte accounting and not just RSS

`RetrainController.__init__` allocates ALL FOUR CL structures for every arm and
seeds them lazily, so the sdft arm still carries an empty `ForecastingReplayBuffer`
and an empty `PPOEWCEngine`. Process RSS therefore does not attribute memory to
the mechanism that caused it — and RSS also carries the dataset, the dataloader
workers, and whatever the allocator has not returned to the OS. RSS is reported
here as an OUTER BOUND, not as the measurement.

The measurement is per-structure tensor storage, split by device, because the
binding constraint is 4 GB of VRAM rather than host RAM.

## What is actually being counted

Sizing walks to the underlying *storage*, not the tensor, and de-duplicates by
`(device, data_ptr)`. This matters in three places that would otherwise inflate
the numbers:

* A dataloader batch can hold views into one shared buffer.
* `PPOEWCEngine.opt_params` are `.clone()`d, so they are genuinely separate —
  the dedup confirms that rather than assuming it.
* `CLTFT._ewc_dev_cache` holds a **device-resident duplicate** of the CPU Fisher
  and opt-params (`trainers.py:340-344`). That duplicate is real, live memory and
  is counted separately, so EWC's true cost is up to four model-sized tensor sets
  once a retrain has started — a fact invisible to anyone reading only `__init__`.

Because dedup is order-dependent once structures share storage, every component
is sized **independently** (its cost if it were the only thing resident) and a
separate `total_deduped` is computed with one shared seen-set. Reporting both is
what makes double-counting detectable instead of silent.

## The counts matter as much as the bytes

`CONFIG["cl"]["replay_buffer_size"]` is 2000, but `add_from_loader` appends one
entry **per batch** (`trainers.py:1013-1023`), and `batch_size` is 256. So the cap
is denominated in batches, not windows, and its nominal ceiling is ~512k encoder
windows — high enough that it plausibly never binds, in which case the buffer
grows with every retrain event instead of plateauing. `n_entries` vs `n_windows`
is reported per snapshot so that question is answered by data rather than by
reading the config key and guessing.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = [
    "tensor_storage_bytes", "sized_bytes", "module_bytes",
    "ComponentSize", "size_forecasting_replay", "size_rl_replay",
    "size_rl_teacher", "size_ppo_ewc", "size_fc_ewc", "size_fc_teacher",
    "cl_state_bytes", "total_deduped_bytes", "process_rss_bytes",
    "gpu_allocated_bytes", "reset_peak", "MemoryLog", "MB",
]

MB = 1024.0 * 1024.0

# A key that identifies one physical allocation. `data_ptr` alone is not enough:
# CPU and CUDA address spaces are independent and can collide numerically.
_StorageKey = Tuple[str, int]


# ── primitives ────────────────────────────────────────────────────────────────

def tensor_storage_bytes(t: torch.Tensor,
                         seen: Optional[Set[_StorageKey]] = None) -> int:
    """Bytes of `t`'s underlying storage, counted once per distinct allocation.

    Uses the storage rather than `numel * element_size` so that a narrow view of
    a large buffer is charged for the buffer it pins, which is what actually
    occupies memory.
    """
    if not isinstance(t, torch.Tensor):
        return 0
    try:
        storage = t.untyped_storage()
    except (AttributeError, RuntimeError):
        # Sparse/meta/fake tensors have no single contiguous storage.
        return int(t.numel() * t.element_size())

    nbytes = int(storage.nbytes())
    if nbytes == 0:
        return 0
    if seen is None:
        return nbytes

    key: _StorageKey = (str(t.device), int(storage.data_ptr()))
    if key in seen:
        return 0
    seen.add(key)
    return nbytes


def _numpy_bytes(arr: np.ndarray, seen: Optional[Set[_StorageKey]] = None) -> int:
    """Bytes of a numpy array's owning buffer, counted once."""
    owner = arr
    while owner.base is not None and isinstance(owner.base, np.ndarray):
        owner = owner.base
    nbytes = int(owner.nbytes)
    if nbytes == 0 or seen is None:
        return nbytes
    key: _StorageKey = ("numpy", int(owner.__array_interface__["data"][0]))
    if key in seen:
        return 0
    seen.add(key)
    return nbytes


def sized_bytes(obj: Any, seen: Optional[Set[_StorageKey]] = None,
                _depth: int = 0) -> int:
    """Recursively total the tensor/array storage reachable from `obj`.

    Deliberately narrow: it descends dicts, lists, tuples and sets, and treats an
    `nn.Module` as its parameters plus buffers. It does NOT walk arbitrary object
    attributes, because a PPO model holds a reference to its environment, which
    holds the dataframe — following that would report the dataset as CL state.
    """
    if _depth > 12:
        return 0
    if isinstance(obj, torch.Tensor):
        return tensor_storage_bytes(obj, seen)
    if isinstance(obj, np.ndarray):
        return _numpy_bytes(obj, seen)
    if isinstance(obj, nn.Module):
        return module_bytes(obj, seen)
    if isinstance(obj, dict):
        return sum(sized_bytes(v, seen, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(sized_bytes(v, seen, _depth + 1) for v in obj)
    return 0


def module_bytes(module: nn.Module,
                 seen: Optional[Set[_StorageKey]] = None) -> int:
    """Parameters + buffers of a module. `duplicate=False` avoids charging twice
    for weights that are tied or otherwise shared between submodules."""
    total = 0
    for p in module.parameters(recurse=True):
        total += tensor_storage_bytes(p, seen)
        if p.grad is not None:
            total += tensor_storage_bytes(p.grad, seen)
    for b in module.buffers(recurse=True):
        total += tensor_storage_bytes(b, seen)
    return total


def _device_split(obj: Any) -> Dict[str, int]:
    """Bytes broken down by device, so VRAM pressure is separable from host RAM.

    VRAM is the binding constraint on a 4 GB card, and a structure that lives on
    the CPU costs nothing against it — a distinction a single total hides.
    """
    out: Dict[str, int] = {}
    for dev, tensors in _collect_tensors(obj).items():
        seen: Set[_StorageKey] = set()
        out[dev] = sum(tensor_storage_bytes(t, seen) for t in tensors)
    return out


def _collect_tensors(obj: Any, acc: Optional[Dict[str, List]] = None,
                     _depth: int = 0) -> Dict[str, List]:
    if acc is None:
        acc = {}
    if _depth > 12:
        return acc
    if isinstance(obj, torch.Tensor):
        acc.setdefault(str(obj.device), []).append(obj)
    elif isinstance(obj, np.ndarray):
        acc.setdefault("numpy", []).append(torch.from_numpy(obj)
                                           if obj.dtype != object else torch.empty(0))
    elif isinstance(obj, nn.Module):
        for p in obj.parameters(recurse=True):
            acc.setdefault(str(p.device), []).append(p)
        for b in obj.buffers(recurse=True):
            acc.setdefault(str(b.device), []).append(b)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_tensors(v, acc, _depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _collect_tensors(v, acc, _depth + 1)
    return acc


# ── per-component sizing ──────────────────────────────────────────────────────

@dataclass
class ComponentSize:
    """One CL structure's footprint at one point in time."""
    component: str
    bytes_total: int = 0
    n_items: int = 0            # entries held (batches / transitions / tensors)
    n_units: int = 0            # underlying units (windows / transitions)
    n_tasks: int = 0
    by_device: Dict[str, int] = field(default_factory=dict)
    note: str = ""

    @property
    def mb(self) -> float:
        return self.bytes_total / MB

    def as_row(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "bytes": int(self.bytes_total),
            "mb": round(self.mb, 4),
            "n_items": int(self.n_items),
            "n_units": int(self.n_units),
            "n_tasks": int(self.n_tasks),
            "bytes_cpu": int(self.by_device.get("cpu", 0)),
            "bytes_gpu": int(sum(v for k, v in self.by_device.items()
                                 if k.startswith("cuda"))),
            "note": self.note,
        }


def _batch_windows(x: Any) -> int:
    """Rows in the leading dimension of a stored batch, i.e. encoder windows.

    This is what separates 'the buffer holds 2000 things' from 'the buffer holds
    2000 x batch_size windows', which is the whole batches-vs-windows question.
    """
    if isinstance(x, torch.Tensor) and x.dim() >= 1:
        return int(x.shape[0])
    if isinstance(x, dict):
        for v in x.values():
            n = _batch_windows(v)
            if n:
                return n
    if isinstance(x, (list, tuple)):
        for v in x:
            n = _batch_windows(v)
            if n:
                return n
    return 0


def size_forecasting_replay(buf: Any) -> ComponentSize:
    """`ForecastingReplayBuffer` — stored TFT batches, keyed by task."""
    cs = ComponentSize("fc_replay")
    by_task = getattr(buf, "buffer_by_task", None)
    if not by_task:
        cs.note = "empty"
        return cs

    seen: Set[_StorageKey] = set()
    total, n_items, n_units = 0, 0, 0
    for task_buf in by_task.values():
        for entry in task_buf:
            total += sized_bytes(entry, seen)
            n_items += 1
            x = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            n_units += _batch_windows(x)

    cs.bytes_total = total
    cs.n_items = n_items                      # stored entries == BATCHES
    cs.n_units = n_units                      # encoder windows inside them
    cs.n_tasks = len([t for t in by_task.values() if t])
    cs.by_device = _device_split(
        [e for task_buf in by_task.values() for e in task_buf])
    cap = getattr(buf, "capacity", None)
    if cap is not None:
        cs.note = f"capacity={cap} entries; binding={n_items >= cap}"
    return cs


def size_rl_replay(buf: Any) -> ComponentSize:
    """`RLReplayBuffer` — (obs, action, reward, next_obs, done) tuples."""
    cs = ComponentSize("rl_replay")
    by_task = getattr(buf, "transitions_by_task", None)
    if not by_task:
        cs.note = "empty"
        return cs

    seen: Set[_StorageKey] = set()
    total, n = 0, 0
    for task_buf in by_task.values():
        for tr in task_buf:
            # obs / next_obs are numpy; action, reward, done are Python scalars
            # whose object overhead the storage walk does not see. Charge them
            # explicitly, or a buffer of millions of transitions reads as free.
            total += sized_bytes(tr, seen) + 3 * 8
            n += 1
    cs.bytes_total = total
    cs.n_items = n
    cs.n_units = n
    cs.n_tasks = len([t for t in by_task.values() if t])
    cs.by_device = {"cpu": total}
    cap = getattr(buf, "capacity", None)
    if cap is not None:
        cs.note = f"capacity={cap} transitions; binding={n >= cap}"
    return cs


def size_rl_teacher(store: Any) -> ComponentSize:
    """`RLTeacherStore` — one frozen deepcopy of the PPO policy."""
    cs = ComponentSize("rl_teacher")
    policy = getattr(store, "teacher_policy", None)
    if policy is None:
        cs.note = "empty"
        return cs
    seen: Set[_StorageKey] = set()
    cs.bytes_total = module_bytes(policy, seen)
    cs.n_items = 1
    cs.n_units = sum(p.numel() for p in policy.parameters())
    cs.by_device = _device_split(policy)
    cs.note = "one frozen policy copy (constant in stream length)"
    return cs


def size_ppo_ewc(engine: Any) -> ComponentSize:
    """`PPOEWCEngine` — diagonal Fisher plus the anchor parameters.

    Two model-sized tensor sets, not one. EWC is routinely described as cheap
    because it stores no data; it stores twice the policy instead.
    """
    cs = ComponentSize("rl_ewc")
    fisher = getattr(engine, "fisher", {}) or {}
    opt = getattr(engine, "opt_params", {}) or {}
    if not fisher and not opt:
        cs.note = "empty"
        return cs
    seen: Set[_StorageKey] = set()
    cs.bytes_total = sized_bytes(fisher, seen) + sized_bytes(opt, seen)
    cs.n_items = len(fisher) + len(opt)
    cs.n_units = sum(t.numel() for t in fisher.values()) + \
                 sum(t.numel() for t in opt.values())
    cs.by_device = _device_split([fisher, opt])
    cs.note = "fisher + anchor = 2x policy"
    return cs


def size_fc_ewc(model: Any) -> ComponentSize:
    """Forecasting EWC state on a `CLTFT`, INCLUDING the device-resident cache.

    `_ewc_dev_cache` is a second copy of Fisher and anchor moved to the compute
    device and held for the life of the fit. Counting only the CPU dicts would
    understate EWC's peak by exactly the amount that lands on the GPU — which is
    the number the 4 GB constraint cares about.
    """
    cs = ComponentSize("fc_ewc")
    fisher = getattr(model, "ewc_fisher", {}) or {}
    opt = getattr(model, "ewc_optparams", {}) or {}
    cache = getattr(model, "_ewc_dev_cache", None)
    if not fisher and not opt and cache is None:
        cs.note = "empty"
        return cs

    seen: Set[_StorageKey] = set()
    total = sized_bytes(fisher, seen) + sized_bytes(opt, seen)
    cached = sized_bytes(cache, seen) if cache is not None else 0
    cs.bytes_total = total + cached
    cs.n_items = len(fisher) + len(opt)
    cs.n_units = sum(t.numel() for t in fisher.values()) + \
                 sum(t.numel() for t in opt.values())
    cs.by_device = _device_split([fisher, opt, cache])
    cs.note = ("fisher + anchor on cpu"
               + (f" + device cache ({cached / MB:.1f} MB)" if cached else
                  " (device cache not yet built)"))
    return cs


def size_fc_teacher(model: Any) -> ComponentSize:
    """SDFT teacher hanging off a `CLTFT` as `.teacher` — a full second TFT."""
    cs = ComponentSize("fc_teacher")
    teacher = getattr(model, "teacher", None)
    if teacher is None:
        cs.note = "empty"
        return cs
    seen: Set[_StorageKey] = set()
    cs.bytes_total = module_bytes(teacher, seen)
    cs.n_items = 1
    cs.n_units = sum(p.numel() for p in teacher.parameters())
    cs.by_device = _device_split(teacher)
    cs.note = "one frozen TFT copy (constant in stream length)"
    return cs


# ── whole-controller snapshot ─────────────────────────────────────────────────

def cl_state_bytes(ctrl: Any) -> List[ComponentSize]:
    """Size every CL structure a `RetrainController` holds.

    Every arm is measured for every component, including the ones its strategy
    never seeds. An empty structure reporting 0 is a result: it is what lets the
    table show that sdft's replay buffer really is empty rather than merely
    unmentioned.
    """
    comps = [
        size_forecasting_replay(getattr(ctrl, "fc_replay", None)),
        size_rl_replay(getattr(ctrl, "rl_buf", None)),
        size_ppo_ewc(getattr(ctrl, "rl_ewc", None)),
        size_rl_teacher(getattr(ctrl, "rl_teacher", None)),
        size_fc_ewc(getattr(ctrl, "forecaster", None)),
        size_fc_teacher(getattr(ctrl, "forecaster", None)),
    ]
    return comps


def total_deduped_bytes(ctrl: Any) -> int:
    """Total across all components with ONE shared seen-set.

    Compare against the sum of the independently-sized components: a gap means
    two structures share storage, and the per-component numbers are then upper
    bounds rather than additive costs.
    """
    seen: Set[_StorageKey] = set()
    total = 0
    for attr in ("fc_replay", "rl_buf", "rl_ewc", "rl_teacher"):
        obj = getattr(ctrl, attr, None)
        if obj is None:
            continue
        for field_name in ("buffer_by_task", "transitions_by_task", "fisher",
                           "opt_params", "teacher_policy"):
            total += sized_bytes(getattr(obj, field_name, None), seen)
    model = getattr(ctrl, "forecaster", None)
    if model is not None:
        total += sized_bytes(getattr(model, "ewc_fisher", None), seen)
        total += sized_bytes(getattr(model, "ewc_optparams", None), seen)
        total += sized_bytes(getattr(model, "_ewc_dev_cache", None), seen)
        teacher = getattr(model, "teacher", None)
        if teacher is not None:
            total += module_bytes(teacher, seen)
    return total


def process_rss_bytes() -> int:
    """Resident set size — the OUTER BOUND, not the measurement.

    Includes the dataset, dataloader state, CUDA context and any memory the
    allocator has not returned to the OS, none of which is CL state.
    """
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


def gpu_allocated_bytes() -> Tuple[int, int]:
    """(current, peak) CUDA bytes allocated by torch, or (0, 0) on CPU."""
    if not torch.cuda.is_available():
        return 0, 0
    return (int(torch.cuda.memory_allocated()),
            int(torch.cuda.max_memory_allocated()))


def reset_peak() -> None:
    """Zero the CUDA peak counter so the next snapshot's `torch_cuda_peak` is the
    peak of ONE retrain rather than of the whole run.

    Without this the peak is monotone by construction and every arm eventually
    reports the same high-water mark, which would say nothing about which
    mechanism caused it.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── the log ───────────────────────────────────────────────────────────────────

class MemoryLog:
    """Long-form record of CL memory over a walk.

    One row per (arm, event, component) so the output matches the shape every
    other result in this project uses and the dashboard already consumes.
    """

    COLUMNS = ["strategy", "event_idx", "date", "event", "model_type",
               "component", "bytes", "mb", "n_items", "n_units", "n_tasks",
               "bytes_cpu", "bytes_gpu", "note"]

    def __init__(self, strategy: str = ""):
        self.strategy = strategy
        self.rows: List[Dict[str, Any]] = []
        self._event_idx = 0

    def snapshot(self, ctrl: Any, *, event: str, date: Any = None,
                 model_type: str = "", collect: bool = False) -> List[Dict]:
        """Record every component's footprint at one moment.

        `collect=True` runs a gc pass first. Off by default: it is slow, and it
        changes RSS, so a walk should either collect at every snapshot or at
        none of them — never mix, or the series is not comparable with itself.
        """
        if collect:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        idx = self._event_idx
        self._event_idx += 1
        stamp = str(pd.Timestamp(date).date()) if date is not None else ""

        new_rows = []
        for cs in cl_state_bytes(ctrl):
            row = {"strategy": self.strategy or getattr(ctrl, "strategy", ""),
                   "event_idx": idx, "date": stamp, "event": event,
                   "model_type": model_type}
            row.update(cs.as_row())
            new_rows.append(row)

        # Aggregates as their own components, so a plot can select one row per
        # event without summing (and without double-counting shared storage).
        deduped = total_deduped_bytes(ctrl)
        rss = process_rss_bytes()
        cur_gpu, peak_gpu = gpu_allocated_bytes()
        for name, value, note in (
            ("cl_state_total", deduped, "all CL structures, storage-deduped"),
            ("process_rss", rss, "outer bound: includes data and CUDA context"),
            ("torch_cuda_allocated", cur_gpu, "torch-tracked current VRAM"),
            ("torch_cuda_peak", peak_gpu, "torch-tracked peak VRAM"),
        ):
            new_rows.append({
                "strategy": self.strategy or getattr(ctrl, "strategy", ""),
                "event_idx": idx, "date": stamp, "event": event,
                "model_type": model_type, "component": name,
                "bytes": int(value), "mb": round(value / MB, 4),
                "n_items": 0, "n_units": 0, "n_tasks": 0,
                "bytes_cpu": 0, "bytes_gpu": 0, "note": note,
            })

        self.rows.extend(new_rows)
        return new_rows

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(self.rows)[self.COLUMNS]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
        return path

    def summary(self) -> pd.DataFrame:
        """Peak bytes per component per arm — the table E4 reports."""
        df = self.to_frame()
        if df.empty:
            return df
        return (df.groupby(["strategy", "component"], as_index=False)
                  .agg(peak_bytes=("bytes", "max"),
                       peak_mb=("mb", "max"),
                       final_mb=("mb", "last"),
                       max_items=("n_items", "max"),
                       max_units=("n_units", "max")))

    def __len__(self) -> int:
        return len(self.rows)

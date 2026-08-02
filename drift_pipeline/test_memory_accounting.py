"""
Tests for E4 memory accounting.

These use hand-built stand-ins for the CL structures rather than the real ones,
so the suite stays fast and needs no data, no checkpoints and no GPU. The
stand-ins mirror the real attribute names (`buffer_by_task`, `transitions_by_task`,
`fisher`/`opt_params`, `teacher_policy`, `ewc_fisher`/`_ewc_dev_cache`, `.teacher`),
which is exactly the contract `memory_accounting` depends on — if a field is
renamed in `drift_pipeline/trainers.py`, the corresponding test here goes to
zero bytes and fails loudly instead of silently under-reporting.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch
import torch.nn as nn

from drift_pipeline.memory_accounting import (
    MB, MemoryLog, cl_state_bytes, module_bytes, sized_bytes,
    size_fc_ewc, size_fc_teacher, size_forecasting_replay, size_ppo_ewc,
    size_rl_replay, size_rl_teacher, tensor_storage_bytes,
    total_deduped_bytes,
)


# ── stand-ins ─────────────────────────────────────────────────────────────────

class FakeFCReplay:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer_by_task = defaultdict(list)

    def add_batch(self, task_id, n_windows, n_features=8, enc_len=60):
        x = {"encoder_cont": torch.zeros(n_windows, enc_len, n_features),
             "encoder_lengths": torch.zeros(n_windows, dtype=torch.long)}
        y = [torch.zeros(n_windows, 14)]
        self.buffer_by_task[task_id].append((x, y))


class FakeRLReplay:
    def __init__(self, capacity=10_000):
        self.capacity = capacity
        self.transitions_by_task = defaultdict(list)

    def add(self, task_id, n, obs_dim=13):
        for _ in range(n):
            self.transitions_by_task[task_id].append(
                (np.zeros(obs_dim, dtype=np.float32), 3, 1.0,
                 np.zeros(obs_dim, dtype=np.float32), False))


class FakeEWCEngine:
    def __init__(self, policy=None):
        self.fisher, self.opt_params = {}, {}
        if policy is not None:
            self.fisher = {n: torch.zeros_like(p)
                           for n, p in policy.named_parameters()}
            self.opt_params = {n: p.detach().clone()
                               for n, p in policy.named_parameters()}


class FakeTeacherStore:
    def __init__(self, policy=None):
        self.teacher_policy = policy


class FakeCLTFT(nn.Module):
    """Only the attributes the accounting reads: EWC dicts, the device cache and
    the `.teacher` submodule."""
    def __init__(self, n_in=32, n_out=16):
        super().__init__()
        self.net = nn.Linear(n_in, n_out)
        self.ewc_fisher = {}
        self.ewc_optparams = {}
        self._ewc_dev_cache = None
        self.teacher = None

    def seed_ewc(self, with_cache=False):
        self.ewc_fisher = {n: torch.zeros_like(p)
                           for n, p in self.net.named_parameters()}
        self.ewc_optparams = {n: p.detach().clone()
                              for n, p in self.net.named_parameters()}
        if with_cache:
            self._ewc_dev_cache = (
                {n: t.clone() for n, t in self.ewc_fisher.items()},
                {n: t.clone() for n, t in self.ewc_optparams.items()},
            )


class FakeController:
    def __init__(self, strategy="sdft"):
        self.strategy = strategy
        self.fc_replay = FakeFCReplay()
        self.rl_buf = FakeRLReplay()
        self.rl_ewc = FakeEWCEngine()
        self.rl_teacher = FakeTeacherStore()
        self.forecaster = FakeCLTFT()


# ── primitives ────────────────────────────────────────────────────────────────

def test_tensor_bytes_is_storage_size():
    assert tensor_storage_bytes(torch.zeros(10, dtype=torch.float32)) == 40
    assert tensor_storage_bytes(torch.zeros(10, dtype=torch.float64)) == 80


def test_same_storage_counted_once():
    t = torch.zeros(100)
    seen = set()
    first = tensor_storage_bytes(t, seen)
    second = tensor_storage_bytes(t, seen)
    assert first == 400 and second == 0


def test_view_is_charged_for_the_buffer_it_pins():
    """A 3-element view of a 1000-element tensor keeps all 1000 alive."""
    big = torch.zeros(1000)
    view = big[2:5]
    assert tensor_storage_bytes(view) == 4000


def test_distinct_clones_are_not_deduped():
    """EWC's anchor is a real second copy; dedup must not hide that."""
    a = torch.zeros(100)
    b = a.detach().clone()
    seen = set()
    assert tensor_storage_bytes(a, seen) + tensor_storage_bytes(b, seen) == 800


def test_module_bytes_counts_params_and_buffers():
    lin = nn.Linear(4, 3)                       # 12 weights + 3 bias, float32
    assert module_bytes(lin) == (12 + 3) * 4


def test_sized_bytes_walks_containers():
    payload = {"a": torch.zeros(10), "b": [torch.zeros(20), (torch.zeros(30),)]}
    assert sized_bytes(payload, set()) == (10 + 20 + 30) * 4


def test_sized_bytes_does_not_follow_arbitrary_attributes():
    """The PPO-holds-the-env trap: walking attributes would charge CL state for
    the entire dataset."""
    class HasEnv:
        def __init__(self):
            self.env = torch.zeros(1_000_000)
    assert sized_bytes(HasEnv(), set()) == 0


def test_numpy_view_charged_once():
    arr = np.zeros(1000, dtype=np.float32)
    seen = set()
    assert sized_bytes(arr, seen) == 4000
    assert sized_bytes(arr[10:20], seen) == 0      # same owning buffer


# ── forecasting replay: the batches-vs-windows question ───────────────────────

def test_fc_replay_counts_batches_as_items_and_windows_as_units():
    """`replay_buffer_size` caps ENTRIES, and one entry is a whole batch. This
    test is the executable form of that distinction."""
    buf = FakeFCReplay(capacity=2000)
    for _ in range(3):
        buf.add_batch(task_id=0, n_windows=256)

    cs = size_forecasting_replay(buf)
    assert cs.n_items == 3                       # three stored entries
    assert cs.n_units == 768                     # ... which hold 768 windows
    assert cs.n_tasks == 1
    assert cs.bytes_total > 0
    assert "binding=False" in cs.note            # 3 entries is nowhere near 2000


def test_fc_replay_grows_with_each_added_batch():
    """If the cap never binds, footprint is monotone in retrain count — the
    claim E4 exists to test."""
    buf = FakeFCReplay()
    sizes = []
    for task in range(4):
        buf.add_batch(task_id=task, n_windows=64)
        sizes.append(size_forecasting_replay(buf).bytes_total)
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_fc_replay_reports_capacity_binding():
    buf = FakeFCReplay(capacity=2)
    for _ in range(2):
        buf.add_batch(task_id=0, n_windows=8)
    assert "binding=True" in size_forecasting_replay(buf).note


def test_empty_structures_report_zero_not_missing():
    ctrl = FakeController()
    for cs in cl_state_bytes(ctrl):
        assert cs.bytes_total == 0
        assert cs.note == "empty" or "not yet built" in cs.note


# ── the model-sized structures ────────────────────────────────────────────────

def test_ppo_ewc_is_two_copies_of_the_policy():
    policy = nn.Linear(32, 11)
    cs = size_ppo_ewc(FakeEWCEngine(policy))
    assert cs.bytes_total == pytest.approx(2 * module_bytes(policy))
    assert "2x policy" in cs.note


def test_rl_teacher_is_one_copy_of_the_policy():
    policy = nn.Linear(32, 11)
    cs = size_rl_teacher(FakeTeacherStore(policy))
    assert cs.bytes_total == module_bytes(policy)
    assert cs.n_items == 1


def test_fc_ewc_counts_the_device_resident_cache():
    """`_ewc_dev_cache` doubles EWC's live footprint once a fit starts. Counting
    only the CPU dicts understates the peak by exactly the GPU-side copy."""
    model = FakeCLTFT()
    model.seed_ewc(with_cache=False)
    without = size_fc_ewc(model).bytes_total

    model.seed_ewc(with_cache=True)
    with_cache = size_fc_ewc(model).bytes_total

    assert with_cache == pytest.approx(2 * without)
    assert "device cache" in size_fc_ewc(model).note


def test_fc_teacher_is_a_full_second_model():
    model = FakeCLTFT()
    model.teacher = FakeCLTFT()
    cs = size_fc_teacher(model)
    assert cs.bytes_total == module_bytes(model.net)
    assert "constant in stream length" in cs.note


def test_rl_replay_charges_scalar_fields():
    """obs/next_obs are numpy, but action/reward/done are Python scalars the
    storage walk cannot see. A million transitions must not read as free."""
    buf = FakeRLReplay()
    buf.add(task_id=0, n=100, obs_dim=13)
    cs = size_rl_replay(buf)
    arrays = 100 * 2 * 13 * 4
    assert cs.bytes_total == arrays + 100 * 24
    assert cs.n_items == 100


# ── aggregation ───────────────────────────────────────────────────────────────

def test_total_deduped_matches_component_sum_when_nothing_is_shared():
    ctrl = FakeController()
    ctrl.fc_replay.add_batch(0, n_windows=32)
    ctrl.rl_teacher = FakeTeacherStore(nn.Linear(16, 4))
    ctrl.forecaster.seed_ewc(with_cache=True)

    component_sum = sum(cs.bytes_total for cs in cl_state_bytes(ctrl))
    assert total_deduped_bytes(ctrl) == component_sum


def test_total_deduped_is_lower_when_storage_is_shared():
    """Shared storage makes per-component numbers upper bounds rather than
    additive costs — the gap is what makes that detectable."""
    ctrl = FakeController()
    shared = torch.zeros(5000)
    ctrl.rl_ewc.fisher = {"w": shared}
    ctrl.rl_ewc.opt_params = {"w": shared}       # same storage, deliberately

    cs = size_ppo_ewc(ctrl.rl_ewc)
    assert cs.bytes_total == 20_000              # counted once within component
    assert total_deduped_bytes(ctrl) == 20_000


# ── the log ───────────────────────────────────────────────────────────────────

def test_snapshot_emits_one_row_per_component_plus_aggregates():
    log = MemoryLog("sdft")
    rows = log.snapshot(FakeController(), event="retrain_fc", date="2015-01-31")
    names = {r["component"] for r in rows}
    assert {"fc_replay", "rl_replay", "rl_ewc", "rl_teacher", "fc_ewc",
            "fc_teacher"} <= names
    assert {"cl_state_total", "process_rss"} <= names
    assert all(r["date"] == "2015-01-31" for r in rows)


def test_frame_has_stable_columns_and_event_ordering():
    log = MemoryLog("replay")
    ctrl = FakeController("replay")
    for i in range(3):
        ctrl.fc_replay.add_batch(task_id=i, n_windows=16)
        log.snapshot(ctrl, event="retrain_fc", model_type="forecasting")

    df = log.to_frame()
    assert list(df.columns) == MemoryLog.COLUMNS
    assert sorted(df["event_idx"].unique().tolist()) == [0, 1, 2]

    growth = df[df["component"] == "fc_replay"].sort_values("event_idx")["bytes"]
    assert growth.is_monotonic_increasing


def test_process_rss_is_recorded_as_an_outer_bound():
    log = MemoryLog("sdft")
    log.snapshot(FakeController(), event="init")
    df = log.to_frame()
    rss = df[df["component"] == "process_rss"]["bytes"].iloc[0]
    cl = df[df["component"] == "cl_state_total"]["bytes"].iloc[0]
    assert rss > cl                     # RSS carries interpreter, torch, data
    assert rss > 10 * MB


def test_summary_reports_peak_per_component():
    log = MemoryLog("replay")
    ctrl = FakeController("replay")
    for i in range(3):
        ctrl.fc_replay.add_batch(task_id=i, n_windows=16)
        log.snapshot(ctrl, event="retrain_fc")

    summary = log.summary()
    row = summary[summary["component"] == "fc_replay"].iloc[0]
    assert row["peak_bytes"] == log.to_frame().query(
        "component == 'fc_replay'")["bytes"].max()
    assert row["max_units"] == 48       # 3 batches x 16 windows


def test_empty_log_still_has_the_schema():
    assert list(MemoryLog("x").to_frame().columns) == MemoryLog.COLUMNS


# ── wiring ────────────────────────────────────────────────────────────────────

def test_real_controller_snapshots_at_init():
    """The stand-ins above only prove the accounting is right about objects that
    look like the real ones. This proves `RetrainController` actually carries a
    MemoryLog and that the attribute names still line up — the failure mode the
    stand-ins cannot catch.

    Slow: importing `retrain` pulls in torch, lightning, pytorch-forecasting and
    stable-baselines3. It needs no data, no checkpoints and no GPU, because
    __init__ only stores what it is handed.
    """
    from drift_pipeline.retrain import RetrainController

    base = {"train_ds": None, "forecaster": FakeCLTFT(), "pricer": None}
    ctrl = RetrainController("sdft", base, {})

    df = ctrl.memlog.to_frame()
    assert not df.empty and set(df["event"]) == {"init"}
    assert df["strategy"].eq("sdft").all()

    # Every CL structure is present and empty at the zero point.
    for component in ("fc_replay", "rl_replay", "rl_ewc", "rl_teacher"):
        assert df.loc[df["component"] == component, "bytes"].iloc[0] == 0
    assert df.loc[df["component"] == "cl_state_total", "bytes"].iloc[0] == 0
    # ... while RSS already carries the interpreter and the ML stack.
    assert df.loc[df["component"] == "process_rss", "bytes"].iloc[0] > 100 * MB


def test_injected_memlog_is_used():
    """One log can span several arms, so a sweep writes a single tidy frame."""
    from drift_pipeline.retrain import RetrainController

    shared = MemoryLog("sweep")
    base = {"train_ds": None, "forecaster": FakeCLTFT(), "pricer": None}
    RetrainController("ewc", base, {}, memlog=shared)
    RetrainController("sdft", base, {}, memlog=shared)

    df = shared.to_frame()
    assert df["strategy"].eq("sweep").all()      # the log's label wins
    assert sorted(df["event_idx"].unique().tolist()) == [0, 1]

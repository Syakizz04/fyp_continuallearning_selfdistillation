"""
E4 - memory footprint of continual-learning state.

    python -m experiments.exp_memory --static          # per-structure cost, no training
    python -m experiments.exp_memory --aggregate       # roll up memory_*.csv from walks

The project's argument for replay-free CL is that the edge cannot afford replay.
FYP1 asserted that and never measured it. This closes the gap.

## Two modes, because there are two different questions

**--static** measures what each CL mechanism costs *per unit of the thing it
stores*, using the real base checkpoints and the real dataloader but **no
training at all**. It answers "how big is a replay batch, a Fisher, a teacher?"
in seconds rather than hours, and it is what makes the growth curve from a real
walk interpretable rather than just a rising line.

**--aggregate** rolls up the `memory_<arm>.csv` files that `RetrainController`
now writes during any walk (see drift_pipeline/memory_accounting.py). That is the
observed trajectory; static is the analytic rate.

## Why static sizing is not merely a shortcut

The replay cap is denominated in *batches*: `add_from_loader` appends one entry
per batch and `replay_buffer_size` is 2000, at `batch_size` 256. Whether the
buffer plateaus at the cap or grows for the whole walk therefore depends on how
many batches the recent window yields, which is a property of the data and not of
the CL method. Measuring one real batch and multiplying is how that gets settled
without waiting for a walk to finish.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drift_pipeline.core_pipeline import CONFIG, DEVICE, console  # noqa: E402
from drift_pipeline.memory_accounting import (  # noqa: E402
    MB, module_bytes, size_fc_ewc, size_fc_teacher, size_forecasting_replay,
    size_ppo_ewc, size_rl_replay, size_rl_teacher, process_rss_bytes,
)

N_PROBE_BATCHES = 8          # enough to average out a short final batch


def _fmt(mb: float) -> str:
    return f"{mb:8.2f} MB"


def measure_static(n_batches: int = N_PROBE_BATCHES) -> pd.DataFrame:
    """Size every CL structure against the real base models. No training."""
    import copy

    import torch

    from drift_pipeline import monitor as mon
    from drift_pipeline.core_pipeline import prepare_drift_data
    from hybrid_pipeline.trainers import (ForecastingReplayBuffer, PPOEWCEngine,
                                          RLReplayBuffer, RLTeacherStore,
                                          build_cltft, make_tft_dataset)

    rss0 = process_rss_bytes()
    console.print(f"[dim]RSS before load: {rss0 / MB:.0f} MB[/dim]")

    # Order matters: `load_base` overwrites CONFIG["forecasting"] with the
    # persisted base meta, in which `adapt_config_to_data` has already dropped
    # constant columns. Preparing data afterwards would then re-drop them and
    # fail. The real pipeline prepares data first for the same reason.
    data = prepare_drift_data()
    base = mon.load_base()
    model, train_ds, pricer = base["forecaster"], base["train_ds"], base["pricer"]

    rows: List[Dict] = []

    def record(name: str, cs, *, unit: str, per_unit_mb: float = 0.0,
               projected_mb: float = 0.0, detail: str = "") -> None:
        row = cs.as_row()
        rows.append({
            "component": name, "mb": round(cs.bytes_total / MB, 4),
            "bytes": int(cs.bytes_total), "n_items": cs.n_items,
            "n_units": cs.n_units, "unit": unit,
            "per_unit_mb": round(per_unit_mb, 6),
            "projected_at_capacity_mb": round(projected_mb, 2),
            # Which memory it consumes is as important as how much: replay is
            # host RAM, the EWC device cache is VRAM, and only one of those is
            # bounded by the 4 GB card.
            "mb_cpu": round(row["bytes_cpu"] / MB, 4),
            "mb_gpu": round(row["bytes_gpu"] / MB, 4),
            "detail": detail or cs.note,
        })

    # ── the model itself, as the yardstick ────────────────────────────────────
    # Params only. `module_bytes` also charges `.grad`, which a live training
    # model carries and a frozen teacher does not — comparing a teacher against
    # a params+grads figure would understate the copy at exactly 2x.
    def _param_bytes(m) -> int:
        return sum(p.numel() * p.element_size() for p in m.parameters())

    tft_mb = _param_bytes(model) / MB
    policy_mb = _param_bytes(pricer.policy) / MB
    rows.append({"component": "tft_model", "mb": round(tft_mb, 4),
                 "bytes": _param_bytes(model), "n_items": 1,
                 "n_units": sum(p.numel() for p in model.parameters()),
                 "unit": "model", "per_unit_mb": round(tft_mb, 6),
                 "projected_at_capacity_mb": round(tft_mb, 2),
                 "detail": "served forecaster, params only - the yardstick"})
    rows.append({"component": "tft_model_training", "mb": round(module_bytes(model) / MB, 4),
                 "bytes": int(module_bytes(model)), "n_items": 1,
                 "n_units": sum(p.numel() for p in model.parameters()),
                 "unit": "model", "per_unit_mb": round(module_bytes(model) / MB, 6),
                 "projected_at_capacity_mb": round(module_bytes(model) / MB, 2),
                 "detail": "same model mid-fit: params + gradients"})
    rows.append({"component": "ppo_policy", "mb": round(policy_mb, 4),
                 "bytes": _param_bytes(pricer.policy), "n_items": 1,
                 "n_units": sum(p.numel() for p in pricer.policy.parameters()),
                 "unit": "model", "per_unit_mb": round(policy_mb, 6),
                 "projected_at_capacity_mb": round(policy_mb, 2),
                 "detail": "served pricer, params only"})

    # ── SDFT: one frozen copy each ────────────────────────────────────────────
    teacher = build_cltft(train_ds, cl_method="naive")
    sd = {k: v for k, v in model.state_dict().items()
          if not k.startswith("teacher.")}
    teacher.load_state_dict(sd)
    teacher.to(DEVICE).eval()
    model.teacher = teacher
    record("fc_teacher", size_fc_teacher(model), unit="model",
           per_unit_mb=tft_mb, projected_mb=tft_mb,
           detail="SDFT: exactly one frozen TFT, flat in stream length")
    model.teacher = None

    store = RLTeacherStore()
    store.store(pricer)
    record("rl_teacher", size_rl_teacher(store), unit="model",
           per_unit_mb=policy_mb, projected_mb=policy_mb,
           detail="SDFT: exactly one frozen policy, flat in stream length")

    # ── EWC: Fisher + anchor, and the device-resident duplicate ───────────────
    # zeros_like has the same footprint as a computed Fisher, so the expensive
    # gradient pass buys no extra information about SIZE. `.cpu()` is NOT
    # cosmetic: `compute_and_store_fisher` stores both dicts on the CPU
    # (trainers.py:394,397), so `_ewc_dev_cache` later makes two genuine GPU
    # copies. Leaving the Fisher on the GPU here would make the cache dedup
    # against it and understate EWC's peak by a full model.
    model.ewc_fisher = {n: torch.zeros_like(p).cpu()
                        for n, p in model.named_parameters() if p.requires_grad}
    model.ewc_optparams = {n: p.detach().cpu().clone()
                           for n, p in model.named_parameters() if p.requires_grad}
    model._ewc_dev_cache = None
    cs_cold = size_fc_ewc(model)
    model._ewc_dev_cache = ({n: f.to(DEVICE) for n, f in model.ewc_fisher.items()},
                            {n: p.to(DEVICE) for n, p in model.ewc_optparams.items()})
    cs_warm = size_fc_ewc(model)
    record("fc_ewc_cold", cs_cold, unit="model",
           per_unit_mb=cs_cold.bytes_total / MB / max(tft_mb, 1e-9),
           projected_mb=cs_cold.bytes_total / MB,
           detail="fisher + anchor, before a fit builds the device cache")
    record("fc_ewc_warm", cs_warm, unit="model",
           per_unit_mb=cs_warm.bytes_total / MB / max(tft_mb, 1e-9),
           projected_mb=cs_warm.bytes_total / MB,
           detail="DURING a fit: cpu pair + device pair both live")
    model.ewc_fisher, model.ewc_optparams, model._ewc_dev_cache = {}, {}, None

    ewc_rl = PPOEWCEngine(CONFIG["cl"]["ewc_lambda"])
    ewc_rl.fisher = {n: torch.zeros_like(p).cpu()
                     for n, p in pricer.policy.named_parameters()}
    ewc_rl.opt_params = {n: p.detach().cpu().clone()
                         for n, p in pricer.policy.named_parameters()}
    record("rl_ewc", size_ppo_ewc(ewc_rl), unit="model",
           per_unit_mb=policy_mb, projected_mb=2 * policy_mb)

    # ── Replay: the one that scales with the stream ───────────────────────────
    df = data["tft_base"]
    ds = make_tft_dataset(df, train=False, training_dataset=train_ds)
    loader = ds.to_dataloader(train=True, shuffle=True,
                              batch_size=CONFIG["forecasting"]["batch_size"],
                              num_workers=0)
    n_loader_batches = len(loader)

    buf = ForecastingReplayBuffer(CONFIG["cl"]["replay_buffer_size"])
    buf.add_from_loader(loader, task_id=0, n_samples=n_batches)
    cs_replay = size_forecasting_replay(buf)
    per_batch_mb = (cs_replay.bytes_total / MB / cs_replay.n_items
                    if cs_replay.n_items else 0.0)
    cap = CONFIG["cl"]["replay_buffer_size"]
    record("fc_replay", cs_replay, unit="batch", per_unit_mb=per_batch_mb,
           projected_mb=per_batch_mb * cap,
           detail=f"measured on {cs_replay.n_items} batches "
                  f"({cs_replay.n_units} windows); one base window yields "
                  f"{n_loader_batches} batches; cap={cap} batches")

    rl_buf = RLReplayBuffer(CONFIG["cl"]["recall_buffer_capacity"])
    obs_dim = int(pricer.observation_space.shape[0])
    import numpy as np
    for _ in range(1000):
        rl_buf.transitions_by_task[0].append(
            (np.zeros(obs_dim, dtype=np.float32), 0, 0.0,
             np.zeros(obs_dim, dtype=np.float32), False))
    cs_rl = size_rl_replay(rl_buf)
    per_tr_mb = cs_rl.bytes_total / MB / max(cs_rl.n_items, 1)
    rl_cap = CONFIG["cl"]["recall_buffer_capacity"]
    record("rl_replay", cs_rl, unit="transition", per_unit_mb=per_tr_mb,
           projected_mb=per_tr_mb * rl_cap,
           detail=f"measured on 1000 transitions at obs_dim={obs_dim}; "
                  f"cap={rl_cap}")

    out = pd.DataFrame(rows)
    out["rss_mb_at_measure"] = round(process_rss_bytes() / MB, 1)
    out["n_loader_batches_base_window"] = n_loader_batches
    return out


def report_static(df: pd.DataFrame) -> None:
    from rich.table import Table

    t = Table(title="E4 static footprint of CL state", header_style="bold cyan")
    for col, just in (("component", "left"), ("resident", "right"),
                      ("per unit", "right"), ("at capacity", "right"),
                      ("what it is", "left")):
        t.add_column(col, justify=just)
    for _, r in df.iterrows():
        t.add_row(r["component"], _fmt(r["mb"]),
                  f"{r['per_unit_mb']:.4f} MB/{r['unit']}",
                  _fmt(r["projected_at_capacity_mb"]), str(r["detail"])[:64])
    console.print(t)

    replay = df[df["component"] == "fc_replay"]
    teacher = df[df["component"] == "fc_teacher"]
    if not replay.empty and not teacher.empty:
        proj = float(replay["projected_at_capacity_mb"].iloc[0])
        tea = float(teacher["mb"].iloc[0])
        n_loader = int(df["n_loader_batches_base_window"].iloc[0])
        cap = CONFIG["cl"]["replay_buffer_size"]
        console.print(
            f"\n[bold]Replay vs SDFT (forecasting):[/bold] a full buffer is "
            f"[bold]{proj:.0f} MB[/bold] against the teacher's "
            f"[bold]{tea:.1f} MB[/bold] -> [bold]{proj / max(tea, 1e-9):.0f}x[/bold].")
        console.print(
            f"[bold]Does the cap bind?[/bold] one base window yields "
            f"{n_loader} batches and the cap is {cap}. "
            + ("[yellow]Cap does NOT bind at one window - the buffer grows with "
               "every retrain until it does.[/yellow]"
               if n_loader < cap else
               "[green]Cap binds within a single window - footprint plateaus."
               "[/green]"))


def aggregate(results_dir: Path) -> pd.DataFrame:
    """Roll up per-arm walk logs into the E4 comparison table."""
    files = sorted(results_dir.glob("memory_*.csv"))
    if not files:
        console.print(f"[yellow]no memory_*.csv in {results_dir} - run a walk "
                      f"first; RetrainController writes them automatically."
                      f"[/yellow]")
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summary = (df.groupby(["strategy", "component"], as_index=False)
                 .agg(peak_mb=("mb", "max"), final_mb=("mb", "last"),
                      n_events=("event_idx", "nunique"),
                      max_items=("n_items", "max"),
                      max_units=("n_units", "max")))
    console.print(f"[green]aggregated {len(files)} arm logs[/green]")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4: CL memory footprint.")
    ap.add_argument("--static", action="store_true",
                    help="measure per-structure cost against the base models")
    ap.add_argument("--aggregate", action="store_true",
                    help="roll up memory_*.csv written by walk arms")
    ap.add_argument("--batches", type=int, default=N_PROBE_BATCHES,
                    help="replay batches to probe for the per-batch rate")
    ap.add_argument("--out", default="e4_memory")
    args = ap.parse_args(argv)

    if not args.static and not args.aggregate:
        args.static = True

    out_dir = Path(CONFIG["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.static:
        df = measure_static(args.batches)
        report_static(df)
        path = out_dir / f"{args.out}_static.csv"
        df.to_csv(path, index=False)
        console.print(f"\n-> {path}")

    if args.aggregate:
        summary = aggregate(out_dir)
        if not summary.empty:
            console.print(summary.to_string(index=False))
            path = out_dir / f"{args.out}_walk.csv"
            summary.to_csv(path, index=False)
            console.print(f"\n-> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

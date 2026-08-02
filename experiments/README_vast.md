# Running E2 on a rented GPU box (vast.ai)

## Why rent at all

Not for speed alone. The measured constraint on the laptop is **host RAM**, not
the GPU: the TFT is 4.40 MB of parameters and peaks at 17.59 MB of VRAM during a
fit, but a replay buffer reaches **~3.1 GB of host RAM**, and the laptop has
~4 GB free. One cell at a time is survivable; running cells concurrently is not.

That is what a rented box buys — enough RAM to run cells in parallel, which is
where the real speedup is. A 3090 is maybe 2–3× faster per cell (the models are
tiny and much of each check is pandas and PPO env rollouts, which a bigger GPU
does not touch), but running 4 cells at once is 4×, and the two multiply.

| | wall clock |
|---|---|
| laptop, serial, 12 cells | ~14 h (and RAM-tight) |
| 3090, 4 concurrent | **~4–5 h** |

At typical 3090 rates that is **$2–3**.

## What to rent

| | minimum | comfortable |
|---|---|---|
| GPU | any 12 GB+ card | RTX 3090 / 4090 |
| **RAM** | **20 GB** (4 cells) | **32 GB+** |
| vCPU | 4 | 8–16 |
| Disk | 30 GB | 40 GB |

**RAM is the spec to filter on, not the GPU.** Budget ~5 GB per concurrent cell.
A cheap box with a fast card and 16 GB of RAM will swap and finish slower than a
slower card with 32 GB.

## Step 1 — get the code and data onto the box

The code is in git. The data and checkpoints are **not** (deliberately — they are
gitignored), so they transfer separately. Total payload is **53 MB**.

```bash
# on the box
git clone <your-repo-url> fyp && cd fyp
```

```powershell
# from the laptop, in the project root
tar -czf payload.tar.gz data/processed_m5_v3 outputs/drift/checkpoints/base_cover
scp -P <port> payload.tar.gz root@<host>:~/fyp/
```

```bash
# on the box
cd ~/fyp && tar -xzf payload.tar.gz && rm payload.tar.gz
```

## Step 2 — environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Most vast images ship torch already. If `pip` pulls a torch that does not match
the image's CUDA, the preflight will catch it at the CUDA check — install the
matching wheel from pytorch.org rather than fighting `requirements.txt`.

## Step 3 — preflight (do not skip)

```bash
python -m experiments.vast_preflight --concurrency 4
```

This is not a formality. It loads the **actual base checkpoint** and confirms the
TFT, PPO and calibration restore, because a bad upload or a mismatched torch is
the failure that wastes the whole rental — and it cannot be detected by checking
that files exist. It also verifies the dataset carries `unmet_demand`, which is
E2's treatment signal: on the v1 dataset the treatment is empty and the sweep
would silently measure nothing.

Exit code 0 means go.

## Step 4 — run

```bash
tmux new -s e2                      # so a dropped SSH session does not kill it
bash experiments/run_e2.sh
# detach: Ctrl-B then D      reattach: tmux attach -t e2
```

Tune with environment variables:

```bash
CONCURRENCY=6 bash experiments/run_e2.sh        # needs ~30 GB RAM
ARMS="frozen sdft" bash experiments/run_e2.sh   # subset
CENSORING="none escrow_quota" bash experiments/run_e2.sh
```

Progress is one line per cell on stdout; full logs land in
`outputs/drift/logs/e2/<censoring>_<arm>.log`.

**If it dies, just run it again.** The driver is resumable — a cell whose
`probe_scores_<arm>.json` exists is complete and gets skipped. That is the
intended recovery path. Do **not** pass `--force`; it re-runs finished cells.

## Step 5 — get the results back

Results are small (CSV/JSON).

```bash
tar -czf results.tar.gz outputs/drift/results/e2
```
```powershell
scp -P <port> root@<host>:~/fyp/results.tar.gz .
```

The file to read is `outputs/drift/results/e2/e2_summary.csv`, one row per
(censoring, arm). The headline column is **`forgetting_mase_base_era`** — each
arm's final model re-scored on base-era probe windows, differenced against
`frozen`, so `>0` means the arm lost pre-drift knowledge. Not `walk_mase_mean`:
mean walk error confounds adapting to the new regime with retaining the old one,
and the second is the question.

`memory_<arm>.csv` in each cell directory is E4's per-event record — that comes
out of the same runs, at no extra cost.

**Destroy the instance once the results are off it.** Storage bills while stopped.

## What to sanity-check before trusting the numbers

- `frozen` should show `forgetting_mase_base_era` of exactly 0 in every cell — it
  is the anchor, so a non-zero value means the tables were built against the
  wrong reference.
- `frozen` should be *identical* across censoring levels. It never retrains, so
  it never ingests censored data; if it moves, censoring has leaked into the
  evaluation path and `assert_uncensored` should have caught it.
- `naive`, `replay` and `sdft` should show a non-zero retrain count. Zero
  retrains means no drift fired and the arms are all secretly `frozen`.

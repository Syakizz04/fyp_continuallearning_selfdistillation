# FYP Continual Learning Dynamic Pricing

This project runs an end-to-end continual learning experiment for demand forecasting and reinforcement-learning-based dynamic pricing.

## Project Structure

```text
fyp/
  fyp_pipeline/          # reusable Python modules
  notebooks/             # interactive experiment notebook
  data/processed/        # processed CSV inputs used by the experiment
  dataset_generator/     # synthetic dataset generation code
  outputs/               # checkpoints, logs, results, and plots
  requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name fyp --display-name "FYP"
```

## Run With Notebook

Open:

```text
notebooks/experiment.ipynb
```

Select the `FYP` kernel, then run the cells in order.

## Run As Script

From the project root:

```powershell
python -m fyp_pipeline.experiment_runner
```

## Vast.ai

On a Vast.ai instance, place the project folder somewhere like `/workspace/fyp`. The CSVs should be under:

```text
/workspace/fyp/data/processed/
```

Then in the notebook:

```python
DATA_DIR = "/workspace/fyp/data/processed"
OUTPUT_DIR = "/workspace/fyp/outputs"
configure_vast_ai(DATA_DIR, OUTPUT_DIR, require_gpu=True)
```

The experiment writes checkpoints, logs, result tables, and plots under `outputs/`.

# ProcessPath_AI

Process mining on the BPI Challenge 2020 Travel Permit dataset — bottleneck analysis, conformance checking, SHAP-explained early warning model, and temporal cross-validation.

**Dataset:** 7,065 cases · 86,581 events · 51 activities · 18 months (TU/e, 2017–2018)

![Summary Dashboard](outputs/figures/report_dashboard.png)

---

## Results summary

| Finding | Detail |
|---|---|
| 991 cases (14%) permanently stuck | Last event is `Send Reminder` — median duration 134d vs 63d for resolved cases |
| 17.1% travel-ordering violations | 746 Type A (departed before permit submitted), 583 Type B (departed before approval) |
| Scheduling dominates duration | 69% of case duration is voluntary employee scheduling, not admin processing |
| Early warning model at k=8 events | AUC **0.810** (leakage-free) — deployable at `Permit FINAL_APPROVED` |
| Data drift confirmed | `elapsed_days` feature halved from 2017Q1 → 2018Q4; k-fold overstates AUC by +0.048 |
| Temporal leakage identified & corrected | `elapsed_days` alone achieves AUC 0.833 — excluded from deployed model (Notebook 10) |

---

## Notebooks

Run in order. Each notebook is self-contained and writes its outputs to `outputs/`.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_initial_exploration.ipynb` | Case/event stats, variant frequency, time coverage |
| 02 | `02_process_structure.ipynb` | Directly-Follows Graph, transition matrix, happy path |
| 03 | `03_process_discovery.ipynb` | Inductive Miner and Heuristics Miner Petri nets |
| 04 | `04_bottleneck_analysis.ipynb` | Waiting/service time split, stuck cases, scheduling vs admin delay |
| 05 | `05_conformance_analysis.ipynb` | Token replay fitness, travel-ordering violations by department |
| 06 | `06_predictive_analytics.ipynb` | XGBoost / RF / LogReg — AUC 0.974 on complete features |
| 07 | `07_shap_prefix.ipynb` | SHAP explanations + prefix-based early warning (k=1–20) |
| 08 | `08_temporal_cv.ipynb` | Temporal cross-validation, optimism bias, feature drift, concept drift |
| 09 | `09_final_report.ipynb` | 6-panel dashboard, priority matrix, 5 findings, 5 recommendations |
| 10 | `10_leakage_calibration.ipynb` | Leakage audit, ablation study, calibration (Brier score, reliability diagram) |

---

## Setup

### 1. Python version

The notebooks require **Python 3.13**. The stack (numpy 2.x, xgboost 3.x, shap 0.52, pm4py 2.7) does not work on Python 3.12 with a standard Anaconda environment due to a numpy ABI conflict.

```bash
# Verify you have Python 3.13
python3.13 --version
```

If you need to install it: https://www.python.org/downloads/

### 2. Clone the repo

```bash
git clone https://github.com/djimrastephane/ProcessPath_AI.git
cd ProcessPath_AI
```

### 3. Create a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 4. Install dependencies

**To run the Streamlit app only:**
```bash
pip install -r requirements.txt
```

**To run the notebooks** (includes pm4py, jupyter):
```bash
pip install -r requirements-dev.txt
```

### 5. Get the data

The raw event log is not included in this repo (33 MB binary). Download it from the 4TU Research Data repository:

**https://data.4tu.nl/articles/dataset/BPI_Challenge_2020/12703980**

> van Dongen, Boudewijn (2020): BPI Challenge 2020: Travel Permit Data. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550

Place the file at:

```
data/raw/PermitLog.xes
```

---

## Running the notebooks

### Register the kernel (once)

```bash
python -m ipykernel install --user --name python313 --display-name "Python 3.13"
```

### Interactive (browser)

```bash
jupyter notebook
```

Open notebooks in order from the `notebooks/` directory. Select kernel **Python 3.13** when prompted.

### Headless (execute all, write outputs)

```bash
for nb in notebooks/0{1..9}*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    --ExecutePreprocessor.kernel_name=python313 \
    "$nb"
done
```

Each notebook writes figures to `outputs/figures/` and tables to `outputs/tables/`. Pre-computed outputs are already committed so you can browse results without re-running.

---

## Repository structure

```
ProcessPath_AI/
├── notebooks/          # 9 analysis notebooks (run in order)
├── src/                # Shared loader and helper functions
│   ├── load_event_log.py
│   ├── inspect_log.py
│   └── process_summary.py
├── outputs/
│   ├── figures/        # 44 PNG charts (pre-computed)
│   └── tables/         # 27 CSV tables (pre-computed)
├── data/
│   └── README.md       # Data download instructions
├── requirements.txt
└── main.py             # CLI entry point (prints dataset summary)
```

---

## Tech stack

| Library | Version | Purpose |
|---|---|---|
| pm4py | 2.7.22.5 | XES loading, DFG, Petri nets, conformance |
| pandas / numpy | 2.x | Data wrangling |
| scikit-learn | ≥1.3 | Preprocessing, CV, metrics |
| xgboost | 3.x | Gradient boosting classifier |
| shap | 0.52 | Feature attribution (TreeExplainer) |
| matplotlib | ≥3.7 | All figures |

---

## Citation

If you use this dataset, please cite:

> van Dongen, Boudewijn (2020): BPI Challenge 2020: Travel Permit Data. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550

---

## License

MIT

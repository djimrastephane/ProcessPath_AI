# ProcessPath_AI

Process mining and workflow analytics on the BPI Challenge 2020 Travel Permit event log.

## Dataset

**BPI Challenge 2020 — Travel Permit Log (`PermitLog.xes`)**

Real-life event log from the reimbursement and travel permit process at Eindhoven University of Technology (TU/e). Covers 2017–2018 across multiple departments. Events capture permit submissions, approvals, rejections, and reimbursement requests by staff.

## Project Roadmap

| Phase | Notebook | Status |
|-------|----------|--------|
| 1 — Exploration | `01_initial_exploration.ipynb` | Active |
| 2 — Process discovery | `02_process_structure.ipynb`, `03_process_discovery.ipynb` | Active |
| 3 — Bottleneck analysis | `04_bottleneck_analysis.ipynb` | Planned |
| 4 — Conformance checking | `05_conformance_analysis.ipynb` | Planned |
| 5 — Predictive process analytics | `06_predictive_analytics.ipynb` | Planned |
| 6 — Interactive application | Streamlit app in `app/` | Planned |

## Current Phase

**Phase 1 & 2:** Data exploration and process structure analysis using PM4Py and pandas. No ML, no dashboards — only validated, reliable exploratory outputs.

## Repository Structure

```
ProcessPath_AI/
├── data/raw/PermitLog.xes
├── notebooks/
│   ├── 01_initial_exploration.ipynb
│   ├── 02_process_structure.ipynb
│   ├── 03_process_discovery.ipynb
│   ├── 04_bottleneck_analysis.ipynb
│   ├── 05_conformance_analysis.ipynb
│   └── 06_predictive_analytics.ipynb
├── src/
│   ├── __init__.py
│   ├── load_event_log.py
│   ├── inspect_log.py
│   └── process_summary.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── requirements.txt
└── main.py
```

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Execution

**Run the CLI summary:**
```bash
python main.py
```

**Run notebooks in order:**
```bash
jupyter notebook notebooks/01_initial_exploration.ipynb
jupyter notebook notebooks/02_process_structure.ipynb
jupyter notebook notebooks/03_process_discovery.ipynb
```

Outputs (CSV tables, PNG figures) are written to `outputs/`.

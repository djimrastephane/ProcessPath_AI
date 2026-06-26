from pathlib import Path
import pm4py
import pandas as pd


def load_xes_log(path: str | Path, legacy: bool = False):
    """Load a XES file. Pass legacy=True to get an EventLog object (required for token replay)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XES file not found: {path}")
    return pm4py.read_xes(str(path), return_legacy_log_object=legacy)


def load_xes_log_legacy(path: str | Path):
    """Convenience wrapper — always returns a legacy EventLog object."""
    return load_xes_log(path, legacy=True)


def convert_to_dataframe(log) -> pd.DataFrame:
    df = pm4py.convert_to_dataframe(log)
    df = df.reset_index(drop=True)
    return df

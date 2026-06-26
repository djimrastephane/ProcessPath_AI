import pandas as pd


def summarize_columns(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notnull().sum(),
        "null": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
    })
    return summary


def summarize_cases(df: pd.DataFrame, case_col: str = "case:concept:name") -> dict:
    n_cases = df[case_col].nunique()
    n_events = len(df)
    events_per_case = df.groupby(case_col).size()
    return {
        "n_cases": n_cases,
        "n_events": n_events,
        "events_per_case_mean": events_per_case.mean(),
        "events_per_case_median": events_per_case.median(),
        "events_per_case_min": events_per_case.min(),
        "events_per_case_max": events_per_case.max(),
    }


def summarize_activities(df: pd.DataFrame, activity_col: str = "concept:name") -> pd.DataFrame:
    freq = df[activity_col].value_counts().reset_index()
    freq.columns = ["activity", "count"]
    freq["pct"] = (freq["count"] / len(df) * 100).round(2)
    return freq

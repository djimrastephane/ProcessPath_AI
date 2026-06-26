import pandas as pd


def get_case_durations(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    time_col: str = "time:timestamp",
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    durations = df.groupby(case_col)[time_col].agg(["min", "max"])
    durations["duration_hours"] = (durations["max"] - durations["min"]).dt.total_seconds() / 3600
    durations["duration_days"] = durations["duration_hours"] / 24
    durations = durations.rename(columns={"min": "start_time", "max": "end_time"})
    return durations.reset_index()


def get_activity_frequency(
    df: pd.DataFrame, activity_col: str = "concept:name"
) -> pd.DataFrame:
    freq = df[activity_col].value_counts().reset_index()
    freq.columns = ["activity", "count"]
    freq["pct"] = (freq["count"] / len(df) * 100).round(2)
    return freq


def get_variants(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    time_col: str = "time:timestamp",
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values([case_col, time_col])
    variant_series = df.groupby(case_col)[activity_col].apply(lambda x: " -> ".join(x))
    variant_counts = variant_series.value_counts().reset_index()
    variant_counts.columns = ["variant", "case_count"]
    variant_counts["pct"] = (variant_counts["case_count"] / variant_counts["case_count"].sum() * 100).round(2)
    return variant_counts

from pathlib import Path
from src.load_event_log import load_xes_log, convert_to_dataframe
from src.inspect_log import summarize_columns, summarize_cases, summarize_activities
from src.process_summary import get_case_durations, get_activity_frequency, get_variants

DATA_PATH = Path("data/raw/PermitLog.xes")


def main():
    print("Loading event log...")
    log = load_xes_log(DATA_PATH)
    df = convert_to_dataframe(log)
    print(f"Loaded {len(df)} events across {df['case:concept:name'].nunique()} cases.\n")

    print("--- Column Summary ---")
    print(summarize_columns(df).to_string())

    print("\n--- Case Summary ---")
    for k, v in summarize_cases(df).items():
        print(f"  {k}: {v}")

    print("\n--- Top 10 Activities ---")
    print(get_activity_frequency(df).head(10).to_string(index=False))

    print("\n--- Top 5 Variants ---")
    print(get_variants(df).head(5).to_string(index=False))


if __name__ == "__main__":
    main()

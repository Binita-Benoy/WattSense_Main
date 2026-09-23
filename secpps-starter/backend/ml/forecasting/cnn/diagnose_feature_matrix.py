"""
Checks the actual feature matrix for real, specific data quality issues,
rather than guessing whether regenerating would help.
"""
import os

import pandas as pd
import numpy as np

FEATURE_MATRIX_PATH = os.getenv(
    "FEATURE_MATRIX_PATH",
    os.path.join(os.path.dirname(__file__), "feature_matrix.csv"),
)

CHECK_COLUMNS = [
    "heating_degree", "cooling_degree", "is_public_holiday",
    "controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh",
]


def main():
    print(f"Loading {FEATURE_MATRIX_PATH}...")
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Total rows: {len(df):,}\n")

    dupes = df.duplicated(subset=["household_id", "timestamp"]).sum()
    print(f"=== Duplicate (household_id, timestamp) rows: {dupes:,} ===")
    if dupes > 0:
        print("  ^ REAL PROBLEM: duplicates will corrupt lag features and double-count sequences.\n")
    else:
        print("  OK -- no duplicates.\n")

    print("=== NaN counts in newer columns ===")
    for col in CHECK_COLUMNS:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            pct = n_nan / len(df) * 100
            flag = "  <-- check this" if pct > 5 else ""
            print(f"  {col}: {n_nan:,} ({pct:.1f}%){flag}")
        else:
            print(f"  {col}: COLUMN MISSING ENTIRELY")
    print()

    print("=== Per-household breakdown ===")
    print(f"{'household_id':<14}{'rows':<8}{'min_kwh':<10}{'max_kwh':<10}{'exact_0':<10}{'near_0(<0.05)':<15}{'negative':<10}")
    summary_rows = []
    for hid, g in df.groupby("household_id"):
        row_count = len(g)
        min_kwh = g["kwh"].min()
        max_kwh = g["kwh"].max()
        exact_zero = (g["kwh"] == 0).sum()
        near_zero = (g["kwh"].abs() < 0.05).sum()
        negative = (g["kwh"] < 0).sum()
        summary_rows.append({
            "household_id": hid, "rows": row_count, "min_kwh": min_kwh, "max_kwh": max_kwh,
            "exact_zero": exact_zero, "near_zero": near_zero, "negative": negative,
        })
        flag = ""
        if row_count < 18000 or row_count > 22000:
            flag += " <-- unusual row count"
        if negative > 0:
            flag += " <-- NEGATIVE CONSUMPTION"
        if exact_zero > row_count * 0.05:
            flag += " <-- many exact zeros"
        print(f"{hid:<14}{row_count:<8}{min_kwh:<10.4f}{max_kwh:<10.2f}{exact_zero:<10}{near_zero:<15}{negative:<10}{flag}")

    print(f"\n=== Overall totals ===")
    total_exact_zero = sum(r["exact_zero"] for r in summary_rows)
    total_near_zero = sum(r["near_zero"] for r in summary_rows)
    total_negative = sum(r["negative"] for r in summary_rows)
    print(f"Total exact-zero kwh rows across all households: {total_exact_zero:,} ({total_exact_zero/len(df)*100:.2f}%)")
    print(f"Total near-zero (<0.05) kwh rows: {total_near_zero:,} ({total_near_zero/len(df)*100:.2f}%)")
    print(f"Total negative kwh rows: {total_negative:,}")

    print(f"\n=== Interpretation ===")
    if total_negative > 0:
        print("REAL PROBLEM: negative consumption values exist -- consumption should never be negative. "
              "Worth checking with Alka whether this is a data pipeline bug.")
    if total_near_zero / len(df) > 0.1:
        print(f"NOTE: {total_near_zero/len(df)*100:.1f}% of all rows are near-zero (<0.05 kWh). "
              "This alone explains why MAPE is so unstable at this pooled scale -- likely a "
              "genuine data characteristic (some hours genuinely have very low consumption), "
              "not a bug. The threshold-masked MAPE fix is the correct response, not a "
              "data regeneration.")
    if dupes == 0 and total_negative == 0:
        print("No structural data corruption found (no duplicates, no negative values). "
              "The near-zero-MAPE issue is very likely a genuine data characteristic, "
              "not a data quality bug -- regenerating the feature matrix would NOT fix this, "
              "since it would recompute from the same underlying real readings.")


if __name__ == "__main__":
    main()

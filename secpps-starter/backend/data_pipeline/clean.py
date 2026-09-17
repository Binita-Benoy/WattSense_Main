from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "interim" / "sgsc_selected_households.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "sgsc_hourly_clean.csv"

ENERGY_COLUMNS = [
    "GENERAL_SUPPLY_KWH",
    "CONTROLLED_LOAD_KWH",
    "GROSS_GENERATION_KWH",
    "NET_GENERATION_KWH",
    "OTHER_KWH",
]


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["READING_DATETIME"] = pd.to_datetime(
        df["READING_DATETIME"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    invalid_count = df["READING_DATETIME"].isna().sum()
    print(f"Invalid timestamps: {invalid_count}")

    df = df.dropna(subset=["READING_DATETIME"])

    return df


def convert_energy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ENERGY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.drop_duplicates(
        subset=["CUSTOMER_ID", "READING_DATETIME"],
        keep="first",
    )

    removed = before - len(df)
    print(f"Duplicate readings removed: {removed}")

    return df


def remove_outliers(series: pd.Series, window: int = 48) -> pd.Series:
    rolling_mean = series.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean()

    rolling_std = series.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).std()

    lower = rolling_mean - (3 * rolling_std)
    upper = rolling_mean + (3 * rolling_std)

    outlier_mask = (series < lower) | (series > upper)

    print(f"Outliers detected: {outlier_mask.sum()}")

    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan

    return cleaned


def preprocess_household(group: pd.DataFrame) -> pd.DataFrame:
    household_id = group["CUSTOMER_ID"].iloc[0]

    group = group.sort_values("READING_DATETIME").copy()
    group = group.set_index("READING_DATETIME")

    start = group.index.min()
    end = group.index.max()

    expected_index = pd.date_range(
        start=start,
        end=end,
        freq="30min",
    )

    original_count = len(group)

    group = group.reindex(expected_index)

    missing_intervals = len(group) - original_count

    print(
        f"Household {household_id}: "
        f"original={original_count}, "
        f"expected={len(group)}, "
        f"missing_intervals={missing_intervals}"
    )

    group["CUSTOMER_ID"] = household_id

    for col in ENERGY_COLUMNS:
        if col not in group.columns:
            continue

        group[col] = group[col].interpolate(
            method="time",
            limit_direction="both",
        )

        group[col] = remove_outliers(group[col])

        group[col] = group[col].interpolate(
            method="time",
            limit_direction="both",
        )

    group.index.name = "READING_DATETIME"

    return group.reset_index()


def resample_hourly(df):
    """
    Resample SGSC 30-minute interval energy readings to hourly data.

    SGSC energy values are kWh consumed/generated during each
    30-minute interval, so hourly values must be calculated using SUM.
    """

    hourly_frames = []

    for customer_id, group in df.groupby("CUSTOMER_ID"):

        group = group.copy()

        # READING_DATETIME must be the DatetimeIndex for resampling.
        if "READING_DATETIME" in group.columns:
            group = group.set_index("READING_DATETIME")

        # Keep only energy columns before resampling.
        energy = group[ENERGY_COLUMNS].copy()

        # Two 30-minute kWh intervals -> one hourly kWh value.
        hourly = energy.resample("1h").sum()

        hourly["CUSTOMER_ID"] = customer_id

        hourly = hourly.reset_index()

        hourly_frames.append(hourly)

    hourly_df = pd.concat(
        hourly_frames,
        ignore_index=True
    )

    return hourly_df


def validate_output(df: pd.DataFrame) -> None:
    print("\nValidation")

    print(f"Rows: {len(df):,}")
    print(f"Households: {df['CUSTOMER_ID'].nunique()}")

    print(
        "Missing GENERAL_SUPPLY_KWH:",
        df["GENERAL_SUPPLY_KWH"].isna().sum(),
    )

    print(
        "Duplicate customer/timestamp pairs:",
        df.duplicated(
            subset=["CUSTOMER_ID", "READING_DATETIME"]
        ).sum(),
    )

    print(
        "Negative GENERAL_SUPPLY_KWH:",
        (df["GENERAL_SUPPLY_KWH"] < 0).sum(),
    )

    non_hour_aligned = (
        (df["READING_DATETIME"].dt.minute != 0)
        | (df["READING_DATETIME"].dt.second != 0)
    ).sum()

    print("Non-hour-aligned timestamps:", non_hour_aligned)


def clean_sgsc(
    input_file: Path = RAW_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    print(f"Reading: {input_file}")

    df = pd.read_csv(input_file)

    # Remove leading/trailing whitespace from SGSC column names.
    df.columns = df.columns.str.strip()

    print(f"Input rows: {len(df):,}")

    df = parse_timestamps(df)
    df = convert_energy_columns(df)
    df = remove_duplicates(df)

    processed_households = []

    for _, group in df.groupby("CUSTOMER_ID"):
        processed = preprocess_household(group)
        processed_households.append(processed)

    cleaned = pd.concat(
        processed_households,
        ignore_index=True,
    )

    hourly = resample_hourly(cleaned)

    hourly = hourly.sort_values(
        ["CUSTOMER_ID", "READING_DATETIME"]
    ).reset_index(drop=True)

    validate_output(hourly)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved cleaned hourly SGSC data to:")
    print(output_file)

    return hourly

if __name__ == "__main__":
    clean_sgsc()
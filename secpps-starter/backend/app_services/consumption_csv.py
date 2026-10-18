
import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

MIN_HOURS = 24 * 14             # hard minimum: two weeks of hourly data
MONTH_FORECAST_MIN_HOURS = 913  # 168 (lag rows) + 24 (lookback) + 720 (1 month) + 1
MAX_HOURS = 24 * 730            # use at most the most recent two years
MAX_GAP_HOURS = 24              # longest run of missing hours we will interpolate
MAX_MISSING_FRACTION = 0.10
STALE_DAYS = 7


class CsvValidationError(ValueError):
    """Raised with a message that is safe to show to the user."""


def _read_table(raw: bytes) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise CsvValidationError("The file is empty.")
    text = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        # sep=None lets pandas detect comma / tab / semicolon from the header line
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
    except Exception:
        raise CsvValidationError(
            "We couldn't read this file as a CSV. Please upload a plain .csv file "
            "with the columns: timestamp, kwh."
        )
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _regularity(parsed: pd.Series):
    """(share of rows that parsed, share of steps that match the most common step)."""
    valid = parsed.dropna()
    fraction = len(valid) / max(len(parsed), 1)
    if len(valid) < 3:
        return (round(fraction, 6), 0.0)
    steps = valid.diff().dropna()
    modes = steps.mode()
    regular = float((steps == modes.iloc[0]).mean()) if len(modes) else 0.0
    return (round(fraction, 6), round(regular, 6))


def _to_datetime(s: pd.Series, dayfirst: bool) -> pd.Series:
    """Fast strict parse first; anything it can't read (e.g. midnight rows exported
    without a time) is retried one by one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")
        retry = parsed.isna() & s.notna()
        if retry.any():
            parsed[retry] = pd.to_datetime(s[retry], dayfirst=dayfirst, format="mixed", errors="coerce")
    return parsed


_DAY_MONTH_FIELDS = r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}"
_YEAR_FIRST = r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"


def _parse_timestamps(series: pd.Series):
    """Returns (timestamps, date_format_label, guessed_without_evidence)."""
    s = series.astype(str).str.strip()
    ambiguous = False

    if s.str.match(_YEAR_FIRST, na=False).mean() > 0.9:
        # 2025-06-01 00:00 or 2025/06/01 00:00: year first, so there is nothing to guess
        s = s.str.replace(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})", r"\1-\2-\3", regex=True)
        parsed = _to_datetime(s, dayfirst=False)
        label = "year-month-day"
    else:
        fields = s.str.extract(_DAY_MONTH_FIELDS).astype(float)
        have = fields[0].notna()
        if have.mean() < 0.5:
            # unusual layout, e.g. "6 Jan 2025 00:00": month names leave nothing ambiguous
            parsed, label = _to_datetime(s, dayfirst=True), "auto-detected"
        else:
            first_big = bool((fields[0][have] > 12).any())   # only a DAY can exceed 12
            second_big = bool((fields[1][have] > 12).any())
            if first_big and second_big:
                raise CsvValidationError(
                    "The dates in the file mix day/month and month/day layouts, so we can't tell "
                    "which is which. Please use one layout throughout (or YYYY-MM-DD)."
                )
            if first_big:
                dayfirst, label = True, "day/month/year"
            elif second_big:
                dayfirst, label = False, "month/day/year"
            else:
                # every value is 12 or less in both positions: use whichever reading gives
                # evenly spaced hourly steps; on a tie assume day/month (Australian format)
                candidates = {"day/month/year": _to_datetime(s, True),
                              "month/day/year": _to_datetime(s, False)}
                scores = {name: _regularity(p) for name, p in candidates.items()}
                ambiguous = scores["day/month/year"] == scores["month/day/year"]
                label = "day/month/year" if ambiguous else max(scores, key=scores.get)
                dayfirst = label == "day/month/year"
            parsed = _to_datetime(s, dayfirst)

    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)  # keep the wall-clock time, drop the offset
    return parsed, label, ambiguous


def normalize_consumption_csv(raw: bytes):
    """
    Returns (hourly_df, info, notes):
      hourly_df  DataFrame with columns timestamp, kwh (one row per hour, no gaps)
      info       small dict describing what was found / changed
      notes      list of human-readable warnings to show the user
    Raises CsvValidationError with a user-friendly message if the file can't be used.
    """
    notes = []
    df = _read_table(raw)

    missing_cols = [c for c in ("timestamp", "kwh") if c not in df.columns]
    if missing_cols:
        found = ", ".join(list(df.columns)[:6]) or "none"
        raise CsvValidationError(
            f"The file needs the columns 'timestamp' and 'kwh' "
            f"(missing: {', '.join(missing_cols)}). Columns found: {found}."
        )

    rows_uploaded = len(df)
    timestamps, date_format, ambiguous = _parse_timestamps(df["timestamp"])
    kwh = pd.to_numeric(df["kwh"].astype(str).str.strip(), errors="coerce")

    data = pd.DataFrame({"timestamp": timestamps, "kwh": kwh})
    bad = data["timestamp"].isna() | data["kwh"].isna()
    if bad.mean() > 0.10:
        first_bad = df.loc[bad.idxmax()]
        raise CsvValidationError(
            f"{int(bad.sum())} of {rows_uploaded} rows have a date or kWh value we couldn't read "
            f"(first one: timestamp '{first_bad['timestamp']}', kwh '{first_bad['kwh']}'). "
            f"Please check the date format and that kwh is a number."
        )
    dropped_rows = int(bad.sum())
    if dropped_rows:
        notes.append(f"{dropped_rows} row(s) with an unreadable date or kWh value were skipped.")
    data = data[~bad]

    if (data["kwh"] < 0).any():
        raise CsvValidationError(
            "Some kwh values are negative. Please upload electricity CONSUMED (imported) per "
            "interval; export or solar-feed-in values aren't supported."
        )

    data = data.sort_values("timestamp")
    duplicates = int(data["timestamp"].duplicated().sum())
    if duplicates:
        notes.append(f"{duplicates} duplicate timestamp(s) found; the first reading for each was kept.")
        data = data.drop_duplicates("timestamp", keep="first")

    if len(data) < 3:
        raise CsvValidationError("The file has too few readings to forecast from.")

    # --- reading interval -------------------------------------------------
    median_step = data["timestamp"].diff().dropna().median()
    minutes = median_step.total_seconds() / 60
    if minutes > 60.5:
        every = (f"{minutes / 60:.0f} hours" if minutes < 1440 else
                 "day" if abs(minutes - 1440) < 1 else f"{minutes / 1440:.0f} days")
        raise CsvValidationError(
            f"The file needs hourly readings (or finer, such as every 30 minutes), but it looks "
            f"like one reading every {every}. Hourly data is needed to forecast usage by the hour."
        )
    if minutes < 1:
        raise CsvValidationError("The readings are less than a minute apart, which isn't supported.")

    aggregated = minutes < 59.5
    expected_per_hour = 1
    if aggregated:
        interval = int(round(minutes))
        if 60 % interval != 0:
            raise CsvValidationError(
                f"Readings every {interval} minutes don't divide evenly into an hour. "
                f"Please upload hourly, 30-minute, 15-minute or 5-minute data."
            )
        expected_per_hour = 60 // interval
        notes.append(f"Readings every {interval} minutes were summed into hourly kWh.")

    series = data.set_index("timestamp")["kwh"]
    sums = series.resample("h").sum(min_count=1)
    counts = series.resample("h").count()
    hourly = sums.where(counts >= expected_per_hour)  # hours with missing intervals count as gaps

    first, last = hourly.first_valid_index(), hourly.last_valid_index()
    hourly = hourly.loc[first:last]
    hourly = hourly.reindex(pd.date_range(first, last, freq="h"))

    # --- gaps ---------------------------------------------------------------
    missing = hourly.isna()
    n_missing = int(missing.sum())
    if n_missing:
        if n_missing / len(hourly) > MAX_MISSING_FRACTION:
            raise CsvValidationError(
                f"{n_missing} of {len(hourly)} hours have no reading ({n_missing / len(hourly):.0%}). "
                f"Too much of the data is missing to forecast reliably."
            )
        run_id = (missing != missing.shift()).cumsum()
        run_lengths = missing.groupby(run_id).transform("sum").where(missing, 0)
        if run_lengths.max() > MAX_GAP_HOURS:
            gap_start = run_lengths.idxmax()
            raise CsvValidationError(
                f"There is a gap of {int(run_lengths.max())} hours with no readings starting "
                f"{gap_start:%d %b %Y %H:%M}. Gaps longer than {MAX_GAP_HOURS} hours can't be filled in."
            )
        hourly = hourly.interpolate(method="linear", limit_direction="both")
        notes.append(f"{n_missing} missing hour(s) were filled in by interpolation.")

    # --- length -------------------------------------------------------------
    if len(hourly) < MIN_HOURS:
        raise CsvValidationError(
            f"At least {MIN_HOURS // 24} days of hourly data are needed; the file has "
            f"{len(hourly) / 24:.1f} days."
        )
    if len(hourly) > MAX_HOURS:
        hourly = hourly.iloc[-MAX_HOURS:]
        notes.append("Only the most recent two years of data were used.")
    if len(hourly) < MONTH_FORECAST_MIN_HOURS:
        notes.append(
            f"With only {len(hourly) / 24:.0f} days of data the 1-month forecast isn't available "
            f"(it needs about {MONTH_FORECAST_MIN_HOURS / 24:.0f}+ days). A full year gives the best results."
        )

    # --- dates that don't make sense ------------------------------------------
    now = datetime.now()
    end = hourly.index.max()
    if end > now + timedelta(days=2):
        raise CsvValidationError(
            f"The latest timestamp in the file is {end:%d %b %Y}, which is in the future. "
            f"Please check the date format (day/month vs month/day)."
        )
    if now - end > timedelta(days=STALE_DAYS):
        notes.append(
            f"Your data ends on {end:%d %b %Y}. The 24-hour and 7-day forecasts use the live weather "
            f"forecast, which only covers the coming days, so for older data they fall back to a single "
            f"fixed weather value. For best results, upload data up to today."
        )
    if ambiguous:
        notes.append(
            "The dates in your file could be read as day/month or month/day; we assumed day/month/year."
        )

    out = hourly.round(4).rename("kwh").rename_axis("timestamp").reset_index()
    info = {
        "rows_uploaded": int(rows_uploaded),
        "hours_used": int(len(out)),
        "start": out["timestamp"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S"),
        "end": out["timestamp"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S"),
        "date_format": date_format,
        "aggregated_to_hourly": bool(aggregated),
        "interpolated_hours": n_missing,
        "skipped_rows": dropped_rows,
    }
    return out, info, notes
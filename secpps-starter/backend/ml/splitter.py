from logging_utils import log_print
import logging
logger = logging.getLogger()
# anomaly_detection/splitter.py
# ─────────────────────────────────────────────────────────────────────────────
# Chronological train / val / test split for time-series data
#
# WHY NOT RANDOM SPLIT?
#   Energy consumption is a time series.
#   A random split allows the model to "see" future data
#   during training — future patterns leak into the past.
#   This inflates accuracy and produces results that would
#   never generalise to real deployment.
#
# WHY CHRONOLOGICAL?
#   In production, the model always predicts the future
#   from the past. The evaluation must mirror this:
#   train on early data, evaluate on later data.
#   This gives an honest estimate of real-world performance.
#
# SPLIT:  70% TRAIN  |  15% VALIDATION  |  15% TEST
#         ← learns normal →  ← early stop →  ← final eval →
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    COL_HOUSEHOLD, SPLIT_INFO_CSV, SPLIT_GRAPH, OUT_GRAPHS
)


def split_household_chronological(
    df_hh: pd.DataFrame
) -> tuple:
    """
    Split one household's data chronologically.

    CRITICAL RULES:
    1. Sort by timestamp first — always
    2. Split by index position not by date
       (handles unequal spacing)
    3. No shuffling — ever
    4. Train ends strictly before val starts
    5. Val ends strictly before test starts

    Args:
        df_hh: single household DataFrame with DatetimeIndex,
               already sorted by time

    Returns:
        train, val, test DataFrames
    """
    df_hh = df_hh.sort_index()
    n     = len(df_hh)

    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df_hh.iloc[:train_end]
    val   = df_hh.iloc[train_end:val_end]
    test  = df_hh.iloc[val_end:]

    return train, val, test


def split_all_households(F: pd.DataFrame) -> dict:
    """
    Apply chronological split to all households independently.

    Each household is split separately because they have
    different date ranges. We never mix households during
    splitting — features from HH A must not appear in
    the train split of HH B.

    Args:
        F: full feature matrix from US-10 (all households)

    Returns:
        dict:
            "train": combined training DataFrame
            "val":   combined validation DataFrame
            "test":  combined test DataFrame
            "info":  per-household split summary DataFrame
    """
    log_print("\n" + "="*60)
    log_print("CHRONOLOGICAL TRAIN / VAL / TEST SPLIT")
    log_print("="*60)
    log_print(f"  Ratios: Train={TRAIN_RATIO:.0%}  "
          f"Val={VAL_RATIO:.0%}  "
          f"Test={TEST_RATIO:.0%}")
    log_print(f"  Strategy: chronological (time-ordered)")
    log_print(f"  Shuffled: NO ← critical for time series\n")

    households   = sorted(F[COL_HOUSEHOLD].unique())
    train_chunks = []
    val_chunks   = []
    test_chunks  = []
    info_rows    = []

    for hh_id in households:
        df_hh = F[F[COL_HOUSEHOLD] == hh_id].copy()
        df_hh = df_hh.sort_index()

        train, val, test = split_household_chronological(df_hh)

        train_chunks.append(train)
        val_chunks.append(val)
        test_chunks.append(test)

        info_rows.append({
            COL_HOUSEHOLD:   hh_id,
            "total_rows":    len(df_hh),
            "train_rows":    len(train),
            "val_rows":      len(val),
            "test_rows":     len(test),
            "train_start":   str(train.index.min().date()),
            "train_end":     str(train.index.max().date()),
            "val_start":     str(val.index.min().date()),
            "val_end":       str(val.index.max().date()),
            "test_start":    str(test.index.min().date()),
            "test_end":      str(test.index.max().date()),
        })

        log_print(f"  HH {hh_id}:")
        log_print(f"    TRAIN  {len(train):>6,} rows  "
              f"{train.index.min().date()} → "
              f"{train.index.max().date()}")
        log_print(f"    VAL    {len(val):>6,} rows  "
              f"{val.index.min().date()} → "
              f"{val.index.max().date()}")
        log_print(f"    TEST   {len(test):>6,} rows  "
              f"{test.index.min().date()} → "
              f"{test.index.max().date()}")

    # Combine across households
    F_train = pd.concat(train_chunks, axis=0).sort_index()
    F_val   = pd.concat(val_chunks,   axis=0).sort_index()
    F_test  = pd.concat(test_chunks,  axis=0).sort_index()
    df_info = pd.DataFrame(info_rows)

    log_print(f"\n  Combined totals:")
    log_print(f"    TRAIN: {len(F_train):,} rows")
    log_print(f"    VAL:   {len(F_val):,} rows")
    log_print(f"    TEST:  {len(F_test):,} rows")

    # Save split info
    df_info.to_csv(SPLIT_INFO_CSV, index=False)
    log_print(f"\n  ✓ Split info saved: {SPLIT_INFO_CSV}")

    # Plot
    plot_split_overview(F_train, F_val, F_test)

    return {
        "train": F_train,
        "val":   F_val,
        "test":  F_test,
        "info":  df_info
    }


def plot_split_overview(F_train: pd.DataFrame,
                        F_val:   pd.DataFrame,
                        F_test:  pd.DataFrame) -> None:
    """
    Visualise the train/val/test split for one household.
    Shows the three segments as colour-coded regions.
    """
    from config import PRIMARY_HH, COL_KWH

    hh_id  = PRIMARY_HH
    tr     = F_train[F_train[COL_HOUSEHOLD] == hh_id][COL_KWH]
    va     = F_val  [F_val  [COL_HOUSEHOLD] == hh_id][COL_KWH]
    te     = F_test [F_test [COL_HOUSEHOLD] == hh_id][COL_KWH]

    # Resample to daily mean for a cleaner plot
    tr_d = tr.resample("D").mean()
    va_d = va.resample("D").mean()
    te_d = te.resample("D").mean()

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("#F9FAFB")

    ax.fill_between(tr_d.index, tr_d, alpha=0.4,
                    color="#1565C0", label="Train (70%)")
    ax.fill_between(va_d.index, va_d, alpha=0.4,
                    color="#F57F17", label="Validation (15%)")
    ax.fill_between(te_d.index, te_d, alpha=0.4,
                    color="#C62828", label="Test (15%)")

    ax.plot(tr_d.index, tr_d, color="#1565C0", lw=0.8)
    ax.plot(va_d.index, va_d, color="#F57F17", lw=0.8)
    ax.plot(te_d.index, te_d, color="#C62828", lw=0.8)

    # Boundary lines
    if len(va_d) > 0:
        ax.axvline(va_d.index[0], color="#333", lw=1.5,
                   ls="--", alpha=0.7)
        ax.text(va_d.index[0], ax.get_ylim()[1] * 0.95,
                f"  Val start\n  {va_d.index[0].date()}",
                fontsize=8, color="#333")
    if len(te_d) > 0:
        ax.axvline(te_d.index[0], color="#333", lw=1.5,
                   ls="--", alpha=0.7)
        ax.text(te_d.index[0], ax.get_ylim()[1] * 0.95,
                f"  Test start\n  {te_d.index[0].date()}",
                fontsize=8, color="#333")

    ax.set_title(
        f"Data Split — Household {hh_id}  |  "
        f"Train: {len(tr):,}  Val: {len(va):,}  "
        f"Test: {len(te):,} hourly readings",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Daily mean kWh")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, ls="--")
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(SPLIT_GRAPH, dpi=150, bbox_inches="tight")
    log_print(f"  ✓ Split plot saved: {SPLIT_GRAPH}")
    plt.close()


if __name__ == "__main__":
    from db_loader import load_all_households
    from feature_engineering.us10_features import build_feature_matrix
    df_all = load_all_households()
    F      = build_feature_matrix(df_all)
    splits = split_all_households(F)

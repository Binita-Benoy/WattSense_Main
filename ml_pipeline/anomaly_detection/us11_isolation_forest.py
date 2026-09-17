# anomaly_detection/us11_isolation_forest.py
# ─────────────────────────────────────────────────────────────────────────────
# US-11: Isolation Forest — with proper train/test split
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from anomaly_detection.config import (
    IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE,
    IF_MODEL_PKL, IF_SCALER_PKL, IF_RESULTS_CSV, IF_GRAPH,
    COL_HOUSEHOLD, COL_KWH, PRIMARY_HH,
    SEVERITY_HIGH, SEVERITY_MEDIUM
)

IF_FEATURES = [
    COL_KWH,
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "rolling_24h_mean", "rolling_24h_std",
    "rolling_7d_mean",
    "deviation_24h", "z_score",
    "lag_1h", "lag_24h",
    "change_1h", "change_24h",
    "is_peak", "is_weekend"
]


def train_isolation_forest(F_train: pd.DataFrame) -> dict:
    """
    Fit scaler and Isolation Forest on TRAINING data only.

    The scaler sees ONLY training data statistics (median, IQR).
    The model learns ONLY what normal looks like in training.
    Neither has any knowledge of validation or test data.

    Args:
        F_train: training split from splitter.py

    Returns:
        dict with fitted model, scaler, feature list
    """
    print("\n" + "="*60)
    print("US-11: TRAINING ISOLATION FOREST")
    print("="*60)
    print(f"  ⚠️  Fitting on TRAIN split only — "
          f"{len(F_train):,} rows")

    available = [c for c in IF_FEATURES if c in F_train.columns]
    X_train   = F_train[available].copy()

    print(f"  Features:      {len(available)}")
    print(f"  Contamination: {IF_CONTAMINATION*100:.0f}%")

    # ── Fit scaler on TRAIN only ───────────────────────────────────────────
    # Using training statistics means the scaler has NO knowledge
    # of test data distribution
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ── Fit model on TRAIN only ───────────────────────────────────────────
    model = IsolationForest(
        n_estimators = IF_N_ESTIMATORS,
        contamination= IF_CONTAMINATION,
        random_state = IF_RANDOM_STATE,
        n_jobs       = -1
    )
    model.fit(X_scaled)

    joblib.dump(model,  IF_MODEL_PKL)
    joblib.dump(scaler, IF_SCALER_PKL)
    print(f"  ✓ Trained and saved (train split only)")

    return {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": available
    }


def predict_if_on_split(F_split:   pd.DataFrame,
                        trained:   dict,
                        split_name: str = "test") -> pd.DataFrame:
    """
    Apply trained IF to a data split.

    The scaler and model are applied WITHOUT refitting —
    they use the statistics learned from training data only.

    This is called separately for:
        - val split  (optional sanity check)
        - test split (final honest evaluation)

    Args:
        F_split:    DataFrame from one split (val or test)
        trained:    output from train_isolation_forest()
        split_name: label for logging ("val" or "test")

    Returns:
        Results DataFrame with all 4 AC columns
    """
    print(f"\n  Predicting on {split_name.upper()} split "
          f"({len(F_split):,} rows)...")
    print(f"  ⚠️  Scaler and model NOT refitted — "
          f"using train statistics only")

    model     = trained["model"]
    scaler    = trained["scaler"]
    available = trained["feature_cols"]

    # Use only columns the model was trained on
    available = [c for c in available if c in F_split.columns]
    X         = F_split[available]

    # transform (NOT fit_transform) — critical distinction
    X_scaled   = scaler.transform(X)

    raw_preds  = model.predict(X_scaled)
    is_anomaly = (raw_preds == -1).astype(int)
    raw_scores = model.score_samples(X_scaled)
    norm_scores= 1 - (
        (raw_scores - raw_scores.min()) /
        (raw_scores.max() - raw_scores.min() + 1e-9)
    )

    results = pd.DataFrame({
        "timestamp":     F_split.index,
        COL_HOUSEHOLD:   F_split[COL_HOUSEHOLD].values,
        "consumption":   F_split[COL_KWH].values,
        "anomaly_score": norm_scores.round(6),
        "is_anomaly":    is_anomaly,
        "z_score":       F_split["z_score"].values
                         if "z_score" in F_split.columns
                         else np.nan,
        "split":         split_name,
        "method":        "isolation_forest"
    }).set_index("timestamp")

    n     = len(results)
    n_anom= results["is_anomaly"].sum()
    print(f"  Anomalies: {n_anom:,} / {n:,} "
          f"({n_anom/n*100:.1f}%)")

    return results


def run_if_pipeline(splits: dict) -> dict:
    """
    Full IF pipeline with correct train/test separation.

    Flow:
        1. Fit on splits["train"]  ← model never sees val/test
        2. Predict on splits["val"]  ← sanity check
        3. Predict on splits["test"] ← final honest evaluation
        4. Save test results for evaluation (US-13)

    Args:
        splits: dict from splitter.split_all_households()

    Returns:
        dict with trained model and results for each split
    """
    # Train
    trained = train_isolation_forest(splits["train"])

    # Predict on val (sanity check — not used for final eval)
    results_val  = predict_if_on_split(
        splits["val"], trained, "val"
    )

    # Predict on test (final evaluation)
    results_test = predict_if_on_split(
        splits["test"], trained, "test"
    )

    # Save TEST results only — these are what US-13 evaluates
    results_test.to_csv(IF_RESULTS_CSV)
    print(f"\n  ✓ Test results saved: {IF_RESULTS_CSV}")
    print(f"  ℹ️  Evaluation (US-13) will use TEST split only")

    # Plot evidence from test split
    plot_if_evidence(results_test)

    # Verify
    verify_us11(results_test)

    return {
        "trained":      trained,
        "results_val":  results_val,
        "results_test": results_test
    }


def plot_if_evidence(results_test: pd.DataFrame,
                     household_id: int = PRIMARY_HH) -> None:
    """
    Evidence graph — plotted from TEST split only.
    Clearly labelled to show these are unseen test readings.
    """
    print(f"\n  Generating evidence graph (TEST split)...")

    df = results_test[
        results_test[COL_HOUSEHOLD] == household_id
    ].copy()

    normal    = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    if len(df) == 0:
        print(f"  ⚠️  No test data for HH {household_id}")
        return

    # Zoom window
    if len(anomalies) > 0:
        first   = anomalies.index[0]
        z_start = max(df.index[0],
                      first - pd.Timedelta(days=2))
        z_end   = z_start + pd.Timedelta(days=7)
    else:
        z_start = df.index[0]
        z_end   = z_start + pd.Timedelta(days=7)

    zoom   = df.loc[z_start:z_end]
    z_norm = zoom[zoom["is_anomaly"] == 0]
    z_anom = zoom[zoom["is_anomaly"] == 1]

    fig, axes = plt.subplots(
        3, 1, figsize=(17, 12),
        gridspec_kw={"height_ratios": [3, 3, 1.5]}
    )
    fig.patch.set_facecolor("#F9FAFB")
    fig.suptitle(
        f"US-11 — Isolation Forest  |  WattSense  |  "
        f"Household {household_id}\n"
        f"TEST SPLIT ONLY (unseen during training)  |  "
        f"{df.index.min().date()} → {df.index.max().date()}",
        fontsize=12, fontweight="bold", y=0.99
    )

    # Panel 1 — Full test period
    ax = axes[0]
    ax.set_facecolor("white")
    ax.set_title(
        f"Test period overview  |  "
        f"{len(anomalies)} anomalies / {len(df)} readings "
        f"({len(anomalies)/max(len(df),1)*100:.1f}%)",
        fontsize=10
    )
    ax.plot(normal.index, normal["consumption"],
            color="#1565C0", lw=0.6, alpha=0.55, label="Normal")
    ax.scatter(anomalies.index, anomalies["consumption"],
               color="#C62828", s=22, zorder=5,
               label=f"Anomaly ({len(anomalies)})",
               edgecolors="#B71C1C", lw=0.3)
    rm = df["consumption"].rolling(24).mean()
    ax.plot(df.index, rm, color="#E65100",
            lw=1.0, ls="--", alpha=0.75, label="24h mean")
    for idx in anomalies.index:
        ax.axvspan(idx, idx + pd.Timedelta(hours=1),
                   alpha=0.10, color="#C62828")
    ax.set_ylabel("kWh", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, ls="--")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Stats box — emphasise TEST split
    ax.text(0.01, 0.97,
            f"⚠️ TEST SPLIT — model never saw this data\n"
            f"Model: Isolation Forest (n={IF_N_ESTIMATORS})\n"
            f"Contamination: {IF_CONTAMINATION*100:.0f}%\n"
            f"Train/Val/Test: 70/15/15%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#FFF9C4",
                      edgecolor="#F9A825", alpha=0.95))

    # Panel 2 — Zoomed view
    ax = axes[1]
    ax.set_facecolor("white")
    ax.set_title(
        f"7-day zoom  |  "
        f"{z_start.date()} → {z_end.date()}  |  "
        f"{len(z_anom)} anomalies",
        fontsize=10
    )
    ax.plot(z_norm.index, z_norm["consumption"],
            color="#1565C0", lw=1.5, alpha=0.7, label="Normal")
    if len(z_anom) > 0:
        ax.scatter(z_anom.index, z_anom["consumption"],
                   color="#C62828", s=100, zorder=5,
                   label=f"Anomaly ({len(z_anom)})",
                   edgecolors="#B71C1C", lw=0.8)
        for idx, row in z_anom.iterrows():
            ax.annotate(
                f"  {row['consumption']:.3f} kWh\n"
                f"  score={row['anomaly_score']:.3f}",
                xy=(idx, row["consumption"]),
                xytext=(8, 10), textcoords="offset points",
                fontsize=7.5, color="#B71C1C",
                arrowprops=dict(arrowstyle="->",
                                color="#B71C1C", lw=0.8)
            )
        for idx in z_anom.index:
            ax.axvspan(idx, idx + pd.Timedelta(hours=1),
                       alpha=0.2, color="#C62828")
    ax.set_ylabel("kWh")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, ls="--")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b\n%H:00")
    )
    ax.xaxis.set_major_locator(mdates.DayLocator())

    # Panel 3 — Scores
    ax = axes[2]
    ax.set_facecolor("white")
    ax.set_title("Anomaly scores (test split)", fontsize=10)
    ax.fill_between(df.index, df["anomaly_score"],
                    alpha=0.2, color="#6A1B9A")
    ax.plot(df.index, df["anomaly_score"],
            color="#6A1B9A", lw=0.7)
    ax.axhline(SEVERITY_MEDIUM, color="#F57F17",
               lw=1.0, ls="--", label=f"Medium ({SEVERITY_MEDIUM})")
    ax.axhline(SEVERITY_HIGH, color="#C62828",
               lw=1.2, ls="--", label=f"High ({SEVERITY_HIGH})")
    ax.scatter(anomalies.index, anomalies["anomaly_score"],
               color="#C62828", s=12, zorder=5)
    ax.set_ylabel("Score (0–1)")
    ax.set_xlabel("Timestamp")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(IF_GRAPH, dpi=150,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓ Graph saved: {IF_GRAPH}")
    plt.show()


def verify_us11(results: pd.DataFrame) -> bool:
    print(f"\n{'='*60}")
    print("US-11 ACCEPTANCE CRITERIA")
    print("="*60)
    criteria = [
        ("timestamp (DatetimeIndex)",
         isinstance(results.index, pd.DatetimeIndex)),
        ("consumption column",
         "consumption" in results.columns),
        ("anomaly_score 0-1",
         "anomaly_score" in results.columns and
         results["anomaly_score"].between(0, 1).all()),
        ("is_anomaly 0/1",
         "is_anomaly" in results.columns),
        ("tested on held-out TEST split",
         "split" in results.columns and
         (results["split"] == "test").all()),
        ("evidence graph saved",
         os.path.exists(IF_GRAPH))
    ]
    all_pass = True
    for name, passed in criteria:
        print(f"  {'✅' if passed else '❌'}  {name}")
        if not passed:
            all_pass = False
    print(f"\n  {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
    return all_pass
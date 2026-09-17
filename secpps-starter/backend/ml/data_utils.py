# Common data utilities for the WattSense anomaly pipeline.
import pandas as pd

from config import FEATURE_CSV, COL_TIMESTAMP


def load_feature_matrix(path: str = FEATURE_CSV) -> pd.DataFrame:
    """Load the feature matrix produced by feature engineering."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = COL_TIMESTAMP
    return df

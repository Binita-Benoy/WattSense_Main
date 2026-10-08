"""
Scaler persistence - the fix for the gap both CNN-LSTM report PDFs flagged:
the proposal requires fitted scalers to be saved alongside model weights "for
correct inverse transformation without needing the original training data,"
but every script in this project currently calls scale_columns() fresh every
time it needs to inverse-transform a prediction, which only works because
the original training data happens to still be sitting in the same database
right next to it. In a real deployment (or just handing the saved .keras
file to someone without DB access) that assumption breaks.

Drop this file next to features.py and import save_scalers/load_scalers
wherever a model's fitted scaler dict needs to survive past the current
process.

Deliberately NOT wired into the CV scripts (timedistributed_cnn_lstm_multihorizon_cv.py,
timedistributed_direct_multistep_1month_cv.py): those intentionally re-fit a
fresh scaler per fold because that's what makes rolling-origin CV valid in
the first place (each fold must only ever see data up to its own cutoff) -
persisting a fold's scaler would have nothing meaningful to be loaded back
for later, since the next real deployment doesn't correspond to any one
fold. Persistence belongs on the deployment-facing single-split scripts
instead, where one specific household's fine-tuned model+scaler pair is what
actually gets saved and reused.
"""
import os
import pickle


def save_scalers(scalers: dict, path: str) -> None:
    """scalers is the dict scale_columns() returns: {column_name: fitted RobustScaler}.
    Pickling is fine here - these are plain sklearn objects with no project-specific
    classes in them, so no custom unpickling logic is needed on the load side."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)


def load_scalers(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved scalers at {path} - this model was saved before scaler "
            f"persistence was added, or save_scalers() was never called for it. "
            f"Either re-run fine-tuning (which now saves scalers automatically) "
            f"or fall back to scale_columns() on the original training data."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def scaler_path_for_household(model_dir: str, variant: str, household_id: int) -> str:
    """One consistent naming scheme so every script's saved scalers live
    findably next to that script's saved .keras model, e.g.:
        saved_models/scalers/timedistributed_recursive_8162493.pkl
        saved_models/scalers/timedistributed_1month_8162493.pkl
    """
    return os.path.join(model_dir, "scalers", f"{variant}_{household_id}.pkl")

"""
utils.py - Helper functions for the Agriculture Decision Support System
"""

import joblib
import numpy as np
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all trained models and preprocessing objects.

    Returns
    -------
    dict with keys: dt, kmeans, lr, scaler, label_encoders,
                    crop_encoder, feature_columns
    """
    paths = {
        "dt":              "decision_tree.pkl",
        "kmeans":          "kmeans_model.pkl",
        "lr":              "linear_regression.pkl",
        "scaler":          "scaler.pkl",
        "label_encoders":  "label_encoders.pkl",
        "crop_encoder":    "crop_encoder.pkl",
        "feature_columns": "feature_columns.pkl",
    }

    missing = [v for v in paths.values()
               if not os.path.exists(os.path.join(MODELS_DIR, v))]
    if missing:
        raise FileNotFoundError(
            f"Run preprocessing.py and model_training.py first. "
            f"Missing: {missing}"
        )

    models = {k: joblib.load(os.path.join(MODELS_DIR, v))
              for k, v in paths.items()}
    print("✅ All models loaded successfully!")
    return models


# ============================================================================
# INPUT VALIDATION  (columns from the agriculture dataset)
# ============================================================================

# Reasonable ranges for each numeric feature
FEATURE_RANGES = {
    "Farm_Area(acres)":         (5.0,    600.0),
    "Fertilizer_Used(tons)":    (0.1,    15.0),
    "Pesticide_Used(kg)":       (0.0,    10.0),
    "Water_Usage(cubic meters)":(1000.0, 120000.0),
}

CATEGORICAL_OPTIONS = {
    "Crop_Type":       ["Wheat", "Rice", "Cotton", "Maize", "Carrot",
                        "Tomato", "Sugarcane", "Barley", "Potato", "Soybean"],
    "Irrigation_Type": ["Drip", "Manual", "Flood", "Sprinkler"],
    "Soil_Type":       ["Loamy", "Sandy", "Silty", "Clay", "Peaty"],
    "Season":          ["Kharif", "Rabi", "Zaid"],
}


def validate_numeric_inputs(values_dict: dict) -> tuple[bool, str]:
    """Validate numeric feature values against allowed ranges.

    Parameters
    ----------
    values_dict : {feature_name: float}

    Returns
    -------
    (True, "OK") or (False, error_message)
    """
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = values_dict.get(feat)
        if val is None:
            continue
        if not (lo <= val <= hi):
            return False, f"'{feat}' should be between {lo} and {hi} (got {val})"
    return True, "All inputs OK"


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_input(raw_inputs: dict, models: dict) -> np.ndarray:
    """Convert a dict of raw user inputs into a scaled feature vector.

    Parameters
    ----------
    raw_inputs : dict  – {column_name: value}
    models     : dict  – output of load_models()

    Returns
    -------
    numpy array of shape (1, n_features)
    """
    import pandas as pd

    feature_columns = models["feature_columns"]
    label_encoders  = models["label_encoders"]
    scaler          = models["scaler"]

    row = {}
    for col in feature_columns:
        val = raw_inputs.get(col)
        if val is None:
            raise ValueError(f"Missing input for feature: '{col}'")
        if col in label_encoders:
            le = label_encoders[col]
            if val not in le.classes_:
                raise ValueError(
                    f"Unknown value '{val}' for '{col}'. "
                    f"Valid options: {list(le.classes_)}"
                )
            row[col] = le.transform([val])[0]
        else:
            row[col] = float(val)

    df_row = pd.DataFrame([row], columns=feature_columns)

    # Scale only the numeric columns that the scaler was fitted on
    numeric_cols = [c for c in feature_columns if c not in label_encoders]
    df_scaled = df_row.copy()
    df_scaled[numeric_cols] = scaler.transform(df_row[numeric_cols])

    return df_scaled.values


# ============================================================================
# SOIL / CLUSTER ADVICE
# ============================================================================

def get_soil_advice(cluster_num: int, n_clusters: int = 3) -> str:
    """Return farming advice for a given soil cluster."""
    # Generic bucket: lower cluster index ≈ lower resource usage
    if n_clusters <= 3:
        advice = {
            0: "🔴 Resource-limited cluster – consider organic compost and irrigation improvements.",
            1: "🟡 Average cluster – apply balanced NPK fertilizer; monitor water usage.",
            2: "🟢 High-input cluster – soil is well-managed; maintain current practices.",
        }
    else:
        frac = cluster_num / (n_clusters - 1)
        if frac < 0.33:
            return "🔴 Low-resource cluster – needs nutrient and water improvements."
        elif frac < 0.67:
            return "🟡 Mid-range cluster – balanced inputs recommended."
        else:
            return "🟢 High-resource cluster – soil is productive; keep monitoring."

    return advice.get(cluster_num, "🟡 Follow standard farming practices.")


# ============================================================================
# PREDICTION HELPERS
# ============================================================================

def get_confidence_range(yield_pred: float, margin: float = 0.15) -> tuple[float, float]:
    """Return a ±margin confidence interval around a yield prediction."""
    return round(yield_pred * (1 - margin), 2), round(yield_pred * (1 + margin), 2)


def yield_category(yield_pred: float, low_bound: float, high_bound: float) -> str:
    """Convert a numeric yield to a human-readable category."""
    if yield_pred <= low_bound:
        return "Low 🔴"
    elif yield_pred <= high_bound:
        return "Medium 🟡"
    else:
        return "High 🟢"


# ============================================================================
# PLOT HELPER
# ============================================================================

def save_plot(fig, name: str):
    """Save a matplotlib figure to the results folder."""
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {name}.png")


# ============================================================================
# SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("Testing utils.py …")
    print(get_soil_advice(1))
    ok, msg = validate_numeric_inputs({
        "Farm_Area(acres)": 120.0,
        "Fertilizer_Used(tons)": 4.5,
        "Pesticide_Used(kg)": 1.2,
        "Water_Usage(cubic meters)": 45000.0,
    })
    print(msg)
    lo, hi = get_confidence_range(25.0)
    print(f"Yield 25.0 tons → range [{lo}, {hi}]")
    print("✅ utils.py OK")

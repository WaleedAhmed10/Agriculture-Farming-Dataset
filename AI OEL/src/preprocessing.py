"""
preprocessing.py - Data cleaning and preprocessing for Farm Yield dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "agriculture_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 50)
print("LOADING DATA")
print("=" * 50)

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}")

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n📊 Checking for missing values...")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "   No missing values found.")

df = df.dropna()
print(f"✅ After cleaning: {df.shape[0]} rows")

# ============================================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================================
if "Farm_ID" in df.columns:
    df = df.drop("Farm_ID", axis=1)
    print("✅ Removed Farm_ID column")

# ============================================================================
# SEPARATE FEATURES AND TARGET
# ============================================================================
TARGET_COLUMN = "Yield(tons)"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape:   {y.shape}")

# ============================================================================
# IDENTIFY COLUMN TYPES
# ============================================================================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"\n📊 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"📊 Numerical columns   ({len(numerical_cols)}): {numerical_cols}")

# ============================================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================================
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X = X.copy()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"✅ Encoded '{col}'  →  classes: {list(le.classes_)}")

# ============================================================================
# SPLIT DATA
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Testing  set: {X_test.shape[0]} samples")

# ============================================================================
# SCALE NUMERICAL FEATURES
# ============================================================================
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

if numerical_cols:
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols]  = scaler.transform(X_test[numerical_cols])
    print(f"✅ Scaled numerical columns: {numerical_cols}")

# ============================================================================
# SAVE PREPROCESSED DATA AND OBJECTS
# ============================================================================
joblib.dump(scaler,         os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_DIR, "feature_columns.pkl"))

X_train_scaled.to_csv(os.path.join(MODELS_DIR, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(MODELS_DIR, "X_test_scaled.csv"),  index=False)
y_train.to_csv(os.path.join(MODELS_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(MODELS_DIR, "y_test.csv"),  index=False)

print("\n✅ All preprocessing files saved in 'models/' folder")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 50)
print("PREPROCESSING SUMMARY")
print("=" * 50)
print(f"Training samples : {len(X_train_scaled)}")
print(f"Testing  samples : {len(X_test_scaled)}")
print(f"Features         : {X_train_scaled.columns.tolist()}")
print("=" * 50)
print("\n📊 Sample (first 3 rows of scaled training data):")
print(X_train_scaled.head(3).to_string())

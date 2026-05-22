"""
model_training.py - Train Decision Tree, K-Means, and Linear Regression
                    on the agriculture dataset.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score,
)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================
print("=" * 50)
print("LOADING PREPROCESSED DATA")
print("=" * 50)

X_train = pd.read_csv(os.path.join(MODELS_DIR, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(MODELS_DIR, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(MODELS_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(MODELS_DIR, "y_test.csv")).values.ravel()

feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
label_encoders  = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))

print(f"✅ X_train : {X_train.shape}")
print(f"✅ X_test  : {X_test.shape}")
print(f"✅ y_train : {y_train.shape}")
print(f"✅ y_test  : {y_test.shape}")
print(f"✅ Features: {feature_columns}")

# ============================================================================
# TASK 1 – DECISION TREE CLASSIFIER  (Yield category prediction)
# ============================================================================
print("\n" + "=" * 50)
print("1. TRAINING DECISION TREE CLASSIFIER")
print("=" * 50)

# Create classification target: Low / Medium / High yield
y_all  = np.concatenate([y_train, y_test])
bins   = np.percentile(y_all, [0, 33.3, 66.6, 100])   # data-driven equal-thirds
labels = ["Low", "Medium", "High"]

y_train_cat = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
y_test_cat  = pd.cut(y_test,  bins=bins, labels=labels, include_lowest=True)

# Handle any NaN edges
y_train_cat = y_train_cat.fillna("Medium")
y_test_cat  = y_test_cat.fillna("Medium")

crop_encoder = LabelEncoder()
y_train_enc  = crop_encoder.fit_transform(y_train_cat)
y_test_enc   = crop_encoder.transform(y_test_cat)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train_enc)

y_pred_dt = dt_model.predict(X_test)
accuracy  = accuracy_score(y_test_enc, y_pred_dt)
precision = precision_score(y_test_enc, y_pred_dt, average="weighted", zero_division=0)
recall    = recall_score(y_test_enc, y_pred_dt, average="weighted", zero_division=0)

print(f"✅ Decision Tree trained!")
print(f"   Accuracy  : {accuracy:.4f}")
print(f"   Precision : {precision:.4f}")
print(f"   Recall    : {recall:.4f}")

importance_df = pd.DataFrame({
    "Feature":    feature_columns,
    "Importance": dt_model.feature_importances_,
}).sort_values("Importance", ascending=False)
print("\n📊 Top 10 Feature Importances:")
print(importance_df.head(10).to_string(index=False))

# ============================================================================
# TASK 2 – K-MEANS CLUSTERING  (Farm segmentation)
# ============================================================================
print("\n" + "=" * 50)
print("2. TRAINING K-MEANS CLUSTERING")
print("=" * 50)

# Use first 4 scaled numeric features for clustering
soil_features      = X_train.iloc[:, :4]
soil_features_test = X_test.iloc[:, :4]

inertias, sil_scores = [], []
K_range = range(2, 8)

for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_tmp.fit(soil_features)
    inertias.append(km_tmp.inertia_)
    sil_scores.append(silhouette_score(soil_features, km_tmp.labels_))

optimal_k = list(K_range)[sil_scores.index(max(sil_scores))]
print(f"✅ Optimal clusters : {optimal_k}  (silhouette = {max(sil_scores):.4f})")

kmeans_model  = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters_train = kmeans_model.fit_predict(soil_features)
clusters_test  = kmeans_model.predict(soil_features_test)

print(f"✅ Cluster distribution (train): {np.bincount(clusters_train)}")

# ============================================================================
# TASK 3 – LINEAR REGRESSION  (Yield prediction)
# ============================================================================
print("\n" + "=" * 50)
print("3. TRAINING LINEAR REGRESSION")
print("=" * 50)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_train = lr_model.predict(X_train)
y_pred_test  = lr_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
mae_train  = mean_absolute_error(y_train, y_pred_train)
mae_test   = mean_absolute_error(y_test,  y_pred_test)
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test,  y_pred_test)

print(f"✅ Linear Regression trained!")
print(f"   Train RMSE : {rmse_train:.4f}   Test RMSE : {rmse_test:.4f}")
print(f"   Train MAE  : {mae_train:.4f}   Test MAE  : {mae_test:.4f}")
print(f"   Train R²   : {r2_train:.4f}   Test R²   : {r2_test:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 50)
print("SAVING MODELS")
print("=" * 50)

joblib.dump(dt_model,     os.path.join(MODELS_DIR, "decision_tree.pkl"))
joblib.dump(kmeans_model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
joblib.dump(lr_model,     os.path.join(MODELS_DIR, "linear_regression.pkl"))
joblib.dump(crop_encoder, os.path.join(MODELS_DIR, "crop_encoder.pkl"))

print("✅ decision_tree.pkl")
print("✅ kmeans_model.pkl")
print("✅ linear_regression.pkl")
print("✅ crop_encoder.pkl")

# ============================================================================
# GENERATE PLOTS
# ============================================================================
print("\n" + "=" * 50)
print("GENERATING PLOTS")
print("=" * 50)

GREEN = "#4AB075"

# -- Plot 1: Feature Importance ------------------------------------------------
plt.figure(figsize=(10, 6))
top10 = importance_df.head(10)
plt.barh(top10["Feature"], top10["Importance"], color=GREEN)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Decision Tree – Feature Importance", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("✅ feature_importance.png")

# -- Plot 2: Elbow + Silhouette ------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(list(K_range), inertias, "bo-")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")

ax2.plot(list(K_range), sil_scores, "go-")
ax2.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "elbow_silhouette.png"), dpi=150)
plt.close()
print("✅ elbow_silhouette.png")

# -- Plot 3: Cluster Scatter ---------------------------------------------------
plt.figure(figsize=(10, 6))
sc = plt.scatter(soil_features.iloc[:, 0], soil_features.iloc[:, 1],
                 c=clusters_train, cmap="viridis", alpha=0.6)
plt.scatter(kmeans_model.cluster_centers_[:, 0],
            kmeans_model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")
plt.xlabel(soil_features.columns[0])
plt.ylabel(soil_features.columns[1])
plt.title(f"K-Means Clustering (k={optimal_k})", fontsize=14, fontweight="bold")
plt.colorbar(sc, label="Cluster")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cluster_scatter.png"), dpi=150)
plt.close()
print("✅ cluster_scatter.png")

# -- Plot 4: Residuals ---------------------------------------------------------
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5, color=GREEN)
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Predicted Yield (tons)")
plt.ylabel("Residuals")
plt.title("Linear Regression – Residual Plot", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "residual_plot.png"), dpi=150)
plt.close()
print("✅ residual_plot.png")

# -- Plot 5: Actual vs Predicted -----------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color=GREEN, label="Predictions")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", linewidth=2, label="Perfect fit")
plt.xlabel("Actual Yield (tons)")
plt.ylabel("Predicted Yield (tons)")
plt.title(f"Actual vs Predicted Yield  (R²={r2_test:.3f})", fontsize=14, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()
print("✅ actual_vs_predicted.png")

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "=" * 50)
print("PERFORMANCE SUMMARY")
print("=" * 50)
summary = pd.DataFrame({
    "Model":  ["Decision Tree (Classifier)", "K-Means", "Linear Regression"],
    "Metric": ["Accuracy", "Best Silhouette Score", "Test R²"],
    "Value":  [f"{accuracy:.4f}", f"{max(sil_scores):.4f}", f"{r2_test:.4f}"],
})
print(summary.to_string(index=False))
print("\n✅ MODEL TRAINING COMPLETE!")
print("=" * 50)

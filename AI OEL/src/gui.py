"""
gui.py - Agriculture Decision Support System
         Tkinter GUI for crop-yield prediction, classification, and clustering.

Features
--------
• Predict Yield (tons) using Linear Regression
• Classify Yield Category (Low / Medium / High) using Decision Tree
• Identify Farm Cluster (soil segmentation) using K-Means
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# ── make sure our src/ package is importable ─────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import (
    load_models,
    prepare_input,
    validate_numeric_inputs,
    get_soil_advice,
    get_confidence_range,
    yield_category,
    CATEGORICAL_OPTIONS,
    FEATURE_RANGES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "#4AB075"
DKGRN  = "#2e7d54"
WHITE  = "#FFFFFF"
LBLGRY = "#f4f4f4"
FONT_H = ("Helvetica", 13, "bold")
FONT_N = ("Helvetica", 11)
FONT_S = ("Helvetica", 10)

NUMERIC_FIELDS = [
    ("Farm_Area(acres)",          "Farm Area (acres)",         "e.g. 120.0"),
    ("Fertilizer_Used(tons)",     "Fertilizer Used (tons)",    "e.g. 4.5"),
    ("Pesticide_Used(kg)",        "Pesticide Used (kg)",       "e.g. 1.2"),
    ("Water_Usage(cubic meters)", "Water Usage (cubic m³)",    "e.g. 45000"),
]

CAT_FIELDS = [
    ("Crop_Type",       "Crop Type"),
    ("Irrigation_Type", "Irrigation Type"),
    ("Soil_Type",       "Soil Type"),
    ("Season",          "Season"),
]


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
class FarmApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌾 Agriculture Decision Support System")
        self.geometry("700x700")
        self.resizable(True, True)
        self.configure(bg=LBLGRY)

        # Load models
        try:
            self.models = load_models()
        except FileNotFoundError as e:
            messagebox.showerror("Models not found",
                                 f"{e}\n\nRun main.py first to train the models.")
            self.destroy()
            return

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Label(self, text="🌾 Farm Yield Decision Support",
                       bg=GREEN, fg=WHITE, font=("Helvetica", 16, "bold"),
                       pady=12)
        hdr.pack(fill=tk.X)

        # Scrollable frame
        container = tk.Frame(self, bg=LBLGRY)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        canvas  = tk.Canvas(container, bg=LBLGRY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=LBLGRY)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        frame = self.scroll_frame

        # ── Categorical inputs ────────────────────────────────────────────────
        tk.Label(frame, text="Farm Details", bg=LBLGRY,
                 font=FONT_H, fg=DKGRN).pack(anchor="w", pady=(10, 4))

        self.cat_vars = {}
        for key, label in CAT_FIELDS:
            row = tk.Frame(frame, bg=LBLGRY)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label + ":", bg=LBLGRY, font=FONT_N,
                     width=22, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=CATEGORICAL_OPTIONS[key][0])
            cb  = ttk.Combobox(row, textvariable=var,
                                values=CATEGORICAL_OPTIONS[key],
                                state="readonly", width=22, font=FONT_S)
            cb.pack(side=tk.LEFT)
            self.cat_vars[key] = var

        # ── Numeric inputs ────────────────────────────────────────────────────
        tk.Label(frame, text="Resource Inputs", bg=LBLGRY,
                 font=FONT_H, fg=DKGRN).pack(anchor="w", pady=(14, 4))

        self.num_entries = {}
        for key, label, placeholder in NUMERIC_FIELDS:
            row = tk.Frame(frame, bg=LBLGRY)
            row.pack(fill=tk.X, pady=3)
            lo, hi = FEATURE_RANGES[key]
            tk.Label(row, text=f"{label} [{lo}–{hi}]:", bg=LBLGRY,
                     font=FONT_N, width=28, anchor="w").pack(side=tk.LEFT)
            entry = tk.Entry(row, font=FONT_S, width=18,
                             fg="#888", relief=tk.FLAT,
                             highlightbackground=GREEN,
                             highlightthickness=1)
            entry.insert(0, placeholder)
            entry.bind("<FocusIn>",  lambda e, ent=entry, ph=placeholder: self._clear_ph(e, ent, ph))
            entry.bind("<FocusOut>", lambda e, ent=entry, ph=placeholder: self._restore_ph(e, ent, ph))
            entry.pack(side=tk.LEFT)
            self.num_entries[key] = entry

        # ── Predict button ────────────────────────────────────────────────────
        btn = tk.Button(frame, text="🔍  Predict",
                        bg=GREEN, fg=WHITE, font=FONT_H,
                        relief=tk.FLAT, cursor="hand2",
                        padx=20, pady=8,
                        command=self._predict)
        btn.pack(pady=16)

        # ── Results area ──────────────────────────────────────────────────────
        tk.Label(frame, text="Prediction Results", bg=LBLGRY,
                 font=FONT_H, fg=DKGRN).pack(anchor="w")

        self.result_box = tk.Text(frame, height=12, font=FONT_N,
                                  bg=WHITE, fg="#222", relief=tk.FLAT,
                                  wrap=tk.WORD, state=tk.DISABLED,
                                  highlightbackground=GREEN,
                                  highlightthickness=1)
        self.result_box.pack(fill=tk.X, pady=(4, 20))

        # Reset button
        tk.Button(frame, text="Reset",
                  bg="#ccc", font=FONT_S, relief=tk.FLAT, cursor="hand2",
                  command=self._reset).pack(pady=(0, 20))

    # ── Placeholder helpers ───────────────────────────────────────────────────
    def _clear_ph(self, event, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="#000")

    def _restore_ph(self, event, entry, placeholder):
        if entry.get().strip() == "":
            entry.insert(0, placeholder)
            entry.config(fg="#888")

    # ── Prediction logic ──────────────────────────────────────────────────────
    def _predict(self):
        # 1. Collect categorical inputs
        raw = {k: v.get() for k, v in self.cat_vars.items()}

        # 2. Collect & validate numeric inputs
        num_vals = {}
        for key, label, placeholder in NUMERIC_FIELDS:
            txt = self.num_entries[key].get().strip()
            if txt == placeholder or txt == "":
                messagebox.showwarning("Missing input",
                                       f"Please enter a value for '{label}'.")
                return
            try:
                num_vals[key] = float(txt)
            except ValueError:
                messagebox.showerror("Invalid input",
                                     f"'{label}' must be a number.")
                return

        ok, msg = validate_numeric_inputs(num_vals)
        if not ok:
            messagebox.showerror("Out-of-range input", msg)
            return

        raw.update(num_vals)

        # 3. Prepare scaled feature vector
        try:
            X_input = prepare_input(raw, self.models)
        except Exception as e:
            messagebox.showerror("Preparation error", str(e))
            return

        # 4. Run models
        try:
            # Linear Regression → yield
            yield_pred = float(self.models["lr"].predict(X_input)[0])
            lo_ci, hi_ci = get_confidence_range(yield_pred)

            # Decision Tree → yield category
            cat_enc  = self.models["dt"].predict(X_input)[0]
            cat_label = self.models["crop_encoder"].inverse_transform([cat_enc])[0]

            # K-Means → farm cluster
            n_clusters = self.models["kmeans"].n_clusters
            # cluster uses only the first 4 numeric-scaled features
            soil_vec = X_input[:, :4]
            cluster  = int(self.models["kmeans"].predict(soil_vec)[0])
            advice   = get_soil_advice(cluster, n_clusters)

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            return

        # 5. Display results
        lines = [
            "─" * 48,
            f"  Crop      : {raw['Crop_Type']}   |   Season: {raw['Season']}",
            f"  Soil      : {raw['Soil_Type']}   |   Irrigation: {raw['Irrigation_Type']}",
            "─" * 48,
            f"  📈 Predicted Yield   :  {yield_pred:.2f} tons",
            f"  📊 95% CI Range      :  [{lo_ci}  –  {hi_ci}] tons",
            f"  🏷️  Yield Category    :  {cat_label}",
            f"  🗺️  Farm Cluster      :  Cluster {cluster}  (of {n_clusters})",
            f"  💡 Advice            :  {advice}",
            "─" * 48,
        ]
        self._show_result("\n".join(lines))

    def _show_result(self, text: str):
        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text)
        self.result_box.config(state=tk.DISABLED)

    def _reset(self):
        for key, _, placeholder in NUMERIC_FIELDS:
            entry = self.num_entries[key]
            entry.delete(0, tk.END)
            entry.insert(0, placeholder)
            entry.config(fg="#888")
        for key, _ in CAT_FIELDS:
            self.cat_vars[key].set(CATEGORICAL_OPTIONS[key][0])
        self._show_result("")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FarmApp()
    app.mainloop()

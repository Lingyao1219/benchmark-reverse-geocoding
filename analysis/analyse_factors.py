# -*- coding: utf-8 -*-
"""
run:
    python calc_feature_weights.py
Outputs:
  feature_weights/
      <model>/
          weights_logit.csv   # value-level OR
          weights_ridge.csv   # value-level β
          field_importance.csv# field-level Σ|β|
          metrics.json
      summary.csv
"""
import json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_absolute_error, r2_score)

# Configuration
MODEL_ORDER = [
    'gpt-4.1-mini','gpt-4.1','o3',
    'gemini-1.5-pro','gemini-2.5-flash','gemini-2.5-pro',
    'llama-4-17b','llama-3.2-11b','llama-3.2-90b',
    'claude-3.5-haiku'
]

ALLOWED = {
    "environment": ["indoor","outdoor"],
    "scene_type" : ["street","scenery","building"],
    "setting"    : ["urban","suburban","rural","natural"],
    "landmark"   : ["yes","no"],
}

OUT_DIR = Path("feature_weights")
OUT_DIR.mkdir(exist_ok=True)

# Read data
df = pd.read_csv("dataset3_result_with_flag.csv")

# Derived columns
df["city_correct"] = (df["city"] == df["true_city"]).astype(int)
df["dist_err"] = np.log1p(
        np.sqrt((df["latitude"]-df["true_latitude"])**2 +
                (df["longitude"]-df["true_longitude"])**2)
)

# Normalize categorical values
def norm_val(col, v):
    v = str(v).lower().strip()
    return v if v in ALLOWED[col] else "mixed"

for c in ["environment","scene_type","setting"]:
    df[c] = df[c].apply(lambda x: norm_val(c, x))
df["landmark"] = df["landmark"].where(df["landmark"].isin(["yes","no"]), "no")

cat_cols = ["environment","scene_type","setting","landmark"]
prep = ColumnTransformer(
        [("cat",
          OneHotEncoder(
              drop=None,
              sparse_output=False,
              handle_unknown="ignore"),
          cat_cols)],
        remainder="drop"
      )

# Main loop
summary = []

for model_name in MODEL_ORDER:
    sub = df[df["model"] == model_name].dropna(subset=["dist_err", "city_correct"])
    if len(sub) < 2:
        continue

    X = sub[cat_cols]

    # 1) Logistic regression
    logit = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            penalty="l2",
            fit_intercept=True))
    ]).fit(X, sub["city_correct"])

    feat_names = logit["prep"].get_feature_names_out()
    beta_raw   = logit["clf"].coef_.flatten()
    intercept  = logit["clf"].intercept_[0]

    # 2) Field-wise zero-mean centering
    beta_adj = beta_raw.copy()

    for field in cat_cols:
        prefix = f"cat__{field}_"
        idx    = [i for i, f in enumerate(feat_names) if f.startswith(prefix)]
        if not idx:
            continue
        mu = beta_adj[idx].mean()
        beta_adj[idx] -= mu
        intercept     += mu

    OR = np.exp(beta_adj)

    # 3) Ridge regression
    ridge = Pipeline([
        ("prep", prep),
        ("reg", Ridge(alpha=1.0))
    ]).fit(X, sub["dist_err"])

    beta_ridge = ridge["reg"].coef_
    pred       = ridge.predict(X)
    mae        = mean_absolute_error(sub["dist_err"], pred)
    r2         = r2_score(sub["dist_err"], pred)

    # 4) Classification metrics
    y_prob = logit.predict_proba(X)[:, 1]
    acc    = accuracy_score(sub["city_correct"], (y_prob >= 0.5))
    auc    = roc_auc_score(sub["city_correct"], y_prob)

    # 5) Save outputs
    df_val = (pd.DataFrame({
                  "feature"   : feat_names,
                  "logit_beta": beta_adj,
                  "logit_OR"  : OR,
                  "ridge_beta": beta_ridge})
              .assign(field=lambda d:
                      d["feature"].str.replace(r"^cat__", "", regex=True)
                                    .str.split("_").str[0])
             )

    field_imp = (df_val.groupby("field")
                 .agg(logit_abs_sum=("logit_beta", lambda x: np.abs(x).sum()),
                      ridge_abs_sum=("ridge_beta", lambda x: np.abs(x).sum()))
                 .sort_values("logit_abs_sum", ascending=False))

    m_dir = OUT_DIR / model_name
    m_dir.mkdir(parents=True, exist_ok=True)

    df_val[["feature", "logit_OR", "logit_beta"]].sort_values(
        "logit_OR", ascending=False).to_csv(m_dir / "weights_logit.csv", index=False)

    df_val[["feature", "ridge_beta"]].sort_values(
        "ridge_beta", key=np.abs, ascending=False).to_csv(m_dir / "weights_ridge.csv", index=False)

    field_imp.to_csv(m_dir / "field_importance.csv")

    with open(m_dir / "metrics.json", "w") as f:
        json.dump({"n": len(sub), "accuracy": acc, "roc_auc": auc,
                   "mae_log_err": mae, "r2": r2}, f, indent=2)

    summary.append([model_name, len(sub), acc, auc, mae, r2])

# Summary table
sum_df = pd.DataFrame(summary,
                      columns=["model", "n", "acc", "auc", "mae_log_err", "r2"])
sum_df.to_csv(OUT_DIR / "summary.csv", index=False)

print("\n=== Summary ===")
print(sum_df.round(4))
print(f"\nAll files written to →  {OUT_DIR.resolve()}")

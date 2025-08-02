# draw_feature_weights.py
import os, re, math, json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"
import numpy as np

ROOT   = Path("dataset3_feature_weights")
TOP_N  = None

cmap = mpl.colormaps["coolwarm"]
positions = [0.1, 0.4, 0.7, 0.9]
c_env, c_scene, c_set, c_land = [cmap(p) for p in positions]

PALETTE = {
    "environment" : c_env,
    "scene_type"  : c_scene,
    "setting"     : c_set,
    "landmark"    : c_land,
    "other"       : "#7F7F7F",
}

def split_feat(col):
    name = col.replace("cat__", "")
    parts = name.split("_")
    if len(parts) >= 2 and "_".join(parts[:2]) in PALETTE:
        group = "_".join(parts[:2])
        value = "_".join(parts[2:]) or "<base>"
    else:
        group = parts[0]
        value = "_".join(parts[1:]) or "<base>"
    if group not in PALETTE:
        group = "other"
    return group, value

def plot_bar1(df, title, outfile, xlabel, legend_loc=(0.60, 0.01)):
    df["feat_val"] = (
        df["group"].str.replace("_", " ") + " : " + df["value"].str.replace("_", " ")
    )

    if TOP_N:
        df = df.iloc[:TOP_N]

    fig, ax = plt.subplots(figsize=(8, 0.35*len(df)+1))

    sns.barplot(
        data=df, y="feat_val", x="weight",
        hue="group", palette=PALETTE,
        dodge=False, edgecolor="k",
        ax=ax, legend=False
    )

    ax.axvline(0, c="k", lw=0.8)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("", fontsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    handles = [
        mpatches.Patch(
            color=PALETTE[g],
            label=g.replace("_", " ")
        )
        for g in ["environment", "scene_type", "setting", "landmark"]
        if g in df["group"].values
    ]

    ax.legend(handles=handles,
              frameon=True,
              fontsize=16,
              bbox_to_anchor=legend_loc,
              loc="lower left")

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"))
    plt.close(fig)

def plot_bar2(df, title, outfile, xlabel, legend_loc=(0.02, 0.01)):
    df["feat_val"] = (
        df["group"].str.replace("_", " ") + " : " + df["value"].str.replace("_", " ")
    )

    if TOP_N:
        df = df.iloc[:TOP_N]

    fig, ax = plt.subplots(figsize=(8, 0.35*len(df)+1))

    sns.barplot(
        data=df, y="feat_val", x="weight",
        hue="group", palette=PALETTE,
        dodge=False, edgecolor="k",
        ax=ax, legend=False
    )

    ax.axvline(0, c="k", lw=0.8)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("", fontsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    handles = [
        mpatches.Patch(
            color=PALETTE[g],
            label=g.replace("_", " ")
        )
        for g in ["environment", "scene_type", "setting", "landmark"]
        if g in df["group"].values
    ]

    ax.legend(handles=handles,
              frameon=True,
              fontsize=16,
              bbox_to_anchor=legend_loc,
              loc="lower left")

    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"))
    plt.close(fig)

for model_dir in ROOT.iterdir():
    if not model_dir.is_dir():
        continue
    logit_path = model_dir / "weights_logit.csv"
    ridge_path = model_dir / "weights_ridge.csv"
    if not logit_path.exists():
        continue

    or_ser = pd.read_csv(logit_path, index_col=0).iloc[:, 0]
    df_or  = (
        or_ser.to_frame("OR")
              .assign(weight=lambda d: d["OR"].apply(math.log))
              .reset_index(names="raw")
    )
    df_or[["group", "value"]] = df_or["raw"].apply(
        lambda s: pd.Series(split_feat(s))
    )
    df_or["feat_val"] = df_or["group"] + " : " + df_or["value"]
    df_or = df_or.sort_values("weight", key=abs, ascending=False)

    plot_bar1(
        df_or,
        f"{model_dir.name}  –  Logistic   (log OR)",
        model_dir / "feature_plot_logit.png",
        "log(OR)   (>0 ↑ hit prob.)"
    )

    if ridge_path.exists():
        beta_ser = pd.read_csv(ridge_path, index_col=0).iloc[:, 0]
        df_b = (beta_ser.to_frame("beta")
                        .rename(columns={"beta": "weight"})
                        .reset_index(names="raw"))
        df_b[["group", "value"]] = df_b["raw"].apply(
            lambda s: pd.Series(split_feat(s))
        )
        df_b["feat_val"] = df_b["group"] + " : " + df_b["value"]
        df_b = df_b.sort_values("weight", key=abs, ascending=False)

        plot_bar2(
            df_b,
            f"{model_dir.name}  –  Ridge   β (log distance error)",
            model_dir / "feature_plot_ridge.png",
            "β   (- ↓ error , + ↑ error)"
        )

print("All plots generated.")

#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(
    "/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/Notes_for_200_cases/3_Outputs"
)

DIR_GPT5_SET = BASE_DIR / "202604_notes_for_200_cases_18_features(4o_5pro_5mini_5nano_5.1pro_7B_70B)"
DIR_MISTRAL7B_SET = BASE_DIR / "202604_notes_for_200_cases_18_features(mistral7b)"

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_PATH = Path(
    "/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/outputs/goal-standard.csv"
)

# ----------------------------------------------------------------------
# Model files (6 out of 7; intentionally excluding 70B)
# ----------------------------------------------------------------------
MODEL_FILES = {
    "GPT-4o": [DIR_GPT5_SET / "notes-for-200-cases-18-features-gpt4o.parquet"],
    "GPT-5": [DIR_GPT5_SET / "notes-for-200-cases-18-features-gpt5.parquet"],
    "GPT-5.1": [
        DIR_GPT5_SET / "notes-for-200-cases-18-features-gpt5-1.parquet",
        DIR_GPT5_SET / "notes-for-200-cases-18-features-gpt5.1.parquet",
    ],
    "GPT-5-mini": [DIR_GPT5_SET / "notes-for-200-cases-18-features-gpt5-mini.parquet"],
    "GPT-5-nano": [DIR_GPT5_SET / "notes-for-200-cases-18-features-gpt5-nano.parquet"],
    "Mistral-7B": [DIR_MISTRAL7B_SET / "ALL_notes-for-200-cases-18-features-mistral7b.parquet"],
}

MODEL_ORDER = ["GPT-5", "GPT-5.1", "GPT-5-mini", "GPT-5-nano", "GPT-4o", "Mistral-7B"]
METRICS_ORDER = ["accuracy", "kappa", "precision", "recall", "f1"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def first_existing_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these candidate files exists: " + ", ".join(str(c) for c in candidates)
    )


def pred_name(gold_col: str) -> str:
    return gold_col.replace("__manual", "")


def build_mapping(gold_label_cols: list[str], model_df: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for g in gold_label_cols:
        p = pred_name(g)
        if p in model_df.columns:
            mapping[g] = p
    return mapping


def evaluate(df_gold: pd.DataFrame, df_model: pd.DataFrame, model_name: str, mapping: dict[str, str]) -> pd.DataFrame:
    if not mapping:
        raise ValueError(f"No overlapping label columns found for model: {model_name}")

    merged = df_gold[["EncounterCsn"] + list(mapping.keys())].merge(
        df_model[["EncounterCsn"] + list(mapping.values())],
        on="EncounterCsn",
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping EncounterCsn rows after merge for model: {model_name}")

    per_label_metrics = []
    for gold_col, pred_col in mapping.items():
        label_name = gold_col.replace("__manual", "")

        y_true = merged[gold_col].astype(str)
        y_pred = merged[pred_col].astype(str)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred)

        per_label_metrics.append([label_name, acc, kappa, precision, recall, f1])

    df_result = pd.DataFrame(
        per_label_metrics,
        columns=["feature", "accuracy", "kappa", "precision", "recall", "f1"],
    )
    df_result["model"] = model_name
    return df_result


def plot_feature_metric(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    pivot = df.pivot(index="feature", columns="model", values=metric)
    pivot = pivot.reindex(columns=MODEL_ORDER)

    fig_height = max(6, len(pivot) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(pivot.index))
    width = 0.13

    for i, model in enumerate(MODEL_ORDER):
        vals = pivot[model].values if model in pivot.columns else np.repeat(np.nan, len(pivot.index))
        ax.barh(y + (i - 2.5) * width, vals, height=width, label=model, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold file not found: {GOLD_PATH}")

    df_gold = pd.read_csv(GOLD_PATH)
    df_gold["EncounterCsn"] = df_gold["EncounterCsn"].astype("int64")
    gold_label_cols = [c for c in df_gold.columns if c.endswith("__manual")]

    model_results = []

    for model_name in MODEL_ORDER:
        model_path = first_existing_path(MODEL_FILES[model_name])
        print(f"[info] Loading {model_name}: {model_path}")
        df_model = pd.read_parquet(model_path)
        df_model["EncounterCsn"] = df_model["EncounterCsn"].astype("int64")

        mapping = build_mapping(gold_label_cols, df_model)
        res = evaluate(df_gold, df_model, model_name, mapping)
        model_results.append(res)

    df_all = pd.concat(model_results, ignore_index=True)
    df_all["model"] = pd.Categorical(df_all["model"], categories=MODEL_ORDER, ordered=True)
    df_all = df_all.sort_values(["model", "feature"]).reset_index(drop=True)

    summary_rows = []
    for model_name, df_sub in df_all.groupby("model", observed=True):
        for metric in METRICS_ORDER:
            summary_rows.append([model_name, metric, df_sub[metric].mean(), df_sub[metric].std()])

    df_summary = pd.DataFrame(summary_rows, columns=["model", "metric", "mean", "std"])
    df_summary["model"] = pd.Categorical(df_summary["model"], categories=MODEL_ORDER, ordered=True)
    df_summary = df_summary.sort_values(["model", "metric"]).reset_index(drop=True)

    df_all.to_csv(OUT_DIR / "per_feature_metrics_6models.csv", index=False)
    df_summary.to_csv(OUT_DIR / "average_metrics_with_std_6models.csv", index=False)

    print("\nSaved per-feature metrics -> per_feature_metrics_6models.csv")
    print("Saved averaged metrics -> average_metrics_with_std_6models.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(METRICS_ORDER))
    width = 0.13

    for i, model in enumerate(MODEL_ORDER):
        sub = df_summary[df_summary["model"] == model].set_index("metric").reindex(METRICS_ORDER)
        ax.bar(
            x + (i - 2.5) * width,
            sub["mean"].values,
            width=width,
            yerr=sub["std"].values,
            capsize=3,
            label=model,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Average performance across features (6 models, excluding 70B)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_average_metrics_by_model_6models.png", dpi=300)
    plt.close()

    plot_feature_metric(
        df_all,
        metric="accuracy",
        filename="figure_per_feature_accuracy_6models.png",
        title="Per-feature accuracy by model (6 models)",
    )
    plot_feature_metric(
        df_all,
        metric="f1",
        filename="figure_per_feature_f1_6models.png",
        title="Per-feature F1 by model (6 models)",
    )
    plot_feature_metric(
        df_all,
        metric="kappa",
        filename="figure_per_feature_kappa_6models.png",
        title="Per-feature kappa by model (6 models)",
    )

    print("\nSaved figures:")
    print("  figure_average_metrics_by_model_6models.png")
    print("  figure_per_feature_accuracy_6models.png")
    print("  figure_per_feature_f1_6models.png")
    print("  figure_per_feature_kappa_6models.png")
    print("\nDone.")


if __name__ == "__main__":
    main()

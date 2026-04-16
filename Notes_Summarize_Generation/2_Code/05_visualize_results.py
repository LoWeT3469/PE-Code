from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"

STAGE1_DIR = OUTPUT_ROOT / "stage1_extractions"
STAGE2_DIR = OUTPUT_ROOT / "stage2_final_json"
FIG_DIR = OUTPUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json_dir(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        print(f"Warning: missing directory {path}")
        return rows

    for f in sorted(path.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f, "r", encoding="utf-8") as fp:
            obj = json.load(fp)
        if "encounter_id" not in obj:
            obj["encounter_id"] = f.stem
        rows.append(obj)
    return rows
    

def flatten_stage1(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def flatten_stage2(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    normalized_rows = []
    for i, row in enumerate(rows):
        r = dict(row)
        if "encounter_id" not in r:
            r["encounter_id"] = None
        normalized_rows.append(r)

    return pd.DataFrame(normalized_rows)


def save_bar(series: pd.Series, title: str, outfile: Path, top_n: int | None = None) -> None:
    s = series.dropna()
    if s.empty:
        print(f"Skip empty plot: {title}")
        return

    counts = s.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def save_hist(series: pd.Series, title: str, outfile: Path, bins: int = 20) -> None:
    s = series.dropna()
    if s.empty:
        print(f"Skip empty histogram: {title}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def explode_list_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="object")

    values = []
    for item in df[col]:
        if isinstance(item, list):
            values.extend(item)
        elif pd.notna(item):
            values.append(item)
    return pd.Series(values, dtype="object")


def summarize_stage1(df: pd.DataFrame) -> None:
    if df.empty:
        print("No Stage 1 data found.")
        return

    print("\n=== Stage 1 overview ===")
    print(f"Rows: {len(df)}")
    print("Columns:", len(df.columns))

    # Core categorical plots
    for col, fname, title in [
        ("risk_factors.prior_vte", "stage1_prior_vte.png", "Stage 1: Prior VTE"),
        ("risk_factors.malignancy", "stage1_malignancy.png", "Stage 1: Malignancy"),
        (
            "diagnostic_data.ctpa_or_ct_chest.status",
            "stage1_ct_status.png",
            "Stage 1: CT / CTPA status",
        ),
        (
            "diagnostic_data.d_dimer.status",
            "stage1_ddimer_status.png",
            "Stage 1: D-dimer status",
        ),
        (
            "diagnostic_data.echo_or_rv_strain.status",
            "stage1_rv_strain_status.png",
            "Stage 1: Echo / RV strain status",
        ),
    ]:
        if col in df.columns:
            save_bar(df[col], title, FIG_DIR / fname)

    # Symptoms
    symptoms = explode_list_column(df, "presenting_features.symptoms")
    if not symptoms.empty:
        save_bar(
            symptoms,
            "Stage 1: Top extracted symptoms",
            FIG_DIR / "stage1_top_symptoms.png",
            top_n=20,
        )

    # Evidence for PE
    evidence_for = explode_list_column(df, "evidence_for_pe")
    if not evidence_for.empty:
        save_bar(
            evidence_for,
            "Stage 1: Top evidence for PE items",
            FIG_DIR / "stage1_evidence_for_pe.png",
            top_n=20,
        )

    # Evidence against PE / alternatives
    evidence_against = explode_list_column(df, "evidence_against_pe_or_alternatives")
    if not evidence_against.empty:
        save_bar(
            evidence_against,
            "Stage 1: Top evidence against PE / alternatives",
            FIG_DIR / "stage1_evidence_against_pe.png",
            top_n=20,
        )

    # Missing key history
    missing_hist = explode_list_column(df, "missing_key_history")
    if not missing_hist.empty:
        save_bar(
            missing_hist,
            "Stage 1: Top missing key history items",
            FIG_DIR / "stage1_missing_key_history.png",
            top_n=20,
        )

    # Uncertainty flags
    uncertainty = explode_list_column(df, "uncertainty_flags")
    if not uncertainty.empty:
        save_bar(
            uncertainty,
            "Stage 1: Top uncertainty flags",
            FIG_DIR / "stage1_uncertainty_flags.png",
            top_n=20,
        )


def summarize_stage2(df: pd.DataFrame) -> None:
    if df.empty:
        print("No Stage 2 data found.")
        return

    print("\n=== Stage 2 overview ===")
    print(f"Rows: {len(df)}")
    print("Columns:", len(df.columns))

    length_cols = [
        "One-line case summary",
        "Evidence for PE",
        "Evidence against PE / alternative diagnoses",
        "Missing key history",
        "Immediate next diagnostic considerations",
        "Teaching note for residents",
        "Confidence / uncertainty statement",
    ]

    for col in length_cols:
        if col in df.columns:
            lengths = df[col].fillna("").astype(str).str.len()
            save_hist(
                lengths,
                f"Stage 2: Length distribution for {col}",
                FIG_DIR / f"stage2_len_{col.lower().replace('/', '_').replace(' ', '_')}.png",
                bins=20,
            )

    # Flag summaries that mention uncertainty
    if "Confidence / uncertainty statement" in df.columns:
        uncertainty_flag = (
            df["Confidence / uncertainty statement"]
            .fillna("")
            .astype(str)
            .str.contains("uncertain|uncertainty|limited|insufficient|not documented", case=False, regex=True)
        )
        save_bar(
            uncertainty_flag.astype(str),
            "Stage 2: Uncertainty statements present",
            FIG_DIR / "stage2_uncertainty_presence.png",
        )


def stage1_stage2_consistency(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1.empty or df2.empty:
        return pd.DataFrame()

    # Merge on encounter_id
    keep_stage1 = ["encounter_id"]
    if "diagnostic_data.ctpa_or_ct_chest.status" in df1.columns:
        keep_stage1.append("diagnostic_data.ctpa_or_ct_chest.status")

    merged = df1[keep_stage1].merge(df2, on="encounter_id", how="inner")

    if "diagnostic_data.ctpa_or_ct_chest.status" in merged.columns and "Evidence for PE" in merged.columns:
        merged["stage2_mentions_pe"] = (
            merged["Evidence for PE"].fillna("").astype(str).str.contains("pe|pulmonary embol", case=False, regex=True)
        )

    return merged


def main() -> None:
    stage1_rows = load_json_dir(STAGE1_DIR)
    stage2_rows = load_json_dir(STAGE2_DIR)

    df1 = flatten_stage1(stage1_rows)
    df2 = flatten_stage2(stage2_rows)

    summarize_stage1(df1)
    summarize_stage2(df2)

    merged = stage1_stage2_consistency(df1, df2)
    if not merged.empty:
        merged.to_csv(OUTPUT_ROOT / "stage1_stage2_consistency.csv", index=False)
        print(f"Saved {OUTPUT_ROOT / 'stage1_stage2_consistency.csv'}")

        if (
            "diagnostic_data.ctpa_or_ct_chest.status" in merged.columns
            and "stage2_mentions_pe" in merged.columns
        ):
            ctab = pd.crosstab(
                merged["diagnostic_data.ctpa_or_ct_chest.status"],
                merged["stage2_mentions_pe"],
                dropna=False,
            )
            print("\n=== Stage1 vs Stage2 consistency table ===")
            print(ctab)

            plt.figure(figsize=(8, 5))
            ctab.plot(kind="bar", stacked=True)
            plt.title("Stage 1 CT status vs Stage 2 PE mention")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "stage1_stage2_consistency.png", dpi=200)
            plt.close()
            print(f"Saved {FIG_DIR / 'stage1_stage2_consistency.png'}")

    print("\nDone.")
    print(f"Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
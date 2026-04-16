from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
NOTES_DIR = PROJECT_ROOT / "1_InputData" / "Notes"
DERIVED_DIR = NOTES_DIR / "derived_tables"

OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

LABELED_200_FILE = NOTES_DIR / "notes-for-200-cases.csv"
ENCOUNTER_MASTER_FILE = DERIVED_DIR / "encounter_master.parquet"
ENCOUNTER_AGG_FILE = DERIVED_DIR / "encounter_text_aggregates.parquet"

EXPECTED_N = 200


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find any of {candidates}. Available columns: {list(df.columns)}")


def find_optional_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def to_jsonable_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled-file", type=Path, default=LABELED_200_FILE)
    p.add_argument("--encounter-master", type=Path, default=ENCOUNTER_MASTER_FILE)
    p.add_argument("--encounter-agg", type=Path, default=ENCOUNTER_AGG_FILE)
    p.add_argument("--out-csv", type=Path, default=OUTPUT_ROOT / "labeled_200_prompt_packets.csv")
    p.add_argument("--out-jsonl", type=Path, default=OUTPUT_ROOT / "labeled_200_prompt_packets.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labeled = load_table(args.labeled_file)
    master = load_table(args.encounter_master)
    agg = load_table(args.encounter_agg)

    labeled_id_col = find_col(
        labeled,
        ["EncounterCsn", "encounter_id", "encounterepiccsn", "encounter_csn", "csn"],
    )
    master_id_col = find_col(
        master,
        ["EncounterCsn", "encounter_id", "encounterepiccsn", "encounter_csn", "csn"],
    )
    agg_id_col = find_col(
        agg,
        ["EncounterCsn", "encounter_id", "encounterepiccsn", "encounter_csn", "csn"],
    )

    labeled[labeled_id_col] = normalize_id(labeled[labeled_id_col])
    master[master_id_col] = normalize_id(master[master_id_col])
    agg[agg_id_col] = normalize_id(agg[agg_id_col])

    labeled_ids = labeled[labeled_id_col].dropna().unique().tolist()
    print(f"Found {len(labeled_ids)} labeled encounter IDs")
    if len(labeled_ids) < EXPECTED_N:
        print(f"Warning: fewer than expected {EXPECTED_N} labeled encounters")

    master_200 = master[master[master_id_col].isin(labeled_ids)].copy()
    agg_200 = agg[agg[agg_id_col].isin(labeled_ids)].copy()

    print(f"master_200 rows: {len(master_200)}")
    print(f"agg_200 rows: {len(agg_200)}")

    merged = master_200.merge(
        agg_200,
        left_on=master_id_col,
        right_on=agg_id_col,
        how="left",
        suffixes=("_master", "_agg"),
    )

    print(f"merged rows: {len(merged)}")

    triage_col = find_optional_col(
        merged,
        [
            "triage_text_concat",
            "triage_text",
            "triage_note_text",
            "triage_notes_text",
            "all_triage_text",
            "triage_concat_text",
            "triage_concat",
            "triage_note",
            "earliest_triage_note_text",
        ],
    )
    provider_col = find_optional_col(
        merged,
        [
            "provider_text_concat",
            "provider_text",
            "provider_note_text",
            "provider_notes_text",
            "all_provider_text",
            "provider_concat_text",
            "provider_concat",
            "provider_note",
            "ed_provider_text",
            "earliest_provider_note_text",
        ],
    )
    ct_col = find_optional_col(
        merged,
        [
            "ct_text_concat",
            "ct_text",
            "ct_report_text",
            "ct_impression_text",
            "ct_narrative_text",
            "all_ct_text",
            "ct_concat",
            "ct_report",
            "ct_impression",
            "ct_narrative",
            "ct_chest_text",
            "earliest_ct_text",
        ],
    )

    print(f"triage_col = {triage_col}")
    print(f"provider_col = {provider_col}")
    print(f"ct_col = {ct_col}")

    if not any([triage_col, provider_col, ct_col]):
        raise ValueError(
            "No triage/provider/CT text columns were found in encounter_text_aggregates after merge."
        )

    if triage_col:
        non_empty = merged[triage_col].fillna("").astype(str).str.strip().str.len().gt(0).sum()
        print(f"Non-empty triage rows: {non_empty}")
    if provider_col:
        non_empty = merged[provider_col].fillna("").astype(str).str.strip().str.len().gt(0).sum()
        print(f"Non-empty provider rows: {non_empty}")
    if ct_col:
        non_empty = merged[ct_col].fillna("").astype(str).str.strip().str.len().gt(0).sum()
        print(f"Non-empty ct rows: {non_empty}")

    keep_cols = [master_id_col]
    for c in [triage_col, provider_col, ct_col]:
        if c and c not in keep_cols:
            keep_cols.append(c)

    for c in [
        "PatientMrn",
        "patientmrn",
        "ArrivalInstant",
        "arrival_time",
        "ed_arrival_time",
        "ChiefComplaint",
        "chief_complaint",
        "has_triage_note",
        "has_provider_note",
        "has_ct_report",
        "included_in_labeled_200",
        "pe_suspected",
    ]:
        oc = find_optional_col(merged, [c])
        if oc and oc not in keep_cols:
            keep_cols.append(oc)

    packet_df = merged[keep_cols].drop_duplicates(subset=[master_id_col]).copy()

    rename_map = {master_id_col: "encounter_id"}
    if triage_col:
        rename_map[triage_col] = "triage_text"
    if provider_col:
        rename_map[provider_col] = "provider_text"
    if ct_col:
        rename_map[ct_col] = "ct_text"

    packet_df = packet_df.rename(columns=rename_map)

    for col in ["triage_text", "provider_text", "ct_text"]:
        if col not in packet_df.columns:
            packet_df[col] = ""
        packet_df[col] = packet_df[col].fillna("").astype(str)

    packet_df["has_any_text"] = (
        packet_df["triage_text"].str.strip().str.len().gt(0)
        | packet_df["provider_text"].str.strip().str.len().gt(0)
        | packet_df["ct_text"].str.strip().str.len().gt(0)
    )
    packet_df = packet_df[packet_df["has_any_text"]].reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    packet_df.to_csv(args.out_csv, index=False)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in packet_df.to_dict(orient="records"):
            row = {k: to_jsonable_value(v) for k, v in row.items()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(packet_df)} rows")
    print(f"CSV:   {args.out_csv}")
    print(f"JSONL: {args.out_jsonl}")


if __name__ == "__main__":
    main()
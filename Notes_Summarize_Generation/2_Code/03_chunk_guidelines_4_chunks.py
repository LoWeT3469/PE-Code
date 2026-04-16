from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"
CHUNK_DIR = OUTPUT_ROOT / "guidelines_chunks"


NATIONAL_STARTER_CHUNKS = [
    {
        "chunk_id": "national_pretest",
        "title": "National: clinical assessment and pretest probability",
        "text": "For suspected pulmonary embolism, assess symptoms, signs, and pretest probability before advanced imaging."
    },
    {
        "chunk_id": "national_ddimer",
        "title": "National: D-dimer use",
        "text": "In low or intermediate pretest probability, D-dimer can help determine whether imaging is needed. Age-adjusted strategies may be appropriate in selected patients."
    },
    {
        "chunk_id": "national_imaging",
        "title": "National: imaging pathway",
        "text": "CT pulmonary angiography is generally preferred when imaging is indicated. V/Q scan is an alternative when CTPA is contraindicated or less suitable."
    },
    {
        "chunk_id": "national_risk",
        "title": "National: RV strain and biomarkers",
        "text": "Hemodynamic findings, right ventricular dysfunction, and biomarkers help characterize severity after PE is diagnosed."
    },
]

LOCAL_STARTER_CHUNKS = [
    {
        "chunk_id": "local_pretest",
        "title": "Michigan Medicine: pretest workflow",
        "text": "Michigan Medicine local workflow uses clinical suspicion and pretest probability assessment with tools such as PERC or Wells when appropriate."
    },
    {
        "chunk_id": "local_ddimer",
        "title": "Michigan Medicine: D-dimer workflow",
        "text": "Michigan Medicine local workflow uses D-dimer with local thresholding and age adjustment when appropriate before proceeding to imaging."
    },
    {
        "chunk_id": "local_imaging",
        "title": "Michigan Medicine: imaging pathway",
        "text": "CTPA is preferred when imaging is needed. V/Q scanning is used when CT contrast or CTPA is less suitable."
    },
    {
        "chunk_id": "local_pesi",
        "title": "Michigan Medicine: PESI and disposition context",
        "text": "PESI and evidence of RV dysfunction inform local monitoring and disposition workflow."
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--national-out",
        type=Path,
        default=CHUNK_DIR / "national_chunks.json",
        help="Output path for national chunks",
    )
    p.add_argument(
        "--local-out",
        type=Path,
        default=CHUNK_DIR / "local_chunks.json",
        help="Output path for local chunks",
    )
    return p.parse_args()


def main():
    args = parse_args()

    national_out = args.national_out
    local_out = args.local_out

    national_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.parent.mkdir(parents=True, exist_ok=True)

    with national_out.open("w", encoding="utf-8") as f:
        json.dump(NATIONAL_STARTER_CHUNKS, f, ensure_ascii=False, indent=2)

    with local_out.open("w", encoding="utf-8") as f:
        json.dump(LOCAL_STARTER_CHUNKS, f, ensure_ascii=False, indent=2)

    print("Guideline chunks created:")
    print(f"  National → {national_out}")
    print(f"  Local    → {local_out}")


if __name__ == "__main__":
    main()
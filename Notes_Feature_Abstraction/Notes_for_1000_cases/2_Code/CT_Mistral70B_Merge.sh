#!/bin/bash
# Merge all OrcaHermes 70B CT shard outputs for Notes_for_1000_cases

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTDIR="${PROJECT_ROOT}/3_Outputs"

cd "${SCRIPT_DIR}"

echo "=== MERGE ORCAHERMES 70B CT SHARDS ==="
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTDIR=${OUTDIR}"

mkdir -p "${OUTDIR}"

# ------------------------------------------------------------------
# Merge shard parquet files
# ------------------------------------------------------------------
python - <<'PY'
from pathlib import Path
import pandas as pd

EXPECTED_SHARDS = 8

script_dir = Path.cwd()
outdir = (script_dir.parent / "3_Outputs").resolve()

files = sorted(outdir.glob("ct-reports-for-1000-cases-ct-schema-orcahermes70b_shard*.parquet"))
print("Found", len(files), "parquet shard files")
for f in files:
    print(" -", f.name)

if len(files) != EXPECTED_SHARDS:
    raise SystemExit(
        f"Expected {EXPECTED_SHARDS} shard parquet files, found {len(files)}"
    )

dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

if "EncounterCsn" in df.columns:
    df = df.sort_values("EncounterCsn").reset_index(drop=True)

out_path = outdir / "ALL_ct-reports-for-1000-cases-ct-schema-orcahermes70b.parquet"
df.to_parquet(out_path, index=False)

print("Wrote", out_path)
print("Rows:", len(df))
print("Columns:", len(df.columns))
PY

# ------------------------------------------------------------------
# Merge shard run logs
# ------------------------------------------------------------------
python - <<'PY'
import json
from pathlib import Path

script_dir = Path.cwd()
outdir = (script_dir.parent / "3_Outputs").resolve()

files = sorted(outdir.glob("ct-reports-for-1000-cases-ct-schema-orcahermes70b_shard*.log.*.json"))
print("Found", len(files), "shard log JSON files")
for f in files:
    print(" -", f.name)

if files:
    merged = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            merged.append(json.load(f))

    out_path = outdir / "ALL_shard_runlogs_orcahermes70b_ct.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("Wrote", out_path)
    print("Merged log records:", len(merged))
else:
    print("No shard log JSON files found.")
PY

# ------------------------------------------------------------------
# Merge debug JSON files
# ------------------------------------------------------------------
python - <<'PY'
import json
from collections import OrderedDict
from pathlib import Path

script_dir = Path.cwd()
outdir = (script_dir.parent / "3_Outputs").resolve()

files = sorted(outdir.glob("debug-ct-reports-for-1000-cases-ct-schema-orcahermes70b-job*-shard*.json"))
print("Found", len(files), "debug JSON files")
for f in files:
    print(" -", f.name)

all_rows = []
for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Skipping unreadable:", fp.name, "error:", e)
        continue

    if isinstance(data, list):
        all_rows.extend(data)
    elif isinstance(data, dict):
        all_rows.append(data)

dedup = OrderedDict()
no_id = []

for r in all_rows:
    if isinstance(r, dict) and r.get("_id") is not None:
        dedup[str(r["_id"])] = r
    else:
        no_id.append(r)

merged = list(dedup.values()) + no_id

out_path = outdir / "ALL_debug_merged_orcahermes70b_ct.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Merged debug records:", len(merged))
print("Wrote", out_path)
PY

echo "=== MERGE COMPLETE ==="
echo "Outputs:"
echo " - ${OUTDIR}/ALL_ct-reports-for-1000-cases-ct-schema-orcahermes70b.parquet"
echo " - ${OUTDIR}/ALL_shard_runlogs_orcahermes70b_ct.json"
echo " - ${OUTDIR}/ALL_debug_merged_orcahermes70b_ct.json"
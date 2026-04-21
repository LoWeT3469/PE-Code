python - <<'PY'
import glob, pandas as pd
files = sorted(glob.glob("../3_Outputs/notes-for-200-cases-18-features-mistral7b_shard*.parquet"))
print("Found", len(files), "parquets")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("EncounterCsn")
df.to_parquet("../3_Outputs/ALL_notes-for-200-cases-18-features-mistral7b.parquet", index=False)
print("Wrote ../3_Outputs/ALL_notes-for-200-cases-18-features-mistral7b.parquet with rows:", len(df))
PY

python - <<'PY'
import glob, json

files = sorted(glob.glob("../3_Outputs/notes-for-200-cases-18-features-mistral7b_shard*.log.*.json"))
print("Found", len(files), "shard log JSONs")
if not files:
    raise SystemExit("No shard log json files found.")

merged = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        merged.append(json.load(f))

out_path = "../3_Outputs/ALL_shard_runlogs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print("Wrote:", out_path)
PY

python - <<'PY'
import glob, json
from collections import OrderedDict

files = sorted(glob.glob("../3_Outputs/debug*.json"))
print("Found", len(files), "debug json files")

all_rows = []
for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Skipping unreadable:", fp, "error:", e)
        continue

    if isinstance(data, list):
        all_rows.extend(data)
    elif isinstance(data, dict):
        all_rows.append(data)
    else:
        print("Skipping unknown type:", fp)

# Optional de-dup by _id
dedup = OrderedDict()
no_id = []
for r in all_rows:
    if isinstance(r, dict) and "_id" in r and r["_id"] is not None:
        dedup[str(r["_id"])] = r
    else:
        no_id.append(r)

merged = list(dedup.values()) + no_id
out_path = "../3_Outputs/ALL_debug_merged.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Merged debug records:", len(merged))
print("Wrote:", out_path)
PY



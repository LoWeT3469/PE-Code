#!/bin/bash
# LWT

#SBATCH --job-name=PE_Mistral7B_1000
#SBATCH --account=atjanke0
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%4
#SBATCH --time=6-23:59:59
#SBATCH --output=./%x-%A_%a.out
#SBATCH --error=./%x-%A_%a.err
#SBATCH --mail-user=liuwent@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load cuda/12.8.1
module load gcc/10.3.0

set +u
source "$HOME/.bashrc"
set -u
conda activate PE
export PYTHONNOUSERSITE=1

# ------------------------------------------------------------------
# Project layout
# ------------------------------------------------------------------
WORKDIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/Notes_for_1000_cases"
CODEDIR="${WORKDIR}/2_Code"
DATADIR="${WORKDIR}/1_Data"
OUTDIR="${WORKDIR}/3_Outputs"

cd "$CODEDIR"
mkdir -p "$OUTDIR"

# Safe default for local bash testing
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SCHEMA_XLSX="/nfs/turbo/umms-atjanke/liuwent/Schema/20260415/pe-schema.xlsx"
echo "Using schema: ${SCHEMA_XLSX}"

# ------------------------------------------------------------------
# Provider / model
# ------------------------------------------------------------------
export LLM_API_PROVIDER="mistral_local"
export MISTRAL_MODEL_DIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/Mistral/7B"

# ------------------------------------------------------------------
# Generation / runtime settings
# ------------------------------------------------------------------
export MISTRAL_USE_4BIT=0
export MISTRAL_DTYPE="bfloat16"
export MISTRAL_MAX_NEW_TOKENS=1200
export MISTRAL_TEMPERATURE=0.0
export MISTRAL_USE_CACHE=0
export MISTRAL_CHUNK_TOKENS=9000
export MISTRAL_MAX_CHUNKS=0

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

echo "=== ENV ==="
date
hostname
echo "WORKDIR=$WORKDIR"
echo "CODEDIR=$CODEDIR"
echo "DATADIR=$DATADIR"
echo "OUTDIR=$OUTDIR"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}"
which python
python --version
nvidia-smi

# ------------------------------------------------------------------
# Smoke test on shard 0 only
# ------------------------------------------------------------------
if [[ "${ARRAY_TASK_ID}" == "0" ]]; then
  echo "=== SINGLE NOTE SMOKE TEST (shard 0 only) ==="
  python -u - << 'PY'
import pandas as pd
import subprocess
import sys
from pathlib import Path

codedir = Path(".").resolve()
datadir = codedir.parent / "1_Data"

df = pd.read_csv(datadir / "notes-for-1000-cases.csv")
note = str(df["Text"].iloc[0])

cmd = [
    sys.executable,
    str(codedir / "llm-chart-abstraction-call_Mistral7B.py"),
    "--api-provider", "mistral_local",
    "--model", "mistral-7b",
    "--repair",
    "--quote-per-var",
    "--var", "smoke_test_field:presence:Does the note mention a clinically relevant PE-related finding?"
]

print("Launching smoke test command:")
print(" ".join(cmd))
sys.stdout.flush()

p = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

p.stdin.write(note)
p.stdin.close()

for line in p.stdout:
    print(line, end="")

rc = p.wait()
print("\nSmoke test returncode:", rc)
if rc != 0:
    raise SystemExit(rc)
PY
fi

# ------------------------------------------------------------------
# Run shard
# ------------------------------------------------------------------
echo "=== RUN BATCH SHARD ==="
NUM_SHARDS="${NUM_SHARDS:-20}"
SHARD="${ARRAY_TASK_ID}"

python -u ./batch-abstract-notes-logged_Mistral7B.py \
  --input ../1_Data/notes-for-1000-cases.csv \
  --output "notes-for-1000-cases-pe-schema-mistral7b_shard${SHARD}.parquet" \
  --note-col Text \
  --id-col EncounterCsn \
  --script ./llm-chart-abstraction-call_Mistral7B.py \
  --model mistral-7b \
  --api-provider mistral_local \
  --num-shards ${NUM_SHARDS} \
  --shard-index ${SHARD} \
  --repair \
  --quote-per-var \
  --rpm 100000 \
  --checkpoint-every 1 \
  --json-out "debug-notes-for-1000-cases-pe-schema-mistral7b-job${SLURM_JOB_ID:-local}-shard${SHARD}.json" \
  --var "shortness_of_breath:presence:Does the note indicate the patient is complaining about shortness of breath?" \
  --var "chest_pain:presence:Does the note indicate the patient is complaining about chest pain?" \
  --var "pleuritic_pain:presence:Does the note indicate that there is a 'pleuritic' pain (a chest, back, or other thoracic or truncal pain that is explicitly worse with breathing)? If the patient does *not* have any pain complaint, then mark 'explicitly absent.'" \
  --var "back_pain:presence:Does the note indicate the patient is complaining about back pain?" \
  --var "cough:presence:Does the note indicate the patient is complaining of cough?" \
  --var "hemoptysis_present:presence:Look for any mention of the patient coughing up blood. Include synonyms like “bloody sputum” or “blood-tinged mucus.” If not described, mark as “not mentioned.” If cough is 'explicitly absent', then mark hemotypsis as 'explicitly absent.'" \
  --var "syncope:presence:Does the note indicate syncope or passing out or similar?" \
  --var "exertional_symptoms:presence:Does the note indicate problems with 'exertion,' like exertional shortness of breath, or feeling weak with walking? If the patient has no symptoms, mark as 'explicitly absent.'" \
  --var "unilateral_leg_swelling_present:presence:Look for a physical exam or history describing one-sided leg swelling or asymmetry. Include phrases like “right calf swelling,” “asymmetric edema.” Do not include joint swelling alone (like 'right knee swelling'), as this is not indicative of one-sided leg swelling or asymmetry. Mark as “present,” “explicitly absent,” or “not mentioned.”" \
  --var "prior_dvt_pe_present:presence:Determine whether the patient has a documented history of DVT or pulmonary embolism. Accept phrases like “prior clot,” “history of PE,” or “old DVT.” Mark as “present,” “explicitly absent,” or “not mentioned.”" \
  --var "recent_surgery:presence:Does the note indicate that the patient has had a recent surgery requiring general anesthesia, stated or implied to be recent, especially in the past 4 weeks?" \
  --var "recent_immobilization:presence:Does the note indicate the patient has had prolonged immobilization, stated or implied to be for 3 days or greater?" \
  --var "recent_travel:presence:Does the note indicate recent long-distance travel, either explicitly 'prolonger' or implied to be lasting for 4 hours or greater?" \
  --var "recent_or_active_malignancy:presence:Does the note indicate whether the patient has recent (prior 6 months) or active malignancy? If malignancy history is specified but reportedly treated/in remission or similar without specified timeframe within preceding 6 months, mark as explicitly absent." \
  --var "estrogen_use_present:presence:Check for current use of estrogen-containing medications, such as oral contraceptives, hormone replacement therapy, or transdermal estrogen. Do NOT include IUD as part of this (like NuvaRing), as we are only looking for oral estrogen. If use is explicitly denied, mark as “explicitly absent.”" \
  --var "pregnancy:yn:Does the note indicate that the patient is pregnant?" \
  --var "perc_mentioned:yn:Does the note mention the PERC or 'PE Rule Out' criteria?" \
  --var "wells_mentioned:yn:Does the note mention the Wells criteria?"

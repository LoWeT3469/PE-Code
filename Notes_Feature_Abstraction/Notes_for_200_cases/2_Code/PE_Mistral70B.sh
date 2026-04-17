#!/bin/bash
# LWT

#SBATCH --job-name=PE_OrcaHermes70B
#SBATCH --account=atjanke0
#SBATCH --qos=normal
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%4
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
WORKDIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/Notes_for_200_cases"
CODEDIR="${WORKDIR}/2_Code"
DATADIR="${WORKDIR}/1_Data"
OUTDIR="${WORKDIR}/3_Outputs"

cd "$CODEDIR"
mkdir -p "$OUTDIR"

# Safe default for local bash testing
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ------------------------------------------------------------------
# Provider / model
# ------------------------------------------------------------------
export LLM_API_PROVIDER="hf_local"
export HF_MODEL_ID="cookinai/OrcaHermes-Mistral-70B-miqu"

# ------------------------------------------------------------------
# Precision / generation
# ------------------------------------------------------------------
export HF_USE_4BIT=1
export HF_DTYPE="bfloat16"
export HF_MAX_NEW_TOKENS=1200
export HF_TEMPERATURE=0.0
export HF_USE_CACHE=0

# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
export HF_CHUNK_TOKENS=512
export HF_MAX_CHUNKS=0

# ------------------------------------------------------------------
# Runtime knobs
# ------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKDIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

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
# Stream logs instead of hiding them
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

df = pd.read_csv(datadir / "notes-for-200-cases.csv")
note = str(df["Text"].iloc[0])

cmd = [
    sys.executable,
    str(codedir / "llm-chart-abstraction-call_Mistral70B.py"),
    "--api-provider", "hf_local",
    "--model", "cookinai/OrcaHermes-Mistral-70B-miqu",
    "--repair",
    "--quote-per-var",
    "--var", "shortness_of_breath:presence:Does the note indicate SOB?"
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
NUM_SHARDS="${NUM_SHARDS:-32}"
SHARD="${ARRAY_TASK_ID}"

python -u ./batch-abstract-notes-logged_Mistral70B.py \
  --input ../1_Data/notes-for-200-cases.csv \
  --output "notes-for-200-cases-18-features-orcahermes70b_shard${SHARD}.parquet" \
  --note-col Text \
  --id-col EncounterCsn \
  --script ./llm-chart-abstraction-call_Mistral70B.py \
  --model cookinai/OrcaHermes-Mistral-70B-miqu \
  --api-provider hf_local \
  --num-shards ${NUM_SHARDS} \
  --shard-index ${SHARD} \
  --repair \
  --quote-per-var \
  --rpm 100000 \
  --checkpoint-every 1 \
  --json-out "debug-${SLURM_JOB_ID}-shard${SHARD}.json" \
  --var "shortness_of_breath:presence:Does the note indicate the patient is complaining about shortness of breath?" \
  --var "chest_pain:presence:Does the note indicate the patient is complaining about chest pain?" \
  --var "pleuritic_pain:presence:Does the note indicate that there is a 'pleuritic' pain (a chest, back, or other pain that is explicitly worse with breathing)? If the patient does *not* have any pain complaint, then mark 'explicitly absent.'" \
  --var "back_pain:presence:Does the note indicate the patient is complaining about back pain?" \
  --var "cough:presence:Does the note indicate the patient is complaining of cough?" \
  --var "hemoptysis_present:presence:Look for any mention of the patient coughing up blood. Include synonyms like 'bloody sputum' or 'blood-tinged mucus.' If not described, mark as 'not mentioned.'" \
  --var "syncope:presence:Does the note indicate syncope or passing out or similar?" \
  --var "exertional_symptoms:presence:Does the note indicate problems with exertion, like exertional shortness of breath?" \
  --var "unilateral_leg_swelling_present:presence:Look for one-sided leg swelling or asymmetry." \
  --var "prior_dvt_pe_present:presence:History of DVT or PE?" \
  --var "recent_surgery:presence:Recent surgery requiring general anesthesia in past 4 weeks?" \
  --var "recent_immobilization:presence:Prolonged immobilization (≥3 days)?" \
  --var "recent_travel:presence:Recent long-distance travel ≥4 hours?" \
  --var "recent_or_active_malignancy:presence:Recent (≤6 months) or active malignancy?" \
  --var "estrogen_use_present:presence:Current oral estrogen use?" \
  --var "pregnancy:yn:Is the patient pregnant?" \
  --var "perc_mentioned:yn:Does the note mention PERC?" \
  --var "wells_mentioned:yn:Does the note mention Wells criteria?"

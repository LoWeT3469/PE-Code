#!/bin/bash

#SBATCH --job-name=CT_OrcaHermes70B_1000
#SBATCH --account=atjanke0
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%4
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
SCHEMA_XLSX="/nfs/turbo/umms-atjanke/liuwent/Schema/ct-chest-schema.xlsx"
echo "Using schema: ${SCHEMA_XLSX}"

# ------------------------------------------------------------------
# Provider / model
# ------------------------------------------------------------------
export LLM_API_PROVIDER="hf_local"
export HF_MODEL_ID="cookinai/OrcaHermes-Mistral-70B-miqu"

# ------------------------------------------------------------------
# Generation / runtime settings
# ------------------------------------------------------------------
export HF_USE_4BIT=1
export HF_DTYPE="bfloat16"
export HF_MAX_NEW_TOKENS=1200
export HF_MIN_NEW_TOKENS=1
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
# ------------------------------------------------------------------
if [[ "${ARRAY_TASK_ID}" == "0" ]]; then
  echo "=== SINGLE REPORT SMOKE TEST (shard 0 only) ==="
  python -u - << 'PY'
import pandas as pd
import subprocess
import sys
from pathlib import Path

codedir = Path(".").resolve()
datadir = codedir.parent / "1_Data"

df = pd.read_csv(datadir / "ct-reports-for-1000-cases.csv")
report = str(df["Text"].iloc[0])

cmd = [
    sys.executable,
    str(codedir / "llm-chart-abstraction-call_Mistral70B.py"),
    "--api-provider", "hf_local",
    "--model", "cookinai/OrcaHermes-Mistral-70B-miqu",
    "--repair",
    "--quote-per-var",
    "--var", "smoke_test_field:presence:Does the report mention a clinically relevant PE-related CT finding?"
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

p.stdin.write(report)
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
NUM_SHARDS="${NUM_SHARDS:-40}"
SHARD="${ARRAY_TASK_ID}"

python -u ./batch-abstract-notes-logged_Mistral70B.py \
  --input ../1_Data/ct-reports-for-1000-cases.csv \
  --output "../3_Outputs/ct-reports-for-1000-cases-ct-schema-orcahermes70b_shard${SHARD}.parquet" \
  --note-col Text \
  --id-col EncounterCsn \
  --script ./llm-chart-abstraction-call_Mistral70B.py \
  --model cookinai/OrcaHermes-Mistral-70B-miqu \
  --api-provider hf_local \
  --num-shards ${NUM_SHARDS} \
  --shard-index ${SHARD} \
  --repair \
  --quote-per-var \
  --timeout-s 1200 \
  --rpm 100000 \
  --checkpoint-every 1 \
  --json-out "../3_Outputs/debug-ct-reports-for-1000-cases-ct-schema-orcahermes70b-job${SLURM_JOB_ID:-local}-shard${SHARD}.json" \
  --var "acute_pe_present:yn:Does the CT chest radiology report indicate an acute pulmonary embolism? Look for explicit language such as acute pulmonary embolism, filling defect consistent with PE, or similar. If the report describes only chronic PE findings without acute findings, mark no." \
  --var "pe_size:presence:What is the most proximal extent of PE described in the report? Use present for a described category and explicitly absent if a more proximal category is ruled out. Only answer meaningfully if acute_pe_present is yes." \
  --var "pe_distribution:presence:Does the report indicate PE involves the left pulmonary vasculature, the right pulmonary vasculature, or both bilateral? A saddle PE should be considered bilateral. If PE is confirmed but not clearly lateralized, mark not mentioned." \
  --var "equivocal_language:presence:Does the report use hedging or uncertainty language regarding PE, such as cannot exclude PE, suspicious for PE, motion artifact, poor contrast bolus, or a recommendation for follow-up imaging or clinical correlation?" \
  --var "rv_strain_present:yn:Does the radiology report indicate right heart strain or right ventricular strain due to pulmonary embolism?" \
  --var "alternative_diagnosis_present:yn:Does the report mention an alternative diagnosis that could explain symptoms, such as pneumonia, pleural effusion, pulmonary edema, pneumothorax, aortic pathology, or another acute chest process?" \
  --var "pneumonia_present:yn:Does the report mention pneumonia, infiltrate suspicious for infection, or consolidation consistent with pneumonia?" \
  --var "pleural_effusion_present:yn:Does the report mention a pleural effusion?" \
  --var "pulmonary_edema_present:yn:Does the report mention pulmonary edema or CHF/fluid overload type lung findings?" \
  --var "pneumothorax_present:yn:Does the report mention pneumothorax?" \
  --var "aortic_pathology_present:yn:Does the report mention acute aortic pathology such as dissection, aneurysm rupture, or intramural hematoma?" \
  --var "chronic_pe_only:yn:Does the report describe chronic pulmonary embolism findings without acute pulmonary embolism?" \
  --var "study_limited:yn:Is the study described as limited or suboptimal for evaluating PE due to motion artifact, poor bolus timing, or another technical issue?" \
  --var "followup_recommended:yn:Does the report recommend follow-up imaging, additional testing, or clinical correlation specifically because PE assessment is uncertain or limited?"

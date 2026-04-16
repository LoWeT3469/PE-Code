#!/bin/bash
# LWT

#SBATCH --job-name=PE_Mistral7B
#SBATCH --account=atjanke0
#SBATCH --qos=normal
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=12:00:00
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

WORKDIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction"
cd "$WORKDIR"
mkdir -p outputs

export LLM_API_PROVIDER="mistral_local"
export MISTRAL_MODEL_DIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/Mistral/7B"

export MISTRAL_USE_4BIT=0
export MISTRAL_MAX_NEW_TOKENS=1200
export MISTRAL_TEMPERATURE=0.0
export MISTRAL_USE_CACHE=0

export MISTRAL_CHUNK_TOKENS=9000
export MISTRAL_MAX_CHUNKS=0

export PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "=== ENV ==="
hostname
echo "WORKDIR=$WORKDIR"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
nvidia-smi

# Optional: keep smoke test for shard 0 only (saves time)
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  echo "=== SINGLE NOTE SMOKE TEST (shard 0 only) ==="
  python - << 'PY'
import pandas as pd, subprocess, sys
df = pd.read_csv("./notes-for-200-cases.csv")
note = str(df["Text"].iloc[0])
cmd = [sys.executable, "./llm-chart-abstraction-call_Mistral7B.py",
       "--api-provider", "mistral_local",
       "--var", "shortness_of_breath:presence:Does the note indicate SOB?",
       "--repair"]
p = subprocess.run(cmd, input=note, text=True, capture_output=True)
print("returncode:", p.returncode)
print("STDERR tail:", (p.stderr or "")[-800:])
print("STDOUT head:", (p.stdout or "")[:500])
PY
fi

echo "=== RUN BATCH SHARD ==="
NUM_SHARDS=8
SHARD=${SLURM_ARRAY_TASK_ID}

python -u ./batch-abstract-notes-logged_Mistral7B.py \
  --input ./notes-for-200-cases.csv \
  --output notes-for-200-cases-18-features-mistral7b_shard${SHARD}.parquet \
  --note-col Text \
  --id-col EncounterCsn \
  --script ./llm-chart-abstraction-call_Mistral7B.py \
  --model mistral-7b \
  --api-provider mistral_local \
  --num-shards ${NUM_SHARDS} \
  --shard-index ${SHARD} \
  --repair \
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

#!/bin/bash
#SBATCH --job-name=PE_stage1_200
#SBATCH --account=atjanke0
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=./%x-%j.out
#SBATCH --error=./%x-%j.err
#SBATCH --mail-user=liuwent@med.umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -eo pipefail

if [[ ${SLURM_JOB_NODELIST:-} ]]; then
  echo "Running on:"
  scontrol show hostnames "$SLURM_JOB_NODELIST"
fi

source "$HOME/.bashrc"
set -u

conda activate PE

WORKDIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation/2_Code"
cd "$WORKDIR" || exit 1

set -a
source "/nfs/turbo/umms-atjanke/liuwent/gpt.env"
set +a

export HTTPS_PROXY="http://proxy1.arc-ts.umich.edu:3128/"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY=""
export no_proxy=""

# Keep these for compatibility/debugging
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
export LLM_API_PROVIDER="${LLM_API_PROVIDER:-azure}"

PROJECT_ROOT="/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation"

mkdir -p "$PROJECT_ROOT/3_Outputs"
mkdir -p "$PROJECT_ROOT/3_Outputs/stage1_extractions"
mkdir -p "$PROJECT_ROOT/3_Outputs/manifests"

echo "DEBUG: PROVIDER=$LLM_API_PROVIDER  BASE_URL=${OPENAI_BASE_URL:-<unset>}  API_BASE=${OPENAI_API_BASE:-<unset>}  MODEL=${MODEL:-<unset>}"

echo "===== Stage 1: Building prompt packets ====="
python -u 01_subset_labeled_200.py

echo "===== Stage 1: Running extraction ====="
python -u 02_stage1_extract.py \
  --api-provider "${LLM_API_PROVIDER}" \
  --model "gpt-5" \
  --max-cases 200 \
  --sleep-seconds 2 \
  --timeout-seconds 180 \
  --max-retries 4 \
  --triage-max-chars 2000 \
  --provider-max-chars 6000 \
  --ct-max-chars 3000 \
  --resume

echo "===== Stage 1 complete ====="
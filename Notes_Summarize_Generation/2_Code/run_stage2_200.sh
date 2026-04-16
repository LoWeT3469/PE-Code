#!/bin/bash
#SBATCH --job-name=PE_stage2_200
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

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"
export LLM_API_PROVIDER="${LLM_API_PROVIDER:-azure}"

PROJECT_ROOT="/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation"

mkdir -p "$PROJECT_ROOT/3_Outputs"
mkdir -p "$PROJECT_ROOT/3_Outputs/stage2_final_json"
mkdir -p "$PROJECT_ROOT/3_Outputs/manifests"
mkdir -p "$PROJECT_ROOT/3_Outputs/guidelines_chunks"

echo "DEBUG: PROVIDER=$LLM_API_PROVIDER  BASE_URL=${OPENAI_BASE_URL:-<unset>}  API_BASE=${OPENAI_API_BASE:-<unset>}  MODEL=${MODEL:-<unset>}"

echo "===== Stage 2: Creating guideline chunks ====="
python -u 03_chunk_guidelines.py

echo "===== Stage 2: Running RAG summarization ====="
python -u 04_stage2_with_rag.py \
  --api-provider "${LLM_API_PROVIDER}" \
  --model "gpt-5" \
  --top-k-national 3 \
  --top-k-local 3 \
  --sleep-seconds 2 \
  --timeout-seconds 180 \
  --max-retries 4 \
  --resume

echo "===== Stage 2 complete ====="
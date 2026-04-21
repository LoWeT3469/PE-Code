#!/bin/bash
# LWT

#SBATCH --job-name=PE_GPT4o
#SBATCH --account=atjanke0
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=./%x-%j
#SBATCH --error=./%x-%j
#SBATCH --mail-user=liuwent@med.umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e -o pipefail

if [[ ${SLURM_JOB_NODELIST:-} ]]; then
  echo "Running on:"; scontrol show hostnames "$SLURM_JOB_NODELIST"
fi

# ---- Conda env ----
source "$HOME/.bashrc"
conda activate PE

# ---- Paths ----
WORKDIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction"
cd "$WORKDIR" || exit 1

# ---- API env + proxy (exactly like your CLI) ----
set -a
source "/nfs/turbo/umms-atjanke/liuwent/gpt.env"
set +a
export HTTPS_PROXY="http://proxy1.arc-ts.umich.edu:3128/"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY=""
export no_proxy=""

# >>> key fix so SDKs/wrappers hit UM-GPT base <<<
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

echo "DEBUG: BASE_URL=$OPENAI_BASE_URL  API_BASE=$OPENAI_API_BASE  MODEL=${MODEL:-<unset>}"

mkdir -p outputs

SCHEMA_XLSX="/nfs/turbo/umms-atjanke/liuwent/Schema/20260415/pe-schema.xlsx"
echo "Using schema: ${SCHEMA_XLSX}"

# ---- Run (match your CLI: use python -u, not srun) ----
python -u ./batch-abstract-notes-logged.py \
  --input ./notes-for-200-cases.csv \
  --output ./notes-for-200-cases-18-features-gpt4o.parquet \
  --note-col Text \
  --id-col EncounterCsn \
  --script ./llm-chart-abstraction-call.py \
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
  --var "wells_mentioned:yn:Does the note mention the Wells criteria?" \
  --exp-all \
  --quote-per-var \
  --repair \
  --rps 2 \
  --checkpoint-every 10 \
  --model gpt-4o \
  --json-out "./outputs/debug-${SLURM_JOB_ID:-local}-gpt4o.json"

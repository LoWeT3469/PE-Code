#!/bin/bash
# Auto-generated GPT script for 1000-case run

#SBATCH --job-name=CT_GPT5Nano_1000
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
  echo "Running on:"
  scontrol show hostnames "$SLURM_JOB_NODELIST"
fi

source "$HOME/.bashrc"
conda activate PE

WORKDIR="/nfs/turbo/umms-atjanke/liuwent/Notes_Feature_Abstraction/Notes_for_1000_cases"
cd "$WORKDIR" || exit 1

set -a
source "/nfs/turbo/umms-atjanke/liuwent/gpt.env"
set +a
export HTTPS_PROXY="http://proxy1.arc-ts.umich.edu:3128/"
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY=""
export no_proxy=""
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

mkdir -p outputs

python -u ./batch-abstract-notes-logged.py \
  --input "./ct-reports-for-1000-cases.csv" \
  --output "ct-reports-for-1000-cases-ct-schema-gpt5nano.parquet" \
  --note-col Text \
  --id-col EncounterCsn \
  --script ./llm-chart-abstraction-call.py \
  --var "acute_pe_present:yn:Does the CT chest radiology report indicate an acute pulmonary embolism (PE)? Look for explicit language such as \"acute pulmonary embolism,\" \"filling defect consistent with PE,\" or similar. If the report describes only chronic PE findings (e.g., \"chronic thrombus,\" \"web or band\") without acute findings, mark \"no.\"" \
  --var "pe_size:presence:What is the most proximal (largest) extent of PE described in the radiology report? \"Saddle\" indicates thrombus straddling the main pulmonary artery bifurcation. \"Main pulmonary artery\" indicates thrombus in the right or left main pulmonary artery. \"Lobar\" indicates thrombus extending to lobar branches. \"Segmental\" indicates thrombus limited to segmental arteries. \"Subsegmental\" indicates thrombus isolated to subsegmental branches. If the report confirms PE but does not specify the anatomic level, mark \"not specified.\" Only answer if acute_pe_present is \"yes.\"" \
  --var "pe_distribution:presence:Does the radiology report indicate the PE involves the left pulmonary vasculature, the right pulmonary vasculature, or both (bilateral)? A saddle PE should be marked \"bilateral.\" If the report confirms PE but does not clearly lateralize or is ambiguous, mark \"not specified.\" Only answer if acute_pe_present is \"yes.\"" \
  --var "equivocal_language:presence:Does the radiology report use equivocal or hedging language regarding the presence of PE? \"No equivocal language\" means the report gives a definitive positive or negative interpretation. \"Possible or suspected PE\" means the report uses language like \"cannot exclude PE,\" \"suspicious for PE,\" \"possible filling defect,\" or \"findings suggestive of PE.\" \"Motion or technical limitation noted\" means the report explicitly cites motion artifact, poor contrast bolus, or suboptimal study quality as limiting PE assessment. \"Low confidence or recommend follow-up\" means the report expresses low diagnostic confidence or recommends additional imaging (e.g., V/Q scan, repeat CTA, or clinical correlation for PE)." \
  --exp-all \
  --quote-per-var \
  --repair \
  --rps 2 \
  --checkpoint-every 10 \
  --model gpt-5-nano \
  --json-out "ct-reports-for-1000-cases-ct-schema-gpt5nano.json"

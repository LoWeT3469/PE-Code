from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"
CHUNK_DIR = OUTPUT_ROOT / "guidelines_chunks"


NATIONAL_CHUNKS = [
    {
        "chunk_id": "national_clinical_assessment_overview",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: overview of PE evaluation and diagnosis",
        "tags": ["diagnosis", "workflow", "overview"],
        "text": (
            "Evaluation of suspected acute pulmonary embolism should begin with structured clinical assessment, "
            "not reflexive imaging. Initial evaluation should consider symptoms, signs, hemodynamic status, and "
            "pretest probability. Diagnostic testing should be selected according to the estimated clinical "
            "probability, with D-dimer reserved for appropriate low- or intermediate-probability settings and "
            "imaging used when clinical probability or test results warrant it."
        ),
    },
    {
        "chunk_id": "national_pretest_probability_tools",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: pretest probability tools",
        "tags": ["pretest probability", "Wells", "Geneva", "YEARS", "PERC"],
        "text": (
            "Pretest probability is central to PE diagnosis. Structured approaches may include clinician gestalt "
            "or validated tools such as Wells, revised Geneva, YEARS, or PERC in the right clinical context. "
            "These tools are intended to reduce unnecessary imaging while maintaining safety. The appropriate "
            "tool depends on the clinical setting and whether the patient is low risk, intermediate risk, or high "
            "risk at presentation."
        ),
    },
    {
        "chunk_id": "national_low_risk_perc",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: PERC in low-risk patients",
        "tags": ["PERC", "low risk", "rule out"],
        "text": (
            "In patients with sufficiently low pretest probability, the pulmonary embolism rule-out criteria "
            "(PERC) can be used to avoid further PE testing when all criteria are satisfied. PERC is not meant "
            "for patients with intermediate or high clinical probability and should not be used as a substitute "
            "for clinical judgment in patients with concerning features."
        ),
    },
    {
        "chunk_id": "national_ddimer_standard_and_age_adjusted",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: D-dimer and age-adjusted strategies",
        "tags": ["D-dimer", "age adjusted", "low risk", "intermediate risk"],
        "text": (
            "D-dimer testing is most useful in patients with low or intermediate pretest probability. A negative "
            "D-dimer in the appropriate clinical setting can support exclusion of PE without imaging. "
            "Age-adjusted D-dimer strategies may be used in selected older adults to reduce unnecessary CT "
            "pulmonary angiography while maintaining diagnostic safety."
        ),
    },
    {
        "chunk_id": "national_years_strategy",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: YEARS-based diagnostic strategy",
        "tags": ["YEARS", "D-dimer", "probability adjusted"],
        "text": (
            "A YEARS-based strategy combines selected clinical features with D-dimer interpretation to simplify "
            "evaluation of suspected PE and potentially reduce imaging. This approach is most relevant in "
            "patients without high-risk hemodynamic instability and should be used as part of a structured "
            "diagnostic pathway rather than as an isolated rule."
        ),
    },
    {
        "chunk_id": "national_high_probability_imaging",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: when high clinical probability should go to imaging",
        "tags": ["high probability", "imaging", "CTPA"],
        "text": (
            "When clinical probability is high, further diagnostic evaluation should generally proceed directly "
            "to imaging rather than relying on D-dimer to exclude disease. A normal D-dimer is less useful in "
            "patients with sufficiently high clinical suspicion, and delayed definitive imaging may be unsafe in "
            "patients with strong concern for PE."
        ),
    },
    {
        "chunk_id": "national_ctpa_preferred",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: CT pulmonary angiography as preferred imaging",
        "tags": ["CTPA", "imaging", "diagnosis"],
        "text": (
            "CT pulmonary angiography is generally the preferred imaging test when PE imaging is indicated. It "
            "can both diagnose PE and identify alternative thoracic pathology. Interpretation must still be "
            "considered in light of pretest probability, especially if scan quality is limited or if results are "
            "discordant with strong clinical suspicion."
        ),
    },
    {
        "chunk_id": "national_vq_when_ctpa_unsuitable",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: V/Q scanning when CTPA is unsuitable",
        "tags": ["V/Q", "CT contraindication", "pregnancy", "contrast"],
        "text": (
            "Ventilation-perfusion scanning is an appropriate alternative when CTPA is contraindicated, not "
            "feasible, or less desirable, such as in selected patients with contrast limitations or situations "
            "where radiation distribution matters. V/Q testing is most informative when paired with an "
            "appropriate pretest probability framework."
        ),
    },
    {
        "chunk_id": "national_ultrasound_supportive_testing",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: lower-extremity ultrasound as supportive testing",
        "tags": ["ultrasound", "DVT", "supportive testing"],
        "text": (
            "Lower-extremity venous ultrasound may support diagnosis in selected patients, especially when "
            "imaging for PE is delayed, contraindicated, or when coexisting DVT would change diagnostic "
            "confidence. It is not a universal substitute for thoracic imaging, but may be helpful in structured "
            "diagnostic algorithms."
        ),
    },
    {
        "chunk_id": "national_hemodynamic_assessment",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: hemodynamic assessment in suspected or confirmed PE",
        "tags": ["hemodynamics", "shock", "hypotension", "risk"],
        "text": (
            "Hemodynamic assessment is essential early in suspected PE. Hypotension, shock, vasopressor "
            "requirement, or other evidence of circulatory compromise identifies patients who need urgent "
            "escalation and risk stratification. Hemodynamic instability is not merely a prognostic detail; it "
            "changes the urgency and structure of diagnostic evaluation."
        ),
    },
    {
        "chunk_id": "national_biomarkers_and_rv_strain",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: biomarkers and RV strain for severity assessment",
        "tags": ["troponin", "BNP", "RV strain", "severity"],
        "text": (
            "After PE is diagnosed, or when severity assessment is needed, biomarkers such as troponin and "
            "natriuretic peptides and imaging evidence of right ventricular dysfunction can help identify "
            "intermediate or higher-risk disease. These findings are used to characterize severity and short-term "
            "risk, not as stand-alone diagnostic tests for PE."
        ),
    },
    {
        "chunk_id": "national_risk_scores_after_diagnosis",
        "source": "2026 national PE guideline",
        "scope": "Section 3",
        "title": "National: risk scores and post-diagnosis stratification",
        "tags": ["PESI", "sPESI", "risk stratification", "disposition"],
        "text": (
            "Once PE is confirmed or strongly suspected, structured risk stratification may incorporate PESI, "
            "simplified PESI, hemodynamics, biomarkers, and RV imaging findings. These tools help separate "
            "lower-risk patients from those needing closer monitoring, higher-acuity placement, or advanced "
            "therapies."
        ),
    },
]

LOCAL_CHUNKS = [
    {
        "chunk_id": "local_diagnostic_workflow_overview",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: overview of PE diagnostic workflow",
        "tags": ["workflow", "diagnosis", "overview"],
        "text": (
            "Michigan Medicine uses a structured PE evaluation pathway that starts with clinical suspicion and "
            "pretest probability, then uses rule-out tools and D-dimer in appropriate patients, followed by "
            "imaging when indicated. The local workflow is intended to reduce unnecessary imaging while "
            "maintaining safety and making escalation pathways clearer."
        ),
    },
    {
        "chunk_id": "local_pretest_probability_and_rule_selection",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: pretest probability and rule selection",
        "tags": ["pretest probability", "PERC", "Wells", "diagnosis"],
        "text": (
            "The local diagnostic pathway emphasizes estimating pretest probability before ordering advanced "
            "testing. PERC is most relevant in sufficiently low-risk patients, while Wells or similar frameworks "
            "help organize the diagnostic approach in patients who cannot be ruled out clinically. The purpose "
            "is to match D-dimer and imaging decisions to the underlying probability of disease."
        ),
    },
    {
        "chunk_id": "local_perc_low_risk_rule_out",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: PERC in low-risk patients",
        "tags": ["PERC", "low risk", "rule out"],
        "text": (
            "In low-risk patients, the local pathway allows PE to be ruled out clinically when PERC is negative. "
            "If PERC is not satisfied, the patient generally moves forward in the diagnostic pathway rather than "
            "being ruled out clinically. PERC is not intended for higher-risk presentations."
        ),
    },
    {
        "chunk_id": "local_wells_and_pretest_escalation",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: Wells-based escalation",
        "tags": ["Wells", "pretest probability", "diagnostic pathway"],
        "text": (
            "The local pathway uses Wells-style pretest framing to decide whether D-dimer is appropriate or "
            "whether imaging should be pursued more directly. This supports a probability-adjusted approach "
            "rather than uniform testing for every patient with chest pain or dyspnea."
        ),
    },
    {
        "chunk_id": "local_ddimer_threshold",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: D-dimer threshold",
        "tags": ["D-dimer", "threshold", "diagnosis"],
        "text": (
            "The local guideline uses a standard negative D-dimer threshold of less than 0.50 mg/L FEU in the "
            "appropriate clinical setting. D-dimer is used to help exclude PE in patients whose pretest "
            "probability is not high enough to require direct imaging."
        ),
    },
    {
        "chunk_id": "local_age_adjusted_ddimer",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: age-adjusted D-dimer",
        "tags": ["age adjusted", "D-dimer", "older adults"],
        "text": (
            "The local guideline supports age-adjusted D-dimer interpretation in older adults when clinically "
            "appropriate. This is intended to reduce unnecessary CT pulmonary angiography while preserving "
            "diagnostic safety, especially in patients with low or intermediate clinical probability."
        ),
    },
    {
        "chunk_id": "local_ctpa_preferred",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: CTPA as preferred imaging",
        "tags": ["CTPA", "imaging", "diagnosis"],
        "text": (
            "When imaging is indicated in the local pathway, CT pulmonary angiography is generally preferred. "
            "It provides direct evaluation for pulmonary embolism and may identify alternative thoracic causes "
            "for symptoms. As with national guidance, interpretation should still be considered in the context "
            "of clinical suspicion."
        ),
    },
    {
        "chunk_id": "local_vq_alternative_pathway",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE diagnostic workflow",
        "title": "Michigan Medicine: V/Q when CTPA is less suitable",
        "tags": ["V/Q", "contrast limitation", "alternative imaging"],
        "text": (
            "The local pathway uses V/Q scanning when CTPA is not appropriate or less suitable, such as when "
            "iodinated contrast or CT-based evaluation should be avoided. This keeps patients within a formal "
            "diagnostic pathway rather than leaving them incompletely evaluated."
        ),
    },
    {
        "chunk_id": "local_rv_strain_and_biomarker_context",
        "source": "Michigan Medicine VTE guideline",
        "scope": "PE risk assessment",
        "title": "Michigan Medicine: RV dysfunction and biomarker context",
        "tags": ["RV strain", "troponin", "BNP", "severity"],
        "text": (
            "The local guideline uses evidence of right ventricular dysfunction and biomarker abnormality as "
            "part of severity assessment after PE is diagnosed or strongly suspected. These findings help guide "
            "monitoring intensity and disposition rather than serving as stand-alone diagnostic rule-out tools."
        ),
    },
    {
        "chunk_id": "local_pesi_disposition_context",
        "source": "Michigan Medicine VTE guideline",
        "scope": "Disposition and monitoring",
        "title": "Michigan Medicine: PESI and disposition context",
        "tags": ["PESI", "disposition", "monitoring"],
        "text": (
            "The local guideline uses PESI or similar structured risk stratification to help determine whether a "
            "patient may be suitable for lower-acuity management or requires admission and closer monitoring. "
            "PESI is interpreted alongside symptoms, cardiopulmonary reserve, RV findings, and practical "
            "disposition concerns."
        ),
    },
    {
        "chunk_id": "local_admission_and_outpatient_considerations",
        "source": "Michigan Medicine VTE guideline",
        "scope": "Disposition and monitoring",
        "title": "Michigan Medicine: admission versus outpatient considerations",
        "tags": ["admission", "outpatient", "disposition"],
        "text": (
            "Local disposition decisions consider more than the presence of PE alone. Patients with marked "
            "symptoms, low cardiopulmonary reserve, hemodynamic concern, significant comorbidity, or other "
            "clinical instability generally need a higher level of monitoring. Lower-risk patients may be "
            "considered for less intensive settings only when the full clinical picture supports it."
        ),
    },
    {
        "chunk_id": "local_pert_and_escalation",
        "source": "Michigan Medicine VTE guideline",
        "scope": "Escalation pathways",
        "title": "Michigan Medicine: PERT and escalation pathways",
        "tags": ["PERT", "escalation", "high risk PE"],
        "text": (
            "Michigan Medicine uses a PE Response Team pathway for urgent consultation in severe or higher-risk "
            "cases. This supports rapid multidisciplinary input when patients have hypotension, shock, major RV "
            "strain, or other features suggesting need for advanced evaluation or intervention. Even when your "
            "current summarization task is diagnosis-focused, this chunk is useful for understanding why certain "
            "cases are escalated locally."
        ),
    },
]


def validate_chunks(chunks: list[dict], label: str) -> None:
    required = {"chunk_id", "source", "scope", "title", "tags", "text"}
    seen_ids: set[str] = set()

    for idx, chunk in enumerate(chunks, start=1):
        missing = required - set(chunk.keys())
        if missing:
            raise ValueError(f"{label} chunk #{idx} missing keys: {sorted(missing)}")
        cid = chunk["chunk_id"]
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk_id in {label}: {cid}")
        seen_ids.add(cid)
        if not isinstance(chunk["tags"], list):
            raise ValueError(f"{label} chunk {cid} has non-list tags")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--national-out",
        type=Path,
        default=CHUNK_DIR / "national_chunks.json",
        help="Output path for national chunks",
    )
    p.add_argument(
        "--local-out",
        type=Path,
        default=CHUNK_DIR / "local_chunks.json",
        help="Output path for local chunks",
    )
    return p.parse_args()


def main():
    args = parse_args()

    validate_chunks(NATIONAL_CHUNKS, "national")
    validate_chunks(LOCAL_CHUNKS, "local")

    national_out = args.national_out
    local_out = args.local_out

    national_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.parent.mkdir(parents=True, exist_ok=True)

    with national_out.open("w", encoding="utf-8") as f:
        json.dump(NATIONAL_CHUNKS, f, ensure_ascii=False, indent=2)

    with local_out.open("w", encoding="utf-8") as f:
        json.dump(LOCAL_CHUNKS, f, ensure_ascii=False, indent=2)

    print("Guideline chunks created:")
    print(f"  National chunks: {len(NATIONAL_CHUNKS)}")
    print(f"  Local chunks:    {len(LOCAL_CHUNKS)}")
    print(f"  National → {national_out}")
    print(f"  Local    → {local_out}")


if __name__ == "__main__":
    main()
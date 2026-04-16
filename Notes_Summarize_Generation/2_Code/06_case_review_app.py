from __future__ import annotations

import streamlit as st

st.set_page_config(layout="wide", page_title="PE Case Review")

import json
from pathlib import Path
from typing import Any

import pandas as pd


def find_project_root() -> Path:
    candidates = [
        Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation"),
        Path("/Volumes/umms-atjanke/liuwent/Notes_Summarize_Generation"),
        Path(__file__).resolve().parent.parent,
    ]
    for p in candidates:
        if (p / "3_Outputs").exists():
            return p
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"

PACKET_FILE = OUTPUT_ROOT / "labeled_200_prompt_packets.jsonl"
STAGE1_DIR = OUTPUT_ROOT / "stage1_extractions"
STAGE2_DIR = OUTPUT_ROOT / "stage2_final_json"


@st.cache_data
def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data
def load_json_dir(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for f in sorted(path.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f, "r", encoding="utf-8") as fp:
            obj = json.load(fp)
        encounter_id = str(obj.get("encounter_id", f.stem))
        out[encounter_id] = obj
    return out


def pretty_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def list_to_lines(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return "\n".join(f"- {x}" for x in value)
    if isinstance(value, dict):
        return pretty_json(value)
    return str(value)


def stage1_readable_text(stage1: dict[str, Any]) -> dict[str, str]:
    if not stage1:
        return {}

    presenting = stage1.get("presenting_features", {}) or {}
    risk = stage1.get("risk_factors", {}) or {}
    diagnostic = stage1.get("diagnostic_data", {}) or {}
    pretest = stage1.get("pretest_probability_elements", {}) or {}

    sections = {
        "Symptoms": list_to_lines(presenting.get("symptoms", [])),
        "Vitals or instability": list_to_lines(presenting.get("vitals_or_instability", [])),
        "Exam findings": list_to_lines(presenting.get("exam_findings", [])),
        "Key negatives": list_to_lines(presenting.get("key_negatives", [])),
        "Evidence for PE": list_to_lines(stage1.get("evidence_for_pe", [])),
        "Evidence against PE / alternatives": list_to_lines(
            stage1.get("evidence_against_pe_or_alternatives", [])
        ),
        "Missing key history": list_to_lines(stage1.get("missing_key_history", [])),
        "Risk factors": pretty_json(risk),
        "Diagnostic data": pretty_json(diagnostic),
        "Pretest probability elements": pretty_json(pretest),
        "Teaching candidates": list_to_lines(stage1.get("teaching_candidates", [])),
        "Uncertainty flags": list_to_lines(stage1.get("uncertainty_flags", [])),
    }
    return sections


def stage2_readable_keys(stage2: dict[str, Any]) -> list[str]:
    preferred = [
        "One-line case summary",
        "Evidence for PE",
        "Evidence against PE / alternative diagnoses",
        "Missing key history",
        "Immediate next diagnostic considerations",
        "Teaching note for residents",
        "Confidence / uncertainty statement",
    ]
    return [k for k in preferred if k in stage2]


def extract_retrieved_chunks(stage2: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieved = stage2.get("retrieved_chunks", {}) if stage2 else {}
    national = retrieved.get("national", []) if isinstance(retrieved, dict) else []
    local = retrieved.get("local", []) if isinstance(retrieved, dict) else []
    return national, local


def build_consistency_checks(stage1: dict[str, Any], stage2: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    if not stage1 or not stage2:
        return checks

    ct_status = (
        stage1.get("diagnostic_data", {})
        .get("ctpa_or_ct_chest", {})
        .get("status", "")
    )
    stage2_for = str(stage2.get("Evidence for PE", "") or "")
    stage2_against = str(stage2.get("Evidence against PE / alternative diagnoses", "") or "")
    uncertainty = str(stage2.get("Confidence / uncertainty statement", "") or "")

    if ct_status == "positive_for_pe" and "pe" not in stage2_for.lower():
        checks.append(
            "CT is positive_for_pe in Stage 1, but Stage 2 evidence-for text may not clearly mention PE."
        )

    if ct_status == "negative_for_pe" and "negative" not in stage2_against.lower() and "alternative" not in stage2_against.lower():
        checks.append(
            "CT is negative_for_pe in Stage 1, but Stage 2 evidence-against text may not clearly reflect that."
        )

    missing_key_history = stage1.get("missing_key_history", [])
    if missing_key_history and len(str(stage2.get("Missing key history", "") or "").strip()) == 0:
        checks.append(
            "Stage 1 contains missing key history items, but Stage 2 missing-history field looks empty."
        )

    if len(uncertainty.strip()) == 0:
        checks.append("Stage 2 confidence / uncertainty statement is empty.")

    return checks


packets_df = load_jsonl(PACKET_FILE)
stage1_map = load_json_dir(STAGE1_DIR)
stage2_map = load_json_dir(STAGE2_DIR)

st.title("PE Case Review")
st.caption(f"Project root: {PROJECT_ROOT}")

if packets_df.empty:
    st.error(f"No packet file found: {PACKET_FILE}")
    st.write("Available output directory:", OUTPUT_ROOT)
    if OUTPUT_ROOT.exists():
        st.write("Files in 3_Outputs:")
        st.write([p.name for p in sorted(OUTPUT_ROOT.iterdir())])
    st.stop()

if "encounter_id" not in packets_df.columns:
    st.error("The packet file does not contain an 'encounter_id' column.")
    st.stop()

packets_df["encounter_id"] = packets_df["encounter_id"].astype(str)

encounter_ids = sorted(packets_df["encounter_id"].tolist())
selected_id = st.sidebar.selectbox("Encounter ID", encounter_ids)

selected_rows = packets_df.loc[packets_df["encounter_id"] == selected_id]
if selected_rows.empty:
    st.error(f"Could not find encounter_id={selected_id} in packet file.")
    st.stop()

row = selected_rows.iloc[0].to_dict()
stage1 = stage1_map.get(selected_id, {})
stage2 = stage2_map.get(selected_id, {})
national_chunks, local_chunks = extract_retrieved_chunks(stage2)
consistency_checks = build_consistency_checks(stage1, stage2)

st.sidebar.write("Available data")
st.sidebar.write(f"Stage 1: {'yes' if stage1 else 'no'}")
st.sidebar.write(f"Stage 2: {'yes' if stage2 else 'no'}")
st.sidebar.write(f"National chunks: {len(national_chunks)}")
st.sidebar.write(f"Local chunks: {len(local_chunks)}")

top_left, top_right = st.columns([1, 1.25])

with top_left:
    st.subheader("Source notes")

    note_tabs = st.tabs(["Triage", "Provider", "CT / Imaging"])

    with note_tabs[0]:
        st.text_area(
            "triage_text",
            row.get("triage_text", ""),
            height=250,
            key="triage_text_box",
            label_visibility="collapsed",
        )

    with note_tabs[1]:
        st.text_area(
            "provider_text",
            row.get("provider_text", ""),
            height=450,
            key="provider_text_box",
            label_visibility="collapsed",
        )

    with note_tabs[2]:
        st.text_area(
            "ct_text",
            row.get("ct_text", ""),
            height=300,
            key="ct_text_box",
            label_visibility="collapsed",
        )

with top_right:
    st.subheader("Model outputs")

    output_tabs = st.tabs(
        [
            "Stage 1 readable",
            "Stage 1 raw JSON",
            "Stage 2 readable",
            "Stage 2 raw JSON",
            "Retrieved chunks",
            "Consistency checks",
        ]
    )

    with output_tabs[0]:
        if stage1:
            readable = stage1_readable_text(stage1)
            for section_name, section_text in readable.items():
                st.markdown(f"**{section_name}**")
                st.text_area(
                    section_name,
                    section_text,
                    height=120 if section_name not in {"Risk factors", "Diagnostic data", "Pretest probability elements"} else 220,
                    key=f"stage1_readable_{section_name}",
                    label_visibility="collapsed",
                )
        else:
            st.info("Stage 1 output missing.")

    with output_tabs[1]:
        if stage1:
            st.text_area(
                "stage1_json",
                pretty_json(stage1),
                height=750,
                key="stage1_json_box",
                label_visibility="collapsed",
            )
        else:
            st.info("Stage 1 output missing.")

    with output_tabs[2]:
        if stage2:
            for key in stage2_readable_keys(stage2):
                st.markdown(f"**{key}**")
                height = 110
                if key in {
                    "Teaching note for residents",
                    "Evidence for PE",
                    "Evidence against PE / alternative diagnoses",
                }:
                    height = 180
                st.text_area(
                    key,
                    "" if stage2[key] is None else str(stage2[key]),
                    height=height,
                    key=f"stage2_readable_{key}",
                    label_visibility="collapsed",
                )

            extra_keys = [
                k for k in stage2.keys()
                if k not in stage2_readable_keys(stage2)
                and k not in {"_usage", "retrieved_chunks", "retrieval_metadata", "encounter_id"}
            ]
            if extra_keys:
                st.markdown("**Additional Stage 2 fields**")
                for key in extra_keys:
                    st.markdown(f"**{key}**")
                    st.text_area(
                        key,
                        "" if stage2[key] is None else str(stage2[key]),
                        height=100,
                        key=f"stage2_extra_{key}",
                        label_visibility="collapsed",
                    )
        else:
            st.info("Stage 2 output missing.")

    with output_tabs[3]:
        if stage2:
            st.text_area(
                "stage2_json",
                pretty_json(stage2),
                height=750,
                key="stage2_json_box",
                label_visibility="collapsed",
            )
            if "_usage" in stage2:
                st.markdown("**Usage metadata**")
                st.write(stage2["_usage"])
            if "retrieval_metadata" in stage2:
                st.markdown("**Retrieval metadata**")
                st.write(stage2["retrieval_metadata"])
        else:
            st.info("Stage 2 output missing.")

    with output_tabs[4]:
        subtabs = st.tabs(["National", "Local"])

        with subtabs[0]:
            if national_chunks:
                for i, chunk in enumerate(national_chunks, start=1):
                    title = chunk.get("title", f"National chunk {i}")
                    st.markdown(f"**{i}. {title}**")
                    st.caption(
                        f"chunk_id={chunk.get('chunk_id', '')} | source={chunk.get('source', '')} | scope={chunk.get('scope', '')}"
                    )
                    st.text_area(
                        f"national_chunk_{i}",
                        chunk.get("text", ""),
                        height=160,
                        key=f"national_chunk_box_{i}",
                        label_visibility="collapsed",
                    )
            else:
                st.info("No retrieved national chunks found in Stage 2 output.")

        with subtabs[1]:
            if local_chunks:
                for i, chunk in enumerate(local_chunks, start=1):
                    title = chunk.get("title", f"Local chunk {i}")
                    st.markdown(f"**{i}. {title}**")
                    st.caption(
                        f"chunk_id={chunk.get('chunk_id', '')} | source={chunk.get('source', '')} | scope={chunk.get('scope', '')}"
                    )
                    st.text_area(
                        f"local_chunk_{i}",
                        chunk.get("text", ""),
                        height=160,
                        key=f"local_chunk_box_{i}",
                        label_visibility="collapsed",
                    )
            else:
                st.info("No retrieved local chunks found in Stage 2 output.")

    with output_tabs[5]:
        if consistency_checks:
            for msg in consistency_checks:
                st.warning(msg)
        else:
            st.success("No simple consistency problems detected by the current heuristic checks.")

        st.markdown("**Quick structured context**")
        quick_context = {
            "encounter_id": selected_id,
            "stage1_ct_status": (
                stage1.get("diagnostic_data", {})
                .get("ctpa_or_ct_chest", {})
                .get("status", None)
                if stage1 else None
            ),
            "stage1_missing_key_history_n": len(stage1.get("missing_key_history", [])) if stage1 else None,
            "stage2_has_uncertainty_statement": bool(
                str(stage2.get("Confidence / uncertainty statement", "") or "").strip()
            ) if stage2 else None,
            "retrieved_national_chunks": len(national_chunks),
            "retrieved_local_chunks": len(local_chunks),
        }
        st.json(quick_context)
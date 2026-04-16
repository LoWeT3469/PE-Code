from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
INPUT_ROOT = PROJECT_ROOT / "1_InputData" / "Notes"
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"

DEFAULT_INPUT = OUTPUT_ROOT / "labeled_200_prompt_packets.jsonl"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "stage1_extractions"
DEFAULT_MANIFEST = OUTPUT_ROOT / "manifests" / "stage1_manifest.csv"


SYSTEM_PROMPT = """
You are a clinical abstraction assistant for retrospective pulmonary embolism chart review.

This is Stage 1 only.
Extract structured facts related to PE evaluation and diagnosis from the notes.

Rules:
- Use only the text provided.
- Do not recommend treatment.
- Do not invent findings.
- If information is absent, use "not_documented".
- Return valid JSON only.
"""

DEVELOPER_PROMPT = """
Return a JSON object with exactly these keys:

{
  "encounter_id": "string",
  "presenting_features": {
    "symptoms": [],
    "vitals_or_instability": [],
    "exam_findings": [],
    "key_negatives": []
  },
  "evidence_for_pe": [],
  "evidence_against_pe_or_alternatives": [],
  "missing_key_history": [],
  "risk_factors": {
    "prior_vte": "present|absent|not_documented",
    "recent_surgery_or_immobilization": "present|absent|not_documented",
    "malignancy": "present|absent|not_documented",
    "estrogen_or_pregnancy_related": "present|absent|not_documented",
    "leg_symptoms_or_dvt_history": "present|absent|not_documented",
    "other": []
  },
  "diagnostic_data": {
    "d_dimer": {"status": "present|absent|not_documented", "value_text": "string"},
    "troponin": {"status": "present|absent|not_documented", "value_text": "string"},
    "bnp_or_ntprobnp": {"status": "present|absent|not_documented", "value_text": "string"},
    "lactate": {"status": "present|absent|not_documented", "value_text": "string"},
    "ctpa_or_ct_chest": {"status": "positive_for_pe|negative_for_pe|alternative_finding_only|not_done|unclear|not_documented", "supporting_text": "string"},
    "vq_scan": {"status": "positive|negative|not_done|unclear|not_documented", "supporting_text": "string"},
    "lower_ext_ultrasound": {"status": "positive_dvt|negative_dvt|not_done|unclear|not_documented", "supporting_text": "string"},
    "echo_or_rv_strain": {"status": "rv_strain_present|rv_strain_absent|not_done|unclear|not_documented", "supporting_text": "string"}
  },
  "pretest_probability_elements": {
    "wells_related_elements": [],
    "perc_related_elements": [],
    "years_related_elements": [],
    "explicit_pretest_probability_statement": "string"
  },
  "teaching_candidates": [],
  "uncertainty_flags": []
}
"""

USER_TEMPLATE = """
Encounter ID: {encounter_id}

TRIAGE NOTE:
{triage_text}

PROVIDER NOTE:
{provider_text}

CT / IMAGING TEXT:
{ct_text}

Extract Stage 1 structured facts only.
Return JSON only.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p.add_argument("--model", default=os.environ.get("MODEL") or os.environ.get("GPT_MODEL", "gpt-4o"))
    p.add_argument("--api-provider", choices=["azure", "openai"], default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-version", default=None)
    p.add_argument("--organization", default=None)

    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=2.0)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=4)

    p.add_argument("--triage-max-chars", type=int, default=2000)
    p.add_argument("--provider-max-chars", type=int, default=6000)
    p.add_argument("--ct-max-chars", type=int, default=3000)

    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def create_llm_client(args: argparse.Namespace) -> AzureOpenAI | OpenAI:
    provider = (args.api_provider or os.environ.get("LLM_API_PROVIDER") or "azure").strip().lower()
    if provider not in {"azure", "openai"}:
        sys.exit(f"Unsupported --api-provider '{provider}'. Use 'azure' or 'openai'.")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Missing API key. Provide via --api-key or OPENAI_API_KEY env var.")

    organization = args.organization or os.environ.get("OPENAI_ORGANIZATION")

    if provider == "azure":
        azure_endpoint = (
            args.api_base
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        api_version = args.api_version or os.environ.get("API_VERSION")
        if not azure_endpoint:
            sys.exit(
                "Missing Azure endpoint. Provide via --api-base or OPENAI_API_BASE/AZURE_OPENAI_ENDPOINT env var."
            )
        if not api_version:
            sys.exit("Missing Azure API version. Provide via --api-version or API_VERSION env var.")
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            organization=organization,
            timeout=args.timeout_seconds,
            max_retries=0,
        )

    base_url = (
        args.api_base
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout_seconds,
        "max_retries": 0,
    }
    if organization:
        client_kwargs["organization"] = organization
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def append_manifest(manifest_path: Path, row: dict[str, Any]) -> None:
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def truncate(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_prompt(
    encounter_id: str,
    row: dict[str, Any],
    triage_max_chars: int,
    provider_max_chars: int,
    ct_max_chars: int,
) -> str:
    triage_text = truncate(row.get("triage_text", ""), triage_max_chars)
    provider_text = truncate(row.get("provider_text", ""), provider_max_chars)
    ct_text = truncate(row.get("ct_text", ""), ct_max_chars)

    return USER_TEMPLATE.format(
        encounter_id=encounter_id,
        triage_text=triage_text,
        provider_text=provider_text,
        ct_text=ct_text,
    )


def extract_chat_text(resp: Any) -> str:
    try:
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Could not extract chat completion text: {e}") from e


def create_completion_with_retry(
    client: AzureOpenAI | OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
) -> Any:
    last_error: Exception | None = None

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{DEVELOPER_PROMPT}"},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                raise
            sleep_for = min(20.0, 2 ** attempt) + random.uniform(0, 1)
            print(
                f"Retry {attempt}/{max_retries - 1} after error: {e} "
                f"(sleeping {sleep_for:.1f}s)"
            )
            time.sleep(sleep_for)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected retry failure without captured exception.")


def main() -> None:
    load_dotenv(PROJECT_ROOT / "gpt.env")
    args = parse_args()

    client = create_llm_client(args)

    input_path = args.input
    output_dir = args.output_dir
    manifest_path = args.manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]

    total = len(rows)
    provider = (args.api_provider or os.environ.get("LLM_API_PROVIDER") or "azure").strip().lower()

    print(f"Loaded {total} encounters from {input_path}")
    print(
        f"Using provider={provider}, model={args.model}, "
        f"timeout={args.timeout_seconds}s, retries={args.max_retries}, sleep={args.sleep_seconds}s"
    )
    print(
        f"API base={args.api_base or os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')}"
    )
    print(
        f"Char limits: triage={args.triage_max_chars}, "
        f"provider={args.provider_max_chars}, ct={args.ct_max_chars}"
    )

    for idx, row in enumerate(rows, start=1):
        encounter_id = str(row.get("encounter_id", f"row_{idx}"))
        out_path = output_dir / f"{safe_name(encounter_id)}.json"

        if args.resume and out_path.exists():
            print(f"[{idx}/{total}] skip existing {encounter_id}")
            continue

        prompt = build_prompt(
            encounter_id=encounter_id,
            row=row,
            triage_max_chars=args.triage_max_chars,
            provider_max_chars=args.provider_max_chars,
            ct_max_chars=args.ct_max_chars,
        )

        started = time.time()
        status = "success"
        error_msg = ""

        try:
            resp = create_completion_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                max_retries=args.max_retries,
            )

            response_text = extract_chat_text(resp)
            result = json.loads(response_text)
            result["encounter_id"] = encounter_id

            if hasattr(resp, "usage") and resp.usage:
                result["_usage"] = {
                    "input_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "output_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"[{idx}/{total}] wrote {out_path.name}")

        except Exception as e:
            status = "failed"
            error_msg = str(e)
            print(f"[{idx}/{total}] FAILED {encounter_id}: {e}")

        append_manifest(
            manifest_path,
            {
                "encounter_id": encounter_id,
                "stage": "stage1",
                "status": status,
                "elapsed_seconds": round(time.time() - started, 2),
                "output_file": str(out_path),
                "provider": provider,
                "model": args.model,
                "timeout_seconds": args.timeout_seconds,
                "max_retries": args.max_retries,
                "triage_max_chars": args.triage_max_chars,
                "provider_max_chars": args.provider_max_chars,
                "ct_max_chars": args.ct_max_chars,
                "error": error_msg,
            },
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
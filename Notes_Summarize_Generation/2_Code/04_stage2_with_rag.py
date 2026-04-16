from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from utils_rag import (
    DEFAULT_LOCAL_CHUNKS,
    DEFAULT_NATIONAL_CHUNKS,
    format_chunks_for_prompt,
    load_chunks,
    retrieve_top_k,
)


PROJECT_ROOT = Path("/nfs/turbo/umms-atjanke/liuwent/Notes_Summarize_Generation")
OUTPUT_ROOT = PROJECT_ROOT / "3_Outputs"

DEFAULT_STAGE1_DIR = OUTPUT_ROOT / "stage1_extractions"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "stage2_final_json"
DEFAULT_MANIFEST = OUTPUT_ROOT / "manifests" / "stage2_manifest.csv"


SYSTEM_PROMPT = """
You are a clinical summarization assistant for retrospective pulmonary embolism case review.

This is Stage 2.
Use:
1) the Stage 1 extraction,
2) retrieved national guideline snippets,
3) retrieved Michigan Medicine local guideline snippets.

Task limits:
- Focus on PE evaluation and diagnosis only.
- Do not recommend treatment.
- Do not invent facts.
- If information is absent, say not documented.
- Return valid JSON only with exactly these keys:
  - One-line case summary
  - Evidence for PE
  - Evidence against PE / alternative diagnoses
  - Missing key history
  - Immediate next diagnostic considerations
  - Teaching note for residents
  - Confidence / uncertainty statement
"""

USER_TEMPLATE = """
STAGE 1 EXTRACTION:
{stage1_json}

RETRIEVED NATIONAL GUIDELINE SNIPPETS:
{national_text}

RETRIEVED LOCAL GUIDELINE SNIPPETS:
{local_text}

Return JSON only.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--national-chunks", type=Path, default=DEFAULT_NATIONAL_CHUNKS)
    p.add_argument("--local-chunks", type=Path, default=DEFAULT_LOCAL_CHUNKS)

    p.add_argument("--model", default=os.environ.get("MODEL") or os.environ.get("GPT_MODEL", "gpt-4o"))
    p.add_argument("--api-provider", choices=["azure", "openai"], default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-version", default=None)
    p.add_argument("--organization", default=None)

    p.add_argument("--top-k-national", type=int, default=3)
    p.add_argument("--top-k-local", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=2.0)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=4)
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


def append_manifest(manifest_path: Path, row: dict[str, Any]) -> None:
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        {"role": "system", "content": SYSTEM_PROMPT},
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

    stage1_dir = args.stage1_dir
    output_dir = args.output_dir
    manifest_path = args.manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    national_chunks = load_chunks(args.national_chunks)
    local_chunks = load_chunks(args.local_chunks)

    files = sorted([p for p in stage1_dir.glob("*.json") if not p.name.startswith("_")])
    if args.max_cases is not None:
        files = files[: args.max_cases]

    total = len(files)
    provider = (args.api_provider or os.environ.get("LLM_API_PROVIDER") or "azure").strip().lower()

    print(f"Loaded {total} Stage 1 files from {stage1_dir}")
    print(
        f"Using provider={provider}, model={args.model}, "
        f"timeout={args.timeout_seconds}s, retries={args.max_retries}, sleep={args.sleep_seconds}s"
    )
    print(
        f"API base={args.api_base or os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')}"
    )

    for idx, path in enumerate(files, start=1):
        stage1 = load_json(path)
        encounter_id = str(stage1.get("encounter_id", path.stem))
        out_path = output_dir / path.name

        if args.resume and out_path.exists():
            print(f"[{idx}/{total}] skip existing {encounter_id}")
            continue

        query_text = json.dumps(stage1, ensure_ascii=False)

        top_national = retrieve_top_k(
            query_text=query_text,
            chunks=national_chunks,
            k=args.top_k_national,
        )
        top_local = retrieve_top_k(
            query_text=query_text,
            chunks=local_chunks,
            k=args.top_k_local,
        )

        national_text = format_chunks_for_prompt(top_national)
        local_text = format_chunks_for_prompt(top_local)

        prompt = USER_TEMPLATE.format(
            stage1_json=json.dumps(stage1, ensure_ascii=False, indent=2),
            national_text=national_text,
            local_text=local_text,
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
            result["retrieved_chunks"] = {
                "national": top_national,
                "local": top_local,
            }
            result["retrieval_metadata"] = {
                "top_k_national": args.top_k_national,
                "top_k_local": args.top_k_local,
                "national_chunk_count": len(top_national),
                "local_chunk_count": len(top_local),
            }

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
                "stage": "stage2",
                "status": status,
                "elapsed_seconds": round(time.time() - started, 2),
                "output_file": str(out_path),
                "provider": provider,
                "model": args.model,
                "timeout_seconds": args.timeout_seconds,
                "max_retries": args.max_retries,
                "top_k_national": args.top_k_national,
                "top_k_local": args.top_k_local,
                "error": error_msg,
            },
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
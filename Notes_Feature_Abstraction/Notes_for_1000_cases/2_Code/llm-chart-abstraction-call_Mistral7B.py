#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Literal, List, Set

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, create_model, ValidationError, conint

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


# -------------------------------------------------------------------
# Environment loading
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ENV_CANDIDATES = [
    SCRIPT_DIR / "gpt.env",
    PROJECT_ROOT / "gpt.env",
    PROJECT_ROOT.parent / "gpt.env",
    SCRIPT_DIR.parent / "gpt.env",
    Path.cwd() / "gpt.env",
]

for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(env_path)
        break


# -------------------------------------------------------------------
# Local Mistral-style chat wrapper
# -------------------------------------------------------------------
class _LocalChatCompletions:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def create(self, model: str, messages):
        system = ""
        user = ""

        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = m.get("content") or ""
            if role == "system":
                system += content.strip() + "\n"
            elif role == "user":
                user += content.strip() + "\n"
            else:
                user += content.strip() + "\n"

        chat_messages = []
        if system.strip():
            chat_messages.append({"role": "system", "content": system.strip()})
        if user.strip():
            chat_messages.append({"role": "user", "content": user.strip()})

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "[SYSTEM]\n"
                f"{system.strip()}\n\n"
                "[USER]\n"
                f"{user.strip()}\n\n"
                "[ASSISTANT]\n"
            )

        max_new = int(os.environ.get("MISTRAL_MAX_NEW_TOKENS", "256"))
        min_new = int(os.environ.get("MISTRAL_MIN_NEW_TOKENS", "1"))
        temperature = float(os.environ.get("MISTRAL_TEMPERATURE", "0.0"))
        use_cache = os.environ.get("MISTRAL_USE_CACHE", "1").strip() not in {"0", "false", "False"}

        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_new,
            "min_new_tokens": min_new,
            "do_sample": (temperature > 0),
            "use_cache": use_cache,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self.model.generate(**inputs, **generate_kwargs)

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        class _Msg:
            pass

        class _Choice:
            pass

        class _Usage:
            pass

        class _Resp:
            pass

        msg = _Msg()
        msg.content = text

        choice = _Choice()
        choice.message = msg

        usage = _Usage()
        usage.prompt_tokens = int(inputs["input_ids"].numel())
        usage.completion_tokens = int(gen_ids.numel())
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        resp = _Resp()
        resp.choices = [choice]
        resp.usage = usage
        return resp


class _LocalClient:
    def __init__(self, model, tokenizer, device):
        self._tokenizer = tokenizer

        class _Chat:
            pass

        class _Completions:
            pass

        self.chat = _Chat()
        self.chat.completions = _Completions()
        self.chat.completions.create = _LocalChatCompletions(model, tokenizer, device).create


# -------------------------------------------------------------------
# Client factory
# -------------------------------------------------------------------
def create_llm_client(args):
    provider = (args.api_provider or os.environ.get("LLM_API_PROVIDER") or "azure").strip().lower()

    if provider == "mistral_local":
        model_dir = os.environ.get("MISTRAL_MODEL_DIR") or args.model
        if not model_dir:
            sys.exit("Missing MISTRAL_MODEL_DIR for mistral_local provider.")

        use_4bit = os.environ.get("MISTRAL_USE_4BIT", "0").strip() in {"1", "true", "True"}

        dtype_name = os.environ.get("MISTRAL_DTYPE", "bfloat16").strip().lower()
        if dtype_name == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "float16":
            torch_dtype = torch.float16
        else:
            sys.exit(f"Unsupported dtype '{dtype_name}'. Use bfloat16 or float16.")

        quant_cfg = None
        if use_4bit:
            if BitsAndBytesConfig is None:
                sys.exit("MISTRAL_USE_4BIT=1 but BitsAndBytesConfig not available.")
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=None if use_4bit else torch_dtype,
            device_map="auto",
            quantization_config=quant_cfg,
        )
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return _LocalClient(model, tokenizer, device)

    if provider not in {"azure", "openai"}:
        sys.exit(f"Unsupported --api-provider '{provider}'. Use 'azure', 'openai', or 'mistral_local'.")

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
            sys.exit("Missing Azure endpoint.")
        if not api_version:
            sys.exit("Missing Azure API version.")
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            organization=organization,
        )

    base_url = args.api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if organization:
        client_kwargs["organization"] = organization
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


# -------------------------------------------------------------------
# Schema definitions
# -------------------------------------------------------------------
KIND_TO_TYPE = {
    "yn": Literal["yes", "no"],
    "ynu": Literal["yes", "no", "uncertain"],
    "presence": Literal["present", "explicitly absent", "not mentioned"],
    "text": str,
    "text_opt": Optional[str],
}

ALLOWED_ENUM_VALUES: Dict[str, Set[str]] = {
    "yn": {"yes", "no"},
    "ynu": {"yes", "no", "uncertain"},
    "presence": {"present", "explicitly absent", "not mentioned"},
}

STRICT_DEFAULTS: Dict[str, Optional[str]] = {
    "yn": "no",
    "ynu": "uncertain",
    "presence": "not mentioned",
}


def _to_lower_str(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).strip().lower()


def norm_yn(value: Any) -> Any:
    if isinstance(value, bool):
        return "yes" if value else "no"
    s = _to_lower_str(value)
    if s in {"yes", "y", "true", "t", "1"}:
        return "yes"
    if s in {"no", "n", "false", "f", "0"}:
        return "no"
    return s


def norm_ynu(value: Any) -> Any:
    normalized = norm_yn(value)
    if normalized in {"yes", "no"}:
        return normalized
    s = _to_lower_str(value)
    if s in {"uncertain", "maybe", "unknown", "indeterminate", "equivocal"}:
        return "uncertain"
    return s


def norm_presence(value: Any) -> Any:
    if isinstance(value, bool):
        return "present" if value else "explicitly absent"
    s = _to_lower_str(value)
    if s in {"present", "pos", "positive", "y", "yes", "true", "1"}:
        return "present"
    if s in {
        "absent",
        "explicitly absent",
        "neg",
        "negative",
        "n",
        "no",
        "false",
        "0",
        "denied",
        "denies",
    }:
        return "explicitly absent"
    if s in {
        "not mentioned",
        "none mentioned",
        "not stated",
        "unknown",
        "uncertain",
        "not documented",
        "no mention",
        "none",
    }:
        return "not mentioned"
    return s


def parse_var(spec: str) -> Tuple[str, str, str]:
    parts = spec.split(":", 2)
    name = parts[0].strip()
    kind = (parts[1].strip() if len(parts) > 1 else "text")
    desc = (parts[2].strip() if len(parts) > 2 else "")
    return name, kind, desc


def build_schema(var_specs: List[str], explain_vars: Set[str], quote_per_var: bool):
    fields: Dict[str, tuple] = {}
    for spec in var_specs:
        name, kind, _ = parse_var(spec)
        if kind not in KIND_TO_TYPE:
            raise ValueError(f"Unknown kind '{kind}' for field '{name}'.")
        fields[name] = (KIND_TO_TYPE[kind], ...)
        fields[f"{name}_confidence_1to3"] = (conint(ge=1, le=3), ...)
        if name in explain_vars:
            fields[f"{name}_explanation"] = (str, ...)
        if quote_per_var:
            fields[f"{name}_quote"] = (Optional[str], None)

    fields["conflict_flag"] = (Literal["yes", "no", "uncertain"], ...)
    fields["conflict_explanation"] = (Optional[str], ...)
    return create_model("NoteResult", __base__=BaseModel, **fields)


def make_prompt(var_specs: List[str], explain_vars: Set[str], quote_per_var: bool):
    constraints = {
        "yn": 'Allowed values: "yes" or "no".',
        "ynu": 'Allowed values: "yes", "no", or "uncertain".',
        "presence": 'Allowed values: "present", "explicitly absent", or "not mentioned".',
        "text": "Free text string.",
        "text_opt": "Optional free text string.",
    }

    lines = [
        "You are an expert clinical note interpreter.",
        "Return ONLY a single valid JSON object with exactly these fields (no extra keys, no prose):"
    ]

    for spec in var_specs:
        name, kind, desc = parse_var(spec)
        lines.append(f'- "{name}": {constraints[kind]} {desc}')
        lines.append(
            f'- "{name}_confidence_1to3": Integer 1–3 assessing confidence specifically for "{name}". '
            "Use 1 if language is unclear or conflicting, 2 for moderate/edge-case evidence, 3 for clear straightforward support."
        )
        if name in explain_vars:
            lines.append(
                f'- "{name}_explanation": Free text string. '
                "Provide 1–2 concise sentences (≤30 words) justifying the value, "
                "ideally including a brief quote (≤15 words) from the note."
            )
        if quote_per_var:
            lines.append(
                f'- "{name}_quote": Exact verbatim substring (≤200 chars) from the note that most directly supports the value for "{name}". '
                'If you cannot identify direct support, return "".'
            )

    lines.extend([
        '- "conflict_flag": One of "yes","no","uncertain". Set to "yes" only when the note contains conflicting clinical information about the patient (e.g., a symptom or finding is both affirmed and denied in different passages). Use "uncertain" if you suspect a clinical contradiction but evidence is unclear. Never use this field for disagreements between people.',
        '- "conflict_explanation": If conflict_flag is "yes" or "uncertain", provide a non-empty 1–2 sentence explanation describing the conflicting clinical statements, referencing the relevant portions of the note. If conflict_flag is "no", return null or "".',
    ])
    lines.append("Return ONLY one JSON object with exactly these fields; no extra keys; no prose or markdown.")
    return "\n".join(lines)


# -------------------------------------------------------------------
# Input / repair / validation helpers
# -------------------------------------------------------------------
def read_note(path: Optional[str]) -> str:
    if path:
        note_path = Path(path)
        if not note_path.is_absolute():
            candidate_paths = [
                Path.cwd() / note_path,
                SCRIPT_DIR / note_path,
                PROJECT_ROOT / note_path,
                PROJECT_ROOT / "1_Data" / note_path.name,
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    note_path = candidate
                    break

        with open(note_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return sys.stdin.read().strip()


def repair_values(parsed: Dict[str, Any], var_specs: List[str]) -> Dict[str, Any]:
    kind_by_name: Dict[str, str] = {}
    for spec in var_specs:
        n, k, _ = parse_var(spec)
        kind_by_name[n] = (k if k else "text")

    repaired = {}
    for k, v in parsed.items():
        if k.endswith("_explanation"):
            repaired[k] = v
            continue
        kind = kind_by_name.get(k)
        if kind == "yn":
            repaired[k] = norm_yn(v)
        elif kind == "ynu":
            repaired[k] = norm_ynu(v)
        elif kind == "presence":
            repaired[k] = norm_presence(v)
        else:
            repaired[k] = v
    return repaired


def enforce_choice_values(parsed: Dict[str, Any], var_specs: List[str]) -> Dict[str, Any]:
    enforced = dict(parsed)
    kind_by_name: Dict[str, str] = {}
    for spec in var_specs:
        n, k, _ = parse_var(spec)
        kind_by_name[n] = k if k else "text"

    for name, kind in kind_by_name.items():
        if kind not in {"yn", "presence"}:
            continue
        if name not in enforced:
            continue

        raw_value = enforced[name]
        if raw_value is None:
            continue

        candidate = norm_yn(raw_value) if kind == "yn" else norm_presence(raw_value)
        allowed = ALLOWED_ENUM_VALUES[kind]

        if isinstance(candidate, str) and candidate in allowed:
            enforced[name] = candidate
        else:
            enforced[name] = STRICT_DEFAULTS.get(kind)

    return enforced


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def ensure_conflict_explanation(parsed: Dict[str, Any]) -> None:
    flag = parsed.get("conflict_flag")
    if flag in {"yes", "uncertain"}:
        explanation = parsed.get("conflict_explanation")
        if not (isinstance(explanation, str) and explanation.strip()):
            parsed["conflict_explanation"] = "Possible conflicting statements in note."
    else:
        parsed["conflict_explanation"] = None


def compute_justification_span(note: str, quote: str) -> Optional[Dict[str, int]]:
    if not quote:
        return None

    raw_quote = quote
    stripped_quote = raw_quote.strip()
    if not stripped_quote:
        return None

    def trim_bounds(s: str, start_idx: int, end_idx: int) -> Tuple[int, int]:
        while start_idx < end_idx and s[start_idx].isspace():
            start_idx += 1
        while end_idx > start_idx and s[end_idx - 1].isspace():
            end_idx -= 1
        return start_idx, end_idx

    start = note.find(raw_quote)
    if start != -1:
        end = start + len(raw_quote)
        start, end = trim_bounds(note, start, end)
        return {"start": start, "end": end}

    start = note.find(stripped_quote)
    if start != -1:
        end = start + len(stripped_quote)
        start, end = trim_bounds(note, start, end)
        return {"start": start, "end": end}

    def normalize_with_spans(text: str) -> Tuple[str, List[Tuple[int, int]]]:
        normalized_chars: List[str] = []
        spans: List[Tuple[int, int]] = []
        i = 0
        length = len(text)
        while i < length:
            ch = text[i]
            if ch.isspace():
                start_ws = i
                while i < length and text[i].isspace():
                    i += 1
                if normalized_chars and normalized_chars[-1] == " ":
                    prev_start, _ = spans[-1]
                    spans[-1] = (prev_start, i)
                else:
                    normalized_chars.append(" ")
                    spans.append((start_ws, i))
            else:
                normalized_chars.append(ch)
                spans.append((i, i + 1))
                i += 1

        while normalized_chars and normalized_chars[0] == " ":
            normalized_chars.pop(0)
            spans.pop(0)
        while normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            spans.pop()

        return "".join(normalized_chars), spans

    normalized_note, spans = normalize_with_spans(note)
    normalized_quote = normalize_ws(stripped_quote)
    if not normalized_note or not normalized_quote:
        return None

    idx = normalized_note.find(normalized_quote)
    if idx == -1:
        return None

    end_idx = idx + len(normalized_quote) - 1
    if idx >= len(spans) or end_idx >= len(spans):
        return None

    span_start = spans[idx][0]
    span_end = spans[end_idx][1]
    span_start, span_end = trim_bounds(note, span_start, span_end)
    if span_start >= span_end:
        return None

    return {"start": span_start, "end": span_end}


HEDGE_TERMS = [
    "uncertain",
    "uncertainty",
    "possible",
    "possibly",
    "suggest",
    "suggests",
    "consider",
    "may represent",
    "cannot exclude",
    "likely",
    "question",
    "equivocal",
    "appears",
]
LOW_CERTAINTY_VALUES = {"uncertain", "not mentioned", "unknown", "n/a", "na", "none", "indeterminate", "equivocal"}
STRONG_TERMS = [
    "definite",
    "definitive",
    "confirmed",
    "diagnosed",
    "clear",
    "clearly",
    "no evidence of",
    "denies",
    "resolved",
    "documented",
    "demonstrates",
]
HIGH_CERTAINTY_VALUES = {"present", "explicitly absent", "yes", "no", "true", "false"}


def _contains_term(text: str, terms: List[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms if term)


def calibrate_confidences(note: str, parsed: Dict[str, Any], var_names: List[str]) -> None:
    conflict_flag = (parsed.get("conflict_flag") or "").strip().lower()

    for name in var_names:
        conf_key = f"{name}_confidence_1to3"
        raw_conf = parsed.get(conf_key)
        try:
            score = int(raw_conf)
        except (TypeError, ValueError):
            score = 2

        value_text = str(parsed.get(name, "") or "").strip()
        value_lower = value_text.lower()
        explanation_text = str(parsed.get(f"{name}_explanation", "") or "").strip()
        quote_text = str(parsed.get(f"{name}_quote", "") or "").strip()

        if conflict_flag == "yes":
            score = 1
        elif conflict_flag == "uncertain":
            score = min(score, 2)

        if not quote_text and not explanation_text:
            score = min(score, 1)
        elif not quote_text or not explanation_text:
            score = min(score, 2)

        if value_lower in LOW_CERTAINTY_VALUES:
            score = 1
        elif quote_text and value_lower in HIGH_CERTAINTY_VALUES:
            score = max(score, 3)

        text_for_hints = " ".join(filter(None, [explanation_text, quote_text, value_text]))
        if text_for_hints:
            if _contains_term(text_for_hints, HEDGE_TERMS):
                score = min(score, 2)
                if _contains_term(text_for_hints, ["very uncertain", "not sure", "unclear"]):
                    score = 1
            if _contains_term(text_for_hints, STRONG_TERMS):
                score = max(score, 3)

        parsed[conf_key] = max(1, min(3, score))


# -------------------------------------------------------------------
# Chunking + merge
# -------------------------------------------------------------------
def chunk_text_by_tokens(tokenizer, text: str, max_tokens: int):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    for i in range(0, len(ids), max_tokens):
        yield tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True)


def merge_json_partials(client, model_name: str, system_prompt: str, partials: List[str]):
    reduce_user = (
        "You are given partial JSON outputs extracted from multiple note chunks.\n"
        "Merge them into ONE final JSON that matches the schema exactly.\n"
        "- Do not invent values.\n"
        "- If multiple chunks support the same field, prefer the best-supported one.\n"
        "- Preserve/merge quotes when present.\n\n"
        "PARTIALS (each element is JSON text):\n"
        f"{json.dumps(partials, ensure_ascii=False, indent=2)}\n"
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reduce_user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    return text, resp


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    final_resp = None

    ap = argparse.ArgumentParser(description="Extract structured vars from a clinical note via GPT Toolkit.")
    ap.add_argument("--note-file", help="Path to note text file (or pipe via stdin).")
    ap.add_argument("--var", action="append", required=True,
                    help="Field spec 'name:kind:desc'. Kinds: yn, ynu, presence, text, text_opt. Repeatable.")
    ap.add_argument("--exp", action="append", default=[],
                    help="Add an explanation field for this variable (repeatable). Example: --exp acute_pe")
    ap.add_argument("--exp-all", action="store_true", help="Add explanations for all variables.")
    ap.add_argument("--model", default=os.environ.get("GPT_MODEL", "gpt-5-nano"))
    ap.add_argument("--api-provider", choices=["azure", "openai", "mistral_local"],
                    help="LLM API provider. Defaults to env LLM_API_PROVIDER or 'azure'.")
    ap.add_argument("--api-key", help="Override API key (defaults to OPENAI_API_KEY env).")
    ap.add_argument("--api-base",
                    help="Override API base URL (Azure endpoint or OpenAI base_url). Defaults to OPENAI_API_BASE/AZURE_OPENAI_ENDPOINT.")
    ap.add_argument("--api-version",
                    help="Override API version (Azure only). Defaults to API_VERSION env.")
    ap.add_argument("--organization",
                    help="Override organization/project ID. Defaults to OPENAI_ORGANIZATION env.")
    ap.add_argument("--repair", action="store_true",
                    help="Auto-normalize booleans/synonyms to required enums before validation.")
    ap.add_argument("--quote-per-var", action="store_true",
                    help="Ask the model to provide a supporting quote for each variable.")
    args = ap.parse_args()

    note = read_note(args.note_file)
    if not note:
        sys.exit("No note provided. Use --note-file or pipe note text via stdin.")

    var_names = [parse_var(s)[0] for s in args.var]
    explain_vars = set(var_names) if args.exp_all else set(args.exp or [])
    explain_vars = {v for v in explain_vars if v in var_names}

    NoteResult = build_schema(args.var, explain_vars, args.quote_per_var)
    prompt = make_prompt(args.var, explain_vars, args.quote_per_var)
    client = create_llm_client(args)

    provider = (args.api_provider or os.environ.get("LLM_API_PROVIDER") or "azure").strip().lower()

    if provider == "mistral_local":
        tokenizer = getattr(client, "_tokenizer", None)
        if tokenizer is None:
            sys.exit("Internal error: local client missing tokenizer.")

        chunk_tokens = int(os.environ.get("MISTRAL_CHUNK_TOKENS", "5500"))
        max_chunks = int(os.environ.get("MISTRAL_MAX_CHUNKS", "0"))

        partials = []
        last_map_resp = None

        for idx, chunk in enumerate(chunk_text_by_tokens(tokenizer, note, chunk_tokens), start=1):
            if max_chunks and idx > max_chunks:
                sys.exit(f"Too many chunks ({idx - 1}); increase MISTRAL_CHUNK_TOKENS or set MISTRAL_MAX_CHUNKS=0.")

            map_user = (
                f"NOTE CHUNK {idx}:\n\n{chunk}\n\n"
                "Return ONLY valid JSON for the schema. "
                "If a field is not supported in this chunk, still return it using the schema defaults.\n"
                "Do not add extra keys."
            )
            resp_i = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": map_user},
                ],
            )
            last_map_resp = resp_i
            partials.append(resp_i.choices[0].message.content.strip())

        if len(partials) == 1:
            text_out = partials[0]
            final_resp = last_map_resp
        else:
            text_out, final_resp = merge_json_partials(client, args.model, prompt, partials)

    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": note},
        ]
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
        )
        text_out = resp.choices[0].message.content.strip()
        final_resp = resp

    try:
        parsed = json.loads(text_out)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[error] Model did not return valid JSON:\n{text_out}\n{e}\n")
        sys.exit(2)

    if args.repair:
        parsed = repair_values(parsed, args.var)

    parsed = enforce_choice_values(parsed, args.var)
    ensure_conflict_explanation(parsed)
    calibrate_confidences(note, parsed, var_names)

    for name in var_names:
        qkey = f"{name}_quote"
        qval = parsed.get(qkey) or ""
        span_key = f"{name}_span"
        parsed[span_key] = compute_justification_span(note, qval) if qval else None

    try:
        core_payload = {k: v for k, v in parsed.items() if k in NoteResult.model_fields}
        validated = NoteResult(**core_payload)
        output = validated.model_dump()
    except ValidationError as e:
        sys.stderr.write(f"[error] Output failed validation:\n{e}\n")
        output = parsed
    else:
        for name in var_names:
            quote_key = f"{name}_quote"
            span_key = f"{name}_span"
            if args.quote_per_var or quote_key in parsed:
                output[quote_key] = parsed.get(quote_key)
            output[span_key] = parsed.get(span_key)

    if final_resp is not None and hasattr(final_resp, "usage") and final_resp.usage:
        output["_usage"] = {
            "input_tokens": final_resp.usage.prompt_tokens,
            "output_tokens": final_resp.usage.completion_tokens,
            "total_tokens": final_resp.usage.total_tokens,
        }
    else:
        output["_usage"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

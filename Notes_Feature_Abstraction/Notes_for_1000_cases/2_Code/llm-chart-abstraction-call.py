#!/usr/bin/env python3
import os, sys, argparse, json
from typing import Tuple, Optional, Dict, Any, Literal, List, Set, Union
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, create_model, ValidationError, conint

# ---- Load env ----
load_dotenv("../gpt.env")

def create_llm_client(args) -> Union[AzureOpenAI, OpenAI]:
    """Instantiate an Azure or OpenAI client based on CLI args/env."""
    provider = (args.api_provider or os.environ.get("LLM_API_PROVIDER") or "azure").strip().lower()
    if provider not in {"azure", "openai"}:
        sys.exit(f"Unsupported --api-provider '{provider}'. Use 'azure' or 'openai'.")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Missing API key. Provide via --api-key or OPENAI_API_KEY env var.")

    organization = args.organization or os.environ.get("OPENAI_ORGANIZATION")
    if provider == "azure":
        azure_endpoint = args.api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = args.api_version or os.environ.get("API_VERSION")
        if not azure_endpoint:
            sys.exit("Missing Azure endpoint. Provide via --api-base or OPENAI_API_BASE/AZURE_OPENAI_ENDPOINT env var.")
        if not api_version:
            sys.exit("Missing Azure API version. Provide via --api-version or API_VERSION env var.")
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

# ---- Kind -> Python type (STRICT) ----
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
    """Return (name, kind, desc)."""
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
        if (name in explain_vars):
            # Add required explanation field as free text
            fields[f"{name}_explanation"] = (str, ...)
        if quote_per_var:
            fields[f"{name}_quote"] = (Optional[str], None)
    fields["conflict_flag"] = (Literal["yes", "no", "uncertain"], ...)
    fields["conflict_explanation"] = (Optional[str], ...)
    return create_model("NoteResult", __base__=BaseModel, **fields)

def make_prompt(var_specs: List[str], explain_vars: Set[str], quote_per_var: bool):
    """Strict prompt insisting on exact labels + JSON only."""
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

def read_note(path: Optional[str]) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return sys.stdin.read().strip()

def repair_values(parsed: Dict[str, Any], var_specs: List[str]) -> Dict[str, Any]:
    """Normalize booleans/synonyms to strict enums (explanation fields untouched)."""
    kind_by_name: Dict[str, str] = {}
    for spec in var_specs:
        n, k, _ = parse_var(spec)
        kind_by_name[n] = (k if k else "text")

    repaired = {}
    for k, v in parsed.items():
        if k.endswith("_explanation"):
            repaired[k] = v  # leave text explanations untouched
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
    """Ensure yn/presence fields return only allowed literals, defaulting when needed."""
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
        if kind == "yn":
            candidate = norm_yn(raw_value)
        else:
            candidate = norm_presence(raw_value)
        if candidate is None:
            continue
        allowed = ALLOWED_ENUM_VALUES[kind]
        if isinstance(candidate, str) and candidate in allowed:
            enforced[name] = candidate
            continue
        fallback = STRICT_DEFAULTS.get(kind)
        if fallback is None:
            continue
        if isinstance(candidate, str) and candidate.strip():
            sys.stderr.write(
                f"[warn] Field '{name}' ({kind}) produced '{candidate}', defaulting to '{fallback}'.\n"
            )
        enforced[name] = fallback
    return enforced

def normalize_ws(text: str) -> str:
    """Collapse consecutive whitespace into single spaces (trimmed)."""
    return " ".join(text.split())

def ensure_conflict_explanation(parsed: Dict[str, Any]) -> None:
    """Ensure conflict_explanation aligns with the conflict_flag semantics."""
    flag = parsed.get("conflict_flag")
    if flag in {"yes", "uncertain"}:
        explanation = parsed.get("conflict_explanation")
        if not (isinstance(explanation, str) and explanation.strip()):
            parsed["conflict_explanation"] = "Possible conflicting statements in note."
            sys.stderr.write("[warn] conflict_flag is set but conflict_explanation was empty; default applied.\n")
    else:
        parsed["conflict_explanation"] = None

def compute_justification_span(note: str, quote: str) -> Optional[Dict[str, int]]:
    """Return first character-span mapping for quote within note, or None if absent."""
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

    # Direct match with original text
    start = note.find(raw_quote)
    if start != -1:
        end = start + len(raw_quote)
        start, end = trim_bounds(note, start, end)
        return {"start": start, "end": end}

    # Try trimmed version
    start = note.find(stripped_quote)
    if start != -1:
        end = start + len(stripped_quote)
        start, end = trim_bounds(note, start, end)
        return {"start": start, "end": end}

    # Normalized whitespace fallback
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
        # Trim leading/trailing spaces
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
    """Normalize per-variable confidence_1to3 values using field-level heuristics."""
    conflict_flag = (parsed.get("conflict_flag") or "").strip().lower()

    for name in var_names:
        conf_key = f"{name}_confidence_1to3"
        raw_conf = parsed.get(conf_key)
        try:
            score = int(raw_conf)
        except (TypeError, ValueError):
            score = None
        if score not in {1, 2, 3}:
            score = 2  # neutral baseline

        value_text = str(parsed.get(name, "") or "").strip()
        value_lower = value_text.lower()
        explanation_text = str(parsed.get(f"{name}_explanation", "") or "").strip()
        quote_text = str(parsed.get(f"{name}_quote", "") or "").strip()
        quote_present = bool(quote_text)
        explanation_present = bool(explanation_text)

        if conflict_flag == "yes":
            score = 1
        elif conflict_flag == "uncertain":
            score = min(score, 2)

        if not quote_present and not explanation_present:
            score = min(score, 1)
        elif not quote_present or not explanation_present:
            score = min(score, 2)

        if value_lower in LOW_CERTAINTY_VALUES:
            score = 1
        elif quote_present and value_lower in HIGH_CERTAINTY_VALUES:
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

def main():
    ap = argparse.ArgumentParser(description="Extract structured vars from a clinical note via GPT Toolkit.")
    ap.add_argument("--note-file", help="Path to note text file (or pipe via stdin).")
    ap.add_argument("--var", action="append", required=True,
                    help="Field spec 'name:kind:desc'. Kinds: yn, ynu, presence, text, text_opt. Repeatable.")
    ap.add_argument("--exp", action="append", default=[],
                    help="Add an explanation field for this variable (repeatable). Example: --exp acute_pe")
    ap.add_argument("--exp-all", action="store_true", help="Add explanations for all variables.")
    ap.add_argument("--model", default=os.environ.get("GPT_MODEL", "gpt-5-nano"))
    ap.add_argument("--api-provider", choices=["azure", "openai"],
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
    # sanity: ignore exp for vars that don't exist
    explain_vars = {v for v in explain_vars if v in var_names}

    NoteResult = build_schema(args.var, explain_vars, args.quote_per_var)
    prompt = make_prompt(args.var, explain_vars, args.quote_per_var)
    client = create_llm_client(args)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": note},
    ]
    resp = client.chat.completions.create(model=args.model, messages=messages)
    text_out = resp.choices[0].message.content.strip()

    # Parse
    try:
        parsed = json.loads(text_out)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[error] Model did not return valid JSON:\n{text_out}\n{e}\n")
        sys.exit(2)

    # Optional repair (explanations left untouched)
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

    # Validate
    try:
        core_payload = {k: v for k, v in parsed.items() if k in NoteResult.model_fields}
        validated = NoteResult(**core_payload)
        output = validated.model_dump()
    except ValidationError as e:
        sys.stderr.write(f"[error] Output failed validation:\n{e}\n")
        output = parsed  # dump raw, even if invalid
    else:
        for name in var_names:
            quote_key = f"{name}_quote"
            span_key = f"{name}_span"
            if args.quote_per_var or quote_key in parsed:
                output[quote_key] = parsed.get(quote_key)
            output[span_key] = parsed.get(span_key)

    # Token usage → embed in JSON
    if hasattr(resp, "usage") and resp.usage:
        output["_usage"] = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

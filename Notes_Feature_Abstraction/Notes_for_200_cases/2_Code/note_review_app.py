import hashlib
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "field"


def gather_var_names(df: pd.DataFrame) -> List[str]:
    suffix = "_quote"
    names = sorted(col[: -len(suffix)] for col in df.columns if col.endswith(suffix))
    if not names:
        raise ValueError("No *_quote columns detected. Re-run extraction with --quote-per-var enabled.")
    return names


SUPPORTED_SUFFIXES = {".parquet", ".csv"}

PRESENT_ABSENT_CHOICES = ["not mentioned", "explicitly absent", "present"]
YES_NO_CHOICES = ["yes", "no"]
YES_NO_UNCERTAIN_CHOICES = ["yes", "no", "uncertain"]

DEFAULT_LABEL_CHOICES = PRESENT_ABSENT_CHOICES
CONFIDENCE_LEVEL_LABELS = {1: "Low", 2: "Moderate", 3: "High"}

COMMON_SECTION_HEADERS = [
    "history",
    "history of present illness",
    "hpi",
    "review of systems",
    "ros",
    "past medical history",
    "pmh",
    "social history",
    "home medications",
    "medications",
    "physical exam",
    "physical examination",
    "ed results",
    "results",
    "ed course",
    "medical decision making",
    "ed course/medical decision making",
    "assessment",
    "plan",
    "disposition",
]
SECTION_HEADER_CANONICAL = {
    re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", header.lower())).strip()
    for header in COMMON_SECTION_HEADERS
}
SECTION_KEYWORD_HINTS = [
    "history",
    "system",
    "exam",
    "assessment",
    "plan",
    "course",
    "decision",
    "disposition",
    "medication",
    "results",
    "reasoning",
    "instructions",
    "differential",
]


def canonicalize_header(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def is_likely_header_token(token: str) -> bool:
    candidate = token.strip()
    if not candidate:
        return False
    canonical = canonicalize_header(candidate)
    if not canonical:
        return False
    if canonical in SECTION_HEADER_CANONICAL:
        return True
    if any(canonical.startswith(entry) for entry in SECTION_HEADER_CANONICAL):
        return True
    words = candidate.split()
    if candidate.isupper() and 1 <= len(words) <= 10:
        return True
    if len(words) <= 8 and any(keyword in canonical for keyword in SECTION_KEYWORD_HINTS):
        return True
    return False


def extract_header_from_line(line: str) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    if not line.strip():
        return None
    leading_ws = len(line) - len(line.lstrip())
    content = line[leading_ws:]
    if not content:
        return None
    colon_idx = content.find(":")
    if 0 <= colon_idx <= 80:
        header_candidate = content[:colon_idx].strip()
        if header_candidate and is_likely_header_token(header_candidate):
            header_start = leading_ws
            header_end = leading_ws + colon_idx + 1
            body_offset = header_end
            while body_offset < len(line) and line[body_offset].isspace():
                body_offset += 1
            if body_offset >= len(line):
                body_offset = None
            return {
                "label": header_candidate,
                "start": header_start,
                "end": header_end,
                "body_offset": body_offset,
            }
    stripped_content = content.rstrip()
    if stripped_content and len(stripped_content) <= 80 and is_likely_header_token(stripped_content):
        header_start = leading_ws
        header_end = header_start + len(stripped_content)
        return {"label": stripped_content, "start": header_start, "end": header_end, "body_offset": None}

    double_space_match = re.match(r"(\s*)([^\s].*?)(\s{2,}|\t+).+", line)
    if double_space_match:
        candidate = double_space_match.group(2).strip()
        if candidate and is_likely_header_token(candidate):
            header_start = len(double_space_match.group(1))
            header_end = header_start + len(candidate)
            body_offset = header_end
            while body_offset < len(line) and line[body_offset].isspace():
                body_offset += 1
            return {
                "label": candidate,
                "start": header_start,
                "end": header_end,
                "body_offset": body_offset if body_offset < len(line) else None,
            }

    dash_match = re.match(r"(\s*)([^\s].*?)(\s*[-–—]\s+).+", line)
    if dash_match:
        candidate = dash_match.group(2).strip()
        if candidate and is_likely_header_token(candidate):
            header_start = len(dash_match.group(1))
            header_end = header_start + len(candidate)
            body_offset = header_end + len(dash_match.group(3))
            return {
                "label": candidate,
                "start": header_start,
                "end": header_end,
                "body_offset": body_offset if body_offset < len(line) else None,
            }
    return None


def detect_note_layout(note: str, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"blocks": [], "boundaries": []}
    if not isinstance(note, str) or not note.strip():
        return {"blocks": [], "boundaries": []}

    length = len(note)
    lines = note.splitlines(keepends=True)
    position = 0
    blocks: List[Dict[str, Any]] = []
    paragraph_start: Optional[int] = None

    def flush_paragraph(end_idx: int) -> None:
        nonlocal paragraph_start
        if paragraph_start is not None and paragraph_start < end_idx:
            blocks.append({"type": "paragraph", "start": paragraph_start, "end": end_idx})
            paragraph_start = None

    for line in lines:
        line_start = position
        line_end = position + len(line)
        line_no_newline = line.rstrip("\r\n")
        header_info = extract_header_from_line(line_no_newline)
        if header_info:
            header_abs_start = line_start + header_info["start"]
            header_abs_end = line_start + header_info["end"]
            flush_paragraph(header_abs_start)
            blocks.append(
                {
                    "type": "header",
                    "start": header_abs_start,
                    "end": header_abs_end,
                    "label": header_info["label"],
                }
            )
            body_offset = header_info.get("body_offset")
            if body_offset is not None and body_offset < len(line_no_newline):
                paragraph_start = line_start + body_offset
            else:
                paragraph_start = None
        elif not line_no_newline.strip():
            flush_paragraph(line_start)
        else:
            if paragraph_start is None:
                paragraph_start = line_start
        position = line_end

    flush_paragraph(length)

    if not blocks:
        blocks = [{"type": "paragraph", "start": 0, "end": length}]

    boundaries = {0, length}
    for block in blocks:
        boundaries.add(block["start"])
        boundaries.add(block["end"])

    return {"blocks": blocks, "boundaries": sorted(boundaries)}


def apply_layout_to_segments(segments: List[Dict[str, Any]], blocks: List[Dict[str, Any]]) -> str:
    if not blocks:
        return "".join(segment["html"] for segment in segments)
    output: List[str] = []
    seg_idx = 0
    total = len(segments)
    for block in blocks:
        block_parts: List[str] = []
        while seg_idx < total and segments[seg_idx]["end"] <= block["start"]:
            output.append(segments[seg_idx]["html"])
            seg_idx += 1
        while seg_idx < total and segments[seg_idx]["start"] < block["end"]:
            block_parts.append(segments[seg_idx]["html"])
            seg_idx += 1
        if not block_parts:
            continue
        class_name = "note-section-header" if block["type"] == "header" else "note-paragraph"
        extra_attr = ""
        if block["type"] == "header" and block.get("label"):
            label_attr = html.escape(str(block["label"]), quote=True)
            extra_attr = f" data-section-label='{label_attr}'"
        output.append(f"<div class='{class_name}'{extra_attr}>" + "".join(block_parts) + "</div>")
    while seg_idx < total:
        output.append(segments[seg_idx]["html"])
        seg_idx += 1
    return "".join(output)


def compute_dataset_signature(df: pd.DataFrame, source_hint: str = "") -> str:
    """Return a deterministic fingerprint for the currently loaded dataset."""
    digest = hashlib.sha256()
    digest.update(str(len(df)).encode("utf-8"))
    column_repr = "||".join(map(str, df.columns))
    digest.update(column_repr.encode("utf-8"))
    preview = df.head(25).to_csv(index=False)
    digest.update(preview.encode("utf-8"))
    if source_hint:
        digest.update(source_hint.encode("utf-8", "ignore"))
    return digest.hexdigest()


def reset_state_for_new_dataset() -> None:
    """Clear persisted per-case session state when a new dataset is loaded."""
    prefixes = ("manual_select::", "reviewed_checkbox::", "note_text::")
    state_keys = list(st.session_state.keys())
    for key in state_keys:
        key_str = str(key)
        if any(key_str.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]
    st.session_state.manual_labels = {}
    st.session_state.reviewed_cases = {}
    st.session_state.case_notes = {}
    st.session_state.case_index = 0


def load_dataframe(source, suffix: Optional[str] = None) -> pd.DataFrame:
    """Load Parquet or CSV data from a path-like or file-like object."""
    suffix = (suffix or "").lower()
    candidates: List[str] = []
    if suffix in SUPPORTED_SUFFIXES:
        candidates = [suffix]
    else:
        candidates = [".parquet", ".csv"]

    last_error: Optional[Exception] = None
    for ext in candidates:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            if ext == ".parquet":
                return pd.read_parquet(source)
            return pd.read_csv(source)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
    raise ValueError(f"Failed to load data as Parquet or CSV: {last_error}")


def resolve_path(path_str: str) -> Path:
    """Allow users to omit the extension while preferring Parquet output."""
    path = Path(path_str).expanduser()
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        for ext in (".parquet", ".csv"):
            candidate = path.with_suffix(ext)
            if candidate.exists():
                return candidate
    if not path.exists() and path.suffix.lower() == ".parquet":
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            return fallback
    return path


def to_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return str(value)


def format_confidence_value(value: Any) -> str:
    """Return a human-readable confidence string (e.g., 'High (3)') or '' if unavailable."""
    num_value = to_int(value)
    if num_value is None:
        raw_text = stringify_value(value).strip()
        return raw_text
    level_label = CONFIDENCE_LEVEL_LABELS.get(num_value)
    if level_label:
        return f"{level_label} ({num_value})"
    return str(num_value)


def normalize_choice(value: Any, options: List[str]) -> str:
    candidate = stringify_value(value).strip()
    if not candidate:
        return ""
    candidate_lower = candidate.lower()
    for option in options:
        if candidate_lower == option.lower():
            return option
    return ""


def collect_annotations(row: pd.Series, var_names: List[str], note_text: Any) -> List[Dict[str, int]]:
    annotations: List[Dict[str, int]] = []
    occupied_spans: List[Tuple[int, int]] = []
    seen_spans: set[Tuple[int, int]] = set()
    for name in var_names:
        start = to_int(row.get(f"{name}_start"))
        end = to_int(row.get(f"{name}_end"))
        if start is not None and end is not None and start < end:
            span = (start, end)
            if span in seen_spans:
                continue
            annotations.append({"name": name, "start": start, "end": end})
            occupied_spans.append(span)
            seen_spans.add(span)

    note_str: str
    if isinstance(note_text, str):
        note_str = note_text
    elif pd.isna(note_text):
        note_str = ""
    else:
        note_str = str(note_text)

    if note_str:
        for name in var_names:
            if any(ann["name"] == name for ann in annotations):
                continue
            quote_text = stringify_value(row.get(f"{name}_quote", "")).strip()
            if not quote_text:
                continue
            escaped = re.escape(quote_text)
            pattern = re.sub(r"\\\s+", r"\\s+", escaped)
            for match in re.finditer(pattern, note_str, flags=re.IGNORECASE):
                start, end = match.span()
                if start >= end:
                    continue
                if any(not (end <= s or start >= e) for s, e in occupied_spans):
                    continue
                annotations.append({"name": name, "start": start, "end": end, "source": "inferred"})
                occupied_spans.append((start, end))
                break

    return sorted(annotations, key=lambda item: item["start"])


def derive_label_options(df: pd.DataFrame, var_names: List[str]) -> Dict[str, List[str]]:
    """Return candidate manual label options for each variable."""
    options_map: Dict[str, List[str]] = {}
    for name in var_names:
        normalized_values: set[str] = set()
        if name in df.columns:
            series = df[name]
            for raw in pd.unique(series.dropna()):
                text = stringify_value(raw).strip()
                if text:
                    normalized_values.add(text.lower())

        if not normalized_values:
            options = list(PRESENT_ABSENT_CHOICES)
        elif normalized_values.issubset({choice.lower() for choice in PRESENT_ABSENT_CHOICES}):
            options = list(PRESENT_ABSENT_CHOICES)
        elif normalized_values.issubset({choice.lower() for choice in YES_NO_CHOICES}):
            options = list(YES_NO_CHOICES)
        elif normalized_values.issubset({choice.lower() for choice in YES_NO_UNCERTAIN_CHOICES}):
            options = list(YES_NO_UNCERTAIN_CHOICES)
        else:
            # Fall back to sorted unique canonical strings if values fall outside supported sets.
            options = sorted({stringify_value(val).strip() for val in normalized_values if val})
            if not options:
                options = list(PRESENT_ABSENT_CHOICES)

        options_map[name] = options
    return options_map


def resolve_case_state(df: pd.DataFrame, position: int, id_col: Optional[str]) -> Tuple[pd.Series, str, str]:
    """Return the row, stable review key, and display ID for the requested case."""
    row = df.iloc[position]
    index_label = df.index[position]

    if id_col and id_col in df.columns:
        raw_value = row.get(id_col)
        if not pd.isna(raw_value):
            case_key = f"{index_label}|{id_col}:{raw_value}"
            note_id = raw_value
            return row, str(case_key), str(note_id)

    case_key = f"{index_label}|row"
    note_id = f"Row {position}"
    return row, str(case_key), note_id


def derive_tint(hex_color: str, blend: float = 0.78) -> str:
    """Blend a color towards white to create a softer tint."""
    stripped = hex_color.lstrip("#")
    if len(stripped) != 6:
        return hex_color
    try:
        r, g, b = (int(stripped[i : i + 2], 16) for i in range(0, 6, 2))
    except ValueError:
        return hex_color
    r = int(r + (255 - r) * blend)
    g = int(g + (255 - g) * blend)
    b = int(b + (255 - b) * blend)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_note_html(
    note: str,
    annotations: List[Dict[str, int]],
    colors: Dict[str, str],
    slug_map: Dict[str, str],
    root_id: str,
    layout_enabled: bool,
) -> str:
    if not isinstance(note, str):
        note = "" if pd.isna(note) else str(note)
    length = len(note)
    safe_note = html.escape(note)
    if not length:
        return f"<div class='note-container' data-note-root='{root_id}'>{safe_note.replace('\n', '<br/>')}</div>"

    layout_info = detect_note_layout(note, layout_enabled)
    layout_blocks: List[Dict[str, Any]] = layout_info.get("blocks", [])
    layout_boundaries: List[int] = layout_info.get("boundaries", [])
    events: List[Tuple[int, int, Optional[str]]] = []
    first_positions: Dict[str, int] = {}
    for ann in annotations:
        name = ann.get("name")
        if not name or name not in slug_map:
            continue
        try:
            start = int(ann.get("start", 0))
            end = int(ann.get("end", 0))
        except (TypeError, ValueError):
            continue
        start = max(0, min(length, start))
        end = max(0, min(length, end))
        if start >= end:
            continue
        events.append((start, 1, name))
        events.append((end, -1, name))
        first_positions[name] = min(first_positions.get(name, start), start)

    for boundary in layout_boundaries:
        events.append((boundary, 0, None))

    events.sort(key=lambda item: item[0])
    segments: List[Dict[str, Any]] = []
    active_counts: Dict[str, int] = {}

    def render_segment(start_idx: int, end_idx: int) -> None:
        if start_idx >= end_idx:
            return
        segment = html.escape(note[start_idx:end_idx]).replace("\n", "<br/>")
        active_names = [name for name, count in active_counts.items() if count > 0]
        if not active_names:
            segments.append({"start": start_idx, "end": end_idx, "html": segment})
            return
        slug_tokens = sorted({slug_map[name] for name in active_names if name in slug_map})
        if not slug_tokens:
            segments.append({"start": start_idx, "end": end_idx, "html": segment})
            return
        primary_name = min(
            (name for name in active_names if name in colors),
            key=lambda nm: (first_positions.get(nm, start_idx), nm),
            default=None,
        )
        color = colors.get(primary_name) if primary_name else None
        if not color:
            color = "#f9844a"
        tint = derive_tint(color)
        slug_attr = html.escape(" ".join(slug_tokens), quote=True)
        segments.append(
            {
                "start": start_idx,
                "end": end_idx,
                "html": (
                    f"<mark class='note-highlight' data-slug='{slug_attr}' "
                    f"style='--accent-color:{color}; --tint-color:{tint};'>{segment}</mark>"
                ),
            }
        )

    if events:
        idx = 0
        last_pos = 0
        while idx < len(events):
            pos = max(0, min(length, events[idx][0]))
            if pos > last_pos:
                render_segment(last_pos, pos)
                last_pos = pos
            same_pos: List[Tuple[int, int, Optional[str]]] = []
            while idx < len(events) and events[idx][0] == pos:
                same_pos.append(events[idx])
                idx += 1
            for _, change, name in same_pos:
                if name is None or change >= 0:
                    continue
                remaining = active_counts.get(name, 0) - 1
                if remaining <= 0:
                    active_counts.pop(name, None)
                else:
                    active_counts[name] = remaining
            for _, change, name in same_pos:
                if name is None or change <= 0:
                    continue
                active_counts[name] = active_counts.get(name, 0) + 1
        if last_pos < length:
            render_segment(last_pos, length)
    else:
        render_segment(0, length)

    if layout_enabled and layout_blocks:
        content_html = apply_layout_to_segments(segments, layout_blocks)
    else:
        content_html = "".join(segment["html"] for segment in segments)

    return f"<div class='note-container' data-note-root='{root_id}'>" + content_html + "</div>"


def build_card_html(
    name: str,
    row: pd.Series,
    colors: Dict[str, str],
    slug_map: Dict[str, str],
    manual_value: str,
    label_options: Dict[str, List[str]],
    root_id: str,
) -> Tuple[str, str, bool]:
    llm_raw = stringify_value(row.get(name, ""))
    quote = stringify_value(row.get(f"{name}_quote", ""))
    confidence_raw = row.get(f"{name}_confidence_1to3")
    explanation = stringify_value(row.get(f"{name}_explanation", ""))
    has_quote = bool(quote.strip())
    color = colors[name]
    tint = derive_tint(color) if has_quote else "#ffffff"
    chip_color = derive_tint(color, 0.65) if has_quote else "rgba(0,0,0,0.05)"
    slug = slug_map[name]
    class_name = "var-card has-quote" if has_quote else "var-card no-quote"
    style_vars = f"--accent-color:{color}; --tint-color:{tint}; --chip-color:{chip_color};"

    options = list(label_options.get(name, DEFAULT_LABEL_CHOICES)) or list(PRESENT_ABSENT_CHOICES)
    llm_normalized = normalize_choice(llm_raw, options)
    llm_display = llm_normalized or llm_raw

    manual_normalized = normalize_choice(manual_value, options) or (llm_normalized or options[0])

    llm_compare = (llm_normalized or llm_raw).strip().lower()
    manual_compare = manual_normalized.strip().lower()
    manual_changed = bool(manual_compare) and manual_compare != llm_compare
    if manual_changed:
        class_name += " manual-changed"

    quote_html = f"<span class='quote-chip'>{html.escape(quote)}</span>" if has_quote else "—"
    quote_attr = html.escape(quote, quote=True)
    explanation_html = html.escape(explanation) if explanation else "—"
    value_html = html.escape(llm_display) if llm_display else "—"
    confidence_text = format_confidence_value(confidence_raw)
    confidence_html = html.escape(confidence_text) if confidence_text else "—"
    manual_html = html.escape(manual_normalized) if manual_normalized else "—"

    card_html = f"""
    <div class='var-card-wrapper' data-card-root='{root_id}'>
      <div class='{class_name}' data-slug='{slug}' data-name='{html.escape(name)}' data-quote='{quote_attr}' data-llm='{html.escape(llm_display, quote=True)}' style='{style_vars}'>
        <div class='var-name'>{html.escape(name)}</div>
        <div class='var-value'><strong>LLM:</strong> {value_html}</div>
        <div class='var-confidence'><strong>Confidence:</strong> {confidence_html}</div>
        <div class='var-quote'><strong>Quote:</strong> {quote_html}</div>
        <div class='var-explanation'><strong>Explanation:</strong> {explanation_html}</div>
        <div class='manual-summary'><strong>Final Label:</strong> <span>{manual_html}</span></div>
      </div>
    </div>
"""
    return card_html, manual_normalized, manual_changed


CSS = """
<style>
:root {
  --maize: #FFCB05;
  --blue: #00274C;
}
.app-banner {
  background: linear-gradient(135deg, rgba(0, 39, 76, 0.96), rgba(0, 39, 76, 0.85));
  border-radius: 10px;
  padding: 1rem 1.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  box-shadow: 0 10px 24px rgba(0, 39, 76, 0.25);
  margin-bottom: 1.25rem;
  border: 1px solid rgba(255, 203, 5, 0.18);
}
.app-title {
  font-size: 2.2rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--maize);
}
.app-subtitle {
  font-size: 1.2rem;
  font-weight: 500;
  color: rgba(247, 249, 252, 0.92);
  letter-spacing: 0.02em;
}
.review-grid {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  --font-scale: 1;
}
.left-col, .right-col {
  flex: 1;
  min-width: 0;
}
.structured-header {
  position: sticky;
  top: 0;
  z-index: 5;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(4px);
  padding: 0.6rem 0.4rem 0.5rem;
  margin-bottom: 0.4rem;
  border-bottom: 1px solid rgba(0, 39, 76, 0.12);
}
.structured-header h3 {
  margin: 0;
}
.structured-header p {
  margin: 0.35rem 0 0;
  font-size: calc(0.85rem * var(--font-scale, 1));
  color: rgba(0, 39, 76, 0.72);
}
.vars-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding-right: 0.4rem;
}
.structured-scroll-column {
  overflow-y: auto !important;
  padding-right: 0.5rem;
  scrollbar-width: thin;
}
.structured-scroll-column::-webkit-scrollbar {
  width: 0.45rem;
}
.structured-scroll-column::-webkit-scrollbar-thumb {
  background: rgba(0, 39, 76, 0.25);
  border-radius: 0.4rem;
}
.structured-scroll-column::-webkit-scrollbar-track {
  background: rgba(0, 39, 76, 0.05);
  border-radius: 0.4rem;
}
.vars-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.75rem;
}
.var-card {
  --accent-color: #dee2e6;
  --tint-color: #ffffff;
  --chip-color: rgba(0,0,0,0.05);
  border: 1px solid rgba(0, 39, 76, 0.08);
  border-left: 6px solid var(--accent-color);
  border-radius: 6px;
  padding: 0.55rem 0.6rem 0.65rem;
  background: var(--tint-color);
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
  cursor: pointer;
  font-size: calc(0.95rem * var(--font-scale, 1));
}
.var-card.has-quote {
  color: #1f2a36;
  border-color: rgba(0, 39, 76, 0.12);
  box-shadow: 0 2px 6px rgba(0, 39, 76, 0.08);
}
.var-card.hovered {
  transform: translateY(-2px);
  box-shadow: 0 6px 14px rgba(0, 39, 76, 0.16);
}
.var-card.selected {
  transform: translateY(-3px);
  box-shadow: 0 8px 18px rgba(0, 39, 76, 0.2);
  border-left-color: var(--accent-color) !important;
  border-color: rgba(255, 203, 5, 0.2);
}
.var-card.manual-changed {
  border-left-color: var(--maize);
  box-shadow: 0 6px 18px rgba(255, 203, 5, 0.22);
}
.var-card.manual-changed .manual-summary span::after {
  content: " (edited)";
  font-weight: 600;
  color: #b23a48;
}
.note-highlight.clicked {
  box-shadow: 0 0 0 3px var(--accent-color, #ff922b);
  transition: box-shadow 0.15s ease, filter 0.15s ease;
  filter: brightness(1.1);
}
.var-name { font-weight: 600; margin-bottom: 0.35rem; font-size: calc(1.05rem * var(--font-scale, 1)); }
.var-value, .var-confidence, .var-quote, .var-explanation { font-size: calc(0.9rem * var(--font-scale, 1)); margin-bottom: 0.25rem; }
.quote-chip {
  background: var(--chip-color);
  border-radius: 4px;
  padding: 0 0.2rem;
  font-size: calc(0.85rem * var(--font-scale, 1));
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
}

.var-card-wrapper {
  margin-bottom: 1.15rem;
}
.var-card-wrapper div[data-testid="stSelectbox"] {
  margin-top: -0.25rem;
}
.var-card-wrapper div[data-testid="stSelectbox"] label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: rgba(0, 39, 76, 0.7);
  margin-bottom: 0.1rem;
}
.var-card-wrapper div[data-testid="stSelectbox"] div[data-baseweb] {
  border-radius: 4px;
}
.manual-summary {
  margin-top: 0.35rem;
  font-size: calc(0.85rem * var(--font-scale, 1));
  color: rgba(0, 39, 76, 0.75);
}
.manual-summary span {
  font-weight: 600;
  color: rgba(0, 39, 76, 0.95);
}
.note-container {
  border: 1px solid rgba(0, 39, 76, 0.16);
  border-radius: 6px;
  background: #ffffff;
  padding: 1rem;
  min-height: 420px;
  max-height: 75vh;
  overflow-y: auto;
  white-space: pre-wrap;
  font-family: "Source Code Pro", "Fira Code", monospace;
  font-size: calc(0.95rem * var(--font-scale, 1));
  line-height: calc(1.45 * var(--font-scale, 1));
}
.note-paragraph {
  margin: 0 0 0.9rem;
  padding: 0.4rem 0.2rem 0.4rem 0.4rem;
  border-left: 3px solid rgba(0, 39, 76, 0.1);
}
.note-section-header {
  margin: 1rem 0 0.35rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--blue);
  background: rgba(0, 39, 76, 0.05);
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
  border-left: 4px solid rgba(0, 39, 76, 0.25);
}
.note-section-header:first-child {
  margin-top: 0;
}
.note-highlight {
  background-color: var(--tint-color, rgba(255,146,43,0.15));
  padding: 0.15rem 0.2rem;
  border-radius: 0.2rem;
  border: 1px solid transparent;
  transition: box-shadow 0.12s ease, filter 0.12s ease, border-color 0.12s ease;
}
.note-highlight.hovered {
  box-shadow: 0 0 0 2px rgba(0, 39, 76, 0.18);
  filter: brightness(1.05);
}
.note-highlight.selected {
  border-color: var(--accent-color, #ff922b);
  box-shadow: 0 0 0 2px var(--accent-color, rgba(0, 39, 76, 0.22));
  filter: brightness(1.1);
}
.left-col h3, .right-col h3 {
  color: var(--blue);
  border-bottom: 2px solid rgba(0, 39, 76, 0.16);
  padding-bottom: 0.25rem;
  margin-bottom: 0.75rem;
  letter-spacing: 0.02em;
}
.left-col em, .left-col code {
  color: var(--blue);
}
</style>
"""

def main() -> None:
    st.set_page_config(page_title="M-AI SCAN", layout="wide")

    with st.sidebar:
        st.markdown(
            """
            <style>
            .sidebar-app-banner {
              background: linear-gradient(135deg, rgba(0, 39, 76, 0.96), rgba(0, 39, 76, 0.85));
              border-radius: 10px;
              padding: 1rem 1.25rem;
              display: flex;
              flex-direction: column;
              align-items: flex-start;
              gap: 0.25rem;
              box-shadow: 0 10px 24px rgba(0, 39, 76, 0.25);
              margin-bottom: 1.5rem;
              border: 1px solid rgba(255, 203, 5, 0.18);
            }
            .sidebar-app-banner .title-line {
              font-size: 2rem;
              font-weight: 700;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              color: #FFCB05;
              line-height: 1;
              margin: 0;
            }
            .sidebar-app-banner .subtitle {
              font-size: 1.05rem;
              font-weight: 500;
              color: rgba(247, 249, 252, 0.92);
              letter-spacing: 0.02em;
              margin-top: 0.35rem;
            }
            </style>
            <div class='sidebar-app-banner'>
              <div class='title-line'>M-AI SCAN</div>
              <div class='subtitle'>Michigan AI Structured Chart Abstraction Navigator</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Data Source")
        uploaded = st.file_uploader("Upload Parquet/CSV", type=["parquet", "csv"])
        review_status_placeholder = st.empty()
        review_status_placeholder.markdown("**0 / 0 Reviewed**")
        notes_download_placeholder = st.empty()
        notes_download_placeholder.caption("Add notes to enable download.")
        labels_download_placeholder = st.empty()
        labels_download_placeholder.caption("Adjust labels to enable download.")

    df: Optional[pd.DataFrame] = None
    try:
        if uploaded is not None:
            uploaded_suffix = Path(getattr(uploaded, "name", "")).suffix
            df = load_dataframe(uploaded, uploaded_suffix)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    if df is None:
        st.info("Provide a Parquet or CSV file via sidebar to start.")
        return

    source_hint = getattr(uploaded, "name", "") if uploaded is not None else ""
    dataset_signature = compute_dataset_signature(df, source_hint)
    previous_signature = st.session_state.get("dataset_signature")
    if previous_signature != dataset_signature:
        reset_state_for_new_dataset()
    st.session_state.dataset_signature = dataset_signature

    try:
        var_names = gather_var_names(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    colors = {name: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, name in enumerate(var_names)}
    slug_map = {name: slugify(name) for name in var_names}
    label_options = derive_label_options(df, var_names)

    note_columns = list(df.columns)
    default_note_idx = note_columns.index("Text") if "Text" in note_columns else note_columns.index("note_text") if "note_text" in note_columns else 0
    note_col = st.sidebar.selectbox("Note column", options=note_columns, index=default_note_idx)

    id_options = ["(none)"] + note_columns
    default_id_idx = id_options.index("EncounterCsn") if "EncounterCsn" in id_options else 0
    id_col = st.sidebar.selectbox("ID column", options=id_options, index=default_id_idx)
    id_col = None if id_col == "(none)" else id_col

    font_scale = st.sidebar.select_slider(
        "Aa",
        options=[0.85, 1.0, 1.15, 1.3],
        value=1.0,
        format_func=lambda v: f"{int(v * 100)}%"
    )
    format_note_sections = st.sidebar.checkbox(
        "Group note into sections",
        value=True,
        help="Adds paragraph spacing and highlights common headers (HPI, ROS, Physical Exam, etc.).",
    )

    if "case_index" not in st.session_state:
        st.session_state.case_index = 0
    if "reviewed_cases" not in st.session_state:
        st.session_state.reviewed_cases = {}
    if "case_notes" not in st.session_state:
        st.session_state.case_notes = {}
    if "manual_labels" not in st.session_state:
        st.session_state.manual_labels = {}
    num_cases = len(df)
    if num_cases == 0:
        st.warning("No cases available in the selected dataset.")
        return

    # Keep the index within bounds before rendering controls.
    st.session_state.case_index = max(0, min(st.session_state.case_index, num_cases - 1))
    prev_disabled = st.session_state.case_index <= 0
    next_disabled = st.session_state.case_index >= num_cases - 1

    def go_prev() -> None:
        st.session_state.case_index = max(0, st.session_state.case_index - 1)

    def go_next() -> None:
        st.session_state.case_index = min(num_cases - 1, st.session_state.case_index + 1)

    prev_col, next_col, info_col = st.columns([1, 1, 8])
    with prev_col:
        st.button("◀ Previous", key="prev_button", disabled=prev_disabled, on_click=go_prev)
    with next_col:
        st.button("Next ▶", key="next_button", disabled=next_disabled, on_click=go_next)

    st.session_state.case_index = max(0, min(st.session_state.case_index, num_cases - 1))
    row_position = st.session_state.case_index
    row, case_key, note_id = resolve_case_state(df, row_position, id_col)

    manual_store: Dict[str, Dict[str, str]] = st.session_state.manual_labels
    case_manual = manual_store.setdefault(case_key, {}).copy()
    card_root_id = slugify(f'case-root-{case_key}')
    for name in var_names:
        options_for_name = label_options.get(name, DEFAULT_LABEL_CHOICES) or list(PRESENT_ABSENT_CHOICES)
        llm_raw_value = stringify_value(row.get(name, ""))
        canonical_default = normalize_choice(llm_raw_value, options_for_name)
        if not canonical_default:
            canonical_default = options_for_name[0]
        case_manual.setdefault(name, canonical_default)


    checkbox_key = f"reviewed_checkbox::{case_key}"
    stored_review_state = st.session_state.reviewed_cases.get(case_key, False)
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = stored_review_state

    with info_col:
        info_left, notes_col = st.columns([3, 2])
        with info_left:
            reviewed_checked = st.checkbox("Marked as reviewed", key=checkbox_key)
            st.session_state.reviewed_cases[case_key] = bool(reviewed_checked)
            st.markdown(f"**Case {row_position + 1} / {num_cases}** — ID: `{note_id}`")
            conflict_flag = row.get("conflict_flag")
            conflict_explanation = row.get("conflict_explanation")
            conflict_flag_text = "" if pd.isna(conflict_flag) else str(conflict_flag).strip()
            conflict_explanation_text = "" if pd.isna(conflict_explanation) else str(conflict_explanation).strip()
            normalized_flag = conflict_flag_text.lower()
            has_conflict = bool(conflict_explanation_text)
            if not has_conflict:
                has_conflict = normalized_flag not in {"", "no", "none", "nan", "0", "false"}
            conflict_details = conflict_explanation_text or (conflict_flag_text if has_conflict else "")
            conflict_caption = (
                f"Conflicting Information: {conflict_details}"
                if has_conflict and conflict_details
                else "Conflicting Information: None"
            )
            st.caption(conflict_caption)

        existing_note_entry = st.session_state.case_notes.get(case_key)
        existing_note_text = ""
        if existing_note_entry:
            existing_note_text = existing_note_entry.get("note") or ""
        note_state_key = f"note_text::{case_key}"
        if note_state_key not in st.session_state:
            st.session_state[note_state_key] = existing_note_text

        with notes_col:
            note_value = notes_col.text_area(
                "Reviewer notes",
                key=note_state_key,
                height=110,
                label_visibility="collapsed",
                placeholder="Notes (saved per case)",
            )

        raw_id_value: Optional[Any] = None
        if id_col:
            raw_id_value = row.get(id_col)
            if pd.isna(raw_id_value):
                raw_id_value = None

        if note_value or existing_note_entry:
            st.session_state.case_notes[case_key] = {
                "note": note_value,
                "note_id": note_id,
                "raw_id": raw_id_value,
                "id_column": id_col,
                "row_index": row_position,
            }
        else:
            st.session_state.case_notes.pop(case_key, None)

    reviewed_total = 0
    for position in range(num_cases):
        _, iter_case_key, _ = resolve_case_state(df, position, id_col)
        if st.session_state.reviewed_cases.get(iter_case_key, False):
            reviewed_total += 1
    review_status_placeholder.markdown(f"**{reviewed_total} / {num_cases} Reviewed**")

    notes_entries: List[Dict[str, Any]] = []
    for stored_case_key, payload in st.session_state.case_notes.items():
        note_text = payload.get("note", "")
        if not isinstance(note_text, str):
            note_text = "" if pd.isna(note_text) else str(note_text)
        if not note_text.strip():
            continue
        record: Dict[str, Any] = {
            "case_key": stored_case_key,
            "case_identifier": payload.get("note_id"),
            "note_text": note_text,
            "row_index": payload.get("row_index"),
        }
        id_column_name = payload.get("id_column")
        raw_id_value = payload.get("raw_id")
        if id_column_name:
            record[id_column_name] = raw_id_value
        notes_entries.append(record)

    notes_entries.sort(key=lambda rec: (rec.get("row_index") is None, rec.get("row_index", 0)))
    if notes_entries:
        notes_df = pd.DataFrame(notes_entries)
        csv_bytes = notes_df.to_csv(index=False).encode("utf-8")
        notes_download_placeholder.download_button(
            "Download notes CSV",
            data=csv_bytes,
            file_name="reviewer_notes.csv",
            mime="text/csv",
        )
    else:
        notes_download_placeholder.caption("Add notes to enable download.")

    note_content = row.get(note_col, "")
    annotations = collect_annotations(row, var_names, note_content)
    note_html = build_note_html(note_content, annotations, colors, slug_map, card_root_id, format_note_sections)

    scale_css = f"<style>:root {{ --font-scale: {font_scale:.2f}; }}</style>"
    st.markdown(CSS + scale_css, unsafe_allow_html=True)

    cards_col, note_col_area = st.columns([1.05, 1.45])

    with cards_col:
        cards_col.markdown(
            f"""
            <div class='structured-header'>
              <h3>Structured Outputs</h3>
              <p><em>Note column:</em> <code>{html.escape(str(note_col))}</code></p>
            </div>
            <div class='vars-wrapper'>
            """,
            unsafe_allow_html=True,
        )
        for name in var_names:
            options = list(label_options.get(name, DEFAULT_LABEL_CHOICES)) or list(PRESENT_ABSENT_CHOICES)
            seen_option_lower: Dict[str, None] = {}
            unique_options: List[str] = []
            for opt in options:
                key = opt.lower()
                if key not in seen_option_lower:
                    seen_option_lower[key] = None
                    unique_options.append(opt)
            options = unique_options or list(PRESENT_ABSENT_CHOICES)

            llm_raw = stringify_value(row.get(name, ""))
            llm_normalized = normalize_choice(llm_raw, options)
            manual_default = case_manual.get(name)
            manual_default = normalize_choice(manual_default, options) if manual_default else None
            if not manual_default:
                manual_default = llm_normalized or options[0]
            case_manual[name] = manual_default

            card_html, manual_default, _ = build_card_html(name, row, colors, slug_map, manual_default, label_options, card_root_id)

            cards_col.markdown(card_html, unsafe_allow_html=True)
            select_key = f"manual_select::{case_key}::{slug_map[name]}"
            default_index = options.index(manual_default) if manual_default in options else 0
            selected_value = cards_col.selectbox(
                "Manual label",
                options=options,
                index=default_index,
                key=select_key,
                label_visibility="collapsed",
            )
            selected_normalized = normalize_choice(selected_value, options) or options[0]
            case_manual[name] = selected_normalized
            st.session_state.manual_labels[case_key][name] = selected_normalized

        cards_col.markdown('</div>', unsafe_allow_html=True)

    with note_col_area:
        note_col_area.markdown("<div class='note-header'><h3>Note</h3></div>", unsafe_allow_html=True)
        note_col_area.markdown(note_html, unsafe_allow_html=True)

    interaction_template = """
    <script>
    (function() {
      const rootId = "__ROOT__";
      const globalKey = "__mai_scan_init_" + rootId;

      const getDoc = () => {
        try {
          if (window.parent && window.parent !== window && window.parent.document) {
            return window.parent.document;
          }
        } catch (err) {
          return document;
        }
        return document;
      };

      const doc = getDoc();
      const globalHost = doc.defaultView || window;

      const previous = globalHost[globalKey];
      if (previous && typeof previous.teardown === "function") {
        try {
          previous.teardown();
        } catch (err) {
          /* ignore */
        }
      }

      const runtime = {
        active: true,
        activeSlugs: [],
        columnBlock: null,
        columnListenersBound: false,
        columnAdjustScheduled: false,
        listeners: [],
        timers: [],
        teardown() {
          if (!runtime.active) {
            return;
          }
          runtime.active = false;
          runtime.listeners.splice(0).forEach(remove => {
            try {
              remove();
            } catch (err) {
              /* ignore */
            }
          });
          runtime.timers.splice(0).forEach(remove => {
            try {
              remove();
            } catch (err) {
              /* ignore */
            }
          });
          if (runtime.columnBlock && runtime.columnBlock.classList) {
            runtime.columnBlock.classList.remove("structured-scroll-column");
          }
        },
      };

      globalHost[globalKey] = runtime;

      const registerListener = (target, type, handler, options) => {
        let active = true;
        const remove = () => {
          if (!active) {
            return;
          }
          active = false;
          target.removeEventListener(type, handler, options);
          runtime.listeners = runtime.listeners.filter(entry => entry !== remove);
        };
        target.addEventListener(type, handler, options);
        runtime.listeners.push(remove);
        return remove;
      };

      const registerTimeout = (fn, delay) => {
        const set = globalHost.setTimeout || window.setTimeout;
        const clear = globalHost.clearTimeout || window.clearTimeout;
        let active = true;
        let id = null;
        const remove = () => {
          if (!active) {
            return;
          }
          active = false;
          if (id !== null) {
            clear(id);
          }
          runtime.timers = runtime.timers.filter(entry => entry !== remove);
        };
        id = set(() => {
          remove();
          if (runtime.active) {
            fn();
          }
        }, delay);
        runtime.timers.push(remove);
        return remove;
      };

      const getCards = () =>
        Array.from(doc.querySelectorAll(`.var-card-wrapper[data-card-root="${rootId}"] .var-card`));
      const getNoteRoot = () => doc.querySelector(`.note-container[data-note-root="${rootId}"]`);
      const getTargets = (slug) => {
        const noteRoot = getNoteRoot();
        if (!noteRoot || !slug) {
          return [];
        }
        return Array.from(noteRoot.querySelectorAll(`[data-slug~="${slug}"]`));
      };
      const normalizeSlugs = (value) => {
        if (Array.isArray(value)) {
          return value
            .map(entry => (entry || "").trim())
            .filter(Boolean);
        }
        if (typeof value === "string") {
          return value
            .split(/\\s+/)
            .map(entry => entry.trim())
            .filter(Boolean);
        }
        return [];
      };
      const collectTargetsForSlugs = (normalizedSlugs) => {
        const nodes = [];
        const seen = new Set();
        normalizedSlugs.forEach(slug => {
          getTargets(slug).forEach(node => {
            if (!seen.has(node)) {
              seen.add(node);
              nodes.push(node);
            }
          });
        });
        return nodes;
      };
      const haveSameSlugSet = (first, second) => {
        if (!Array.isArray(first) || !Array.isArray(second) || first.length !== second.length) {
          return false;
        }
        const a = [...first].sort();
        const b = [...second].sort();
        return a.every((slug, idx) => slug === b[idx]);
      };

      const toggleClassesForSlugs = (slugs, on) => {
        const normalized = normalizeSlugs(slugs);
        if (!normalized.length) {
          return;
        }
        const slugSet = new Set(normalized);
        getCards()
          .filter(card => slugSet.has(card.dataset.slug))
          .forEach(el => el.classList.toggle("selected", on));
        collectTargetsForSlugs(normalized).forEach(el => {
          el.classList.toggle("selected", on);
          if (!on) {
            el.classList.remove("clicked");
          }
        });
      };

      const revealTarget = (slugs) => {
        const normalized = normalizeSlugs(slugs);
        if (!normalized.length) {
          return;
        }
        const targets = getTargets(normalized[0]);
        if (!targets.length) {
          return;
        }
        const target = targets[0];
        target.classList.add("clicked");
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      };

      const setState = (slugInput) => {
        const normalized = normalizeSlugs(slugInput);
        if (!normalized.length || !runtime.active) {
          return;
        }
        const current = runtime.activeSlugs || [];
        if (haveSameSlugSet(current, normalized)) {
          toggleClassesForSlugs(current, false);
          runtime.activeSlugs = [];
          return;
        }
        if (current.length) {
          toggleClassesForSlugs(current, false);
        }
        runtime.activeSlugs = normalized.slice();
        toggleClassesForSlugs(normalized, true);
        revealTarget(normalized);
      };

      const setHover = (slugInput, on) => {
        if (!runtime.active || (runtime.activeSlugs && runtime.activeSlugs.length)) {
          return;
        }
        const normalized = normalizeSlugs(slugInput);
        if (!normalized.length) {
          return;
        }
        const slugSet = new Set(normalized);
        getCards()
          .filter(card => slugSet.has(card.dataset.slug))
          .forEach(el => el.classList.toggle("hovered", on));
        collectTargetsForSlugs(normalized).forEach(el => el.classList.toggle("hovered", on));
      };

      const bindOnce = (element, type, handler, options) => {
        const key = `__maiScanBound_${type}_${rootId}`;
        if (element[key] && typeof element[key] !== "function") {
          delete element[key];
        }
        if (typeof element[key] === "function") {
          return;
        }
        let active = true;
        const remove = () => {
          if (!active) {
            return;
          }
          active = false;
          element.removeEventListener(type, handler, options);
          if (element[key] === remove) {
            delete element[key];
          }
        };
        element.addEventListener(type, handler, options);
        element[key] = remove;
        runtime.listeners.push(remove);
      };

      const findColumnBlock = (cards) => {
        if (!cards.length) {
          return null;
        }
        const first = cards[0];
        const column =
          first.closest('[data-testid="column"]') ||
          first.closest('[data-testid="stColumn"]') ||
          first.closest('[data-testid="stVerticalBlock"]');
        if (!column) {
          return null;
        }
        const verticalBlock =
          column.querySelector('[data-testid="stVerticalBlock"]') ||
          column.querySelector('[data-testid="stVerticalBlockContainer"]');
        return verticalBlock || column;
      };

      const scheduleColumnAdjust = () => {
        if (!runtime.columnBlock || runtime.columnAdjustScheduled || !runtime.active) {
          return;
        }
        runtime.columnAdjustScheduled = true;
        registerTimeout(() => {
          runtime.columnAdjustScheduled = false;
          const column = runtime.columnBlock;
          if (!runtime.active || !column || !column.isConnected) {
            runtime.columnBlock = null;
            return;
          }
          const rect = column.getBoundingClientRect();
          const viewportHeight = globalHost.innerHeight || window.innerHeight || 0;
          const margin = 32;
          const maxHeight = Math.max(320, viewportHeight - rect.top - margin);
          if (maxHeight > 0) {
            column.style.maxHeight = `${maxHeight}px`;
            column.style.overflowY = "auto";
            column.style.paddingRight = "0.55rem";
            column.style.boxSizing = "border-box";
            column.style.scrollbarWidth = "thin";
          }
        }, 20);
      };

      const bind = () => {
        const cards = getCards();
        const noteRoot = getNoteRoot();
        if (!cards.length || !noteRoot || !runtime.active) {
          return false;
        }

        if (!runtime.columnBlock || !runtime.columnBlock.isConnected) {
          runtime.columnBlock = findColumnBlock(cards);
          if (runtime.columnBlock) {
            runtime.columnBlock.classList.add("structured-scroll-column");
            scheduleColumnAdjust();
            if (!runtime.columnListenersBound) {
              runtime.columnListenersBound = true;
              registerListener(globalHost, "resize", scheduleColumnAdjust);
              registerListener(globalHost, "scroll", scheduleColumnAdjust, true);
            }
          }
        } else {
          scheduleColumnAdjust();
        }

        cards.forEach(card => {
          const slug = card.dataset.slug;
          if (!slug) {
            return;
          }
          card.setAttribute("tabindex", card.getAttribute("tabindex") || "0");
          if (card.classList.contains("has-quote")) {
            bindOnce(card, "click", () => setState(slug));
            bindOnce(card, "keydown", evt => {
              if (evt.key === "Enter" || evt.key === " ") {
                evt.preventDefault();
                setState(slug);
              }
            });
          }
          bindOnce(card, "mouseenter", () => setHover(slug, true));
          bindOnce(card, "mouseleave", () => setHover(slug, false));
          bindOnce(card, "focus", () => setHover(slug, true));
          bindOnce(card, "blur", () => setHover(slug, false));
        });

        const highlightNodes = Array.from(noteRoot.querySelectorAll("[data-slug]"));
        highlightNodes.forEach(node => {
          const slugs = normalizeSlugs(node.dataset.slug);
          if (!slugs.length) {
            return;
          }
          bindOnce(node, "mouseenter", () => setHover(slugs, true));
          bindOnce(node, "mouseleave", () => setHover(slugs, false));
          bindOnce(node, "click", () => setState(slugs));
        });

        return true;
      };

      const loop = () => {
        if (!runtime.active) {
          return;
        }
        const ready = bind();
        if (ready) {
          scheduleColumnAdjust();
        }
        registerTimeout(loop, 250);
      };

      loop();
    })();
    </script>
    """
    interaction_js = interaction_template.replace("__ROOT__", card_root_id)
    components.html(interaction_js, height=0, scrolling=False)



    manual_export_records: List[Dict[str, Any]] = []
    for position in range(num_cases):
        iter_row, iter_case_key, iter_note_id = resolve_case_state(df, position, id_col)
        manual_for_case = st.session_state.manual_labels.get(iter_case_key, {})
        record: Dict[str, Any] = {
            "case_key": iter_case_key,
            "case_identifier": iter_note_id,
            "row_index": position,
        }
        if id_col:
            record[id_col] = stringify_value(iter_row.get(id_col))
        for name in var_names:
            llm_value = stringify_value(iter_row.get(name, ""))
            manual_value_raw = manual_for_case.get(name)
            manual_value = stringify_value(manual_value_raw if manual_value_raw is not None else llm_value)
            record[f"{name}__llm"] = llm_value
            record[f"{name}__manual"] = manual_value
        manual_export_records.append(record)

    if manual_export_records:
        manual_df = pd.DataFrame(manual_export_records)
        manual_csv_bytes = manual_df.to_csv(index=False).encode("utf-8")
        labels_download_placeholder.download_button(
            "Download labels CSV",
            data=manual_csv_bytes,
            file_name="manual_vs_llm_labels.csv",
            mime="text/csv",
        )
    else:
        labels_download_placeholder.caption("Adjust labels to enable download.")


COLOR_PALETTE = [
    "#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f",
    "#90be6d", "#43aa8b", "#577590", "#277da1", "#4d908e",
    "#577590", "#7209b7", "#b5179e", "#ffb703", "#fb8500",
    "#219ebc", "#8ecae6", "#ff6f91", "#845ec2", "#2c73d2",
]

if __name__ == "__main__":
    main()

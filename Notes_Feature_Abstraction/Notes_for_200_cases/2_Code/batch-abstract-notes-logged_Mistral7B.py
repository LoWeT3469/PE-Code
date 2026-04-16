#!/usr/bin/env python3
import argparse, json, subprocess, sys, time, signal, os, datetime
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Deque

import pandas as pd

# ---------------- Rate Limiter (token bucket-ish) ----------------
class RateLimiter:
    """
    Enforces a max requests-per-minute (RPM) or requests-per-second (RPS).
    Uses a timestamp deque; sleeps as needed to avoid exceeding limits.
    """
    def __init__(self, rps: Optional[float] = None, rpm: Optional[int] = None):
        self.rps = rps
        self.rpm = rpm
        self.win_s = 1.0 if rps else 60.0
        self.limit = rps if rps else (rpm if rpm else None)
        self.ts: Deque[float] = deque()

    def wait(self):
        if not self.limit:
            return
        now = time.monotonic()
        # drop old timestamps
        while self.ts and (now - self.ts[0] > self.win_s):
            self.ts.popleft()
        if len(self.ts) >= self.limit:
            sleep_for = self.win_s - (now - self.ts[0]) + 0.001
            if sleep_for > 0:
                time.sleep(sleep_for)

    def hit(self):
        self.ts.append(time.monotonic())

# ---------------- Subprocess call wrapper ----------------
def run_cli_once(
    script_path: Path,
    note_text: str,
    model: Optional[str],
    var_args: List[str],
    exp_args: List[str],
    exp_all: bool,
    repair: bool,
    quote_per_var: bool,
    api_provider: Optional[str],
    api_key: Optional[str],
    api_base: Optional[str],
    api_version: Optional[str],
    organization: Optional[str],
    max_retries: int = 3,
    base_backoff_s: float = 1.5,
) -> Dict[str, Any]:
    """Call the single-note CLI via subprocess, return parsed JSON or an error dict."""
    cmd = [sys.executable, str(script_path)]
    for v in var_args:
        cmd += ["--var", v]
    for e in exp_args:
        cmd += ["--exp", e]
    if exp_all:
        cmd.append("--exp-all")
    if repair:
        cmd.append("--repair")
    if model:
        cmd += ["--model", model]
    if quote_per_var:
        cmd.append("--quote-per-var")
    if api_provider:
        cmd += ["--api-provider", api_provider]
    if api_key:
        cmd += ["--api-key", api_key]
    if api_base:
        cmd += ["--api-base", api_base]
    if api_version:
        cmd += ["--api-version", api_version]
    if organization:
        cmd += ["--organization", organization]

    attempt = 0
    while True:
        attempt += 1
        p = subprocess.run(
            cmd, input=note_text, text=True, capture_output=True, check=False
        )
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()

        if p.returncode == 0:
            try:
                return json.loads(out)
            except json.JSONDecodeError as je:
                # Non-JSON from child; retry a couple times then return error
                if attempt > max_retries:
                    return {"_error": f"json_decode_error: {je}", "_raw_stdout": out, "_raw_stderr": err}
        else:
            # Bubble up child's error after retries
            if attempt > max_retries:
                return {"_error": f"cli_exit_{p.returncode}", "_raw_stdout": out, "_raw_stderr": err}

        # Exponential backoff before retry
        time.sleep(base_backoff_s ** attempt)

# ---------------- Var helpers ----------------
def extract_var_names(var_specs: List[str]) -> List[str]:
    """Return variable names from --var specs (text before first colon)."""
    names: List[str] = []
    for spec in var_specs:
        if not spec:
            continue
        name = spec.split(":", 1)[0].strip()
        if name:
            names.append(name)
    return names

# ---------------- Result augmentation ----------------
def enrich_result(
    res: Optional[Dict[str, Any]],
    var_names: List[str],
    note_identifier: str,
) -> Dict[str, Any]:
    """Inject per-variable quote/span fields into the result dict."""
    enriched: Dict[str, Any] = dict(res) if res is not None else {}
    flag = enriched.get("conflict_flag")
    enriched["conflict_flag"] = flag
    enriched["conflict_explanation"] = enriched.get("conflict_explanation")
    for name in var_names:
        conf_key = f"{name}_confidence_1to3"
        quote_key = f"{name}_quote"
        span_key = f"{name}_span"
        start_key = f"{name}_start"
        end_key = f"{name}_end"
        conf_val = enriched.get(conf_key)
        quote_val = enriched.get(quote_key)
        span_val: Any = enriched.get(span_key) or {}
        start_val = span_val.get("start") if isinstance(span_val, dict) else None
        end_val = span_val.get("end") if isinstance(span_val, dict) else None
        enriched[conf_key] = conf_val
        enriched[quote_key] = quote_val
        enriched[start_key] = start_val
        enriched[end_key] = end_val
        enriched.pop(span_key, None)
    if flag and flag != "no":
        # Clarify intent: conflict_flag reflects disagreement within the clinical facts, not interpersonal disputes.
        sys.stderr.write(
            f"\n[warn] conflict_flag='{flag}' (clinical information disagreement) detected for note {note_identifier}\n"
        )
    return enriched

# ---------------- Checkpointing helpers ----------------
def write_outputs(
    base_df: pd.DataFrame,
    results: List[Optional[Dict[str, Any]]],
    output_path: Path,
    json_out: Optional[Path],
    var_names: List[str],
):
    # Flatten result rows (expand _usage into flat columns)
    flat_rows: List[Dict[str, Any]] = []
    for r in results:
        flat = dict(r) if r is not None else {}
        u = flat.pop("_usage", None)
        if isinstance(u, dict):
            flat["usage_input_tokens"] = u.get("input_tokens")
            flat["usage_output_tokens"] = u.get("output_tokens")
            flat["usage_total_tokens"] = u.get("total_tokens")
        for name in var_names:
            flat.setdefault(f"{name}_confidence_1to3", None)
            flat.setdefault(f"{name}_quote", None)
            flat.setdefault(f"{name}_start", None)
            flat.setdefault(f"{name}_end", None)
        flat_rows.append(flat)

    res_df = pd.DataFrame(flat_rows)
    base_part = base_df.reset_index(drop=True)
    res_part = res_df.reset_index(drop=True)
    out_df = pd.concat([base_part, res_part], axis=1)

    base_columns = list(base_df.columns)
    var_columns: List[str] = []
    for name in var_names:
        var_columns.extend([
            f"{name}_confidence_1to3",
            f"{name}_quote",
            f"{name}_start",
            f"{name}_end",
        ])
    other_columns = [
        col for col in out_df.columns
        if col not in base_columns and col not in var_columns
    ]
    final_columns = base_columns + other_columns + var_columns
    out_df = out_df.reindex(columns=final_columns)
    out_df = out_df.where(pd.notnull(out_df), None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(output_path, index=False)
    except (ImportError, ValueError) as exc:
        sys.exit(
            "Parquet output requires pyarrow or fastparquet. "
            f"Install one of them and retry. Original error: {exc}"
        )

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch-call chart abstraction CLI over a CSV of notes with rate limiting & checkpoints.")
    ap.add_argument("--input", required=True, help="Path to input CSV.")
    ap.add_argument("--output", required=True, help="Filename for Parquet output (written to ./outputs).")
    ap.add_argument("--note-col", required=True, help="Column containing the clinical note text.")
    ap.add_argument("--script", required=True, help="Path to llm-chart-abstraction-call.py.")
    ap.add_argument("--model", default=None, help="Model deployment name to pass through.")
    ap.add_argument("--var", action="append", required=True, help="Repeatable: 'name:kind:desc'.")
    ap.add_argument("--exp", action="append", default=[], help="Repeatable: variable names to add _explanation for.")
    ap.add_argument("--exp-all", action="store_true", help="Add _explanation for all variables.")
    ap.add_argument("--repair", action="store_true", help="Normalize values before validation in the child CLI.")
    ap.add_argument("--quote-per-var", action="store_true", help="Request supporting quotes for each variable.")
    ap.add_argument("--api-provider", choices=["azure", "openai", "mistral_local"], help="Override provider passed to the child CLI.")
    ap.add_argument("--api-key", help="Override OPENAI_API_KEY for child CLI (note: appears in process list).")
    ap.add_argument("--api-base", help="Override API base URL (Azure endpoint or OpenAI base).")
    ap.add_argument("--api-version", help="Override Azure API version.")
    ap.add_argument("--organization", help="Override organization/project identifier.")
    ap.add_argument("--id-col", default=None, help="Optional ID column for traceability (copied through).")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit number of input rows to process (for testing)")

    # Throughput / resilience
    ap.add_argument("--rps", type=float, default=None, help="Max requests per second (mutually exclusive with --rpm).")
    ap.add_argument("--rpm", type=int, default=60, help="Max requests per minute. Default 60.")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries per note on transient errors.")
    ap.add_argument("--checkpoint-every", type=int, default=25, help="Write partial outputs every N rows.")
    ap.add_argument("--json-out", default=None, help="Also write raw JSON list to this path (checkpoint-safe).")
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    ap.add_argument("--shard-index", type=int, default=0, help="This shard index [0..num-shards-1].")
    args = ap.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        sys.exit(f"Script not found: {script_path}")

    input_path = Path(args.input)
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)
    provided_name = Path(args.output).name
    output_path = output_dir / provided_name
    if output_path.suffix.lower() != ".parquet":
        output_path = output_path.with_suffix(".parquet")
        sys.stderr.write(f"[info] Output extension adjusted to .parquet -> {output_path.name}\n")
    json_out_path = output_dir / Path(args.json_out).name if args.json_out else None

    df = pd.read_csv(input_path)
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError("--shard-index must be in [0, num-shards).")

        # Deterministic sharding: row i → shard (i % num_shards)
        df = df.iloc[args.shard_index::args.num_shards].reset_index(drop=True)
    if args.max_rows:
        df = df.head(args.max_rows)
    if args.note_col not in df.columns:
        sys.exit(f"note column '{args.note_col}' not found in CSV.")
    if args.id_col and args.id_col not in df.columns:
        sys.exit(f"id column '{args.id_col}' not found in CSV.")

    var_names = extract_var_names(args.var)
    n_total = len(df)
    results: List[Optional[Dict[str, Any]]] = [None] * n_total  # placeholder list (same length)
    limiter = RateLimiter(rps=args.rps, rpm=None if args.rps else args.rpm)

    start_time = time.monotonic()
    last_tick = start_time
    processed = 0
    token_in = token_out = token_total = 0

    # Graceful shutdown: on Ctrl-C, dump what we have
    interrupted = {"flag": False}
    def _handle_sigint(signum, frame):
        interrupted["flag"] = True
        sys.stderr.write("\n[interrupt] Caught Ctrl-C. Writing checkpoint...\n")
        write_outputs(df.iloc[:processed], results[:processed], output_path, json_out_path, var_names)
        sys.stderr.write("[interrupt] Checkpoint written. Exiting.\n")
        sys.exit(130)
    signal.signal(signal.SIGINT, _handle_sigint)

    for i, row in df.iterrows():
        note_text = str(row[args.note_col]) if pd.notnull(row[args.note_col]) else ""
        note_identifier = str(row[args.id_col]) if args.id_col else f"row {i}"
        if not note_text.strip():
            empty_record: Dict[str, Any] = {"_error": "empty_note"}
            if args.id_col:
                empty_record["_id"] = row[args.id_col]
            results[i] = enrich_result(empty_record, var_names, note_identifier)
            processed += 1
            continue

        limiter.wait()
        t0 = time.monotonic()
        res = run_cli_once(
            script_path=script_path,
            note_text=note_text,
            model=args.model,
            var_args=args.var,
            exp_args=args.exp or [],
            exp_all=args.exp_all,
            repair=args.repair,
            quote_per_var=args.quote_per_var,
            api_provider=args.api_provider,
            api_key=args.api_key,
            api_base=args.api_base,
            api_version=args.api_version,
            organization=args.organization,
            max_retries=args.max_retries,
        )
        limiter.hit()

        # enrich with id (if provided)
        if args.id_col:
            res = dict(res)
            res["_id"] = row[args.id_col]

        # accumulate usage
        u = res.get("_usage", {})
        token_in += int(u.get("input_tokens", 0) or 0)
        token_out += int(u.get("output_tokens", 0) or 0)
        token_total += int(u.get("total_tokens", 0) or 0)

        results[i] = enrich_result(res, var_names, note_identifier)
        processed += 1

        # progress line
        now = time.monotonic()
        elapsed = now - start_time
        avg_s_per = elapsed / processed if processed else 0.0
        remaining = n_total - processed
        eta_s = remaining * avg_s_per
        notes_per_min = (processed / elapsed * 60.0) if elapsed > 0 else 0.0

        sys.stderr.write(
            f"\r[{processed}/{n_total}] "
            f"{notes_per_min:5.1f} notes/min | "
            f"avg {avg_s_per:4.2f}s/note | "
            f"ETA {eta_s/60:5.1f} min | "
            f"tokens in/out/total: {token_in}/{token_out}/{token_total}    "
        )
        sys.stderr.flush()

        # periodic checkpoint
        if processed % args.checkpoint_every == 0:
            # Only include rows processed so far
            write_outputs(df.iloc[:processed], results[:processed], output_path, json_out_path, var_names)

    # final write (all rows)
    write_outputs(df, results, output_path, json_out_path, var_names)

    total_elapsed = time.monotonic() - start_time
    sys.stderr.write(
        f"\nDone. Wrote {n_total} rows to {output_path}.\n"
        f"Total time {total_elapsed/60:.2f} min | "
        f"throughput {(n_total/total_elapsed*60):.1f} notes/min | "
        f"tokens in/out/total: {token_in}/{token_out}/{token_total}\n"
    )


    # --- Write run log JSON next to output ---
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = Path(output_path).parent if isinstance(output_path, (str, Path)) else Path(".")
        # Name log after output stem for easier pairing
        log_path = base_dir / f"{Path(output_path).stem}.log.{timestamp}.json"
        log_data = {
            "timestamp": timestamp,
            "input_file": str(args.input) if hasattr(args, "input") else None,
            "output_file": str(output_path),
            "json_output_file": str(json_out_path),
            "variables_abstracted": list(args.var) if hasattr(args, "var") else None,
            "explanation_vars": list(args.exp) if hasattr(args, "exp") else None,
            "explanation_all": bool(getattr(args, "exp_all", False)),
            "quote_per_var": bool(getattr(args, "quote_per_var", False)),
            "num_cases_processed": int(n_total),
            "elapsed_seconds": float(total_elapsed),
            "throughput_notes_per_min": (n_total/total_elapsed*60.0) if total_elapsed > 0 else None,
            "rps": float(args.rps) if getattr(args, "rps", None) is not None else None,
            "rpm": int(args.rpm) if getattr(args, "rpm", None) is not None else None,
            "tokens_in": int(token_in),
            "tokens_out": int(token_out),
            "tokens_total": int(token_total),
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        sys.stderr.write(f"✅ Log written to {log_path}\n")
    except Exception as e:
        sys.stderr.write(f"⚠️ Failed to write log: {e}\n")
if __name__ == "__main__":
    main()

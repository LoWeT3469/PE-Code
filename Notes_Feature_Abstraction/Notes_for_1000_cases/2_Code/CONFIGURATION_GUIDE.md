# Mistral Batch Configuration Guide (7B + 70B)

This guide explains how to configure and run the updated batch scripts.

## 1) What changed

### 7B (1000-case)
- `--exp-all` removed from default 7B PE/CT runs to reduce output burden.
- Runtime defaults are now lighter:
  - `MISTRAL_MAX_NEW_TOKENS=128`
  - `MISTRAL_CHUNK_TOKENS=1500`
  - `MISTRAL_MAX_CHUNKS=5`
- Local 7B child process now supports timeout control via batch runner `--timeout-s`.

### 70B
- More shard parallelism by default:
  - 200-case PE: array `0-31%4`, `NUM_SHARDS=32` default
  - 1000-case PE/CT: array `0-39%4`, `NUM_SHARDS=40` default
- You can override `NUM_SHARDS` at submit time with environment export.

## 2) Quick start commands

Run from each script's `2_Code` directory.

### 1000-case 7B PE
```bash
sbatch PE_Mistral7B.sh
```

### 1000-case 7B CT
```bash
sbatch CT_Mistral7B.sh
```

### 1000-case 70B PE
```bash
sbatch PE_Mistral70B.sh
```

### 1000-case 70B CT
```bash
sbatch CT_Mistral70B.sh
```

### 200-case 70B PE
```bash
sbatch ../../Notes_for_200_cases/2_Code/PE_Mistral70B.sh
```

## 3) How to override shard count and array size

> Important: `NUM_SHARDS` and SLURM `--array` should match.

### Example: 1000-case 70B with 80 shards, 8 running concurrently
```bash
NUM_SHARDS=80 sbatch --array=0-79%8 PE_Mistral70B.sh
```

### Example: 200-case 70B with 40 shards, 6 running concurrently
```bash
NUM_SHARDS=40 sbatch --array=0-39%6 ../../Notes_for_200_cases/2_Code/PE_Mistral70B.sh
```

## 4) How to tune runtime speed vs quality

### 7B speed-focused overrides
```bash
export MISTRAL_MAX_NEW_TOKENS=96
export MISTRAL_CHUNK_TOKENS=1200
export MISTRAL_MAX_CHUNKS=4
```

### 7B quality-focused overrides
```bash
export MISTRAL_MAX_NEW_TOKENS=192
export MISTRAL_CHUNK_TOKENS=1800
export MISTRAL_MAX_CHUNKS=6
```

### 70B speed-focused suggestion
- Increase shards (`NUM_SHARDS`) and array concurrency (`%N`) first.
- Consider removing `--exp-all` if explanations are not required.

## 5) Timeout configuration (7B batch runner)

`batch-abstract-notes-logged_Mistral7B.py` supports:
- `--timeout-s` (default `600` seconds per note)
- retry/backoff via `--max-retries`

Example direct run:
```bash
python -u ./batch-abstract-notes-logged_Mistral7B.py \
  --input ../1_Data/notes-for-1000-cases.csv \
  --output test.parquet \
  --note-col Text \
  --script ./llm-chart-abstraction-call_Mistral7B.py \
  --var "smoke_test_field:presence:demo" \
  --api-provider mistral_local \
  --timeout-s 900 \
  --max-retries 2
```

## 6) Sanity checks before full runs

```bash
bash -n PE_Mistral7B.sh
bash -n CT_Mistral7B.sh
bash -n PE_Mistral70B.sh
bash -n CT_Mistral70B.sh
```

For 200-case PE 70B:
```bash
bash -n ../../Notes_for_200_cases/2_Code/PE_Mistral70B.sh
```


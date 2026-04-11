# OneShot-ID: System Documentation (English)

This document describes the **InstantID + SDXL + InsightFace** pipeline: configuration, CLIs, outputs, selection policy, edge cases, and how to read reports. For Chinese, see [project_documentation_zh.md](project_documentation_zh.md).

## 1. System overview

- **Generation**: Single reference face → InstantID (IP-Adapter identity + ControlNet landmarks) on an SDXL checkpoint.
- **Validation**: Each candidate is analyzed with **InsightFace (AntelopeV2)**; cosine similarity to the reference embedding is compared to a threshold.
- **Orchestration**: `run_pipeline` loops generation rounds until enough images pass, then selects **8–12** finals, copies files into `kept/` and `rejected/`, and writes **CSV / JSON / Markdown** reports under `reports/`.

Default compute device in config is **`cuda:1`** (`runtime.device` in `configs/default.yaml`).

## 2. Architecture (generation → validation → selection)

1. Extract reference embedding and landmarks (failure if no face).
2. For each round, generate up to `candidates_per_round` images from the prompt bank (first round: full set; retries: **retryable** specs not yet passed).
3. Validate each new image; track the latest result per `prompt_id`.
4. Stop early if the number of **passed** prompt slots ≥ `min_keep`, or when `max_rounds` is exhausted.
5. From all **passed** results, keep at most `max_keep`, ranking by **similarity** (then `prompt_id` for ties). Copy finals to `kept/`; failures and excess passes to `rejected/`.
6. Emit **`validation_results.csv`**, **`validation_summary.json`**, and **`validation_report.md`**.

Standalone tools: **`run_generate`** (candidates only) and **`run_validate`** (validate an existing folder).

## 3. Configuration reference (`configs/default.yaml`)

| Key | Role |
| --- | --- |
| `runtime.device` | Torch device string, e.g. **`cuda:1`** |
| `runtime.seed` | Base RNG seed (pipeline offsets per round) |
| `models.insightface_dir` | InsightFace model root |
| `models.instantid_dir` | InstantID assets (ControlNet, IP-Adapter, etc.) |
| `models.sdxl_path` | SDXL single-file checkpoint path |
| `generation.min_keep` | Minimum number of passing images required (default 8) |
| `generation.max_keep` | Maximum images to keep (default 12) |
| `generation.candidates_per_round` | Cap on prompts per generation round (default 12) |
| `generation.max_rounds` | Max retry rounds when below `min_keep` (default 3) |
| `validation.similarity_threshold` | Cosine similarity cutoff (default 0.45) |
| `output.base_dir` | Root for `runs/<run_name>/` (default `outputs`) |

## 4. CLI usage

All commands accept `--config` (default `configs/default.yaml`).

### 4.1 End-to-end: `run_pipeline`

```bash
python app/run_pipeline.py --input path/to/ref.jpg [--output_name my_run] [--seed 42]
```

Produces the full tree under `outputs/runs/<run_name>/`, including **`validation_report.md`**.

### 4.2 Generation only: `run_generate`

```bash
python app/run_generate.py --input path/to/ref.jpg [--run-name my_gen] [--seed 42]
```

Writes candidates and `candidate_manifest.jsonl` only; no `kept/` selection.

### 4.3 Validation only: `run_validate`

```bash
python app/run_validate.py \
  --reference path/to/ref.jpg \
  --candidate-dir outputs/runs/my_gen/candidates \
  [--report-dir path/to/reports]
```

Writes the same three report files; the Markdown file is **`validation_report.md`**.

## 5. Output structure

```
outputs/runs/<run_name>/
├── candidates/
│   └── candidate_manifest.jsonl
├── kept/
├── rejected/
└── reports/
    ├── validation_results.csv
    ├── validation_summary.json
    └── validation_report.md
```

## 6. Selection strategy (8–12 with retries)

- **Goal**: Between `min_keep` and `max_keep` images that **pass** the similarity threshold.
- **Retries**: While passed count &lt; `min_keep`, run additional rounds (up to `max_rounds`) using retryable prompts that are not yet passed, up to `candidates_per_round` per round.
- **Overflow**: If passed count &gt; `max_keep`, keep the **top `max_keep`** by similarity; copy extra passes to `rejected/`.
- **Success**: Pipeline returns success only if final `kept` count ≥ `min_keep`.

## 7. Edge cases

- **No face in reference**: `run_pipeline`, **`run_generate`**, and **`run_validate`** exit with error if the reference has no detectable face.
- **No face in candidate**: `failed_no_face`; image copied to `rejected/` in the full pipeline.
- **Multi-face**: The **largest** face (by bounding-box area) is used for embedding and similarity. If similarity is still below threshold, status is `failed_multi_face_low_similarity` with an explanatory `failure_reason`.
- **Identity drift**: Extreme pose, occlusion, or prompt/SDXL bias can lower scores; see the Chinese doc for mitigation ideas.

## 8. Report interpretation

- **`validation_results.csv`**: One row per image — `status`, `failure_reason`, `similarity`, `face_count`, `prompt_id`. Use for spreadsheets or custom analytics.
- **`validation_summary.json`**: Batch totals, similarity mean/min/max/std, and `failure_reasons` counts. Use for dashboards or scripts.
- **`validation_report.md`**: Human-readable mirror of the summary plus per-image table; suitable for handoffs and reviews. This is the Markdown **validation report** artifact next to CSV/JSON.

Cross-reference: project root [README.md](../README.md) lists quick start and links to **project_documentation_en.md** / **project_documentation_zh.md**.

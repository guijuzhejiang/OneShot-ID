# OneShot-ID: Identity-Consistent Face Generation & Verification

**Default README**: [README.md](README.md)  
中文说明: [README_ZH.md](README_ZH.md)

OneShot-ID is an end-to-end pipeline for **identity-consistent face image generation from a single reference photo**, plus **quantitative identity verification**.

- **Generation**: InstantID (IP-Adapter identity + ControlNet landmarks) on an SDXL checkpoint
- **Verification**: InsightFace (AntelopeV2) embeddings + **cosine similarity** threshold
- **Deliverable**: A final set of **8–12 qualified images** (by default) under `outputs/runs/<run_name>/kept/`, with CSV/JSON/Markdown reports

For deeper technical details, see:
- [docs/project_documentation_en.md](docs/project_documentation_en.md)
- [docs/project_documentation_zh.md](docs/project_documentation_zh.md)

## Quick start

### 1) Environment

```bash
conda create -n py312_cu121 python=3.12 -y
conda activate py312_cu121
pip install -r requirements.txt
```

### 2) Configure model paths (required)

Edit `configs/default.yaml`:

- **Device / seed**: `runtime.device`, `runtime.seed`
- **Models**:
  - `models.insightface_dir`  
    - Download (AntelopeV2): [antelopev2.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip)
  - `models.instantid_dir`  
    - Download: [InstantX/InstantID](https://huggingface.co/InstantX/InstantID)
  - `models.sdxl_path`  
    - Example SDXL checkpoint: [Juggernaut XL (Civitai)](https://civitai.com/models/133005/juggernaut-xl)

The repo ships with the author’s local absolute paths; update them to match your machine.

### 3) Prepare a reference image

Put the reference image anywhere (many users create a `data/` folder). The pipeline works best with:
- a clear, front-facing single face
- no heavy occlusion, extreme blur, or multiple prominent faces

## Run the pipeline (recommended)

End-to-end (generate → validate → select → report):

```bash
python app/run_pipeline.py --input data/my_face.jpg --output_name my_run_01
```

Optional:

```bash
python app/run_pipeline.py --input data/my_face.jpg --config configs/default.yaml --seed 1234
```

**Success condition**: the pipeline returns exit code 0 only if final `kept/` count is at least `generation.min_keep` (default 8).

## Other CLIs

### Generate candidates only

```bash
python app/run_generate.py --input data/my_face.jpg --config configs/default.yaml --run-name my_gen --seed 42
```

This writes candidates + a manifest, but does not run selection into `kept/`.

### Validate an existing candidate folder only

```bash
python app/run_validate.py \
  --reference data/my_face.jpg \
  --candidate-dir outputs/runs/my_gen/candidates \
  --config configs/default.yaml \
  --report-dir outputs/runs/my_gen/reports
```

## Where outputs are written

By default, all runs go under `outputs/runs/<run_name>/` (config key: `output.base_dir`):

```
outputs/runs/<run_name>/
├── candidates/               # All generated candidates (possibly across rounds)
│   └── candidate_manifest.jsonl
├── faces/                    # Cropped largest-face previews per prompt_id (for human review)
├── kept/                     # Final qualified images (passed threshold and selected into 8–12)
├── rejected/                 # Failed candidates + extra passes beyond max_keep
└── reports/
    ├── validation_results.csv
    ├── validation_summary.json
    └── validation_report.md
```

**Qualified / final images** are in **`kept/`**.

## How identity verification works (high level)

- The reference face is analyzed with InsightFace to get an embedding.
- Each candidate image is analyzed; if multiple faces exist, the **largest** face is used.
- Cosine similarity is computed and compared against `validation.similarity_threshold` (default `0.45`).
- Results are written to:
  - **`reports/validation_results.csv`** (per-image: status, failure_reason, similarity, face_count, prompt_id)
  - **`reports/validation_summary.json`** (aggregated stats and failure reasons)
  - **`reports/validation_report.md`** (human-readable report)

## Configuration cheatsheet (`configs/default.yaml`)

- **Identity keep policy**
  - `generation.min_keep`: minimum number of passing images required (default 8)
  - `generation.max_keep`: cap on finals (default 12); if more pass, keep the top by similarity
  - `generation.max_rounds`: retry rounds when passed < min_keep
- **Verification**
  - `validation.similarity_threshold`: cosine similarity cutoff
- **Outputs**
  - `output.base_dir`: output root directory (default `outputs`)

## Troubleshooting

- **“No face detected in reference image”**: change to a clearer reference image (single face, higher resolution, less occlusion).
- **Low similarity / many rejects**: adjust `validation.similarity_threshold`, try a different SDXL checkpoint, or use a better reference photo.
- **Wrong GPU**: set `runtime.device` (default is `cuda:1`).

## Tests

```bash
pytest tests/
```


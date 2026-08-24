# Car Crash Detection & Insurance Claim Analyzer

An end-to-end AI system that combines computer vision, claim-document extraction, hybrid policy RAG, and explainable insurance claim decision support.

## Pipeline

```text
Vehicle Image
     ↓
Image Quality Gate
     ↓
YOLOv8 Damage Detection
     ↓
Damage Classification
     ↓
Severity Estimation
     │
     ├───────────────┐
     ↓               ↓
Claim OCR       Policy Document
+ Metadata      Hybrid RAG
     │               │
     └───────┬───────┘
             ↓
       Claim Decision
       Risk + Coverage
             ↓
      Explainable Report
```

## Project Structure

```text
Car-Crash-Detection-Insurance-Claim-Analyzer/
├── app.py
├── config.py
├── chainlit.md
├── requirements.txt
├── .env.example
├── src/
│   └── car_crash_claim_analyzer/
│       ├── vision/
│       ├── claim/
│       ├── rag/
│       ├── decision/
│       ├── pipeline.py
│       └── schemas.py
├── data/
│   ├── policies/
│   ├── samples/
│   └── damage_dataset.yaml
├── models/
├── scripts/
├── tests/
└── docs/
```

## Components

- **Computer Vision:** YOLOv8 damage detection and interpretable severity estimation.
- **Claim Intelligence:** OCR, field extraction, normalization, and validation.
- **Policy RAG:** PDF parsing, chunking, dense FAISS retrieval, BM25-style sparse retrieval, query expansion, reranking, and context compression.
- **Decision Engine:** deterministic, evidence-grounded preliminary coverage and risk assessment with explicit uncertainty.
- **UI:** Chainlit.

## Current YOLO Baseline

The existing 8-class model has been evaluated on the held-out test set:

| Metric | Score |
|---|---:|
| Precision | 79.75% |
| Recall | 70.59% |
| mAP@50 | 68.34% |
| mAP@50-95 | 45.48% |

The model includes seven specific damage classes plus `unknown`. The `unknown` class is treated by the application as an unclassified damage result rather than a vehicle class or policy condition.

A dataset-audit script and an optional clean 7-class dataset builder are provided under `scripts/`. The original dataset is never modified by the cleaning workflow. Retraining is not required to run the application.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Put the trained YOLO model at `models/best.pt` (model weights are intentionally not committed).

Start the application:

```bash
chainlit run app.py
```

The structured decision engine does not depend on the local Ollama model. If Ollama is unavailable, the UI still returns the deterministic claim assessment and retrieved policy evidence instead of stopping the pipeline.

## Development

Run the smoke tests:

```bash
pytest -q
```

Run YOLO evaluation on CPU:

```bash
python scripts/evaluate_yolo.py
```

See `docs/ARCHITECTURE.md` for the system design and `docs/PROJECT_STRUCTURE.md` for repository conventions.

## Status

Functional end-to-end prototype with evaluated YOLO baseline, policy RAG, OCR/claim extraction, deterministic decision support, and Chainlit UI.

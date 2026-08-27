# 🚗 Car Damage Detection & Insurance Claim Analyzer

An end-to-end AI decision-support prototype that combines **computer vision, claim-document extraction, policy RAG, deterministic decision logic, and grounded LLM explanations** for vehicle insurance claims.

> **Important:** This is a decision-support prototype, not an automated insurance approval/denial system. Final claim decisions require human review.

## Overview

The application accepts a vehicle image and, optionally, a claim document. It processes the inputs through the following pipeline:

```text
Vehicle Image
      │
      ▼
Image Quality Gate
      │
      ▼
YOLOv8 Damage Detection
      │
      ▼
Damage Classification + Severity
      │
      ├───────────────┐
      ▼               ▼
Claim OCR       Policy Documents
+ Extraction        │
      │             ▼
      │         Hybrid RAG
      │             │
      └───────┬─────┘
              ▼
      Deterministic Decision
              │
              ▼
       Grounded Explanation
```

## Key Features

- **Damage Detection:** YOLOv8-based vehicle damage detection.
- **Image Quality Gate:** checks image quality before running vision inference.
- **Severity Estimation:** derives an interpretable severity assessment from detection evidence.
- **Claim Extraction:** OCR, field extraction, normalization, and validation for claim documents.
- **Policy RAG:** PDF parsing, chunking, dense FAISS retrieval, sparse retrieval, query expansion, reranking, and context compression.
- **Decision Engine:** deterministic coverage/risk assessment with explicit uncertainty and manual-review states.
- **Grounded LLM Explanation:** explains the structured result using retrieved evidence without overriding the decision engine.
- **Resilience:** structured decision support remains available when the optional local LLM is unavailable.
- **Testing & CI:** pytest tests and GitHub Actions validation.

## YOLO Baseline

The existing 8-class model was evaluated on the held-out test set:

| Metric | Score |
|---|---:|
| Precision | **79.75%** |
| Recall | **70.59%** |
| mAP@50 | **68.34%** |
| mAP@50-95 | **45.48%** |

The model contains seven specific damage classes plus `unknown`. The application treats `unknown` as **unclassified damage** and does not interpret it as a vehicle class or policy condition.

## Project Structure

```text
Car-Damage-Detection-and-Insurance-Claim-Analyzer/
│
├── README.md
├── app.py
├── config.py
├── chainlit.md
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
│
├── src/
│   └── car_crash_claim_analyzer/
│       ├── vision/
│       ├── claim/
│       ├── rag/
│       ├── decision/
│       ├── application.py
│       └── ...
│
├── data/
│   ├── policies/
│   ├── samples/
│   ├── damage_dataset.yaml
│   └── README.md
│
├── models/
│   └── README.md
│
├── scripts/
│   ├── audit_yolo_dataset.py
│   ├── evaluate_yolo.py
│   └── train_yolo.py
│
├── tests/
├── docs/
└── .github/
    └── workflows/
```

Development-only datasets, caches, generated artifacts, local environments, model weights, and temporary files are excluded from the repository.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Atul1127/Car-Damage-Detection-and-Insurance-Claim-Analyzer.git
cd Car-Damage-Detection-and-Insurance-Claim-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and provide the required API configuration for the optional LLM functionality.

Never commit `.env` or API keys to GitHub.

### 5. Add model weights

Place the trained YOLO weights at:

```text
models/best.pt
```

Model weights are intentionally excluded from Git.

### 6. Run the application

```bash
chainlit run app.py
```

## Testing & Evaluation

Run the test suite:

```bash
pytest -q
```

Evaluate the YOLO model:

```bash
python scripts/evaluate_yolo.py
```

Audit the YOLO dataset:

```bash
python scripts/audit_yolo_dataset.py
```

Training is optional and is not required to run the existing application:

```bash
python scripts/train_yolo.py
```

## Decision-Safety Design

The system deliberately separates **prediction, retrieval, decision, and explanation**:

1. Vision models report detected damage and severity.
2. Retrieval supplies policy evidence with source/page metadata.
3. The deterministic decision engine produces the structured claim assessment.
4. The LLM explains the result from available evidence without overriding the structured decision.
5. Missing or uncertain information results in warnings/manual review rather than fabricated values.

This design reduces the risk of an LLM turning uncertain visual predictions or unrelated policy text into an automatic insurance decision.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system architecture and data flow.
- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) — repository organization.
- [`data/README.md`](data/README.md) — dataset and policy-data information.
- [`models/README.md`](models/README.md) — model-weight instructions.

## Limitations

- The YOLO baseline is a project prototype, not a production-grade damage estimator.
- Damage severity is an interpretable project-level estimate and should not be treated as a professional repair assessment.
- OCR quality depends on document quality and layout.
- Policy applicability depends on retrieved policy text and available claim metadata.
- Missing claim information can result in manual review.
- The optional local LLM may be unavailable; deterministic decision support can still operate without it.

## Tech Stack

**Python · YOLOv8 · OpenCV · OCR · FAISS · BM25/Sparse Retrieval · Reranking · RAG · LLM · Chainlit · Pytest · GitHub Actions**

## Status

**Completed functional prototype.**

The repository is intentionally kept as a clean, reproducible portfolio project rather than an actively evolving production system.
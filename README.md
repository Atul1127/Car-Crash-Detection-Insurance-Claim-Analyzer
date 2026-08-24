# 🚗 Car Damage Detection & Insurance Claim Analyzer

An end-to-end AI system that combines **computer vision, claim-document extraction, hybrid policy RAG, and deterministic insurance-claim decision support** in a Chainlit application.

> **Important:** This is a decision-support prototype, not an automated insurance approval/denial system. Final claim decisions require human review.

## 🎯 What the system does

Upload a vehicle image and optionally a claim document. The system runs:

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
      Deterministic Decision
       Risk + Coverage
             ↓
      Grounded Explanation
```

## ✨ Key engineering features

- **Computer Vision:** YOLOv8 damage detection with seven specific damage classes plus `unknown`.
- **Image Quality:** resolution/blur quality gate before inference.
- **Severity:** interpretable severity score derived from detection evidence.
- **Claim Intelligence:** OCR, field extraction, normalization, and validation for common claim metadata.
- **Policy RAG:** PDF parsing, chunking, dense FAISS retrieval, sparse retrieval, query expansion, reranking, and context compression.
- **Decision Engine:** deterministic coverage/risk assessment with explicit `manual_review` and uncertainty states.
- **LLM Reasoning:** generates an explanation from retrieved evidence; it does **not** replace the deterministic decision engine.
- **Resilience:** the structured assessment can still be returned when the local LLM is unavailable.
- **Testing/CI:** automated pytest suite and GitHub Actions validation.

## 📊 YOLO baseline

The existing 8-class model was evaluated on the held-out test set:

| Metric | Score |
|---|---:|
| Precision | **79.75%** |
| Recall | **70.59%** |
| mAP@50 | **68.34%** |
| mAP@50-95 | **45.48%** |

The model includes seven specific damage classes plus `unknown`. The application maps `unknown` to **unclassified damage**; it is never interpreted as a vehicle class or policy condition.

A dataset-audit script and optional clean 7-class dataset builder are provided under `scripts/`. The original dataset is not modified by the cleaning workflow. **Retraining is not required to run the application.**

## 🏗️ Project structure

```text
Car-Damage-Detection-and-Insurance-Claim-Analyzer/
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
│   └── samples/
├── models/
├── scripts/
├── tests/
└── docs/
```

## 🚀 Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Put the trained YOLO weights at:

```text
models/best.pt
```

Model weights are intentionally not committed to Git.

Start Chainlit:

```bash
chainlit run app.py
```

## 🧪 Development and evaluation

Run tests:

```bash
pytest -q
```

Evaluate YOLO on CPU:

```bash
python scripts/evaluate_yolo.py
```

Audit the dataset:

```bash
python scripts/audit_yolo_dataset.py
```

See:

- `docs/ARCHITECTURE.md` — system design
- `docs/PROJECT_STRUCTURE.md` — repository conventions

## 🔐 Decision-safety design

The system deliberately separates **prediction**, **retrieval**, **decision**, and **explanation**:

1. Vision models report detected damage and severity.
2. Retrieval supplies policy evidence with source/page metadata.
3. The deterministic decision engine produces the structured decision.
4. The LLM explains the evidence without overriding the structured decision.
5. Missing information results in explicit warnings/manual review rather than fabricated values.

This prevents an LLM from turning a weak visual prediction or unrelated policy clause into an automatic approval or denial.

## ⚠️ Known limitations

- The current YOLO baseline is a project prototype rather than a production-grade damage estimator.
- `unknown`/unclassified damage correctly leads to uncertainty rather than a fabricated damage subtype.
- OCR quality depends on document scan quality and layout.
- Policy applicability depends on the retrieved policy text and available claim metadata.
- A missing policy number, incident date, or other claim field can force manual review.
- The local LLM is optional; deterministic decision support remains available without it.

## 📌 Portfolio talking points

This project demonstrates more than model training:

- Computer vision inference and evaluation
- Dataset auditing and reproducible evaluation
- OCR and structured information extraction
- Hybrid retrieval and reranking
- Evidence-grounded LLM workflows
- Deterministic business rules around an LLM
- Failure handling and uncertainty propagation
- Automated testing and CI
- Interactive AI application development with Chainlit

## Status

**Functional end-to-end prototype** with an evaluated YOLO baseline, image quality gate, severity estimation, claim OCR/extraction, policy RAG, deterministic decision support, grounded LLM explanation, Chainlit UI, automated tests, and CI.

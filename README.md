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
- **Decision Engine:** Evidence-grounded preliminary coverage and risk assessment with explicit uncertainty.
- **UI:** Chainlit.

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

## Development

Run the smoke tests:

```bash
pytest -q
```

See `docs/ARCHITECTURE.md` for the system design and `docs/PROJECT_STRUCTURE.md` for repository conventions.

## Status

Active development toward a production-oriented multimodal insurance claim intelligence system.

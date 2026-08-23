# Car-Crash-Detection-Insurance-Claim-Analyzer

An end-to-end AI system for vehicle damage detection, insurance policy analysis, claim intelligence, and explainable claim assessment. The project combines computer vision, document retrieval, multimodal reasoning, OCR, risk analysis, and policy-grounded decision support.

## Target Architecture

```text
CAR DAMAGE IMAGE
       │
       ▼
Image Quality Check
       │
       ▼
YOLOv8 Damage Detection
       │
       ▼
Damage Classification
       │
       ▼
Severity Estimation
       │
       ├──────────────────────┐
       ▼                      ▼
Claim Information        Policy Documents
OCR + Metadata           Existing / Advanced RAG
       │                      │
       └──────────┬───────────┘
                  ▼
       Multimodal / Context RAG
                  │
                  ▼
          Claim Decision Engine
          Risk + Coverage
                  │
                  ▼
       Explainable Claim Report
```

## Current Baseline

The initial system already provides:

- **Damage Detection** — Fine-tuned YOLOv8 model for vehicle damage.
- **Damage Classification** — bumper dents/scratches, door dents/scratches, glass shatter, lamps, and unknown damage.
- **Policy RAG** — FAISS vector retrieval over an insurance policy document.
- **LLM Reasoning** — Policy-grounded claim assessment.
- **Chainlit Interface** — Interactive application for image and policy analysis.

## Upgrade Roadmap

### Phase 1 — Foundation

- Typed claim data contracts
- Image-quality validation
- Reusable YOLO detector interface
- Pipeline orchestration
- Modular project architecture

### Phase 2 — Advanced Computer Vision

- Robust image-quality assessment
- Improved YOLOv8 damage detection
- Damage classification
- Severity estimation
- Confidence calibration
- Detection and classification evaluation

### Phase 3 — Claim Intelligence

- OCR pipeline
- Claim-form extraction
- Policy-number and vehicle metadata extraction
- Structured claim validation

### Phase 4 — Advanced RAG

- Policy parsing and metadata
- Dense retrieval
- BM25 sparse retrieval
- Hybrid search
- FAISS indexing
- Metadata filtering
- Cross-encoder reranking
- Context compression

### Phase 5 — Multimodal Decision Engine

- Image evidence
- Claim metadata
- Retrieved policy evidence
- Coverage reasoning
- Risk scoring
- Anomaly and fraud signals

### Phase 6 — Explainable Claim Report

- Damage summary
- Severity assessment
- Policy evidence
- Coverage decision
- Risk score
- Reasoning trace
- Recommendations

## Tech Stack

| Component | Technology |
|---|---|
| Computer Vision | YOLOv8, Ultralytics, OpenCV, Pillow |
| OCR | Planned OCR extraction pipeline |
| Retrieval | FAISS, dense embeddings, BM25, hybrid search |
| Reranking | Cross-encoder reranker |
| LLM | Local/hosted LLM through a modular interface |
| Orchestration | Python, LangChain/LCEL where appropriate |
| Frontend | Chainlit |
| Evaluation | Precision, Recall, mAP, classification metrics, retrieval metrics |

## Project Structure

```text
Car-Crash-Detection-Insurance-Claim-Analyzer/
├── app.py
├── config.py
├── chain.py
├── embeddings.py
├── loader.py
├── llm.py
├── train_cnn.py
├── data.yaml
├── chainlit.md
│
├── car_crash_claim_analyzer/
│   ├── __init__.py
│   ├── schemas.py
│   ├── pipeline.py
│   └── vision/
│       ├── __init__.py
│       ├── quality.py
│       └── detector.py
│
├── docs/
│   └── ARCHITECTURE.md
│
└── Terms and Conditions for Private Car_2.pdf
```

## Development

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Run the existing Chainlit application with:

```bash
chainlit run app.py
```

## Project Status

**Active development.** The repository is being upgraded incrementally from the original car-damage detection and policy-RAG prototype into a modular multimodal insurance claim intelligence system.

## Internship

Built and extended during the TCS Research Internship — IIT Kharagpur, June 2026.

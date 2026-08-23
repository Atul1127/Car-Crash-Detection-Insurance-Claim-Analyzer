# Car Crash Claim Analyzer — Architecture

## Target pipeline

```text
CAR DAMAGE IMAGE
      |
      v
Image Quality Check
      |
      v
YOLOv8 Damage Detection
      |
      v
Damage Classification
      |
      v
Severity Estimation
      |
      +----------------------+----------------------+
      |                                             |
      v                                             v
Claim Information                            Policy Documents
OCR + Metadata                               Existing RAG
      |                                             |
      +----------------------+----------------------+
                             |
                             v
                 Multimodal / Context RAG
                             |
                             v
                    Claim Decision Engine
                    Risk + Coverage
                             |
                             v
                  Explainable Claim Report
```

## Implementation phases

1. **Foundation** — typed schemas, image-quality gate, detector adapter, orchestration.
2. **Vision** — production YOLO inference, damage normalization, severity model and evaluation.
3. **Claim ingestion** — OCR, metadata validation and structured claim extraction.
4. **Retrieval** — policy ingestion, hybrid retrieval, metadata filtering and reranking.
5. **Decision intelligence** — deterministic coverage checks plus multimodal reasoning and calibrated risk scoring.
6. **Explainability** — evidence-linked reports, confidence, warnings and audit trail.
7. **Evaluation/MLOps** — test datasets, metrics, experiment tracking, monitoring and deployment.

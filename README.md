# Car Crash Detection & Insurance Claim Analyzer

An end-to-end AI system that analyzes vehicle damage from images and determines insurance claim eligibility based on a policy document — built with YOLOv8, LangChain, FAISS, GPT-4o, and Chainlit.

---

## What it does

1. **Damage Detection** — User uploads a photo of a damaged vehicle. A fine-tuned YOLOv8 model detects damage type and confidence.
2. **Damage Classification** — Detected damage is classified into one of 8 categories: `bumper_dent`, `bumper_scratch`, `door_dent`, `door_scratch`, `glass_shatter`, `head_lamp`, `tail_lamp`, `unknown`.
3. **Policy Lookup (RAG)** — A RAG pipeline retrieves relevant clauses from a real insurance policy PDF using FAISS vector search and sentence embeddings.
4. **Claim Decision** — GPT-4o reasons over the detected damage and retrieved policy context to give the user a structured claim eligibility verdict.

---

## Tech Stack

| Component | Technology |
|---|---|
| Damage Detection | YOLOv8 (fine-tuned on Roboflow dataset) |
| Vector Store | FAISS |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| LLM | GPT-4o via LangChain |
| RAG Chain | LangChain `ConversationalRetrievalChain` |
| Frontend | Chainlit |
| Policy Document | National Insurance Company — Private Car Terms & Conditions |

---

## Project Structure

```
car_crash/
├── app.py              # Chainlit app — main entry point
├── chain.py            # LangChain RAG chain setup
├── embeddings.py       # FAISS index creation and loading
├── loader.py           # PDF loader and chunking
├── llm.py              # LLM initialization
├── config.py           # Paths and configuration constants
├── test.py             # Model inference testing
├── train_cnn.py        # YOLOv8 fine-tuning script
├── data.yaml           # YOLO dataset config
├── chainlit.md         # Chainlit welcome screen
└── Terms and Conditions for Private Car_2.pdf  # Insurance policy document
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/abhiram1024/car-crash-insurance.git
cd car-crash-insurance
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

### 5. Add model weights
Download the fine-tuned YOLOv8 weights (`trained.pt`) and place in the project root. Not included in the repo due to file size.

### 6. Run
```bash
chainlit run app.py
```

---

## Model Training

The YOLOv8 model was fine-tuned on a car damage dataset sourced from [Roboflow Universe](https://universe.roboflow.com), covering 8 damage categories across bumpers, doors, glass, and lamps.

To retrain:
```bash
python train_cnn.py
```

---

## How the RAG pipeline works

1. The insurance PDF is chunked and embedded using `all-MiniLM-L6-v2`
2. Chunks are stored in a FAISS index (cached to disk)
3. On each query, relevant policy sections are retrieved based on the detected damage type
4. GPT-4o receives: detected damage + confidence + retrieved policy clauses → outputs claim verdict

---

## Built during

TCS Research Internship — IIT Kharagpur, June 2026

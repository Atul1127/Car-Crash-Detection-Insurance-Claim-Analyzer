# config.py

# ── Sources ───────────────────────────────────────────────────────────────────
# Replace this path with the actual location of your Insurance Policy Document
SOURCES: list[str] = [
    r"C:\Users\abhir\projects\car_crash\Terms and Conditions for Private Car_2.pdf"
]

SOURCE_LABEL: str = "multi-source"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "llama3.1"
LOCAL_EMBED_MODEL: str = "C:\\Users\\abhir\\projects\\RAG\\FULL\\models\\all-MiniLM-L6-v2"
YOLO_MODEL_PATH: str = "best.pt"  # Your newly trained local model from runs/

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K: int = 4
MEMORY_WINDOW: int = 5

# ── FAISS persistence ─────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = "./faiss_index"
FAISS_INDEX_URL_MAP_PATH: str = "./faiss_url_map.json"
# embeddings.py
import os
import json
from hashlib import md5
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def _load_url_map() -> dict[str, str]:
    try:
        with open(config.FAISS_INDEX_URL_MAP_PATH, "r") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_url_map(mapping: dict[str, str]) -> None:
    tmp = config.FAISS_INDEX_URL_MAP_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(mapping, indent=2))
    os.replace(tmp, config.FAISS_INDEX_URL_MAP_PATH)


def _sources_key(sources: list[str]) -> str:
    combined = "|".join(sorted(sources))
    return md5(combined.encode()).hexdigest()[:12]


def get_or_build_faiss(
    sources: list[str] | None = None,
    chunks: list[Document] | None = None,
) -> FAISS:
    if sources is None:
        sources = config.SOURCES

    cache_key = _sources_key(sources)
    url_map = _load_url_map()
    embeddings = get_embedding_model()

    if cache_key in url_map:
        index_path = url_map[cache_key]
        if os.path.exists(index_path):
            try:
                print(f"[embeddings] Loading cached local index from {index_path}")
                return FAISS.load_local(
                    index_path, embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"[embeddings] Local index load failed ({e}), building...")

    if chunks is None:
        raise ValueError("No cached index found. Pass chunks to initialize a new instance.")

    index_path = os.path.join(config.FAISS_INDEX_PATH, cache_key)
    os.makedirs(index_path, exist_ok=True)

    print(f"[embeddings] Constructing vector index for {len(chunks)} text fragments...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)

    url_map[cache_key] = index_path
    _save_url_map(url_map)
    return vector_store
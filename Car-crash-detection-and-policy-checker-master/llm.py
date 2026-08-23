# llm.py
from langchain_ollama import OllamaLLM
import config

def get_llm() -> OllamaLLM:
    return OllamaLLM(model=config.MODEL_NAME)
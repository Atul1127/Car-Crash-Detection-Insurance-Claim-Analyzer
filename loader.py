import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import the exact variables matching your config.py
from config import SOURCES, CHUNK_SIZE, CHUNK_OVERLAP, LOCAL_EMBED_MODEL, FAISS_INDEX_PATH 

def build_vector_database():
    # 1. Validate your source file path from config
    if not SOURCES or not os.path.exists(SOURCES[0]):
        print(f"[{'ERROR':^10}] Could not find the policy file at the path specified in config.py.")
        if SOURCES:
            print(f"Attempted path: {SOURCES[0]}")
        return

    target_pdf = SOURCES[0]
    print(f"[{'Loader':^10}] Target Document Path: {target_pdf}")

    # 2. Extract raw text structure from your specific policy document
    print(f"[{'Loader':^10}] Extracting text from PDF...")
    loader = PyPDFLoader(target_pdf)
    raw_documents = loader.load()
    print(f"[{'Loader':^10}] Successfully loaded {len(raw_documents)} document pages.")

    # 3. Partition text into optimal semantic chunks using your config sizes
    print(f"[{'Loader':^10}] Splitting document into semantic text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    semantic_chunks = text_splitter.split_documents(raw_documents)
    print(f"[{'Loader':^10}] Created {len(semantic_chunks)} refined text segments.")
    print("\n--- SAMPLE CHUNK 1 ---")
    print(semantic_chunks[0].page_content)
    print("----------------------\n")
    
    # 4. Initialize your local Hugging Face Embedding Model from your custom local directory
    print(f"[{'Loader':^10}] Loading local embedding model from: {LOCAL_EMBED_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 5. Compute mathematical vector states and compile the FAISS Index
    print(f"[{'Loader':^10}] Computing vector matrix maps and building FAISS database...")
    vector_db = FAISS.from_documents(semantic_chunks, embedding_model)
    
    # Save the index to the folder specified in your config (./faiss_index)
    vector_db.save_local(FAISS_INDEX_PATH)
    print(f"[{'SUCCESS':^10}] Vector database successfully stored locally at: {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    build_vector_database()
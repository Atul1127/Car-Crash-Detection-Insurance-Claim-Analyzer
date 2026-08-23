import os
import io
from PIL import Image
import chainlit as cl
from ultralytics import YOLO

# Modern LangChain Core & Integration components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import your central configuration framework
from config import (
    YOLO_MODEL_PATH, 
    FAISS_INDEX_PATH, 
    LOCAL_EMBED_MODEL, 
    MODEL_NAME, 
    RETRIEVAL_K
)

# ── 1. INITIALIZE COMPUTER VISION MODEL ──────────────────────────────────────
# Resolves path whether best.pt sits in root directory or under runs/ train outputs
WEIGHTS_PATH = os.path.join("runs", "detect", "train-3", "weights", YOLO_MODEL_PATH) if not os.path.exists(YOLO_MODEL_PATH) else YOLO_MODEL_PATH
yolo_model = YOLO(WEIGHTS_PATH)

# ── 2. INITIALIZE KNOWLEDGE GRAPH EMBEDDINGS & FAISS ─────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name=LOCAL_EMBED_MODEL,
    model_kwargs={'device': 'cpu'}  # Keeps GPU available entirely for YOLO runs
)

# Load the local FAISS matrix configuration safely
vector_db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# ── 3. SETUP LOCAL INFERENCE GENERATION & SYSTEM PROMPTS ─────────────────────
llm = Ollama(model=MODEL_NAME)

prompt_template = """
You are an expert insurance claim adjuster AI. Your job is to analyze the vehicle damage detected by the computer vision system against the official policy clauses provided below.

Rules:
1. Base your assessment heavily on the provided Policy Context clauses.
2. Glass damage (like glass_shatter) typically has 0% depreciation (Nil depreciation) under standard rules.
3. Be professional, direct, and itemize any financial depreciation rules if applicable.

Policy Context:
{context}

Question:
{question}

Answer with clear details regarding coverage eligibility, depreciation structural brackets, or exceptions:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Compile LCEL chain pipeline execution map
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 4. CHAINLIT CORE APPLICATION LOGIC ───────────────────────────────────────
@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="🚗 **Vehicle Damage Assessment & Insurance RAG System Fully Loaded** 🚗\n\n"
                "• **Step 1:** Upload a vehicle image via the clip icon to analyze physical damages.\n"
                "• **Step 2:** Ask any standard policy compliance text questions directly in the chat panel."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    detected_damages = []

    # --- PART A: Handle Image Upload Visual Diagnostics ---
    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                await cl.Message(content="🔍 Analyzing vehicle image for damage features...").send()
                
                # Run YOLO directly via the local disk cache path Chainlit provisions
                results = yolo_model.predict(source=element.path, conf=0.25, save=False, device="0")
                
                for result in results:
                    # Construct canvas matrix box coordinates on image
                    im_array = result.plot()
                    output_image = Image.fromarray(im_array[..., ::-1]) # Reorient BGR map back to RGB
                    
                    # Convert canvas into streaming buffer for interface rendering
                    buffer = io.BytesIO()
                    output_image.save(buffer, format="JPEG")
                    buffer.seek(0)
                    
                    cl_image = cl.Image(
                        content=buffer.read(),
                        name="damage_assessment.jpg",
                        display="inline",
                        size="large"
                    )
                    
                    # Map tensor list components to human class strings
                    detected_damages = [yolo_model.names[int(box.cls[0])] for box in result.boxes]
                    
                    summary_text = f"✅ **Visual Analysis Complete!**\n"
                    if detected_damages:
                        summary_text += f"Captured Damage Anomalies: `{', '.join(set(detected_damages))}`\n"
                        summary_text += f"Total Damage Instances Counted: **{len(detected_damages)}**"
                    else:
                        summary_text += "No structural damage patterns matching training classes identified."
                        
                    await cl.Message(content=summary_text, elements=[cl_image]).send()

    # --- PART B: Handle Unified RAG Retrieval Pipelines ---
    # If vision framework hit a positive match, auto-phrase policy target lookup question
    if detected_damages:
        damage_list_str = ", ".join(set(detected_damages))
        rag_query = f"Does the policy cover accidental vehicle damage like {damage_list_str}? What are the depreciation rates or limitations mentioned for these parts?"
        await cl.Message(content=f"📑 **RAG Pipeline Triggered:** Fetching policy document boundaries regarding `{damage_list_str}` terms...").send()
    else:
        # Otherwise treat input as a pure standalone text question
        rag_query = message.content

    # Execute text query logic against vector space coordinates if string contains text characters
    if rag_query and rag_query.strip():
        # Change .invoke() to await ... .ainvoke() to prevent thread blocking
        response = await rag_chain.ainvoke(rag_query)
        
        await cl.Message(content=f"🤖 **Policy Assessment Answer:**\n\n{response}").send()
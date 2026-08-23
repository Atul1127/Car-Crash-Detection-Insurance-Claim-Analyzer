# chain.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

import config
from llm import get_llm

GROUNDING_PROMPT_TEMPLATE = """You are an expert insurance claims adjuster and advisor. 
Review the vehicle damage scan report alongside the provided insurance policy chunks to evaluate whether the requested damages are covered under the current policy terms.

Rules:
1. Provide a thorough, direct, and explicit explanation outlining if the claim is valid, partially valid, or denied.
2. Rely strictly on the attached document context. Do not invent any outside clauses or terms.
3. If the context does not provide explicit parameters to evaluate the damage, output exactly: NO_ANSWER
4. Do not write anything before or after NO_ANSWER if the information is unavailable.

Context:
{context}

Question: {question}
Answer:"""

CONDENSE_PROMPT_TEMPLATE = """Rephrase the follow-up question as a standalone query using the chat history.
Return only the question text.

Chat History:
{chat_history}

Follow-up: {question}
Standalone question:"""


def _format_chat_history(chat_history: list[tuple[str, str]]) -> str:
    return "\n".join(f"Human: {h}\nAssistant: {a}" for h, a in chat_history)


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _is_no_answer(text: str) -> bool:
    t = text.strip().upper()
    if t == "NO_ANSWER":
        return True
    refusal_phrases = [
        "NO ANSWER CAN BE FOUND",
        "I DON'T KNOW",
        "I DO NOT KNOW",
        "THE CONTEXT DOES NOT",
        "NOT FOUND IN THE CONTEXT",
        "CANNOT ANSWER",
    ]
    if len(t) < 200 and any(phrase in t for phrase in refusal_phrases):
        return True
    return False


class ConversationalRAGChain:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.llm = get_llm()
        self.chat_history: list[tuple[str, str]] = []
        self.condense_prompt = PromptTemplate.from_template(CONDENSE_PROMPT_TEMPLATE)
        self.answer_prompt = PromptTemplate.from_template(GROUNDING_PROMPT_TEMPLATE)
        self.parser = StrOutputParser()
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.RETRIEVAL_K, "fetch_k": config.RETRIEVAL_K * 4, "lambda_mult": 0.5},
        )

    def _condense_question(self, question: str) -> str:
        if not self.chat_history:
            return question
        chain = self.condense_prompt | self.llm | self.parser
        return chain.invoke({
            "chat_history": _format_chat_history(self.chat_history),
            "question": question,
        }).strip()

    def invoke(self, inputs: dict) -> dict:
        question = inputs["question"]
        standalone_question = self._condense_question(question)

        source_documents = self.retriever.invoke(standalone_question)
        
        if not source_documents:
            answer = "NO_ANSWER"
            context = ""
        else:
            context = _format_docs(source_documents)
            answer_chain = self.answer_prompt | self.llm | self.parser
            raw = answer_chain.invoke(
                {"context": context, "question": standalone_question}
            ).strip()

            if _is_no_answer(raw):
                answer = "NO_ANSWER"
            else:
                answer = raw

        history_answer = "I don't know" if answer == "NO_ANSWER" else answer
        self.chat_history.append((question, history_answer))
        if len(self.chat_history) > config.MEMORY_WINDOW:
            self.chat_history = self.chat_history[-config.MEMORY_WINDOW:]

        return {
            "answer": answer,
            "source_documents": source_documents,
        }


def build_chain(vector_store: FAISS) -> ConversationalRAGChain:
    return ConversationalRAGChain(vector_store)
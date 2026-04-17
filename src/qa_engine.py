"""
qa_engine.py
------------
Conversational Q&A over documents using LangChain's ConversationalRetrievalChain.
Maintains conversation history (memory) across multiple questions.
"""

from typing import Dict, List, Optional, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ── System prompt that guides the assistant's behavior ────────────────────────

QA_SYSTEM_PROMPT = """You are an intelligent document analysis assistant.
Your job is to answer questions based ONLY on the provided document context.

Guidelines:
- Answer clearly and directly based on the document content
- If the answer is not in the documents, say: "I don't see that information in the uploaded documents."
- Always cite the page number when possible (e.g., "According to page 3...")
- For complex questions, break down your answer into numbered points
- Be concise but complete — don't leave out important details
- If asked to compare or analyze, structure your response clearly

Context from documents:
{context}

Conversation history:
{chat_history}

Current question: {question}

Answer:"""


class QAEngine:
    """
    Provides conversational Q&A over a FAISS vector store with memory.
    Uses Groq (FREE) for fast LLM inference.
    """

    def __init__(
        self,
        vector_store: FAISS,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        retrieval_k: int = 5,
        memory_window: int = 10,
    ):
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.retrieval_k = retrieval_k
        self.conversation_history: List[Tuple[str, str]] = []

        # Groq LLM — free and very fast
        self.llm = ChatGroq(model=model, temperature=temperature)

        # Retriever — fetches top-k most relevant chunks per question
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k},
        )

        # Memory — keeps last N exchanges so the model can refer to earlier answers
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=memory_window,
        )

        # The full conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
        )

    def ask(self, question: str) -> Dict:
        """
        Ask a question about the documents.

        Returns a dict with:
        - answer: str
        - sources: list of source document metadata
        - question: the original question
        """
        if not question.strip():
            return {"answer": "Please enter a question.", "sources": [], "question": question}

        result = self.qa_chain.invoke({"question": question})

        answer = result.get("answer", "I couldn't generate an answer.")
        source_docs = result.get("source_documents", [])

        # Extract unique source citations
        sources = []
        seen = set()
        for doc in source_docs:
            meta = doc.metadata
            file_name = meta.get("source_file", meta.get("source", "Unknown"))
            page_num = meta.get("page", "?")
            key = f"{file_name}:p{page_num}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": file_name,
                    "page": page_num + 1 if isinstance(page_num, int) else page_num,
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                })

        # Save to local history
        self.conversation_history.append((question, answer))

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
        }

    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """Return the full conversation history as (question, answer) tuples."""
        return self.conversation_history

    def clear_memory(self) -> None:
        """Reset conversation memory and history."""
        self.memory.clear()
        self.conversation_history = []

    def get_follow_up_suggestions(self, last_answer: str) -> List[str]:
        """
        Generate 3 suggested follow-up questions based on the last answer.
        Helps users explore documents more deeply.
        """
        prompt = f"""Based on this answer about a document:

"{last_answer}"

Generate exactly 3 short, insightful follow-up questions a user might ask.
Return only the questions, one per line, no numbering or bullets."""

        response = self.llm.invoke(prompt)
        lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        return lines[:3]

    def ask_with_suggestions(self, question: str) -> Dict:
        """Ask a question and also generate follow-up suggestions."""
        result = self.ask(question)
        suggestions = self.get_follow_up_suggestions(result["answer"])
        result["suggestions"] = suggestions
        return result
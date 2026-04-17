"""
pdf_processor.py
----------------
Handles PDF loading, text extraction, chunking, and vector store creation.
Uses FREE local embeddings (sentence-transformers) + FAISS — no API cost for embeddings.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class PDFProcessor:
    """
    Loads PDFs, splits them into chunks, and builds a FAISS vector store
    for semantic search and retrieval.

    Uses HuggingFace sentence-transformers for embeddings — completely FREE,
    runs locally on your machine, no API key needed for this part.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []

        # Free local embeddings — downloads once (~80MB), runs offline after that
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load a single PDF and return a list of LangChain Document objects."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        loader = PyPDFLoader(str(path))
        pages = loader.load()

        # Add filename metadata to every page
        for page in pages:
            page.metadata["source_file"] = path.name
            page.metadata["total_pages"] = len(pages)

        return pages

    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Load multiple PDFs and combine all pages into one list."""
        all_docs = []
        for path in pdf_paths:
            docs = self.load_pdf(path)
            all_docs.extend(docs)
        return all_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller overlapping chunks for better retrieval."""
        chunks = self.text_splitter.split_documents(documents)
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

    def build_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Embed all chunks using OpenAI embeddings and store them in a FAISS index.
        This enables fast semantic similarity search.
        """
        if not chunks:
            raise ValueError("No chunks to embed. Load a PDF first.")

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def process_pdfs(self, pdf_paths: List[str]) -> FAISS:
        """
        Full pipeline: load -> chunk -> embed -> store.
        Returns a ready-to-query FAISS vector store.
        """
        raw_docs = self.load_multiple_pdfs(pdf_paths)
        self.documents = raw_docs
        chunks = self.chunk_documents(raw_docs)
        vector_store = self.build_vector_store(chunks)
        return vector_store

    def save_vector_store(self, save_path: str) -> None:
        """Persist the FAISS index to disk so you don't have to re-embed every time."""
        if self.vector_store is None:
            raise RuntimeError("No vector store to save. Run process_pdfs first.")
        self.vector_store.save_local(save_path)

    def load_vector_store(self, load_path: str) -> FAISS:
        """Load a previously saved FAISS index from disk."""
        self.vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve the top-k most relevant chunks for a given query."""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Run process_pdfs first.")
        return self.vector_store.similarity_search(query, k=k)

    def get_document_stats(self) -> dict:
        """Return basic stats about the loaded documents."""
        if not self.documents:
            return {}
        files = list({doc.metadata.get("source_file", "Unknown") for doc in self.documents})
        return {
            "total_pages": len(self.documents),
            "files_loaded": files,
            "num_files": len(files),
        }
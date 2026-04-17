"""
tests/test_basic.py
-------------------
Basic unit tests for the document intelligence system.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPDFProcessor:

    def test_import(self):
        from pdf_processor import PDFProcessor
        assert PDFProcessor is not None

    def test_init_defaults(self):
        with patch("pdf_processor.HuggingFaceEmbeddings"):
            from pdf_processor import PDFProcessor
            proc = PDFProcessor()
            assert proc.chunk_size == 1000
            assert proc.chunk_overlap == 200

    def test_init_custom(self):
        with patch("pdf_processor.HuggingFaceEmbeddings"):
            from pdf_processor import PDFProcessor
            proc = PDFProcessor(chunk_size=500, chunk_overlap=50)
            assert proc.chunk_size == 500
            assert proc.chunk_overlap == 50

    def test_load_nonexistent_pdf_raises(self):
        with patch("pdf_processor.HuggingFaceEmbeddings"):
            from pdf_processor import PDFProcessor
            proc = PDFProcessor()
            with pytest.raises(FileNotFoundError):
                proc.load_pdf("/tmp/does_not_exist.pdf")

    def test_load_non_pdf_raises(self):
        with patch("pdf_processor.HuggingFaceEmbeddings"):
            from pdf_processor import PDFProcessor
            dummy = "/tmp/test_doc.txt"
            Path(dummy).write_text("hello")
            proc = PDFProcessor()
            with pytest.raises(ValueError):
                proc.load_pdf(dummy)

    def test_similarity_search_without_store_raises(self):
        with patch("pdf_processor.HuggingFaceEmbeddings"):
            from pdf_processor import PDFProcessor
            proc = PDFProcessor()
            with pytest.raises(RuntimeError):
                proc.similarity_search("test query")

    def test_get_document_stats_empty(self):
        with patch("pdf_processor.HuggingFaceEmbeddings"):
            from pdf_processor import PDFProcessor
            proc = PDFProcessor()
            stats = proc.get_document_stats()
            assert stats == {}


class TestDocumentSummarizer:

    def test_import(self):
        from summarizer import DocumentSummarizer
        assert DocumentSummarizer is not None

    def test_summarize_empty_returns_message(self):
        with patch("summarizer.ChatGroq"):
            from summarizer import DocumentSummarizer
            ds = DocumentSummarizer()
            result = ds.summarize([])
            assert "No documents" in result

    def test_extract_insights_empty_returns_message(self):
        with patch("summarizer.ChatGroq"):
            from summarizer import DocumentSummarizer
            ds = DocumentSummarizer()
            result = ds.extract_insights([])
            assert "No documents" in result

    def test_compare_documents_needs_two(self):
        with patch("summarizer.ChatGroq"):
            from summarizer import DocumentSummarizer
            ds = DocumentSummarizer()
            result = ds.compare_documents(["only one"])
            assert "at least 2" in result


class TestQAEngine:

    def test_import(self):
        from qa_engine import QAEngine
        assert QAEngine is not None

    def test_ask_empty_question(self):
        with patch("qa_engine.ChatGroq"), \
             patch("qa_engine.ConversationalRetrievalChain"):
            from qa_engine import QAEngine
            mock_store = MagicMock()
            mock_store.as_retriever.return_value = MagicMock()
            engine = QAEngine(vector_store=mock_store)
            result = engine.ask("")
            assert "Please enter" in result["answer"]

    def test_conversation_history_starts_empty(self):
        with patch("qa_engine.ChatGroq"), \
             patch("qa_engine.ConversationalRetrievalChain"):
            from qa_engine import QAEngine
            mock_store = MagicMock()
            mock_store.as_retriever.return_value = MagicMock()
            engine = QAEngine(vector_store=mock_store)
            assert engine.get_conversation_history() == []

    def test_clear_memory(self):
        with patch("qa_engine.ChatGroq"), \
             patch("qa_engine.ConversationalRetrievalChain"):
            from qa_engine import QAEngine
            mock_store = MagicMock()
            mock_store.as_retriever.return_value = MagicMock()
            engine = QAEngine(vector_store=mock_store)
            engine.conversation_history = [("q1", "a1"), ("q2", "a2")]
            engine.clear_memory()
            assert engine.get_conversation_history() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
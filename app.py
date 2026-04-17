"""
app.py
------
Main Streamlit application for the AI Document Intelligence System.
Run with: streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from qa_engine import QAEngine
from summarizer import DocumentSummarizer

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #fff !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255,255,255,0.05);
    border: 1.5px dashed rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 0.5rem;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
}
.hero-badges { display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Feature cards ── */
.feature-card {
    background: linear-gradient(145deg, #ffffff, #f8faff);
    border: 1px solid #e8eaf6;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(99,102,241,0.06);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.75rem; }
.feature-title { font-size: 1rem; font-weight: 700; color: #1e1b4b; margin-bottom: 0.4rem; }
.feature-desc { font-size: 0.85rem; color: #64748b; line-height: 1.5; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9; border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #4f46e5 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08) !important;
}

/* ── Content cards ── */
.summary-card {
    background: linear-gradient(145deg, #fefefe, #f8f9ff);
    border: 1px solid #e0e7ff;
    border-radius: 16px;
    padding: 1.75rem;
    line-height: 1.8;
    color: #1e293b;
    box-shadow: 0 2px 16px rgba(99,102,241,0.06);
}
.insight-card {
    background: linear-gradient(145deg, #fffbeb, #fff8e1);
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 0 14px 14px 0;
    padding: 1.5rem;
    color: #1c1917;
    line-height: 1.8;
}

/* ── Chat bubbles — FIXED COLORS ── */
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.75rem 0;
}
.chat-user-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    padding: 0.85rem 1.15rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 3px 15px rgba(99,102,241,0.3);
}
.chat-user-label {
    color: rgba(255,255,255,0.65);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.chat-user-text { color: #ffffff; }
.chat-ai {
    display: flex;
    justify-content: flex-start;
    margin: 0.75rem 0;
}
.chat-ai-bubble {
    background: #ffffff;
    color: #1e293b;
    padding: 0.85rem 1.15rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.7;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 15px rgba(0,0,0,0.06);
}
.chat-ai-label {
    color: #6366f1;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.chat-ai-text { color: #1e293b; }

/* ── Source box ── */
.source-box {
    background: #f8faff;
    border: 1px solid #c7d2fe;
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #374151;
}

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 2.5rem 1rem; }
.empty-icon { font-size: 2.8rem; margin-bottom: 0.75rem; }
.empty-text { font-size: 1.05rem; font-weight: 500; color: #64748b; }
.empty-sub { font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem; }

/* ── Primary buttons ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────────

def check_api_key() -> bool:
    key = os.getenv("GROQ_API_KEY") or st.session_state.get("api_key", "")
    if key and key.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = key
        return True
    return False


def initialize_session_state():
    defaults = {
        "processor": None, "summarizer": None, "qa_engine": None,
        "vector_store": None, "documents_loaded": False,
        "summary": None, "insights": None, "chat_history": [],
        "api_key": "", "doc_stats": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploaded_files(uploaded_files) -> bool:
    if not uploaded_files:
        return False
    with st.spinner("⚙️ Embedding your PDFs into the vector store..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_paths = []
            for file in uploaded_files:
                tmp_path = os.path.join(tmp_dir, file.name)
                with open(tmp_path, "wb") as f:
                    f.write(file.read())
                temp_paths.append(tmp_path)

            processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
            try:
                vector_store = processor.process_pdfs(temp_paths)
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")
                return False

            st.session_state.processor = processor
            st.session_state.vector_store = vector_store
            st.session_state.doc_stats = processor.get_document_stats()
            st.session_state.documents_loaded = True
            st.session_state.summary = None
            st.session_state.insights = None
            st.session_state.chat_history = []
            st.session_state.summarizer = DocumentSummarizer(model="llama-3.1-8b-instant")
            st.session_state.qa_engine = QAEngine(
                vector_store=vector_store, model="llama-3.1-8b-instant", retrieval_k=5,
            )
    return True


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    initialize_session_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 0.5rem'>
            <div style='font-size:2.2rem'>🧠</div>
            <div style='font-size:1.1rem;font-weight:700;color:#fff'>DocMind AI</div>
            <div style='font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:2px'>Groq + LangChain + FAISS</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**⚙️ API Key**")
        if not check_api_key():
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="FREE — get yours at console.groq.com",
            )
            if api_key:
                st.session_state.api_key = api_key
                os.environ["GROQ_API_KEY"] = api_key
                st.success("✅ Saved!")
        else:
            st.success("✅ Groq connected — free tier")

        st.divider()
        st.markdown("**📂 Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown(f"<div style='font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:6px'>{len(uploaded_files)} file(s) ready</div>", unsafe_allow_html=True)
            if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
                if not check_api_key():
                    st.error("Enter Groq API key first.")
                else:
                    if process_uploaded_files(uploaded_files):
                        st.success(f"✅ Done!")
                        st.rerun()

        if st.session_state.documents_loaded:
            stats = st.session_state.doc_stats
            st.divider()
            st.markdown("**📊 Loaded**")
            c1, c2 = st.columns(2)
            c1.metric("Files", stats.get("num_files", 0))
            c2.metric("Pages", stats.get("total_pages", 0))
            for f in stats.get("files_loaded", []):
                st.markdown(f"<div style='font-size:0.75rem;color:rgba(255,255,255,0.45)'>📄 {f}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear All", use_container_width=True):
                for k in ["processor","summarizer","qa_engine","vector_store","summary","insights"]:
                    st.session_state[k] = None
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.session_state.doc_stats = {}
                st.rerun()

        st.divider()
        st.markdown("<div style='font-size:0.7rem;color:rgba(255,255,255,0.25);text-align:center'>v1.0 · MIT License</div>", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">🧠 DocMind AI</div>
        <div class="hero-sub">Upload any PDF — get summaries, insights, and chat with your documents</div>
        <div class="hero-badges">
            <span class="badge">⚡ Groq LLaMA 3.1</span>
            <span class="badge">🔗 LangChain RAG</span>
            <span class="badge">🔍 FAISS Search</span>
            <span class="badge">🆓 100% Free</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Landing ───────────────────────────────────────────────────────────────
    if not st.session_state.documents_loaded:
        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc in zip(
            [c1, c2, c3],
            ["📋", "💡", "💬"],
            ["Smart Summarization", "Insight Extraction", "Conversational Q&A"],
            [
                "Generates comprehensive summaries of any PDF length using AI map-reduce.",
                "Extracts themes, facts, action items, and document type — all structured.",
                "Ask anything in plain English. Answers include exact page citations.",
            ],
        ):
            with col:
                st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="empty-state" style="margin-top:2rem">
            <div class="empty-icon">👈</div>
            <div class="empty-text">Upload a PDF in the sidebar to get started</div>
            <div class="empty-sub">Research papers · Reports · Books · Manuals — any PDF works</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋  Summary", "💡  Key Insights", "💬  Chat"])

    # ── Summary tab ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-label'>AI-Generated Document Summary</div>", unsafe_allow_html=True)
        if st.session_state.summary is None:
            st.markdown('<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-text">No summary yet</div><div class="empty-sub">Click below to generate a comprehensive AI summary</div></div>', unsafe_allow_html=True)
            if st.button("✨ Generate Summary", type="primary"):
                with st.spinner("Summarizing — please wait ~20 seconds..."):
                    try:
                        summary = st.session_state.summarizer.summarize(st.session_state.processor.documents)
                        st.session_state.summary = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.markdown(f'<div class="summary-card">{st.session_state.summary}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Regenerate"):
                st.session_state.summary = None
                st.rerun()

    # ── Insights tab ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-label'>Structured Key Insight Extraction</div>", unsafe_allow_html=True)
        if st.session_state.insights is None:
            st.markdown('<div class="empty-state"><div class="empty-icon">💡</div><div class="empty-text">No insights yet</div><div class="empty-sub">Extracts themes, facts, action items, and more</div></div>', unsafe_allow_html=True)
            if st.button("🔍 Extract Key Insights", type="primary"):
                with st.spinner("Analyzing your document..."):
                    try:
                        insights = st.session_state.summarizer.extract_insights(st.session_state.processor.documents)
                        st.session_state.insights = insights
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.markdown(f'<div class="insight-card">{st.session_state.insights}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Re-extract"):
                st.session_state.insights = None
                st.rerun()

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-label'>Ask anything — answers grounded in your document with page citations</div>", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown('<div class="empty-state"><div class="empty-icon">💬</div><div class="empty-text">No messages yet</div><div class="empty-sub">Try: "What is this document about?" or "List the key findings"</div></div>', unsafe_allow_html=True)

        # Render chat bubbles
        for q, a in st.session_state.chat_history:
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-user-bubble">
                    <div class="chat-user-label">You</div>
                    <div class="chat-user-text">{q}</div>
                </div>
            </div>
            <div class="chat-ai">
                <div class="chat-ai-bubble">
                    <div class="chat-ai-label">🧠 DocMind AI</div>
                    <div class="chat-ai-text">{a}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input(
                "question",
                placeholder="Type your question here...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

        if submitted and question.strip():
            with st.spinner("Searching document and generating answer..."):
                try:
                    result = st.session_state.qa_engine.ask_with_suggestions(question)
                    st.session_state.chat_history.append((question, result["answer"]))

                    if result.get("sources"):
                        st.markdown("<div class='section-label' style='margin-top:1rem'>📍 Sources</div>", unsafe_allow_html=True)
                        for src in result["sources"]:
                            st.markdown(f'<div class="source-box">📄 <strong>{src["file"]}</strong> — Page {src["page"]}<br><em>{src["excerpt"]}</em></div>', unsafe_allow_html=True)

                    if result.get("suggestions"):
                        st.markdown("<div class='section-label' style='margin-top:1rem'>💡 Follow-up suggestions</div>", unsafe_allow_html=True)
                        cols = st.columns(len(result["suggestions"]))
                        for i, s in enumerate(result["suggestions"]):
                            with cols[i]:
                                if st.button(f"💬 {s}", key=f"sug_{len(st.session_state.chat_history)}_{i}", use_container_width=True):
                                    st.session_state._suggestion_clicked = s
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if hasattr(st.session_state, "_suggestion_clicked"):
            q = st.session_state._suggestion_clicked
            del st.session_state._suggestion_clicked
            with st.spinner("Answering..."):
                res = st.session_state.qa_engine.ask(q)
                st.session_state.chat_history.append((q, res["answer"]))
                st.rerun()

        if st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.qa_engine.clear_memory()
                st.rerun()


if __name__ == "__main__":
    main()
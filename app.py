import os
import streamlit as st
import tempfile
import json
import logging
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #134e5e, #71b280);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0; }
.hero p { opacity: 0.85; margin: 0.5rem 0 0 0; font-size: 1.05rem; }

.feature-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 0.5rem 0.2rem 0;
}

.doc-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.doc-icon { font-size: 1.5rem; }
.doc-info { flex: 1; }
.doc-name { font-weight: 600; font-size: 0.9rem; color: #134e5e; }
.doc-meta { font-size: 0.78rem; color: #888; }

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stat-val { font-size: 2rem; font-weight: 700; color: #134e5e; }
.stat-lbl { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }

.entity-chip {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.15rem;
}
.chip-org { background: #dbeafe; color: #1e40af; }
.chip-date { background: #dcfce7; color: #166534; }
.chip-money { background: #fef9c3; color: #854d0e; }
.chip-topic { background: #f3e8ff; color: #6b21a8; }
.chip-loc { background: #ffedd5; color: #9a3412; }

.insight-card {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.6rem 0;
    border-left: 4px solid #22c55e;
    line-height: 1.6;
}

.answer-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid #e0e0e0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    margin: 0.5rem 0;
    line-height: 1.7;
}

.source-chip {
    display: inline-block;
    background: #e0f2fe;
    color: #075985;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 0.15rem;
    font-weight: 500;
}

.empty-state {
    text-align: center;
    padding: 3rem;
    color: #999;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "entities" not in st.session_state:
    st.session_state.entities = {}
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "comparison" not in st.session_state:
    st.session_state.comparison = None


@st.cache_resource
def get_embeddings():
    """Load HuggingFace embeddings model (cached across sessions)."""
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        logger.error("Failed to load embeddings model: %s", e)
        raise RuntimeError(
            "Could not load the sentence-transformers model. "
            "Ensure sentence-transformers is installed: pip install sentence-transformers"
        ) from e


def get_llm(api_key: str, temp: float = 0) -> ChatGroq:
    """Instantiate a Groq LLM client with basic validation."""
    if not api_key or not api_key.strip():
        raise ValueError("A valid Groq API key is required.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temp,
        groq_api_key=api_key.strip(),
        max_tokens=1024,
    )


def process_pdf(uploaded_file) -> list:
    """Parse an uploaded PDF into a list of LangChain Document objects.

    Writes the file to a temporary path, loads it via PyPDFLoader, then
    removes the temporary file before returning.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        if not docs:
            raise ValueError(f"No readable pages found in '{uploaded_file.name}'. "
                             "The PDF may be scanned or password-protected.")
        return docs
    except Exception as e:
        logger.error("PDF processing failed for %s: %s", uploaded_file.name, e)
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def rebuild_vectorstore(documents: dict) -> FAISS:
    """Split all stored documents into chunks and build a unified FAISS index."""
    if not documents:
        raise ValueError("No documents available to index.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for fname, data in documents.items():
        chunks = splitter.split_documents(data["docs"])
        for chunk in chunks:
            chunk.metadata["source_file"] = fname
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError("Documents produced no text chunks. Check that PDFs contain extractable text.")
    embeddings = get_embeddings()
    return FAISS.from_documents(all_chunks, embeddings)


def summarise(text: str, fname: str, api_key: str) -> str:
    """Generate a concise 4-5 sentence summary of a document."""
    if not text.strip():
        return "No text content available to summarise."
    llm = get_llm(api_key, 0.2)
    try:
        response = llm.invoke(
            f"Summarise this document in 4-5 sentences. Be specific and factual.\n\n"
            f"Document ({fname}): {text[:4000]}\n\nSummary:"
        )
        return response.content.strip()
    except Exception as e:
        logger.error("Summarisation failed for %s: %s", fname, e)
        return f"Summarisation failed: {e}"


def extract_entities(text: str, api_key: str) -> dict:
    """Extract named entities from document text as a structured dict.

    Returns a dict with keys: organisations, dates, monetary_values,
    key_topics, locations. Falls back to empty lists on parse failure.
    """
    empty = {
        "organisations": [],
        "dates": [],
        "monetary_values": [],
        "key_topics": [],
        "locations": [],
    }
    if not text.strip():
        return empty
    llm = get_llm(api_key, 0)
    try:
        response = llm.invoke(
            f"Extract key entities. Return ONLY valid JSON, no other text.\n\n"
            f"Document: {text[:3000]}\n\n"
            f'Return exactly:\n{{"organisations": [], "dates": [], "monetary_values": [], '
            f'"key_topics": [], "locations": []}}'
        )
        content = response.content.strip()
        # Strip markdown code fences if present
        if "```" in content:
            parts = content.split("```")
            # parts[1] is inside the first code fence
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        # Ensure all expected keys are present and are lists
        for key in empty:
            if key not in parsed or not isinstance(parsed[key], list):
                parsed[key] = []
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("Entity JSON parse error: %s", e)
        return empty
    except Exception as e:
        logger.error("Entity extraction failed: %s", e)
        return empty


def compare_docs(summaries: dict, api_key: str) -> str:
    """Generate a structured comparative analysis across multiple document summaries."""
    if len(summaries) < 2:
        return "At least two document summaries are required for comparison."
    llm = get_llm(api_key, 0.3)
    docs_text = "\n\n".join(
        [f"**{name}**: {summary}" for name, summary in summaries.items()]
    )
    try:
        response = llm.invoke(
            f"Compare these documents comprehensively.\n\n{docs_text}\n\n"
            "Provide:\n"
            "## Comparison Analysis\n\n"
            "### Common Themes\n[What do these documents share?]\n\n"
            "### Key Differences\n[How do they differ?]\n\n"
            "### Relationships & Connections\n[How do they relate to each other?]\n\n"
            "### Synthesis\n[What do these documents together reveal that individual documents don't?]\n\n"
            "### Recommendations\n[Based on all documents, what are the key takeaways?]"
        )
        return response.content.strip()
    except Exception as e:
        logger.error("Cross-document comparison failed: %s", e)
        return f"Comparison failed: {e}"


def cross_qa(vectorstore: FAISS, question: str, api_key: str) -> tuple:
    """Answer a question using retrieval-augmented generation across the vector index.

    Returns a (answer_text, source_documents) tuple.
    """
    if not question.strip():
        return "Please enter a question.", []
    if vectorstore is None:
        return "No documents have been indexed yet. Upload at least one PDF first.", []
    llm = get_llm(api_key, 0)
    prompt_template = (
        "Answer the question using ONLY information from the provided document excerpts.\n"
        "Always cite which document each piece of information comes from.\n"
        "If the answer spans multiple documents, synthesise them clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (cite documents):"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        source_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return answer, source_docs
    except Exception as e:
        logger.error("Q&A failed: %s", e)
        return f"Could not generate an answer: {e}", []


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Document Intelligence Platform</h1>
    <p>Upload multiple documents — extract entities, generate summaries, compare insights, and ask questions across your entire library</p>
    <div>
        <span class="feature-badge">📋 Auto Summarisation</span>
        <span class="feature-badge">🏷️ Entity Extraction</span>
        <span class="feature-badge">🔄 Cross-Doc Comparison</span>
        <span class="feature-badge">💬 Unified Q&A</span>
        <span class="feature-badge">🔍 Source Attribution</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    # Allow the API key to be pre-loaded from the environment (useful for deployments)
    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )

    st.divider()
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files (text-based, not scanned images)",
    )

    if uploaded_files and api_key:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.documents]
        if new_files:
            progress = st.progress(0)
            for idx, f in enumerate(new_files):
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        docs = process_pdf(f)
                        full_text = " ".join([d.page_content for d in docs])
                        st.session_state.documents[f.name] = {
                            "docs": docs,
                            "text": full_text,
                            "pages": len(docs),
                            "added": datetime.now().strftime("%H:%M"),
                        }
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")
                progress.progress((idx + 1) / len(new_files))

            if st.session_state.documents:
                with st.spinner("Building unified search index..."):
                    try:
                        st.session_state.vectorstore = rebuild_vectorstore(st.session_state.documents)
                        progress.empty()
                        st.success(f"✅ {len(new_files)} document(s) ready!")
                    except Exception as e:
                        progress.empty()
                        st.error(f"Failed to build search index: {e}")
    elif uploaded_files and not api_key:
        st.warning("Enter your Groq API key above before uploading documents.")

    if st.session_state.documents:
        st.divider()
        st.header("📚 Document Library")
        for fname, data in st.session_state.documents.items():
            display_name = fname[:30] + ("..." if len(fname) > 30 else "")
            st.markdown(
                f'<div class="doc-card">'
                f'<div class="doc-icon">📄</div>'
                f'<div class="doc-info">'
                f'<div class="doc-name">{display_name}</div>'
                f'<div class="doc-meta">{data["pages"]} pages · Added {data["added"]}</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.documents = {}
            st.session_state.vectorstore = None
            st.session_state.summaries = {}
            st.session_state.entities = {}
            st.session_state.qa_history = []
            st.session_state.comparison = None
            st.rerun()

# ── Guard: require API key ────────────────────────────────────
if not api_key:
    st.info(
        "👈 Enter your Groq API key in the sidebar to get started. "
        "Free keys are available at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# ── Guard: require at least one document ─────────────────────
if not st.session_state.documents:
    st.markdown("""
<div class="empty-state">
    <div class="icon">📄</div>
    <h3>No documents uploaded yet</h3>
    <p>Upload one or more PDFs in the sidebar to begin</p>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── Dashboard stats ───────────────────────────────────────────
total_pages = sum(d["pages"] for d in st.session_state.documents.values())
total_entities = sum(len(e.get("key_topics", [])) for e in st.session_state.entities.values())
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, len(st.session_state.documents), "Documents"),
    (c2, total_pages, "Total Pages"),
    (c3, total_entities, "Topics Found"),
    (c4, len(st.session_state.qa_history), "Questions Asked"),
]:
    col.markdown(
        f'<div class="stat-card"><div class="stat-val">{val}</div>'
        f'<div class="stat-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Summaries", "🏷️ Entity Extraction", "🔄 Cross-Doc Analysis", "💬 Q&A"]
)

# ── Tab 1: Summaries ──────────────────────────────────────────
with tab1:
    st.subheader("AI-Generated Document Summaries")
    if st.button("📋 Generate All Summaries", type="primary"):
        for fname, data in st.session_state.documents.items():
            if fname not in st.session_state.summaries:
                with st.spinner(f"Summarising {fname}..."):
                    st.session_state.summaries[fname] = summarise(data["text"], fname, api_key)
    if st.session_state.summaries:
        for fname, summary in st.session_state.summaries.items():
            st.markdown(
                f'<div class="insight-card"><strong>📄 {fname}</strong><br><br>{summary}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Click 'Generate All Summaries' to analyse your documents.")

# ── Tab 2: Entity Extraction ──────────────────────────────────
with tab2:
    st.subheader("Automated Entity Extraction")
    if st.button("🏷️ Extract Entities from All Documents", type="primary"):
        for fname, data in st.session_state.documents.items():
            if fname not in st.session_state.entities:
                with st.spinner(f"Extracting from {fname}..."):
                    st.session_state.entities[fname] = extract_entities(data["text"], api_key)
    if st.session_state.entities:
        entity_labels = [
            ("🏢 Orgs", "organisations", "chip-org"),
            ("📅 Dates", "dates", "chip-date"),
            ("💰 Money", "monetary_values", "chip-money"),
            ("📌 Topics", "key_topics", "chip-topic"),
            ("📍 Locations", "locations", "chip-loc"),
        ]
        for fname, ents in st.session_state.entities.items():
            st.markdown(f"**📄 {fname}**")
            cols = st.columns(5)
            for col, (label, key, cls) in zip(cols, entity_labels):
                with col:
                    st.caption(label)
                    items = ents.get(key, [])
                    if items:
                        for item in items[:5]:
                            st.markdown(
                                f'<span class="entity-chip {cls}">{item}</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("—")
            st.divider()
    else:
        st.info("Click 'Extract Entities' to identify key information from your documents.")

# ── Tab 3: Cross-Document Analysis ───────────────────────────
with tab3:
    st.subheader("Cross-Document Comparison & Analysis")
    if len(st.session_state.documents) < 2:
        st.info("📌 Upload at least 2 documents to compare them.")
    else:
        if st.button("🔄 Compare All Documents", type="primary"):
            # Ensure summaries exist for all documents before comparing
            missing = [
                fname for fname in st.session_state.documents
                if fname not in st.session_state.summaries
            ]
            if missing:
                with st.spinner("Generating missing summaries first..."):
                    for fname in missing:
                        text = st.session_state.documents[fname]["text"]
                        st.session_state.summaries[fname] = summarise(text, fname, api_key)
            with st.spinner("Running cross-document analysis..."):
                st.session_state.comparison = compare_docs(st.session_state.summaries, api_key)
        if st.session_state.comparison:
            formatted = st.session_state.comparison.replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card">{formatted}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "💾 Download Analysis",
                st.session_state.comparison,
                file_name="comparison_analysis.md",
                mime="text/markdown",
            )

# ── Tab 4: Q&A ───────────────────────────────────────────────
with tab4:
    st.subheader("Ask Questions Across All Documents")
    st.caption(
        f"🔍 Searching across {len(st.session_state.documents)} document(s) "
        "simultaneously with source attribution"
    )

    for qa in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(qa["question"])
        with st.chat_message("assistant"):
            st.write(qa["answer"])
            if qa.get("sources"):
                source_files = list(
                    {s.metadata.get("source_file", "Unknown") for s in qa["sources"]}
                )
                chips = " ".join(
                    [f'<span class="source-chip">📄 {s}</span>' for s in source_files]
                )
                st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)

    question = st.chat_input("Ask anything about your documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching across all documents..."):
                answer, sources = cross_qa(st.session_state.vectorstore, question, api_key)
            st.write(answer)
            source_files = list(
                {s.metadata.get("source_file", "Unknown") for s in sources}
            )
            chips = " ".join(
                [f'<span class="source-chip">📄 {s}</span>' for s in source_files]
            )
            st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)
        st.session_state.qa_history.append(
            {"question": question, "answer": answer, "sources": sources}
        )

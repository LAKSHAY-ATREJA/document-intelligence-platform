import os
import streamlit as st
import tempfile
import json
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from datetime import datetime

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
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_llm(api_key, temp=0):
    return ChatGroq(model_name="llama3-8b-8192", temperature=temp, groq_api_key=api_key, max_tokens=1000)


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return docs


def rebuild_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for fname, data in documents.items():
        chunks = splitter.split_documents(data["docs"])
        for chunk in chunks:
            chunk.metadata["source_file"] = fname
        all_chunks.extend(chunks)
    embeddings = get_embeddings()
    return FAISS.from_documents(all_chunks, embeddings)


def summarise(text, fname, api_key):
    llm = get_llm(api_key, 0.2)
    response = llm.invoke(f"""Summarise this document in 4-5 sentences. Be specific and factual.

Document ({fname}): {text[:4000]}

Summary:""")
    return response.content


def extract_entities(text, api_key):
    llm = get_llm(api_key, 0)
    response = llm.invoke(f"""Extract key entities. Return ONLY valid JSON, no other text.

Document: {text[:3000]}

Return exactly:
{{"organisations": [], "dates": [], "monetary_values": [], "key_topics": [], "locations": []}}""")
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except:
        return {"organisations": [], "dates": [], "monetary_values": [], "key_topics": [], "locations": []}


def compare_docs(summaries, api_key):
    llm = get_llm(api_key, 0.3)
    docs_text = "\n\n".join([f"**{name}**: {summary}" for name, summary in summaries.items()])
    response = llm.invoke(f"""Compare these documents comprehensively.

{docs_text}

Provide:
## Comparison Analysis

### Common Themes
[What do these documents share?]

### Key Differences
[How do they differ?]

### Relationships & Connections
[How do they relate to each other?]

### Synthesis
[What do these documents together reveal that individual documents don't?]

### Recommendations
[Based on all documents, what are the key takeaways?]""")
    return response.content


def cross_qa(vectorstore, question, api_key):
    llm = get_llm(api_key, 0)
    prompt_template = """Answer the question using ONLY information from the provided document excerpts.
Always cite which document each piece of information comes from.
If the answer spans multiple documents, synthesise them clearly.

Context:
{context}

Question: {question}

Answer (cite documents):"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa.invoke({"query": question})
    return result["result"], result["source_documents"]


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
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Free at console.groq.com")

    st.divider()
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )

    if uploaded_files and api_key:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.documents]
        if new_files:
            progress = st.progress(0)
            for idx, f in enumerate(new_files):
                with st.spinner(f"Processing {f.name}..."):
                    docs = process_pdf(f)
                    full_text = " ".join([d.page_content for d in docs])
                    st.session_state.documents[f.name] = {
                        "docs": docs,
                        "text": full_text,
                        "pages": len(docs),
                        "added": datetime.now().strftime("%H:%M")
                    }
                progress.progress((idx + 1) / len(new_files))
            with st.spinner("Building unified search index..."):
                st.session_state.vectorstore = rebuild_vectorstore(st.session_state.documents)
            progress.empty()
            st.success(f"✅ {len(new_files)} document(s) ready!")

    if st.session_state.documents:
        st.divider()
        st.header("📚 Document Library")
        for fname, data in st.session_state.documents.items():
            st.markdown(f"""
<div class="doc-card">
    <div class="doc-icon">📄</div>
    <div class="doc-info">
        <div class="doc-name">{fname[:30]}{'...' if len(fname) > 30 else ''}</div>
        <div class="doc-meta">{data['pages']} pages · Added {data['added']}</div>
    </div>
</div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear All", use_container_width=True):
            for key in ["documents", "vectorstore", "summaries", "entities", "qa_history", "comparison"]:
                st.session_state[key] = {} if key != "vectorstore" and key != "comparison" else None
                if key == "qa_history":
                    st.session_state[key] = []
            st.rerun()

# ── Main ──────────────────────────────────────────────────────
if not api_key:
    st.info("👈 Enter your Groq API key to get started. Free at [console.groq.com](https://console.groq.com)")
    st.stop()

if not st.session_state.documents:
    st.markdown("""
<div class="empty-state">
    <div class="icon">📄</div>
    <h3>No documents uploaded yet</h3>
    <p>Upload one or more PDFs in the sidebar to begin</p>
</div>""", unsafe_allow_html=True)
    st.stop()

# Stats
total_pages = sum(d["pages"] for d in st.session_state.documents.values())
total_entities = sum(len(e.get("key_topics", [])) for e in st.session_state.entities.values())
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state.documents)}</div><div class="stat-lbl">Documents</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card"><div class="stat-val">{total_pages}</div><div class="stat-lbl">Total Pages</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card"><div class="stat-val">{total_entities}</div><div class="stat-lbl">Topics Found</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state.qa_history)}</div><div class="stat-lbl">Questions Asked</div></div>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Summaries", "🏷️ Entity Extraction", "🔄 Cross-Doc Analysis", "💬 Q&A"])

with tab1:
    st.subheader("AI-Generated Document Summaries")
    if st.button("📋 Generate All Summaries", type="primary"):
        for fname, data in st.session_state.documents.items():
            if fname not in st.session_state.summaries:
                with st.spinner(f"Summarising {fname}..."):
                    st.session_state.summaries[fname] = summarise(data["text"], fname, api_key)
    if st.session_state.summaries:
        for fname, summary in st.session_state.summaries.items():
            st.markdown(f"""
<div class="insight-card">
    <strong>📄 {fname}</strong><br><br>
    {summary}
</div>""", unsafe_allow_html=True)
    else:
        st.info("Click 'Generate All Summaries' to analyse your documents.")

with tab2:
    st.subheader("Automated Entity Extraction")
    if st.button("🏷️ Extract Entities from All Documents", type="primary"):
        for fname, data in st.session_state.documents.items():
            if fname not in st.session_state.entities:
                with st.spinner(f"Extracting from {fname}..."):
                    st.session_state.entities[fname] = extract_entities(data["text"], api_key)
    if st.session_state.entities:
        for fname, ents in st.session_state.entities.items():
            st.markdown(f"**📄 {fname}**")
            cols = st.columns(5)
            labels = [("🏢 Orgs", "organisations", "chip-org"), ("📅 Dates", "dates", "chip-date"),
                      ("💰 Money", "monetary_values", "chip-money"), ("📌 Topics", "key_topics", "chip-topic"),
                      ("📍 Locations", "locations", "chip-loc")]
            for col, (label, key, cls) in zip(cols, labels):
                with col:
                    st.caption(label)
                    items = ents.get(key, [])
                    if items:
                        for item in items[:5]:
                            st.markdown(f'<span class="entity-chip {cls}">{item}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("—")
            st.divider()
    else:
        st.info("Click 'Extract Entities' to identify key information from your documents.")

with tab3:
    st.subheader("Cross-Document Comparison & Analysis")
    if len(st.session_state.documents) < 2:
        st.info("📌 Upload at least 2 documents to compare them.")
    else:
        if st.button("🔄 Compare All Documents", type="primary"):
            if len(st.session_state.summaries) < len(st.session_state.documents):
                with st.spinner("Generating summaries first..."):
                    for fname, data in st.session_state.documents.items():
                        if fname not in st.session_state.summaries:
                            st.session_state.summaries[fname] = summarise(data["text"], fname, api_key)
            with st.spinner("Running cross-document analysis..."):
                st.session_state.comparison = compare_docs(st.session_state.summaries, api_key)
        if st.session_state.comparison:
            st.markdown(f'<div class="answer-card">{st.session_state.comparison.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)
            st.download_button("💾 Download Analysis", st.session_state.comparison,
                               file_name="comparison_analysis.md", mime="text/markdown")

with tab4:
    st.subheader("Ask Questions Across All Documents")
    st.caption(f"🔍 Searching across {len(st.session_state.documents)} document(s) simultaneously with source attribution")

    for qa in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(qa["question"])
        with st.chat_message("assistant"):
            st.write(qa["answer"])
            if qa.get("sources"):
                sources = list(set([s.metadata.get("source_file", "Unknown") for s in qa["sources"]]))
                st.markdown("**Sources:** " + " ".join([f'<span class="source-chip">📄 {s}</span>' for s in sources]),
                            unsafe_allow_html=True)

    question = st.chat_input("Ask anything about your documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching across all documents..."):
                answer, sources = cross_qa(st.session_state.vectorstore, question, api_key)
            st.write(answer)
            source_files = list(set([s.metadata.get("source_file", "Unknown") for s in sources]))
            st.markdown("**Sources:** " + " ".join([f'<span class="source-chip">📄 {s}</span>' for s in source_files]),
                        unsafe_allow_html=True)
        st.session_state.qa_history.append({"question": question, "answer": answer, "sources": sources})

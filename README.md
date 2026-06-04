# 🧠 Document Intelligence Platform

> Upload multiple documents — extract entities, generate summaries, compare insights, and ask questions across your entire document library.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

## 🚀 Live Demo

**[👉 Try it live here](https://your-app.streamlit.app)** ← Update after deployment

---

## 🏗️ Architecture

```
Multiple PDFs
     │
     ▼
PyPDF Parser (per document)
     │
     ▼
RecursiveCharacterTextSplitter
(chunk_size=500, overlap=50)
     │
     ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2)
     │
     ▼
FAISS Unified Vector Index
(all documents in one index)
     │
     ├──► Summarisation Pipeline  → AI summaries per document
     ├──► Entity Extraction       → Orgs, dates, money, topics, locations
     ├──► Cross-Doc Comparison    → Themes, differences, relationships
     └──► Unified Q&A             → Source-attributed answers across all docs
```

## ✨ Features

| Feature | Description |
|---|---|
| 📋 Auto Summarisation | AI-generated 4-5 sentence summaries per document |
| 🏷️ Entity Extraction | Extracts organisations, dates, monetary values, topics, locations |
| 🔄 Cross-Doc Comparison | Identifies themes, differences, and relationships across documents |
| 💬 Unified Q&A | Ask questions across ALL documents simultaneously |
| 🔍 Source Attribution | Every answer shows which document it came from |
| 📊 Live Metrics | Document count, pages, entities, questions tracked |

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (Llama3-8b-8192) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (unified cross-document index) |
| PDF Parsing | PyPDF |
| Orchestration | LangChain RetrievalQA |

## ⚡ Quick Start

```bash
git clone https://github.com/LAKSHAY-ATREJA/document-intelligence-platform
cd document-intelligence-platform

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

## 🌐 Deploy Free on Streamlit Cloud

1. Fork this repo
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select this repo → deploy

## 💡 Use Cases

- Legal contract review and cross-referencing
- Research paper analysis and comparison
- Business report intelligence extraction
- Policy document processing
- Due diligence document review

---

Built by [Lakshay Atreja](https://linkedin.com/in/lakshay-atreja) | [GitHub](https://github.com/LAKSHAY-ATREJA)

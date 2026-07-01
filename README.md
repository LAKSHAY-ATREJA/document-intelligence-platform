# Document Intelligence Platform

A multi-document analysis application that allows you to upload several PDF files at once and extract structured knowledge from them collectively. The platform generates per-document summaries, identifies named entities, compares documents against each other, and answers natural-language questions with precise source attribution — all through a web interface that runs locally or on a free cloud host.

---

## What it does

Most document analysis tools process one file at a time. This platform treats an entire collection of PDFs as a unified knowledge base. Once you upload your files, every feature operates across all of them simultaneously:

- Summarisation produces a 4-5 sentence factual summary for each document independently.
- Entity extraction identifies organisations, dates, monetary figures, key topics, and locations within each document.
- Cross-document comparison synthesises common themes, key differences, relationships between documents, and strategic takeaways.
- Unified Q&A lets you ask free-form questions and receive answers drawn from whichever documents contain the relevant information, with the source document cited for every claim.

---

## Architecture

PDF files are parsed page by page using PyPDF. Each page is split into overlapping 500-character chunks using LangChain's RecursiveCharacterTextSplitter. All chunks from all documents are embedded with the HuggingFace `all-MiniLM-L6-v2` sentence-transformer model and indexed together in a single FAISS vector store. Every query to the Q&A system retrieves the five most semantically similar chunks from that shared index, regardless of which file they came from.

Summarisation, entity extraction, comparison, and question answering all use the Groq-hosted Llama3-8b-8192 model via the LangChain Groq integration. Groq provides a free API tier with generous rate limits, which means the entire platform runs at zero cost.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM inference | Groq API (Llama3-8b-8192) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector store | FAISS (CPU) |
| PDF parsing | PyPDF |
| Orchestration | LangChain RetrievalQA |

---

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/LAKSHAY-ATREJA/document-intelligence-platform
cd document-intelligence-platform

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first time you run the application, the sentence-transformer model (~90 MB) is downloaded automatically from HuggingFace and cached locally.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key used for all LLM calls |

If the `GROQ_API_KEY` variable is set in the shell environment (or in a `.env` file loaded before startup), the sidebar API key field is pre-filled automatically. Otherwise, the key can be entered directly in the application UI.

To obtain a free Groq API key, create an account at https://console.groq.com, navigate to API Keys, and generate a new key. The free tier allows several hundred requests per day at no cost.

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key_here
```

---

## Running locally

```bash
streamlit run app.py
```

Streamlit opens a browser tab at `http://localhost:8501`. Enter your Groq API key in the sidebar, upload one or more PDF files, and use the four tabs — Summaries, Entity Extraction, Cross-Doc Analysis, and Q&A — to analyse your documents.

---

## Running the demo script

`demo.py` runs the three core analytical functions (summarisation, entity extraction, cross-document comparison) against two short sample texts that are embedded directly in the script. No PDF upload is needed. Use this to confirm that your installation and API key are working before uploading real documents.

```bash
python3 demo.py
```

If `GROQ_API_KEY` is not set in your environment, the script will prompt you to enter it interactively.

Sample output excerpt:

```
======================================================================
  STEP 1: Document Summarisation
======================================================================

  Summarising: quarterly_report_q1.txt ...

  Summary:
  Acme Corporation reported strong Q1 2024 results with total revenue of
  $4.8 billion, a 12% year-over-year increase. The Cloud Services division
  led growth at 34%, contributing $1.9 billion. The company completed the
  acquisition of DataSync Inc. for $340 million and reaffirmed full-year
  revenue guidance of $19.5–20.0 billion.

======================================================================
  STEP 3: Cross-Document Comparison
======================================================================

  ## Comparison Analysis

  ### Common Themes
  Both documents focus on cloud infrastructure growth ...
```

---

## Deployment

### Streamlit Community Cloud (recommended, free)

1. Fork this repository to your GitHub account.
2. Visit https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select your fork, set the main file to `app.py`, and click "Deploy".
4. In the app's Settings under "Secrets", add `GROQ_API_KEY = "gsk_your_key_here"`.

Live demo: coming soon

### Render (free tier)

The repository includes `render.yaml` with the correct build and start commands. To deploy:

1. Create a free account at https://render.com.
2. Click "New Web Service" and connect your GitHub repository.
3. Render detects `render.yaml` automatically. Click "Create Web Service".
4. Under Environment Variables, add `GROQ_API_KEY` with your key value.

The free Render tier spins down after 15 minutes of inactivity; the first request after that may take 30-60 seconds to respond.

### Any platform with Docker or a Procfile

The `Procfile` contains:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

This is compatible with Heroku, Railway, Fly.io, and any platform that respects the `$PORT` convention.

---

## Use cases

- Legal and compliance teams reviewing contracts, policies, or regulations across multiple versions or jurisdictions.
- Researchers comparing papers, reports, or datasets from different sources.
- Business analysts extracting figures and entities from earnings reports or market research.
- Due diligence workflows where multiple vendor documents need to be compared quickly.
- Internal knowledge management when important information is spread across many uploaded PDFs.

---

## Notes on PDF compatibility

The parser works on text-based PDFs where text is stored as selectable characters. Scanned PDFs (images of pages) produce no extractable text and will result in empty summaries. If you need to process scanned documents, run them through an OCR tool such as `ocrmypdf` before uploading.

---

Built by Lakshay Atreja — https://github.com/LAKSHAY-ATREJA


"""
demo.py — Document Intelligence Platform demonstration script.

This script exercises the core analytical functions (summarisation, entity
extraction, and cross-document comparison) against two small in-memory sample
documents without requiring any PDF uploads.  It is intended as a quick sanity
check that the environment is configured correctly and that a valid Groq API key
is present.

Usage:
    python3 demo.py

The GROQ_API_KEY environment variable must be set, or you will be prompted to
enter the key interactively.
"""

import os
import sys
import json

# ---------------------------------------------------------------------------
# Sample documents (embedded inline — no file I/O needed)
# ---------------------------------------------------------------------------

SAMPLE_DOCS = {
    "quarterly_report_q1.txt": """
    Acme Corporation — First Quarter Financial Report (Q1 2024)
    Filed: April 15, 2024  |  New York, United States

    Acme Corporation reported total revenue of $4.8 billion for Q1 2024,
    representing a 12% year-over-year increase driven primarily by strong
    performance in its Cloud Services division ($1.9 billion, up 34%) and
    Hardware segment ($1.6 billion, up 5%).  Net income reached $620 million,
    yielding a net margin of 12.9%, compared with 11.4% in Q1 2023.

    The company completed the acquisition of DataSync Inc. on February 3, 2024
    for $340 million in cash.  Integration costs of $22 million were expensed
    in the quarter.  DataSync contributed $47 million in revenue over the
    partial period.

    Headcount grew by 1,200 employees to 38,400 globally.  The company opened
    new offices in Singapore and Dublin to support international expansion.
    Capital expenditure totalled $310 million, with $180 million allocated to
    data-centre infrastructure in North Virginia.

    Full-year 2024 guidance is reaffirmed: revenue of $19.5–20.0 billion and
    adjusted EBITDA margin of 22–24%.
    """,

    "industry_trends_2024.txt": """
    Global Cloud Computing Market — Industry Outlook 2024
    Published by: TechResearch Group  |  March 2024  |  San Francisco, CA

    The worldwide cloud computing market is forecast to reach $1.1 trillion in
    2024, growing at a compound annual rate of 18% through 2027.  Enterprise
    adoption continues to accelerate; 78% of Fortune 500 companies now operate
    multi-cloud strategies, up from 61% in 2022.

    Infrastructure-as-a-Service (IaaS) remains the largest segment at $410 billion,
    followed by Software-as-a-Service (SaaS) at $380 billion and
    Platform-as-a-Service (PaaS) at $210 billion.

    Key trends shaping the landscape include generative AI workload migration
    (cited by 64% of CIOs as a primary driver of new cloud spend), edge
    computing adoption in manufacturing and logistics, and heightened regulatory
    scrutiny of data residency in the European Union and Singapore.

    Labour market data show cloud architect and DevOps engineer roles command
    median salaries of $175,000 and $155,000 respectively in the United States.
    Hiring demand for these profiles grew 23% in Q1 2024 relative to Q1 2023.
    """,
}

# ---------------------------------------------------------------------------
# Helper: lazy import of application functions
# ---------------------------------------------------------------------------

def _import_app_functions():
    """Import analytical functions from app.py located in the same directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from app import summarise, extract_entities, compare_docs
        return summarise, extract_entities, compare_docs
    except ImportError as exc:
        print(f"[ERROR] Could not import from app.py: {exc}")
        print("Ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


def _get_api_key() -> str:
    """Return the Groq API key from the environment or prompt the user."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    print("\nGROQ_API_KEY environment variable is not set.")
    key = input("Enter your Groq API key (gsk_...): ").strip()
    if not key:
        print("[ERROR] No API key provided. Exiting.")
        sys.exit(1)
    return key


def _section(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_demo(api_key: str) -> None:
    """Execute the full demonstration pipeline."""
    summarise, extract_entities, compare_docs = _import_app_functions()

    _section("DOCUMENT INTELLIGENCE PLATFORM — DEMO")
    print(f"Running against {len(SAMPLE_DOCS)} sample documents.")
    print("No PDF upload required; text is embedded directly in this script.\n")

    # ------------------------------------------------------------------ #
    # Step 1 — Summarisation                                               #
    # ------------------------------------------------------------------ #
    _section("STEP 1: Document Summarisation")
    summaries = {}
    for fname, text in SAMPLE_DOCS.items():
        print(f"\n  Summarising: {fname} ...")
        summary = summarise(text, fname, api_key)
        summaries[fname] = summary
        print(f"\n  Summary:\n  {summary}\n")

    # ------------------------------------------------------------------ #
    # Step 2 — Entity Extraction                                           #
    # ------------------------------------------------------------------ #
    _section("STEP 2: Entity Extraction")
    for fname, text in SAMPLE_DOCS.items():
        print(f"\n  Extracting entities from: {fname} ...")
        entities = extract_entities(text, api_key)
        print(f"\n  Entities (JSON):\n  {json.dumps(entities, indent=4)}\n")

    # ------------------------------------------------------------------ #
    # Step 3 — Cross-Document Comparison                                   #
    # ------------------------------------------------------------------ #
    _section("STEP 3: Cross-Document Comparison")
    print("\n  Comparing documents against each other ...")
    comparison = compare_docs(summaries, api_key)
    print(f"\n  Comparison Analysis:\n\n{comparison}\n")

    # ------------------------------------------------------------------ #
    # Done                                                                  #
    # ------------------------------------------------------------------ #
    _section("DEMO COMPLETE")
    print(
        "All three analytical functions executed successfully.\n"
        "To use the full interactive application, run:\n\n"
        "    streamlit run app.py\n"
    )


if __name__ == "__main__":
    api_key = _get_api_key()
    run_demo(api_key)

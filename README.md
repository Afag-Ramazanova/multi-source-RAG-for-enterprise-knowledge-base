# Multi-Source RAG for Enterprise Knowledge Base Q&A

This is a full-stack portfolio project that demonstrates how to build a practical Retrieval-Augmented Generation (RAG) system for enterprise-style knowledge workflows.

The app ingests heterogeneous sources (PDFs, website content, and CSV files), retrieves evidence using multiple complementary retrievers, re-ranks candidates with a cross-encoder, and generates grounded answers with citations. The goal is to show not just that the system can answer simple factual questions, but also that it can:

- retrieve precisely for narrow fact lookups,
- recall broader context across sections,
- combine evidence across documents for mixed-domain questions,
- stay faithful when information is missing.

In short, this project is designed to mirror real-world knowledge-assistant requirements in regulated and document-heavy domains like banking and compliance.

## Architecture

```text
                    +------------------------+
                    |      Streamlit UI      |
                    | - Upload PDFs/CSVs/URL |
                    | - Ask questions         |
                    +-----------+------------+
                                |
                                v
                    +------------------------+
                    |      FastAPI API       |
                    |  /ingest  /query       |
                    +-----------+------------+
                                |
           +--------------------+--------------------+
           |                                         |
           v                                         v
 +------------------------+               +--------------------------+
 | Ingestion Orchestrator |               | Retrieval Orchestrator   |
 | - PDF (PyMuPDF)        |               | - Vector (FAISS)         |
 | - URL (requests+BS4)   |               | - Sentence Window        |
 | - CSV (pandas rows)    |               | - BM25 Sparse Retriever  |
 +-----------+------------+               | - RRF Fusion             |
             |                            | - Cross-Encoder Re-rank  |
             v                            +------------+-------------+
 +-----------------------------+                        |
 | LlamaIndex Nodes + Metadata |                        v
 +-----------------------------+            +-------------------------+
             |                               | Context Assembly + LLM  |
             v                               | (gpt-5.4-mini)          |
 +-----------------------------+             +------------+------------+
 | FAISS Local Index           |                          |
 | persisted at data/index_vector/ + data/index_sentence/ |                          v
 +-----------------------------+               +----------------------+
                                               | Answer + Citations   |
                                               +----------------------+
```

## Project Structure

```text
multi_source_rag/
├── backend/
│   ├── main.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── generator.py
├── frontend/
│   └── app.py
├── data/
│   ├── index_vector/
│   ├── index_sentence/
│   └── sample/
│       ├── employee_handbook.csv
│       └── security_policy_sample.pdf
├── requirements.txt
└── .env.example
```

## Retrieval Strategy (Why Multi-Retriever)

- `vector`: semantic similarity over dense embeddings in FAISS; best for meaning-based matches.
- `sentence_window`: retrieves sentence-level hits and expands to surrounding context (`±2`) to preserve local meaning.
- `bm25`: sparse lexical retrieval; strong for exact terms, acronyms, and policy IDs.
- `RRF fusion`: combines rankings from all three, reducing single-retriever blind spots.
- `cross-encoder re-rank`: scores query-document pairs directly to produce a higher precision final set of citations.

Using multiple retrievers improves recall across different query types (semantic, exact keyword, and local-context heavy), while re-ranking improves precision.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure environment variables:

```bash
cp .env.example .env
```

1. Add your LLM API key in `.env`.

## Run Locally

Start backend:

```bash
uvicorn multi_source_rag.backend.main:app --reload --port 8000
```

Start frontend (new terminal):

```bash
streamlit run multi_source_rag/frontend/app.py
```

## API Endpoints

- `GET /health`: health check.
- `POST /ingest`: upload `pdf_files`, `csv_files`, optional `url` to rebuild knowledge base.
- `POST /query`: JSON body `{"question": "..."}`; returns answer, latency, and citations.

## Citation Payload

Each citation includes:

- `source_name`
- `text_excerpt` (first 200 chars)
- `score` (cross-encoder score)
- `retriever_type` (`vector`, `sentence_window`, or `bm25`)

## Evaluation Guide (Test Prompts)

Use the sample PDFs in `files_to_test/` and run the following prompts directly in the app.

These prompts are intentionally grouped by capability so you can evaluate different parts of the pipeline (precision, recall, multi-source fusion, and faithfulness).

### Single-source fact retrieval 

These should be answered with a clear fact from one source and a tight citation.

- "What is the dollar threshold for filing a Currency Transaction Report?"
- "What AUC-ROC score did the CreditRisk model achieve on the test set?"
- "How long must FinTrust Bank retain copies of filed SARs?"

### Multi-section synthesis 

These require finding multiple relevant sections within a document (or across nearby chunks) and summarizing them.

- "What are all the retrieval strategies described in the RAG guide and how do they differ?"
- "What are the physical and transition risks described in the climate report?"

### Cross-document reasoning (tests multi-source fusion)

These test whether retrieval and generation can combine evidence from different PDFs.

- "How does the credit risk model relate to AML compliance requirements at FinTrust Bank?"
- "What does the RAG guide say about re-ranking, and how could that apply to reviewing SAR filings?"

### Ambiguous / stress queries (tests faithfulness)

These test whether the assistant avoids hallucinations and stays grounded in provided evidence.

- "What is the penalty for structuring transactions?" (should stay evidence-based and avoid inventing details)
- "What is the green bond market size in 2030?" (should respond: "I don't have enough information to answer this.")

## Why These Questions

- **Precision prompts** validate dense/sparse retrieval quality for exact facts.
- **Recall prompts** validate chunk coverage and sentence-window usefulness.
- **Fusion prompts** validate RRF + re-ranking + cross-source synthesis behavior.
- **Faithfulness prompts** validate safe fallback behavior when evidence is partial or absent.

If the system performs well across all four categories, it is a strong signal that the RAG stack is robust enough for realistic enterprise knowledge tasks.

## Notes

- FAISS index is local-only and rebuilt on each `/ingest`.
- No cloud vector DB or LangChain is used.


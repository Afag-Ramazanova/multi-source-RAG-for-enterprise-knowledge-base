from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .generator import AnswerGenerator
from .ingestion import build_ingestion_bundle
from .retrieval import RetrievalOrchestrator


app = FastAPI(title="Multi-Source Enterprise RAG API")
retrieval = RetrievalOrchestrator()
generator = AnswerGenerator()

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    """Parse multipart manually so every repeated ``pdf_files`` / ``csv_files`` part is kept.

    ``UploadFile | list[UploadFile]`` in FastAPI can bind only a single file for duplicate
    field names, which made multi-upload look like a one-document ingest.
    """
    form = await request.form()
    pdf_payloads: list[tuple[str, bytes]] = []
    csv_payloads: list[tuple[str, bytes]] = []
    empty_pdf_names: list[str] = []
    empty_csv_names: list[str] = []

    for upload in form.getlist("pdf_files"):
        if not hasattr(upload, "read"):
            continue
        raw = await upload.read()
        name = getattr(upload, "filename", None) or "uploaded.pdf"
        if not raw:
            empty_pdf_names.append(name)
            continue
        pdf_payloads.append((name, raw))

    for upload in form.getlist("csv_files"):
        if not hasattr(upload, "read"):
            continue
        raw = await upload.read()
        name = getattr(upload, "filename", None) or "uploaded.csv"
        if not raw:
            empty_csv_names.append(name)
            continue
        csv_payloads.append((name, raw))

    url_field = form.get("url")
    url = (
        url_field.strip()
        if isinstance(url_field, str) and url_field.strip()
        else None
    )

    if not pdf_payloads and not csv_payloads and not url:
        raise HTTPException(status_code=400, detail="Provide at least one PDF, CSV, or URL.")

    try:
        bundle = build_ingestion_bundle(pdf_files=pdf_payloads, csv_files=csv_payloads, url=url)
        retrieval.build_indexes(
            vector_nodes=bundle.vector_nodes, sentence_window_nodes=bundle.sentence_window_nodes
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    sources_ingested = sorted(
        {n.metadata.get("source_name", "") for n in bundle.vector_nodes if n.metadata.get("source_name")}
    )
    return {
        "status": "ingested",
        "vector_nodes": len(bundle.vector_nodes),
        "sentence_window_nodes": len(bundle.sentence_window_nodes),
        "sources_ingested": sources_ingested,
        "pdf_files_received": len(pdf_payloads),
        "csv_files_received": len(csv_payloads),
        "empty_pdf_uploads": empty_pdf_names,
        "empty_csv_uploads": empty_csv_names,
    }


@app.post("/query")
async def query(payload: dict[str, str]) -> dict[str, Any]:
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    try:
        citations, elapsed = retrieval.query(question)
        answer = generator.generate(question, citations)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return {
        "question": question,
        "answer": answer,
        "latency_seconds": round(elapsed, 3),
        "citations": [
            {
                "source_name": c.source_name,
                "text_excerpt": c.text_excerpt,
                "score": c.score,
                "retriever_type": c.retriever_type,
            }
            for c in citations
        ],
    }

from __future__ import annotations

import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"


def call_ingest(pdf_files, csv_files, url: str):
    files = []
    for pdf in pdf_files:
        files.append(("pdf_files", (pdf.name, pdf.getvalue(), "application/pdf")))
    for csv in csv_files:
        files.append(("csv_files", (csv.name, csv.getvalue(), "text/csv")))

    data = {"url": url} if url else {}
    return requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, timeout=300)


def call_query(question: str):
    return requests.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=120)


def render_citations(citations: list[dict]):
    for i, citation in enumerate(citations, start=1):
        title = f"Citation {i} - {citation['source_name']} ({citation['retriever_type']})"
        with st.expander(title, expanded=False):
            st.write(f"**Score:** {citation['score']:.4f}")
            st.write(citation["text_excerpt"])


def main() -> None:
    st.set_page_config(page_title="Enterprise Multi-Source RAG", layout="wide")
    st.title("Enterprise Knowledge Base Q&A")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Knowledge Base Builder")
        pdf_files = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        csv_files = st.file_uploader(
            "Upload CSV files", type=["csv"], accept_multiple_files=True
        )
        url = st.text_input("Website URL")

        if st.button("Build Knowledge Base"):
            with st.spinner("Ingesting sources and rebuilding index..."):
                response = call_ingest(pdf_files or [], csv_files or [], url.strip())
                if response.ok:
                    payload = response.json()
                    names = payload.get("sources_ingested") or []
                    n_selected_pdf = len(pdf_files or [])
                    n_recv_pdf = int(payload.get("pdf_files_received") or 0)
                    st.success(
                        f"Ingested {payload['vector_nodes']} vector chunks from "
                        f"{len(names)} source(s): {', '.join(names)}. "
                        f"Sentence-window nodes: {payload['sentence_window_nodes']}. "
                        f"(PDFs received by API: {n_recv_pdf})"
                    )
                    if n_selected_pdf and n_recv_pdf != n_selected_pdf:
                        st.warning(
                            f"You selected {n_selected_pdf} PDF(s) but the API ingested "
                            f"{n_recv_pdf}. Re-select files and build again, or check the backend logs."
                        )
                    empty_pdfs = payload.get("empty_pdf_uploads") or []
                    if empty_pdfs:
                        st.warning(f"These PDF uploads had no bytes (skipped): {empty_pdfs}")
                else:
                    st.error(response.text)

    st.subheader("Chat")
    question = st.chat_input("Ask a question about your enterprise knowledge base...")
    if question:
        with st.spinner("Retrieving and generating answer..."):
            response = call_query(question)
            if response.ok:
                payload = response.json()
                st.session_state.chat_history.append(payload)
            else:
                st.error(response.text)

    for item in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            st.caption(f"Retrieved in {item['latency_seconds']:.2f} seconds")
            render_citations(item.get("citations", []))


if __name__ == "__main__":
    main()

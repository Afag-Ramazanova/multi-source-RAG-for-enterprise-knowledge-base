from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import io
import uuid

import fitz  # PyMuPDF
import pandas as pd
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser, TokenTextSplitter


WINDOW_NODE_PARSER = SentenceWindowNodeParser.from_defaults(
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
VECTOR_TEXT_SPLITTER = TokenTextSplitter(chunk_size=512, chunk_overlap=50)


@dataclass
class IngestionBundle:
    vector_nodes: list
    sentence_window_nodes: list


def _extract_pdf_text(file_bytes: bytes) -> str:
    text_parts: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
        for page in pdf_doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _scrape_url_text(url: str, timeout: int = 20) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body = soup.find("main") or soup.find("article") or soup.body or soup
    text = body.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _chunk_url_document(url: str, text: str) -> list[Document]:
    chunks = VECTOR_TEXT_SPLITTER.split_text(text)
    return [
        Document(
            text=chunk,
            id_=str(uuid.uuid4()),
            metadata={
                "source_name": url,
                "source_type": "url",
                "retriever_types": ["vector", "bm25"],
            },
        )
        for chunk in chunks
        if chunk.strip()
    ]


def _csv_rows_to_documents(filename: str, file_bytes: bytes) -> list[Document]:
    frame = pd.read_csv(io.BytesIO(file_bytes))
    docs: list[Document] = []
    for idx, row in frame.iterrows():
        pieces = [f"{column}: {row[column]}" for column in frame.columns]
        text = ", ".join(pieces)
        docs.append(
            Document(
                text=text,
                id_=str(uuid.uuid4()),
                metadata={
                    "source_name": filename,
                    "source_type": "csv",
                    "row_index": int(idx),
                    "retriever_types": ["vector", "bm25"],
                },
            )
        )
    return docs


def _pdf_to_documents(filename: str, file_bytes: bytes) -> list[Document]:
    text = _extract_pdf_text(file_bytes)
    if not text:
        return []
    return [
        Document(
            text=text,
            id_=str(uuid.uuid4()),
            metadata={
                "source_name": filename,
                "source_type": "pdf",
                "retriever_types": ["vector", "bm25"],
            },
        )
    ]


def build_ingestion_bundle(
    pdf_files: Sequence[tuple[str, bytes]],
    csv_files: Sequence[tuple[str, bytes]],
    url: str | None = None,
) -> IngestionBundle:
    source_documents: list[Document] = []

    for filename, file_bytes in pdf_files:
        source_documents.extend(_pdf_to_documents(filename, file_bytes))

    for filename, file_bytes in csv_files:
        source_documents.extend(_csv_rows_to_documents(filename, file_bytes))

    if url:
        scraped_text = _scrape_url_text(url)
        source_documents.extend(_chunk_url_document(url, scraped_text))

    vector_nodes = VECTOR_TEXT_SPLITTER.get_nodes_from_documents(source_documents)
    for node in vector_nodes:
        node.metadata["retriever_types"] = ["vector", "bm25"]

    sentence_window_nodes = WINDOW_NODE_PARSER.get_nodes_from_documents(source_documents)
    for node in sentence_window_nodes:
        node.metadata["retriever_types"] = ["sentence_window"]

    return IngestionBundle(vector_nodes=vector_nodes, sentence_window_nodes=sentence_window_nodes)

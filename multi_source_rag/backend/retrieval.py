from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore
from sentence_transformers import CrossEncoder


# Paths are anchored to this package so index read/write does not depend on process cwd.
_BACKEND_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _BACKEND_DIR.parent
VECTOR_INDEX_DIR = _PACKAGE_ROOT / "data" / "index_vector"
SENTENCE_INDEX_DIR = _PACKAGE_ROOT / "data" / "index_sentence"
_LEGACY_INDEX_DIR = _PACKAGE_ROOT / "data" / "index"

RRF_K = 60
TOP_K_PER_RETRIEVER = 12
TOP_K_FUSED = 24
TOP_K_RERANKED = 7
MAX_PER_SOURCE = 2
# Cross-encoder logits (ms-marco): keep chunks within this margin of the best head candidate.
RERANK_SCORE_MARGIN = 6.5
RERANK_HEAD = 16


@dataclass
class RetrievedCitation:
    source_name: str
    text_excerpt: str
    score: float
    retriever_type: str
    full_text: str


def _clear_index_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _is_bridge_question(question: str) -> bool:
    lowered = question.lower()
    bridge_cues = ("relate", "relationship", "compare", "impact", "versus", "trade-off")
    return any(cue in lowered for cue in bridge_cues) or (" and " in lowered and "how" in lowered)


def _select_diverse_citations(reranked: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """Prefer one strong hit per document, then fill by score with per-source caps and a relevance band."""
    if not reranked:
        return []

    head = reranked[:RERANK_HEAD]
    best_score = max(item.get("rerank_score", float("-inf")) for item in head)
    score_floor = best_score - RERANK_SCORE_MARGIN

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_texts: set[str] = set()
    per_source: Counter[str] = Counter()
    target_unique_sources = 2 if _is_bridge_question(question) else 1

    def eligible(item: dict[str, Any]) -> bool:
        node = item["node"]
        nid = node.node_id
        if nid in selected_ids:
            return False
        content = node.get_content().strip()
        if not content or content in used_texts:
            return False
        src = node.metadata.get("source_name", "unknown")
        if per_source[src] >= MAX_PER_SOURCE:
            return False
        score = item.get("rerank_score", float("-inf"))
        if score < score_floor:
            return False
        return True

    def push(item: dict[str, Any]) -> None:
        node = item["node"]
        selected.append(item)
        selected_ids.add(node.node_id)
        used_texts.add(node.get_content().strip())
        per_source[node.metadata.get("source_name", "unknown")] += 1

    # Round 1: at most one citation per source (best global rank first).
    seen_source_round1: set[str] = set()
    for item in head:
        if len(selected) >= TOP_K_RERANKED:
            break
        node = item["node"]
        src = node.metadata.get("source_name", "unknown")
        if src in seen_source_round1:
            continue
        if not eligible(item):
            continue
        push(item)
        seen_source_round1.add(src)

    # Round 2: fill remaining slots by rerank order.
    for item in reranked:
        if len(selected) >= TOP_K_RERANKED:
            break
        if not eligible(item):
            continue
        push(item)

    # If this looks like a bridge question, force at least two sources when available.
    if _is_bridge_question(question) and len(per_source) < target_unique_sources:
        for item in reranked:
            if len(selected) >= TOP_K_RERANKED:
                break
            node = item["node"]
            src = node.metadata.get("source_name", "unknown")
            if src in per_source:
                continue
            if not eligible(item):
                continue
            push(item)
            if len(per_source) >= target_unique_sources:
                break

    return selected


class RetrievalOrchestrator:
    def __init__(self) -> None:
        self.vector_index: VectorStoreIndex | None = None
        self.sentence_window_index: VectorStoreIndex | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.sentence_window_postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _build_faiss_store(self, embedding_dim: int) -> FaissVectorStore:
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        return FaissVectorStore(faiss_index=faiss_index)

    def build_indexes(self, vector_nodes: list, sentence_window_nodes: list) -> None:
        if _LEGACY_INDEX_DIR.exists():
            shutil.rmtree(_LEGACY_INDEX_DIR)

        _clear_index_dir(VECTOR_INDEX_DIR)
        _clear_index_dir(SENTENCE_INDEX_DIR)

        sample_embedding = Settings.embed_model.get_text_embedding("dimension probe")
        dim = len(sample_embedding)

        vector_store = self._build_faiss_store(dim)
        vector_storage = StorageContext.from_defaults(vector_store=vector_store)
        self.vector_index = VectorStoreIndex(nodes=vector_nodes, storage_context=vector_storage)
        self.vector_index.storage_context.persist(persist_dir=str(VECTOR_INDEX_DIR))

        sentence_store = self._build_faiss_store(dim)
        sentence_storage = StorageContext.from_defaults(vector_store=sentence_store)
        self.sentence_window_index = VectorStoreIndex(
            nodes=sentence_window_nodes,
            storage_context=sentence_storage,
        )
        self.sentence_window_index.storage_context.persist(persist_dir=str(SENTENCE_INDEX_DIR))

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=vector_nodes,
            similarity_top_k=min(TOP_K_PER_RETRIEVER, max(1, len(vector_nodes))),
        )

    def ensure_loaded(self) -> None:
        if self.vector_index is not None and self.sentence_window_index is not None:
            return
        if not VECTOR_INDEX_DIR.exists() or not SENTENCE_INDEX_DIR.exists():
            raise RuntimeError("Knowledge base is empty. Run /ingest first.")

        vector_ctx = StorageContext.from_defaults(
            persist_dir=str(VECTOR_INDEX_DIR),
            vector_store=FaissVectorStore.from_persist_dir(str(VECTOR_INDEX_DIR)),
        )
        self.vector_index = load_index_from_storage(vector_ctx)

        sentence_ctx = StorageContext.from_defaults(
            persist_dir=str(SENTENCE_INDEX_DIR),
            vector_store=FaissVectorStore.from_persist_dir(str(SENTENCE_INDEX_DIR)),
        )
        self.sentence_window_index = load_index_from_storage(sentence_ctx)

        all_vector_nodes = list(self.vector_index.docstore.docs.values())
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_vector_nodes,
            similarity_top_k=min(TOP_K_PER_RETRIEVER, max(1, len(all_vector_nodes))),
        )

    def _rrf(self, result_buckets: dict[str, list[NodeWithScore]]) -> list[dict[str, Any]]:
        fused_scores: dict[str, dict[str, Any]] = {}
        for retriever_name, nodes in result_buckets.items():
            for rank, node_with_score in enumerate(nodes, start=1):
                node_id = node_with_score.node.node_id
                entry = fused_scores.setdefault(
                    node_id,
                    {
                        "node": node_with_score.node,
                        "rrf_score": 0.0,
                        "retriever_types": set(),
                    },
                )
                entry["rrf_score"] += 1.0 / (RRF_K + rank)
                entry["retriever_types"].add(retriever_name)

        fused = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return fused[:TOP_K_FUSED]

    def query(self, question: str) -> tuple[list[RetrievedCitation], float]:
        self.ensure_loaded()
        assert self.vector_index is not None
        assert self.sentence_window_index is not None
        assert self.bm25_retriever is not None

        start = perf_counter()
        top_k = min(TOP_K_PER_RETRIEVER, max(1, len(self.vector_index.docstore.docs)))

        vector_results = self.vector_index.as_retriever(similarity_top_k=top_k).retrieve(question)
        bm25_results = self.bm25_retriever.retrieve(question)
        sentence_results = self.sentence_window_index.as_retriever(similarity_top_k=top_k).retrieve(
            question
        )
        sentence_results = self.sentence_window_postprocessor.postprocess_nodes(sentence_results)

        fused = self._rrf(
            {
                "vector": vector_results,
                "bm25": bm25_results,
                "sentence_window": sentence_results,
            }
        )

        pairs = [[question, item["node"].get_content()] for item in fused]
        rerank_scores = self.cross_encoder.predict(pairs) if pairs else []
        for item, score in zip(fused, rerank_scores):
            item["rerank_score"] = float(score)

        reranked = sorted(fused, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        final_ranked = _select_diverse_citations(reranked, question)
        elapsed = perf_counter() - start

        citations: list[RetrievedCitation] = []
        for item in final_ranked:
            node = item["node"]
            source_name = node.metadata.get("source_name", "unknown")
            excerpt = node.get_content()[:200]
            retriever_types = sorted(item["retriever_types"])
            primary_retriever = "+".join(retriever_types) if retriever_types else "vector"
            citations.append(
                RetrievedCitation(
                    source_name=source_name,
                    text_excerpt=excerpt,
                    score=float(item.get("rerank_score", 0.0)),
                    retriever_type=primary_retriever,
                    full_text=node.get_content(),
                )
            )
        return citations, elapsed

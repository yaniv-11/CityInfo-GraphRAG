from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import exp
from typing import Protocol

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from torch.nn import Identity

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_THRESHOLD = 0.7


class CrossEncoderLike(Protocol):
    def predict(self, sentences: list[list[str]]) -> Iterable[float]: ...


def build_bm25_retriever(documents: Sequence[Document], top_k: int) -> BM25Retriever:
    if not documents:
        raise ValueError("Cannot build a BM25 retriever without documents.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    retriever = BM25Retriever.from_documents(list(documents))
    retriever.k = top_k
    return retriever


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, activation_fn=Identity())


def merge_documents(*groups: Iterable[Document]) -> list[Document]:
    merged = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for document in group:
            chunk_id = document.metadata.get("chunk_id")
            key = ("id", str(chunk_id)) if chunk_id is not None else ("text", document.page_content)
            if key not in seen:
                seen.add(key)
                merged.append(document)
    return merged


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + exp(-logit))
    exp_logit = exp(logit)
    return exp_logit / (1.0 + exp_logit)


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    reranker: CrossEncoderLike,
    threshold: float = RERANKER_THRESHOLD,
) -> list[tuple[Document, float]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not documents:
        return []

    logits = list(reranker.predict([[query, document.page_content] for document in documents]))
    if len(logits) != len(documents):
        raise ValueError("Cross-encoder returned a different number of scores than input chunks.")

    ranked = [
        (document, _sigmoid(float(logit)))
        for document, logit in zip(documents, logits, strict=True)
    ]
    ranked.sort(key=lambda result: result[1], reverse=True)
    return [result for result in ranked if result[1] >= threshold]


def augment_with_triplets(
    documents: Sequence[Document],
    triplets: Sequence[Mapping[str, str]],
) -> list[str]:
    augmented = []
    for document in documents:
        chunk_text = document.page_content
        chunk_lower = chunk_text.lower()
        related = [
            f"{triplet['subject']} {triplet['relation']} {triplet['object']}"
            for triplet in triplets
            if triplet.get("subject", "").lower() in chunk_lower
            or triplet.get("object", "").lower() in chunk_lower
        ][:10]

        if related:
            chunk_text += "\n\nRelated Info:\n" + "\n".join(related)
        augmented.append(chunk_text)
    return augmented
